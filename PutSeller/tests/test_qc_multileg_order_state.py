"""
PutSeller QuantConnect multi-leg entry order-state-machine regression tests.

Targets (both loaded directly from disk, no package/install required):
  - lean_workspace/PutSeller_IronCondor/main.py
  - PutSeller/quantconnect/main.py

Both files do `from AlgorithmImports import *` (QuantConnect's LEAN SDK, not
installed in this workspace/CI). A minimal fake AlgorithmImports module is
injected into sys.modules before loading the target files via importlib, so
the pure-Python order-state logic can be unit tested with no QC/LEAN install
and no real broker/API calls -- everything here is mocked.

State-machine cases covered (per underlying+side pending entry):
  - both legs FILLED                      -> spread tracked, cleanly closed
  - neither leg filled (CANCELED/INVALID) -> dropped cleanly, nothing naked
  - one leg FILLED, other CANCELED        -> orphan: blocked + closed + alert
  - one leg FILLED, other INVALID         -> orphan (same as above)
  - one leg PARTIALLY_FILLED, other CANCELED -> orphan on the filled quantity
  - cancel/fill race resolving to both FILLED -> recovered, NOT dropped
  - cancel/fill race resolving asymmetrically -> orphan next cycle
  - stale pending order, no fill yet      -> cancel requested, NOT deleted
                                              this cycle (race safety)
  - unknown/ambiguous order status        -> fail-safe: no action, alert only
  - blocked_sides gate                    -> new entries suppressed
"""
import importlib.util
import json
import os
import sys
import types
import unittest
from datetime import datetime, timedelta as _timedelta
from unittest.mock import MagicMock


# ── Fake AlgorithmImports (minimal QC/LEAN SDK stub) ─────────────────

class OrderStatus:
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    INVALID = "INVALID"
    NONE = "NONE"
    CANCEL_PENDING = "CANCEL_PENDING"
    UPDATE_SUBMITTED = "UPDATE_SUBMITTED"


# A status value that is NOT part of the recognized set above -- simulates an
# unknown/ambiguous broker response the port has never seen before.
UNKNOWN_STATUS = "SOME_FUTURE_STATUS_NOT_YET_SUPPORTED"


class Leg:
    @staticmethod
    def create(symbol, qty):
        return {"symbol": symbol, "qty": qty}


class OptionStrategies:
    @staticmethod
    def bull_put_spread(*a, **k):
        return "BULL_PUT_SPREAD"

    @staticmethod
    def bear_call_spread(*a, **k):
        return "BEAR_CALL_SPREAD"


class OptionRight:
    PUT = "put"
    CALL = "call"


class Resolution:
    MINUTE = "minute"


class _FakeEvent:
    def __iadd__(self, handler):
        return self


class TradeBarConsolidator:
    def __init__(self, *a, **k):
        self.data_consolidated = _FakeEvent()


class RollingWindow:
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, size):
        self.size = size


class QCAlgorithm:
    """Empty base -- tests bypass __init__/initialize() via object.__new__."""
    pass


def _make_fake_algorithm_imports():
    mod = types.ModuleType("AlgorithmImports")
    mod.QCAlgorithm = QCAlgorithm
    mod.OrderStatus = OrderStatus
    mod.Leg = Leg
    mod.OptionStrategies = OptionStrategies
    mod.OptionRight = OptionRight
    mod.Resolution = Resolution
    mod.TradeBarConsolidator = TradeBarConsolidator
    mod.RollingWindow = RollingWindow
    mod.timedelta = _timedelta
    return mod


def _load_algo_module(path, name):
    sys.modules["AlgorithmImports"] = _make_fake_algorithm_imports()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LEAN_PATH = os.path.join(_REPO_ROOT, "lean_workspace", "PutSeller_IronCondor", "main.py")
_QC_PATH = os.path.join(_REPO_ROOT, "PutSeller", "quantconnect", "main.py")

lean_main = _load_algo_module(_LEAN_PATH, "putseller_lean_main")
qc_main = _load_algo_module(_QC_PATH, "putseller_qc_main")


# ── Test doubles ──────────────────────────────────────────────────

class FakeTicket:
    """Minimal stand-in for a QC OrderTicket."""

    def __init__(self, status, quantity_filled=0, average_fill_price=0.0, order_id=1):
        self.status = status
        self.quantity_filled = quantity_filled
        self.average_fill_price = average_fill_price
        self.order_id = order_id

    def cancel(self, reason=""):
        # Mirrors real broker semantics: canceling an already-FILLED order is
        # a no-op -- this is what makes the cancel/fill race test meaningful.
        if self.status != OrderStatus.FILLED:
            self.status = OrderStatus.CANCELED


def make_engine(module, cls_name="PutSellerIronCondorAlgorithm"):
    cls = getattr(module, cls_name)
    eng = object.__new__(cls)
    eng.spreads = {"XYZ": []}
    eng.last_exit = {"XYZ": {"put": None, "call": None}}
    eng.pending_entries = {}
    eng.blocked_sides = {}
    eng.incidents = []
    eng.orphan_closes = {}
    eng._incident_seq = 0
    eng.time = datetime(2026, 1, 1, 12, 0, 0)
    eng.debug = MagicMock()
    eng.error = MagicMock()
    eng.market_order = MagicMock()
    eng.combo_limit_order = MagicMock()
    eng.object_store = MagicMock()
    eng.transactions = MagicMock()
    return eng


def make_pending(short_ticket, long_ticket, ticker="XYZ", right="put",
                  placed_minutes_ago=0, **overrides):
    pend = {
        "tickets": (short_ticket, long_ticket),
        "strategy": "BULL_PUT_SPREAD",
        "short_symbol": f"{ticker}_SHORT",
        "long_symbol": f"{ticker}_LONG",
        "short_strike": 350.0,
        "expiry": datetime(2026, 3, 1),
        "mid_credit": 1.50,
        "target_credit": 1.30,
        "fraction": 0.85,
        "placed_time": datetime(2026, 1, 1, 12, 0, 0) - _timedelta(minutes=placed_minutes_ago),
    }
    pend.update(overrides)
    return (ticker, right), pend


def make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT",
                          ticker="XYZ", right="put"):
    """Drive a real orphan-leg incident into existence via the actual
    _handle_orphan_leg() code path (one leg FILLED, the other CANCELED), with
    a controllable FakeTicket standing in for the resulting ORPHAN_LEG_CLOSE
    order. Returns the incident dict (the same object referenced in
    eng.incidents / eng.blocked_sides / eng.orphan_closes).
    """
    eng.market_order.return_value = close_ticket
    if symbol.endswith("_SHORT"):
        short_t = FakeTicket(OrderStatus.FILLED, quantity_filled=naked_qty, average_fill_price=3.50, order_id=201)
        long_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=202)
    else:
        short_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=201)
        long_t = FakeTicket(OrderStatus.FILLED, quantity_filled=naked_qty, average_fill_price=3.50, order_id=202)
    key, pend = make_pending(short_t, long_t, ticker=ticker, right=right)
    eng.pending_entries[key] = pend

    eng._process_pending_entries()

    return eng.incidents[-1]


class _FakeChain:
    underlying = type("U", (), {"price": 100.0})()

    def __iter__(self):
        return iter([])


# ── Shared state-machine test cases (run against BOTH ported files) ──

class _MultiLegOrderStateMachineTests:
    module = None

    def _engine(self):
        return make_engine(self.module)

    def test_both_legs_filled_creates_spread(self):
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.FILLED, quantity_filled=-1, average_fill_price=3.50)
        long_t = FakeTicket(OrderStatus.FILLED, quantity_filled=1, average_fill_price=1.20)
        key, pend = make_pending(short_t, long_t)
        eng.pending_entries[key] = pend

        eng._process_pending_entries()

        self.assertNotIn(key, eng.pending_entries)
        self.assertEqual(len(eng.spreads["XYZ"]), 1)
        self.assertAlmostEqual(eng.spreads["XYZ"][0]["credit"], 2.30)
        self.assertEqual(eng.blocked_sides, {})
        self.assertEqual(eng.incidents, [])
        eng.market_order.assert_not_called()
        eng.error.assert_not_called()

    def test_neither_leg_filled_drops_cleanly(self):
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0)
        long_t = FakeTicket(OrderStatus.INVALID, quantity_filled=0)
        key, pend = make_pending(short_t, long_t)
        eng.pending_entries[key] = pend

        eng._process_pending_entries()

        self.assertNotIn(key, eng.pending_entries)
        self.assertEqual(eng.spreads["XYZ"], [])
        self.assertEqual(eng.blocked_sides, {})
        self.assertEqual(eng.incidents, [])
        eng.market_order.assert_not_called()
        eng.error.assert_not_called()

    def test_one_leg_filled_other_canceled_is_orphan(self):
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.FILLED, quantity_filled=-1, average_fill_price=3.50, order_id=101)
        long_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=102)
        key, pend = make_pending(short_t, long_t)
        eng.pending_entries[key] = pend

        eng._process_pending_entries()

        self.assertNotIn(key, eng.pending_entries)
        self.assertEqual(eng.spreads["XYZ"], [])  # never tracked as a real spread
        self.assertIn(key, eng.blocked_sides)      # new entries on this side now blocked
        self.assertEqual(len(eng.incidents), 1)
        incident = eng.incidents[0]
        self.assertEqual(incident["short_order_id"], 101)
        self.assertEqual(incident["long_order_id"], 102)
        self.assertEqual(incident["short_quantity_filled"], -1)
        self.assertEqual(incident["long_quantity_filled"], 0)
        eng.error.assert_called()
        # closes EXACTLY the filled short leg's quantity, opposite sign, nothing on the long leg
        eng.market_order.assert_called_once_with(pend["short_symbol"], 1, tag="ORPHAN_LEG_CLOSE")
        eng.object_store.save.assert_called()  # incident + running state snapshot both persisted

    def test_one_leg_filled_other_invalid_is_orphan(self):
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.INVALID, quantity_filled=0)
        long_t = FakeTicket(OrderStatus.FILLED, quantity_filled=1, average_fill_price=1.20)
        key, pend = make_pending(short_t, long_t)
        eng.pending_entries[key] = pend

        eng._process_pending_entries()

        self.assertNotIn(key, eng.pending_entries)
        self.assertIn(key, eng.blocked_sides)
        eng.market_order.assert_called_once_with(pend["long_symbol"], -1, tag="ORPHAN_LEG_CLOSE")
        eng.error.assert_called()

    def test_partial_fill_on_one_leg_is_orphan(self):
        eng = self._engine()
        # requested short qty larger than what filled before its sibling canceled
        short_t = FakeTicket(OrderStatus.PARTIALLY_FILLED, quantity_filled=-1, average_fill_price=3.50)
        long_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0)
        key, pend = make_pending(short_t, long_t)
        eng.pending_entries[key] = pend

        eng._process_pending_entries()

        self.assertNotIn(key, eng.pending_entries)
        self.assertIn(key, eng.blocked_sides)
        eng.market_order.assert_called_once_with(pend["short_symbol"], 1, tag="ORPHAN_LEG_CLOSE")

    def test_cancel_fill_race_recovers_to_filled_next_cycle(self):
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0)
        long_t = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0)
        key, pend = make_pending(short_t, long_t, placed_minutes_ago=999)
        eng.pending_entries[key] = pend

        # Cycle 1: stale timeout requests a cancel on both legs, but must NOT
        # delete tracking -- cancel() can race with an actual fill.
        eng._process_pending_entries()
        self.assertIn(key, eng.pending_entries)
        eng.error.assert_not_called()

        # The broker actually filled both legs an instant before the cancel
        # took effect -- a genuine cancel/fill race.
        short_t.status = OrderStatus.FILLED
        short_t.quantity_filled = -1
        short_t.average_fill_price = 3.50
        long_t.status = OrderStatus.FILLED
        long_t.quantity_filled = 1
        long_t.average_fill_price = 1.20

        # Cycle 2: re-evaluated with fresh state -- correctly recognized as a
        # completed spread, NOT dropped and NOT treated as orphaned.
        eng._process_pending_entries()
        self.assertNotIn(key, eng.pending_entries)
        self.assertEqual(len(eng.spreads["XYZ"]), 1)
        self.assertEqual(eng.blocked_sides, {})
        self.assertEqual(eng.incidents, [])
        eng.market_order.assert_not_called()

    def test_cancel_fill_race_one_leg_orphaned_next_cycle(self):
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0)
        long_t = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0)
        key, pend = make_pending(short_t, long_t, placed_minutes_ago=999)
        eng.pending_entries[key] = pend

        eng._process_pending_entries()  # requests cancel, doesn't delete
        self.assertIn(key, eng.pending_entries)

        # Race resolves asymmetrically: short leg fills, long leg's cancel succeeds.
        short_t.status = OrderStatus.FILLED
        short_t.quantity_filled = -1
        short_t.average_fill_price = 3.50
        long_t.status = OrderStatus.CANCELED

        eng._process_pending_entries()
        self.assertNotIn(key, eng.pending_entries)
        self.assertIn(key, eng.blocked_sides)
        eng.market_order.assert_called_once_with(pend["short_symbol"], 1, tag="ORPHAN_LEG_CLOSE")

    def test_stale_pending_with_no_fill_does_not_delete_immediately(self):
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0)
        long_t = FakeTicket(OrderStatus.NEW, quantity_filled=0)
        key, pend = make_pending(short_t, long_t, placed_minutes_ago=999)
        eng.pending_entries[key] = pend

        eng._process_pending_entries()

        self.assertIn(key, eng.pending_entries)  # still tracked, never silently dropped
        self.assertEqual(eng.blocked_sides, {})
        eng.error.assert_not_called()

    def test_unknown_ambiguous_status_is_fail_safe(self):
        eng = self._engine()
        short_t = FakeTicket(UNKNOWN_STATUS, quantity_filled=0)
        long_t = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0)
        key, pend = make_pending(short_t, long_t)
        eng.pending_entries[key] = pend

        eng._process_pending_entries()

        self.assertIn(key, eng.pending_entries)  # never acted on an unrecognized status
        self.assertEqual(eng.blocked_sides, {})
        self.assertEqual(eng.incidents, [])
        eng.market_order.assert_not_called()
        eng.error.assert_called()  # still alerted, just took no destructive action

    # ── ORPHAN_LEG_CLOSE order tracking/reconciliation ──────────────
    # The close order submitted by _handle_orphan_leg() must itself be
    # tracked and its own broker-confirmed fill state verified -- submission
    # succeeding is not the same as the naked leg actually being closed.

    def test_orphan_close_fully_fills_resolves_incident(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0, order_id=301)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        oc_key = (incident["incident_id"], "XYZ_SHORT")
        self.assertIn(oc_key, eng.orphan_closes)
        self.assertEqual(incident["naked_legs"]["XYZ_SHORT"], -1)
        self.assertEqual(incident["status"], "OPEN")

        # Broker confirms the close fully filled (buy back 1 to close short qty -1).
        close_ticket.status = OrderStatus.FILLED
        close_ticket.quantity_filled = 1

        eng._reconcile_orphan_closes()

        self.assertNotIn(oc_key, eng.orphan_closes)
        self.assertNotIn("XYZ_SHORT", incident["naked_legs"])
        self.assertEqual(incident["status"], "RESOLVED")
        self.assertIn(incident, eng.incidents)  # record retained, never deleted

    def test_orphan_close_partially_fills_keeps_incident_open(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0, order_id=302)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-3, symbol="XYZ_SHORT")
        oc_key = (incident["incident_id"], "XYZ_SHORT")

        close_ticket.status = OrderStatus.PARTIALLY_FILLED
        close_ticket.quantity_filled = 1  # only 1 of 3 bought back so far

        eng._reconcile_orphan_closes()

        self.assertIn(oc_key, eng.orphan_closes)
        self.assertEqual(incident["naked_legs"]["XYZ_SHORT"], -2)  # 2 contracts still naked
        self.assertEqual(incident["status"], "OPEN")
        self.assertIn(incident, eng.incidents)

    def test_residual_naked_quantity_remains_after_partial_fill(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.PARTIALLY_FILLED, quantity_filled=2, order_id=303)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-5, symbol="XYZ_SHORT")

        eng._reconcile_orphan_closes()

        self.assertEqual(incident["naked_legs"]["XYZ_SHORT"], -3)
        self.assertEqual(incident["status"], "OPEN")

    def test_orphan_close_canceled_resubmits_and_alerts(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=304)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        calls_before = eng.market_order.call_count
        errors_before = eng.error.call_count
        new_ticket = FakeTicket(OrderStatus.SUBMITTED, order_id=305)
        eng.market_order.return_value = new_ticket

        eng._reconcile_orphan_closes()

        self.assertEqual(eng.market_order.call_count, calls_before + 1)
        eng.market_order.assert_called_with("XYZ_SHORT", 1, tag="ORPHAN_LEG_CLOSE")
        self.assertEqual(incident["status"], "OPEN")
        self.assertIn(incident, eng.incidents)
        new_oc_key = (incident["incident_id"], "XYZ_SHORT")
        self.assertIn(new_oc_key, eng.orphan_closes)
        self.assertIs(eng.orphan_closes[new_oc_key]["ticket"], new_ticket)
        self.assertGreater(eng.error.call_count, errors_before)  # persistent CRITICAL alert

    def test_orphan_close_rejected_invalid_resubmits_and_alerts(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.INVALID, quantity_filled=0, order_id=306)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        eng.market_order.return_value = FakeTicket(OrderStatus.SUBMITTED, order_id=307)

        eng._reconcile_orphan_closes()

        self.assertEqual(incident["status"], "OPEN")
        eng.market_order.assert_called_with("XYZ_SHORT", 1, tag="ORPHAN_LEG_CLOSE")
        eng.error.assert_called()

    def test_orphan_close_unknown_status_does_not_assume_success(self):
        eng = self._engine()
        close_ticket = FakeTicket(UNKNOWN_STATUS, quantity_filled=0, order_id=308)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        oc_key = (incident["incident_id"], "XYZ_SHORT")
        calls_before = eng.market_order.call_count

        eng._reconcile_orphan_closes()

        self.assertIn(oc_key, eng.orphan_closes)  # nothing resolved or resubmitted
        self.assertEqual(incident["naked_legs"]["XYZ_SHORT"], -1)  # left untouched
        self.assertEqual(incident["status"], "OPEN")
        self.assertEqual(eng.market_order.call_count, calls_before)  # no resubmission
        eng.error.assert_called()

    def test_duplicate_close_event_delivery_is_idempotent(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.FILLED, quantity_filled=1, order_id=309)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        oc_key = (incident["incident_id"], "XYZ_SHORT")

        eng._reconcile_orphan_closes()
        self.assertNotIn(oc_key, eng.orphan_closes)
        self.assertEqual(incident["status"], "RESOLVED")
        calls_after_first = eng.market_order.call_count

        # Same FILLED ticket state "delivered" again on a later cycle -- must
        # be a safe no-op: already resolved, nothing left tracked for this key.
        eng._reconcile_orphan_closes()

        self.assertEqual(incident["status"], "RESOLVED")
        self.assertEqual(eng.market_order.call_count, calls_after_first)  # no duplicate resubmission
        self.assertEqual(eng.incidents.count(incident), 1)  # never duplicated/re-added

    def test_incident_persists_until_broker_verified_flat(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=310)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        eng.market_order.return_value = FakeTicket(OrderStatus.SUBMITTED, order_id=311)

        eng._reconcile_orphan_closes()  # canceled -> resubmit, stays OPEN, never deleted
        self.assertIn(incident, eng.incidents)
        self.assertEqual(incident["status"], "OPEN")

        # The resubmitted close order now confirms filled.
        new_ticket = eng.orphan_closes[(incident["incident_id"], "XYZ_SHORT")]["ticket"]
        new_ticket.status = OrderStatus.FILLED
        new_ticket.quantity_filled = 1

        eng._reconcile_orphan_closes()

        self.assertEqual(incident["status"], "RESOLVED")
        self.assertIn(incident, eng.incidents)  # persisted record retained even after resolution

    # ── Retry cap + restart recovery ─────────────────────────────
    # A close order itself can keep failing (CANCELED/INVALID) forever if
    # left unbounded -- these tests enforce a hard cap, escalation instead of
    # silent resolution, and that unresolved state survives a process
    # restart without ever assuming a prior close succeeded.

    def test_retry_count_increments_only_after_confirmed_submission(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=401)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        oc_key = (incident["incident_id"], "XYZ_SHORT")
        self.assertEqual(eng.orphan_closes[oc_key]["attempts"], 1)  # initial submission = attempt 1
        self.assertEqual(eng.orphan_closes[oc_key]["order_ids"], [401])

        new_ticket = FakeTicket(OrderStatus.SUBMITTED, order_id=402)
        eng.market_order.return_value = new_ticket

        eng._reconcile_orphan_closes()  # CANCELED, under cap -> confirmed resubmission

        oc = eng.orphan_closes[oc_key]
        self.assertEqual(oc["attempts"], 2)
        self.assertEqual(oc["order_ids"], [401, 402])
        self.assertIs(oc["ticket"], new_ticket)

    def test_retry_cap_escalates(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=410)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        oc_key = (incident["incident_id"], "XYZ_SHORT")

        # Drive attempts up to the cap -- each still-CANCELED retry resubmits.
        for i in range(eng.MAX_ORPHAN_CLOSE_ATTEMPTS - 1):
            eng.market_order.return_value = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=411 + i)
            eng._reconcile_orphan_closes()

        self.assertEqual(eng.orphan_closes[oc_key]["attempts"], eng.MAX_ORPHAN_CLOSE_ATTEMPTS)
        self.assertEqual(incident["status"], "OPEN")  # not yet escalated

        calls_before = eng.market_order.call_count
        eng._reconcile_orphan_closes()  # cap-th close order also CANCELED

        self.assertEqual(eng.market_order.call_count, calls_before)  # no further resubmission
        self.assertEqual(incident["status"], "ESCALATED")
        self.assertIsNotNone(eng.orphan_closes[oc_key]["escalation_reason"])

    def test_escalated_incidents_remain_blocked(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=420)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        key = ("XYZ", "put")

        for i in range(eng.MAX_ORPHAN_CLOSE_ATTEMPTS):
            eng.market_order.return_value = FakeTicket(OrderStatus.CANCELED, quantity_filled=0, order_id=421 + i)
            eng._reconcile_orphan_closes()

        self.assertEqual(incident["status"], "ESCALATED")
        self.assertIn(key, eng.blocked_sides)
        self.assertIs(eng.blocked_sides[key], incident)
        self.assertIn(incident, eng.incidents)

        eng._trend = MagicMock(return_value=None)
        eng._spy_vol_spiked = MagicMock(return_value=False)
        eng._find_credit_spread = MagicMock()
        eng._execute_spread = MagicMock()
        eng._in_cooldown = MagicMock(return_value=False)
        eng._total_open = MagicMock(return_value=0)

        eng._open_new_spreads("XYZ", _FakeChain())

        # blocked_sides only gates the exact (ticker, right) that escalated --
        # the put side here never reaches _find_credit_spread/_execute_spread
        # (the call side, unaffected, is free to proceed as normal).
        self.assertEqual(eng._execute_spread.call_args.args[2], "call")
        for call in eng._find_credit_spread.call_args_list:
            self.assertEqual(call.args[3], OptionRight.CALL)

    def test_unknown_status_does_not_increment_or_resubmit(self):
        eng = self._engine()
        close_ticket = FakeTicket(UNKNOWN_STATUS, quantity_filled=0, order_id=430)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")
        oc_key = (incident["incident_id"], "XYZ_SHORT")
        attempts_before = eng.orphan_closes[oc_key]["attempts"]
        calls_before = eng.market_order.call_count

        eng._reconcile_orphan_closes()

        self.assertEqual(eng.orphan_closes[oc_key]["attempts"], attempts_before)
        self.assertEqual(eng.market_order.call_count, calls_before)
        self.assertEqual(incident["status"], "OPEN")
        eng.error.assert_called()

    def test_partial_fill_does_not_create_duplicate_close_orders(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.PARTIALLY_FILLED, quantity_filled=1, order_id=440)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-3, symbol="XYZ_SHORT")
        oc_key = (incident["incident_id"], "XYZ_SHORT")
        calls_before = eng.market_order.call_count

        eng._reconcile_orphan_closes()
        eng._reconcile_orphan_closes()  # multiple cycles while still partially filled

        self.assertEqual(eng.market_order.call_count, calls_before)  # no new close order submitted
        self.assertIs(eng.orphan_closes[oc_key]["ticket"], close_ticket)  # same ticket still tracked
        self.assertEqual(incident["naked_legs"]["XYZ_SHORT"], -2)
        self.assertEqual(incident["status"], "OPEN")

    def test_restart_reloads_unresolved_incident_state(self):
        eng = self._engine()
        snapshot = {
            "incident_seq": 5,
            "incidents": [{
                "incident_id": 5, "ticker": "XYZ", "right": "put",
                "short_symbol": "XYZ_SHORT", "long_symbol": "XYZ_LONG",
                "status": "OPEN", "naked_legs": {"XYZ_SHORT": -1},
                "short_quantity_filled": -1, "long_quantity_filled": 0,
            }],
            "orphan_closes": [{
                "incident_id": 5, "symbol": "XYZ_SHORT", "naked_qty": -1,
                "order_ids": [501], "attempts": 1,
                "first_detected": "2026-01-01 12:00:00",
                "last_attempt_time": "2026-01-01 12:00:00",
                "last_status": OrderStatus.SUBMITTED, "escalation_reason": None,
            }],
        }
        eng.object_store.contains_key.return_value = True
        eng.object_store.read.return_value = json.dumps(snapshot)
        restored_ticket = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0, order_id=501)
        eng.transactions.get_order_ticket.return_value = restored_ticket

        eng._restore_incident_state()

        self.assertEqual(len(eng.incidents), 1)
        self.assertIn(("XYZ", "put"), eng.blocked_sides)
        oc_key = (5, "XYZ_SHORT")
        self.assertIn(oc_key, eng.orphan_closes)
        self.assertIs(eng.orphan_closes[oc_key]["ticket"], restored_ticket)
        self.assertEqual(eng._incident_seq, 5)
        eng.transactions.get_order_ticket.assert_called_with(501)

    def test_resolved_incidents_remain_in_incident_archive(self):
        eng = self._engine()
        close_ticket = FakeTicket(OrderStatus.SUBMITTED, quantity_filled=0, order_id=450)
        incident = make_orphan_incident(eng, close_ticket, naked_qty=-1, symbol="XYZ_SHORT")

        close_ticket.status = OrderStatus.FILLED
        close_ticket.quantity_filled = 1
        eng._reconcile_orphan_closes()

        self.assertEqual(incident["status"], "RESOLVED")
        self.assertIn(incident, eng.incidents)
        self.assertEqual(len(eng.incidents), 1)  # never removed, even after resolution

    def test_duplicate_events_after_restart_remain_idempotent(self):
        eng = self._engine()
        snapshot = {
            "incident_seq": 7,
            "incidents": [{
                "incident_id": 7, "ticker": "XYZ", "right": "put",
                "short_symbol": "XYZ_SHORT", "long_symbol": "XYZ_LONG",
                "status": "OPEN", "naked_legs": {"XYZ_SHORT": -1},
                "short_quantity_filled": -1, "long_quantity_filled": 0,
            }],
            "orphan_closes": [{
                "incident_id": 7, "symbol": "XYZ_SHORT", "naked_qty": -1,
                "order_ids": [601], "attempts": 1,
                "first_detected": "2026-01-01 12:00:00",
                "last_attempt_time": "2026-01-01 12:00:00",
                "last_status": OrderStatus.FILLED, "escalation_reason": None,
            }],
        }
        eng.object_store.contains_key.return_value = True
        eng.object_store.read.return_value = json.dumps(snapshot)
        restored_ticket = FakeTicket(OrderStatus.FILLED, quantity_filled=1, order_id=601)
        eng.transactions.get_order_ticket.return_value = restored_ticket

        eng._restore_incident_state()  # reconciles immediately -> residual 0 -> resolves

        incident = eng.incidents[0]
        self.assertEqual(incident["status"], "RESOLVED")
        self.assertNotIn((7, "XYZ_SHORT"), eng.orphan_closes)

        # The same FILLED event "redelivered" on a later cycle must be a no-op.
        eng._reconcile_orphan_closes()

        self.assertEqual(incident["status"], "RESOLVED")
        self.assertEqual(len(eng.incidents), 1)
        eng.market_order.assert_not_called()

    def test_blocked_sides_prevents_new_entry_scan(self):
        eng = self._engine()
        eng.blocked_sides[("XYZ", "put")] = {"reason": "test"}
        eng.blocked_sides[("XYZ", "call")] = {"reason": "test"}
        eng._trend = MagicMock(return_value=None)
        eng._spy_vol_spiked = MagicMock(return_value=False)
        eng._find_credit_spread = MagicMock()
        eng._execute_spread = MagicMock()
        eng._in_cooldown = MagicMock(return_value=False)
        eng._total_open = MagicMock(return_value=0)

        eng._open_new_spreads("XYZ", _FakeChain())

        eng._find_credit_spread.assert_not_called()
        eng._execute_spread.assert_not_called()


class LeanWorkspacePutSellerTests(_MultiLegOrderStateMachineTests, unittest.TestCase):
    module = lean_main

    def test_neither_leg_filled_drops_cleanly(self):
        """Override: this port relaxes/retries a clean (non-naked) double-
        cancel instead of dropping outright -- see test_clean_double_cancel_
        relaxes_price for that path. Confirm it's still not treated as an
        orphan (no block, no incident, no naked-leg close order)."""
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0)
        long_t = FakeTicket(OrderStatus.INVALID, quantity_filled=0)
        key, pend = make_pending(short_t, long_t)
        eng.pending_entries[key] = pend
        eng.combo_limit_order.return_value = (FakeTicket(OrderStatus.SUBMITTED), FakeTicket(OrderStatus.SUBMITTED))

        eng._process_pending_entries()

        self.assertEqual(eng.spreads["XYZ"], [])
        self.assertEqual(eng.blocked_sides, {})
        self.assertEqual(eng.incidents, [])
        eng.market_order.assert_not_called()
        eng.error.assert_not_called()

    def test_clean_double_cancel_relaxes_price(self):
        """This port retries a stale-canceled (but clean, non-naked) entry at
        a relaxed credit -- must still work once the orphan check passes."""
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0)
        long_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0)
        key, pend = make_pending(short_t, long_t, fraction=0.85)
        eng.pending_entries[key] = pend
        new_short_t = FakeTicket(OrderStatus.SUBMITTED)
        new_long_t = FakeTicket(OrderStatus.SUBMITTED)
        eng.combo_limit_order.return_value = (new_short_t, new_long_t)

        eng._process_pending_entries()

        self.assertIn(key, eng.pending_entries)
        self.assertLess(eng.pending_entries[key]["fraction"], 0.85)
        eng.combo_limit_order.assert_called_once()
        eng.market_order.assert_not_called()


class QuantConnectPutSellerTests(_MultiLegOrderStateMachineTests, unittest.TestCase):
    module = qc_main

    def test_clean_double_cancel_gives_up(self):
        """This port has no relax/retry -- a clean (non-naked) double-cancel
        just drops the pending entry outright."""
        eng = self._engine()
        short_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0)
        long_t = FakeTicket(OrderStatus.CANCELED, quantity_filled=0)
        key, pend = make_pending(short_t, long_t)
        eng.pending_entries[key] = pend

        eng._process_pending_entries()

        self.assertNotIn(key, eng.pending_entries)
        eng.combo_limit_order.assert_not_called()
        eng.market_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
