"""
PutSeller Order-Safety Tests
Regression tests for the audit findings:

  F1  get_order() returned enum reprs ("OrderStatus.FILLED") instead of raw
      strings ("filled"), breaking every fill check downstream.
  F2  Entry cancel/fill race: an order that filled during/just before the
      cancel was dropped, leaving a real-but-untracked position that the
      scanner then duplicated.
  F3  Entry idempotency: deterministic client_order_id so a retry after an
      ambiguous network failure is rejected by the broker as a duplicate.
  F4  Adoption fabricated credit for debit spreads (width*0.15) and used
      Windows-only C:\\ sibling paths — root cause of the stuck-CIFR incident.
  F5  P&L basis used scan-time mid instead of actual fill price.
  F6  One-sided quotes (bid=0 or ask=0) produced garbage mids that triggered
      false TAKE_PROFIT/STOP_LOSS exits.
  F7  Ghost positions (closed at broker, still tracked locally) retried a
      doomed close every 5 minutes forever; reconciliation removes them.
  F8  Close-failure circuit breaker: exponential backoff + reconcile after
      3 consecutive failures.
  F9  Pre-order gate: fresh risk check + broker leg-conflict check right
      before submission.

All external dependencies are mocked — no real API calls.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.api_client import PutSellerAPI
from core.put_engine import PutSellerEngine


# ── Helpers ──────────────────────────────────────────────

def make_engine():
    """Build a PutSellerEngine without running its heavy __init__."""
    eng = object.__new__(PutSellerEngine)
    cfg = MagicMock()
    cfg.ORDER_PREFIX = "ic_"
    cfg.TAKE_PROFIT_PCT = 0.50
    cfg.STOP_LOSS_MULT = 2.0
    cfg.MIN_DTE_EXIT = 14
    cfg.EMERGENCY_BUFFER_PCT = 0.02
    cfg.REGIME_BEAR_BUFFER_MULT = 1.5
    cfg.DELTA_EXIT_THRESHOLD = 0.40
    cfg.IV_SPIKE_EXIT_MULT = 1.5
    cfg.LEVERAGED_ETFS = set()
    cfg.LEVERAGED_QTY_CAP = 1
    cfg.BASE_URL = "https://paper-api.example.test"
    cfg.API_KEY = "test-key"
    cfg.API_SECRET = "test-secret"
    eng.config = cfg
    eng.api = MagicMock()
    eng.risk = MagicMock()
    eng.positions = {}
    eng._entry_attempts = {}
    eng._regime_flip_state = {}
    eng._current_regime = None
    eng._save_positions = MagicMock()
    eng._log_trade = MagicMock()
    eng.ml = None
    eng.meta = None
    eng.features = None
    eng.sentiment = None
    return eng


def make_spread(underlying="XYZ", credit=1.30):
    return {
        "underlying": underlying,
        "spread_type": "put",
        "credit": credit,
        "max_loss_per_contract": 370.0,
        "short_symbol": f"{underlying}260918P00350000",
        "long_symbol": f"{underlying}260918P00340000",
        "short_strike": 350.0,
        "long_strike": 340.0,
        "spread_width": 10.0,
        "expiration": "2026-09-18",
        "dte": 45,
        "roc_annual": 0.35,
    }


def make_position(pos_id="XYZ_2026-09-18_P350", **overrides):
    pos = {
        "underlying": "XYZ",
        "spread_type": "put",
        "short_symbol": "XYZ260918P00350000",
        "long_symbol": "XYZ260918P00340000",
        "short_strike": 350.0,
        "long_strike": 340.0,
        "spread_width": 10.0,
        "expiration": (date.today() + timedelta(days=45)).isoformat(),
        "qty": 3,
        "credit_per_share": 1.30,
        "total_credit": 390.0,
        "current_debit": 1.30,
        "open_date": date.today().isoformat(),
    }
    pos.update(overrides)
    return pos_id, pos


# ── F1: get_order enum regression ────────────────────────

class FakeEnum:
    """Mimics an alpaca-py enum: str() gives 'OrderStatus.FILLED', .value gives 'filled'."""
    def __init__(self, name, value):
        self._name = name
        self.value = value

    def __str__(self):
        return self._name


class TestGetOrderEnumFix(unittest.TestCase):
    def _make_api(self, status_enum, side_enum):
        api = object.__new__(PutSellerAPI)
        api.config = MagicMock()
        api._last_call_ts = time.monotonic()  # skip throttle sleep path
        import threading
        api._rate_lock = threading.Lock()
        order = MagicMock()
        order.id = "abc-123"
        order.symbol = "XYZ"
        order.side = side_enum
        order.status = status_enum
        order.qty = "3"
        order.filled_qty = "3"
        order.filled_avg_price = "1.25"
        api._trading = MagicMock()
        api._trading.get_order_by_id.return_value = order
        return api

    def test_status_is_raw_string_not_enum_repr(self):
        """F1: 'filled' — never 'OrderStatus.FILLED'."""
        api = self._make_api(FakeEnum("OrderStatus.FILLED", "filled"),
                             FakeEnum("OrderSide.BUY", "buy"))
        result = api.get_order("abc-123")
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["side"], "buy")
        # The exact downstream comparison that silently failed for 2 days:
        self.assertTrue(result["status"] == "filled")

    def test_none_status_preserved(self):
        api = self._make_api(None, None)
        result = api.get_order("abc-123")
        self.assertIsNone(result["status"])
        self.assertIsNone(result["side"])

    def test_filled_avg_price_is_float(self):
        api = self._make_api(FakeEnum("OrderStatus.FILLED", "filled"),
                             FakeEnum("OrderSide.SELL", "sell"))
        result = api.get_order("abc-123")
        self.assertEqual(result["filled_avg_price"], 1.25)


# ── F3: entry idempotency keys ───────────────────────────

class TestIdempotencyKey(unittest.TestCase):
    def test_same_inputs_same_key(self):
        """F3: identical legs+qty on the same day must reuse the same key."""
        eng = make_engine()
        spread = make_spread()
        self.assertEqual(eng._entry_idempotency_key(spread, 3),
                         eng._entry_idempotency_key(spread, 3))

    def test_key_has_order_prefix(self):
        eng = make_engine()
        key = eng._entry_idempotency_key(make_spread(), 3)
        self.assertTrue(key.startswith("ic_"))

    def test_qty_changes_key(self):
        eng = make_engine()
        spread = make_spread()
        self.assertNotEqual(eng._entry_idempotency_key(spread, 3),
                            eng._entry_idempotency_key(spread, 2))

    def test_attempt_counter_changes_key(self):
        """After a CONFIRMED cancel/reject, the next attempt gets a new key."""
        eng = make_engine()
        spread = make_spread()
        key1 = eng._entry_idempotency_key(spread, 3)
        legs = f"{spread['short_symbol']}|{spread['long_symbol']}|3"
        eng._entry_attempts[legs] = 1  # simulate confirmed rejection
        key2 = eng._entry_idempotency_key(spread, 3)
        self.assertNotEqual(key1, key2)


# ── F9: pre-order gate ───────────────────────────────────

class TestPreOrderGate(unittest.TestCase):
    def _gate_setup(self, broker_legs):
        eng = make_engine()
        eng.api.get_account.return_value = {"equity": 100_000.0}
        eng.risk.can_open_position.return_value = (True, "ok")
        eng.api.get_option_positions.return_value = broker_legs
        return eng

    def test_blocks_when_short_leg_already_at_broker(self):
        """F9/F2: a leg already at the broker means an untracked fill — block."""
        spread = make_spread()
        eng = self._gate_setup({spread["short_symbol"]: -3})
        ok, reason = eng._pre_order_gate(spread, 3)
        self.assertFalse(ok)
        self.assertIn("already exists at broker", reason)

    def test_blocks_when_long_leg_already_at_broker(self):
        spread = make_spread()
        eng = self._gate_setup({spread["long_symbol"]: 3})
        ok, _ = eng._pre_order_gate(spread, 3)
        self.assertFalse(ok)

    def test_blocks_when_broker_positions_unverifiable(self):
        """None (API failure) must NOT be treated like {} (flat account)."""
        eng = self._gate_setup(None)
        ok, reason = eng._pre_order_gate(make_spread(), 3)
        self.assertFalse(ok)
        self.assertIn("could not verify", reason)

    def test_blocks_when_fresh_risk_check_fails(self):
        eng = self._gate_setup({})
        eng.risk.can_open_position.return_value = (False, "daily loss limit")
        ok, reason = eng._pre_order_gate(make_spread(), 3)
        self.assertFalse(ok)
        self.assertIn("daily loss limit", reason)

    def test_blocks_when_account_fetch_fails(self):
        eng = make_engine()
        eng.api.get_account.side_effect = RuntimeError("timeout")
        ok, _ = eng._pre_order_gate(make_spread(), 3)
        self.assertFalse(ok)

    def test_passes_when_all_clear(self):
        eng = self._gate_setup({})
        ok, reason = eng._pre_order_gate(make_spread(), 3)
        self.assertTrue(ok)
        # Risk check ran against FRESH equity
        eng.risk.update_allocation.assert_called_once_with(100_000.0)


# ── F2 + F5: execute_spread fill handling ────────────────

@patch("core.put_engine._time.sleep", lambda *_: None)
class TestExecuteSpread(unittest.TestCase):
    def _exec_setup(self):
        eng = make_engine()
        eng.risk.size_position.return_value = 3
        eng._pre_order_gate = MagicMock(return_value=(True, "OK"))
        eng.api.submit_credit_spread.return_value = "order-1"
        return eng

    def test_cancel_race_fill_is_tracked(self):
        """F2: order fills during the cancel — position MUST be tracked."""
        eng = self._exec_setup()
        calls = {"n": 0}

        def get_order(_):
            calls["n"] += 1
            if calls["n"] <= 24:  # entire 120s wait loop: not filled
                return {"status": "new", "filled_avg_price": 0}
            # post-cancel re-check: it actually filled
            return {"status": "filled", "filled_avg_price": 1.11}

        eng.api.get_order.side_effect = get_order
        result = eng._execute_spread(make_spread(), capital_in_use=0)
        self.assertTrue(result)
        self.assertEqual(len(eng.positions), 1)
        pos = next(iter(eng.positions.values()))
        self.assertEqual(pos["credit_per_share"], 1.11)

    def test_unfilled_and_canceled_is_not_tracked(self):
        eng = self._exec_setup()
        eng.api.get_order.return_value = {"status": "new", "filled_avg_price": 0}
        result = eng._execute_spread(make_spread(), capital_in_use=0)
        self.assertFalse(result)
        self.assertEqual(len(eng.positions), 0)
        # Confirmed no-fill → attempt counter bumps so next try gets a new key
        self.assertEqual(sum(eng._entry_attempts.values()), 1)

    def test_duplicate_order_sentinel_skips(self):
        """F3: broker-rejected duplicate client_order_id must not open a position."""
        eng = self._exec_setup()
        eng.api.submit_credit_spread.return_value = "DUPLICATE_ORDER"
        result = eng._execute_spread(make_spread(), capital_in_use=0)
        self.assertFalse(result)
        self.assertEqual(len(eng.positions), 0)
        # Attempt counter must NOT increment — the original key's order may be live
        self.assertEqual(sum(eng._entry_attempts.values()), 0)

    def test_pnl_basis_is_actual_fill_not_scan_mid(self):
        """F5: credit_per_share = filled price, not the scan-time mid."""
        eng = self._exec_setup()
        eng.api.get_order.return_value = {"status": "filled", "filled_avg_price": 1.05}
        result = eng._execute_spread(make_spread(credit=1.30), capital_in_use=0)
        self.assertTrue(result)
        pos = next(iter(eng.positions.values()))
        self.assertEqual(pos["credit_per_share"], 1.05)
        self.assertEqual(pos["total_credit"], 1.05 * 3 * 100)
        # Max loss must also use the actual fill
        self.assertEqual(pos["max_loss_total"], (10.0 - 1.05) * 100 * 3)

    def test_blocked_by_pre_order_gate(self):
        """F9: gate failure prevents submission entirely."""
        eng = self._exec_setup()
        eng._pre_order_gate = MagicMock(return_value=(False, "leg exists"))
        result = eng._execute_spread(make_spread(), capital_in_use=0)
        self.assertFalse(result)
        eng.api.submit_credit_spread.assert_not_called()

    def test_idempotency_key_passed_to_submit(self):
        eng = self._exec_setup()
        eng.api.get_order.return_value = {"status": "filled", "filled_avg_price": 1.20}
        eng._execute_spread(make_spread(), capital_in_use=0)
        _, kwargs = eng.api.submit_credit_spread.call_args
        self.assertTrue(kwargs["client_order_id"].startswith("ic_"))


# ── F6: one-sided quote guard ────────────────────────────

class TestOneSidedQuoteGuard(unittest.TestCase):
    def test_one_sided_quote_skips_exit_check(self):
        """F6: bid=0 or ask=0 → no exit decision on a garbage mid."""
        eng = make_engine()
        pos_id, pos = make_position()
        eng.api.get_option_quote.side_effect = [
            {"symbol": "s", "bid": 0.0, "ask": 2.60, "mid": 0, "one_sided": True},
            {"symbol": "l", "bid": 0.55, "ask": 0.65, "mid": 0.60, "one_sided": False},
        ]
        self.assertIsNone(eng._check_exit(pos))

    def test_two_sided_profit_still_exits(self):
        """Control: healthy quotes at 50%+ profit still trigger TAKE_PROFIT."""
        eng = make_engine()
        pos_id, pos = make_position()  # credit 1.30
        eng.api.get_option_quote.side_effect = [
            {"symbol": "s", "bid": 0.90, "ask": 1.00, "mid": 0.95, "one_sided": False},
            {"symbol": "l", "bid": 0.40, "ask": 0.50, "mid": 0.45, "one_sided": False},
        ]  # debit 0.50 → profit 61%
        eng.api.get_latest_price.return_value = None
        eng.api.get_option_snapshot.return_value = None
        reason = eng._check_exit(pos)
        self.assertIsNotNone(reason)
        self.assertIn("TAKE_PROFIT", reason)

    def test_quote_mid_zero_when_one_sided(self):
        """F6 (api layer): mid must be 0, never ask/2, on a one-sided book."""
        api = object.__new__(PutSellerAPI)
        api.config = MagicMock()
        api._last_call_ts = time.monotonic()
        import threading
        api._rate_lock = threading.Lock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "quotes": {"XYZ260918P00350000": {"bp": 0, "ap": 2.60, "bs": 0, "as": 5}}
        }
        fake_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=fake_resp):
            q = api.get_option_quote("XYZ260918P00350000")
        self.assertEqual(q["mid"], 0)
        self.assertTrue(q["one_sided"])


# ── F7: ghost reconciliation ─────────────────────────────

class TestReconciliation(unittest.TestCase):
    def test_ghost_removed_when_broker_flat(self):
        """F7: neither leg at broker → position removed + trade recorded."""
        eng = make_engine()
        pos_id, pos = make_position(current_debit=0.20)
        eng.positions[pos_id] = pos
        eng.api.get_option_positions.return_value = {}
        removed = eng._reconcile_position(pos_id)
        self.assertTrue(removed)
        self.assertNotIn(pos_id, eng.positions)
        eng.risk.record_trade.assert_called_once()
        logged = eng._log_trade.call_args[0][0]
        self.assertIn("RECONCILED_GHOST", logged["exit_reason"])
        # est PnL from last mark: (1.30 - 0.20) * 3 * 100
        self.assertAlmostEqual(logged["pnl"], 330.0)

    def test_kept_when_leg_still_at_broker(self):
        eng = make_engine()
        pos_id, pos = make_position()
        eng.positions[pos_id] = pos
        eng.api.get_option_positions.return_value = {pos["short_symbol"]: -3}
        self.assertFalse(eng._reconcile_position(pos_id))
        self.assertIn(pos_id, eng.positions)
        eng.risk.record_trade.assert_not_called()

    def test_skipped_when_broker_unverifiable(self):
        """None (API failure) must never be treated as 'broker flat'."""
        eng = make_engine()
        pos_id, pos = make_position()
        eng.positions[pos_id] = pos
        eng.api.get_option_positions.return_value = None
        self.assertFalse(eng._reconcile_position(pos_id))
        self.assertIn(pos_id, eng.positions)

    def test_sweep_removes_only_ghosts(self):
        eng = make_engine()
        ghost_id, ghost = make_position("GHOST_1")
        live_id, live = make_position(
            "LIVE_1",
            short_symbol="ABC260918P00100000",
            long_symbol="ABC260918P00095000",
        )
        eng.positions = {ghost_id: ghost, live_id: live}
        eng.api.get_option_positions.return_value = {live["short_symbol"]: -3,
                                                     live["long_symbol"]: 3}
        eng._reconcile_all_positions()
        self.assertNotIn(ghost_id, eng.positions)
        self.assertIn(live_id, eng.positions)


# ── F8: close-failure backoff + circuit breaker ──────────

@patch("core.put_engine._time.sleep", lambda *_: None)
class TestCloseFailureBreaker(unittest.TestCase):
    def _failing_close_engine(self):
        eng = make_engine()
        eng.api.close_credit_spread.return_value = None   # MLEG submit fails
        eng.api.close_individual_legs.return_value = False  # leg fallback fails
        eng._reconcile_position = MagicMock(return_value=False)
        return eng

    def test_first_failure_sets_backoff(self):
        """F8: failed close schedules a retry instead of retrying every cycle."""
        eng = self._failing_close_engine()
        pos_id, pos = make_position()
        eng.positions[pos_id] = pos
        eng._close_position(pos_id, "TAKE_PROFIT")
        self.assertEqual(pos["close_failures"], 1)
        self.assertGreater(pos["next_close_retry_ts"], time.time() + 250)  # ~5m

    def test_backoff_escalates(self):
        eng = self._failing_close_engine()
        pos_id, pos = make_position(close_failures=1)
        eng.positions[pos_id] = pos
        eng._close_position(pos_id, "TAKE_PROFIT")
        self.assertEqual(pos["close_failures"], 2)
        self.assertGreater(pos["next_close_retry_ts"], time.time() + 550)  # ~10m

    def test_skips_close_during_backoff_window(self):
        eng = self._failing_close_engine()
        pos_id, pos = make_position(next_close_retry_ts=time.time() + 600)
        eng.positions[pos_id] = pos
        eng._close_position(pos_id, "TAKE_PROFIT")
        eng.api.close_credit_spread.assert_not_called()

    def test_reconcile_triggered_after_three_failures(self):
        """F8/F7: 3 strikes → verify ground truth at the broker."""
        eng = self._failing_close_engine()
        pos_id, pos = make_position(close_failures=2)
        eng.positions[pos_id] = pos
        eng._close_position(pos_id, "TAKE_PROFIT")
        self.assertEqual(pos["close_failures"], 3)
        eng._reconcile_position.assert_called_once_with(pos_id)

    def test_successful_close_clears_failure_state(self):
        eng = make_engine()
        pos_id, pos = make_position(close_failures=2,
                                    next_close_retry_ts=0,
                                    current_debit=0.60)
        eng.positions[pos_id] = pos
        eng.api.close_credit_spread.return_value = "close-1"
        eng.api.get_order.return_value = {"status": "filled", "filled_avg_price": 0.60}
        eng._close_position(pos_id, "TAKE_PROFIT")
        self.assertNotIn(pos_id, eng.positions)
        eng.risk.record_trade.assert_called_once()
        # PnL from actual close fill: (1.30 - 0.60) * 3 * 100
        pnl = eng.risk.record_trade.call_args[0][0]
        self.assertAlmostEqual(pnl, 210.0)


# ── F4: adoption — debit-spread skip + sibling-bot paths ─

class TestAdoptionSafety(unittest.TestCase):
    """_adopt_orphaned_spreads must never adopt debit spreads or sibling-bot
    positions. Uses a temp cwd with a fake ../CallBuyer state file to prove
    the relative (non-Windows) sibling path works."""

    SHORT = "CIFR251219C00010000"
    LONG = "CIFR251219C00012000"

    def setUp(self):
        self._old_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        # layout: <tmp>/PutSeller (cwd), <tmp>/CallBuyer/data/state/
        self.bot_dir = os.path.join(self._tmp.name, "PutSeller")
        self.cb_state = os.path.join(self._tmp.name, "CallBuyer", "data", "state")
        os.makedirs(self.bot_dir)
        os.makedirs(self.cb_state)
        os.chdir(self.bot_dir)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def _broker_positions(self, short_entry, long_entry):
        return [
            {"symbol": self.SHORT, "qty": "-1", "asset_class": "us_option",
             "avg_entry_price": str(short_entry), "market_value": "0"},
            {"symbol": self.LONG, "qty": "1", "asset_class": "us_option",
             "avg_entry_price": str(long_entry), "market_value": "0"},
        ]

    def _run_adoption(self, eng, broker_positions):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = broker_positions
        with patch("requests.get", return_value=resp):
            eng._adopt_orphaned_spreads()

    def test_debit_spread_is_not_adopted(self):
        """F4: net credit <= 0 means it's another strategy's debit spread —
        adopting it with fabricated credit corrupts every exit rule (CIFR)."""
        eng = make_engine()
        # short collected 0.38, long cost 0.76 → net -0.38 (a DEBIT spread)
        self._run_adoption(eng, self._broker_positions(0.38, 0.76))
        self.assertEqual(len(eng.positions), 0)

    def test_credit_spread_is_adopted(self):
        """Control: a genuine orphaned credit spread IS adopted."""
        eng = make_engine()
        self._run_adoption(eng, self._broker_positions(0.76, 0.38))
        self.assertEqual(len(eng.positions), 1)
        pos = next(iter(eng.positions.values()))
        self.assertAlmostEqual(pos["credit_per_share"], 0.38)
        self.assertEqual(pos["order_id"], "ADOPTED")

    def test_sibling_bot_contract_is_not_adopted(self):
        """F4: contracts tracked in ../CallBuyer/data/state/positions.json are
        foreign — the relative sibling path must actually resolve (the old
        C:\\ path silently never matched on Linux)."""
        with open(os.path.join(self.cb_state, "positions.json"), "w") as f:
            json.dump({"cb_pos_1": {"contract": self.SHORT}}, f)
        eng = make_engine()
        # A valid credit spread — but the short leg belongs to CallBuyer
        self._run_adoption(eng, self._broker_positions(0.76, 0.38))
        self.assertEqual(len(eng.positions), 0)

    def test_already_tracked_spread_not_readopted(self):
        eng = make_engine()
        eng.positions["existing"] = {"short_symbol": self.SHORT,
                                     "long_symbol": self.LONG}
        self._run_adoption(eng, self._broker_positions(0.76, 0.38))
        self.assertEqual(len(eng.positions), 1)  # unchanged


if __name__ == "__main__":
    unittest.main(verbosity=2)
