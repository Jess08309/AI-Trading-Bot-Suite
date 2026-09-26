"""Unit tests for tools/portfolio_report.py reconciliation logic.

Focused on the phantom/orphan detection fix requested in PR #11 review:
multi-leg (spread) positions must be checked leg-by-leg, not as a naive
symbol-set difference, and a broker-reported qty==0 row (settlement lag)
must not be treated as present or absent inconsistently.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio_report import PositionClaim, _reconcile  # noqa: E402


def _broker_pos(symbol, qty="1"):
    return {"symbol": symbol, "qty": qty, "side": "long",
            "market_value": None, "unrealized_pl": None}


class TestReconcileSpreads(unittest.TestCase):
    """PutSeller / AlpacaBot-style two-leg spread positions."""

    def test_full_spread_present_no_alert(self):
        claims = {
            "PutSeller": [PositionClaim("spread-1", ["SPY_SHORT", "SPY_LONG"])],
        }
        broker = [_broker_pos("SPY_SHORT"), _broker_pos("SPY_LONG")]
        phantoms, orphans, one_leg = _reconcile(claims, broker)
        self.assertEqual(phantoms, [])
        self.assertEqual(orphans, [])
        self.assertEqual(one_leg, [])

    def test_one_leg_missing_is_critical_naked_leg(self):
        claims = {
            "PutSeller": [PositionClaim("spread-1", ["SPY_SHORT", "SPY_LONG"])],
        }
        # Short leg missing at broker -> naked long, dangerous state.
        broker = [_broker_pos("SPY_LONG")]
        phantoms, orphans, one_leg = _reconcile(claims, broker)
        self.assertEqual(phantoms, [])
        self.assertEqual(len(one_leg), 1)
        self.assertIn("ONE-LEG PHANTOM ALERT (bot PutSeller)", one_leg[0])
        self.assertIn("SPY_SHORT", one_leg[0])

    def test_both_legs_missing_is_full_phantom(self):
        claims = {
            "PutSeller": [PositionClaim("spread-1", ["SPY_SHORT", "SPY_LONG"])],
        }
        phantoms, orphans, one_leg = _reconcile(claims, [])
        self.assertEqual(one_leg, [])
        self.assertEqual(len(phantoms), 1)
        self.assertIn("PHANTOM ALERT (bot PutSeller)", phantoms[0])

    def test_alpacabot_spread_short_leg_checked(self):
        # AlpacaBot spreads are keyed by the long leg but also carry a
        # short_leg_symbol that must independently be checked at broker.
        claims = {
            "AlpacaBot": [PositionClaim("AAPL_LONG_LEG", ["AAPL_LONG_LEG", "AAPL_SHORT_LEG"])],
        }
        broker = [_broker_pos("AAPL_LONG_LEG")]  # short leg missing
        phantoms, orphans, one_leg = _reconcile(claims, broker)
        self.assertEqual(len(one_leg), 1)
        self.assertIn("AAPL_SHORT_LEG", one_leg[0])


class TestReconcileSingleLeg(unittest.TestCase):
    """AlpacaBot single-leg / CallBuyer / CryptoBot single-symbol positions."""

    def test_single_leg_phantom(self):
        claims = {"CallBuyer": [PositionClaim("NFLX240119C00500000", ["NFLX240119C00500000"])]}
        phantoms, orphans, one_leg = _reconcile(claims, [])
        self.assertEqual(one_leg, [])
        self.assertEqual(len(phantoms), 1)
        self.assertIn("PHANTOM ALERT (bot CallBuyer)", phantoms[0])

    def test_single_leg_matched_no_alert(self):
        claims = {"CryptoBot": [PositionClaim("BTC/USD", ["BTCUSD"])]}
        broker = [_broker_pos("BTCUSD")]
        phantoms, orphans, one_leg = _reconcile(claims, broker)
        self.assertEqual(phantoms, [])
        self.assertEqual(one_leg, [])
        self.assertEqual(orphans, [])


class TestReconcileOrphansAndZeroQty(unittest.TestCase):
    def test_orphan_detected(self):
        claims = {"AlpacaBot": [PositionClaim("AAPL_POS", ["AAPL_POS"])]}
        broker = [_broker_pos("AAPL_POS"), _broker_pos("TSLA_UNCLAIMED")]
        phantoms, orphans, one_leg = _reconcile(claims, broker)
        self.assertEqual(phantoms, [])
        self.assertEqual(one_leg, [])
        self.assertEqual(len(orphans), 1)
        self.assertIn("TSLA_UNCLAIMED", orphans[0])

    def test_zero_qty_broker_row_is_not_orphan(self):
        # A position mid-settlement can transiently show qty=0; it must
        # not be reported as an ORPHAN (nor as matched).
        claims = {}
        broker = [_broker_pos("SETTLING_OUT", qty="0")]
        phantoms, orphans, one_leg = _reconcile(claims, broker)
        self.assertEqual(phantoms, [])
        self.assertEqual(one_leg, [])
        self.assertEqual(orphans, [])

    def test_zero_qty_broker_row_does_not_satisfy_local_claim(self):
        # If the bot still thinks it holds a position but the broker's row
        # for that symbol has qty==0, that must count as a phantom, not a
        # match.
        claims = {"AlpacaBot": [PositionClaim("AAPL_POS", ["AAPL_POS"])]}
        broker = [_broker_pos("AAPL_POS", qty="0")]
        phantoms, orphans, one_leg = _reconcile(claims, broker)
        self.assertEqual(len(phantoms), 1)
        self.assertEqual(one_leg, [])
        self.assertEqual(orphans, [])

    def test_broker_none_skips_reconciliation_entirely(self):
        claims = {"AlpacaBot": [PositionClaim("AAPL_POS", ["AAPL_POS"])]}
        phantoms, orphans, one_leg = _reconcile(claims, None)
        self.assertEqual(phantoms, [])
        self.assertEqual(orphans, [])
        self.assertEqual(one_leg, [])


if __name__ == "__main__":
    unittest.main()
