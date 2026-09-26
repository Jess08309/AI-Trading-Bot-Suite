import importlib.util
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "portfolio_report.py"
SPEC = importlib.util.spec_from_file_location("portfolio_report", MODULE_PATH)
portfolio_report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = portfolio_report
SPEC.loader.exec_module(portfolio_report)


class TestPortfolioReportReconciliation(unittest.TestCase):
    def test_alpacabot_spread_with_both_legs_present_has_no_alert(self):
        broker_positions = [
            {"symbol": "NFLX250101C00500000", "qty": "1"},
            {"symbol": "NFLX250101C00510000", "qty": "-1"},
        ]
        bot_positions = {
            "AlpacaBot": {
                "NFLX250101C00500000": {
                    "symbol": "NFLX250101C00500000",
                    "short_leg_symbol": "NFLX250101C00510000",
                }
            }
        }

        phantoms, orphans = portfolio_report._reconcile(bot_positions, broker_positions)

        self.assertEqual(phantoms, [])
        self.assertEqual(orphans, [])

    def test_putseller_one_leg_missing_is_phantom(self):
        broker_positions = [
            {"symbol": "SPY260101P00590000", "qty": "-1"},
        ]
        bot_positions = {
            "PutSeller": {
                "SPY_credit_spread": {
                    "short_symbol": "SPY260101P00590000",
                    "long_symbol": "SPY260101P00585000",
                }
            }
        }

        phantoms, orphans = portfolio_report._reconcile(bot_positions, broker_positions)

        self.assertEqual(len(phantoms), 1)
        self.assertIn("PutSeller", phantoms[0])
        self.assertIn("SPY260101P00585000", phantoms[0])
        self.assertIn("other leg still present", phantoms[0])
        self.assertEqual(orphans, [])

    def test_zero_qty_broker_row_does_not_create_orphan(self):
        broker_positions = [
            {"symbol": "AAPL260101C00200000", "qty": "0"},
        ]

        phantoms, orphans = portfolio_report._reconcile({}, broker_positions)

        self.assertEqual(phantoms, [])
        self.assertEqual(orphans, [])

    def test_zero_qty_broker_row_is_absent_for_local_position(self):
        broker_positions = [
            {"symbol": "AAPL260101C00200000", "qty": "0"},
        ]
        bot_positions = {
            "CallBuyer": {
                "AAPL260101C00200000": {
                    "contract": "AAPL260101C00200000",
                }
            }
        }

        phantoms, orphans = portfolio_report._reconcile(bot_positions, broker_positions)

        self.assertEqual(len(phantoms), 1)
        self.assertIn("CallBuyer", phantoms[0])
        self.assertIn("AAPL260101C00200000", phantoms[0])
        self.assertEqual(orphans, [])


if __name__ == "__main__":
    unittest.main()
