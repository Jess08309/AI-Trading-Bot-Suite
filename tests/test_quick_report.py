import unittest

from quick_report import build_report


class QuickReportTests(unittest.TestCase):
    def test_build_report_basic_metrics(self):
        trades = [
            {"time": "2026-01-01T00:00:00", "symbol": "BTC-USD", "side": "CLOSE", "pnl": 100.0},
            {"time": "2026-01-02T00:00:00", "symbol": "BTC-USD", "side": "CLOSE", "pnl": -50.0},
            {"time": "2026-01-03T00:00:00", "symbol": "ETH-USD", "side": "CLOSE", "pnl": 75.0},
        ]
        report = build_report(trades, initial_balance=5000.0)

        self.assertEqual(report["total_trades"], 3)
        self.assertAlmostEqual(report["total_pnl"], 125.0)
        self.assertIn("symbols", report)
        self.assertIn("BTC-USD", report["symbols"])
        self.assertIn("monthly_pnl", report)
        self.assertIsNotNone(report["best_symbol"])
        self.assertIsNotNone(report["worst_symbol"])


if __name__ == "__main__":
    unittest.main()
