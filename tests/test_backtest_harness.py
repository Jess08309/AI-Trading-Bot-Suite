import unittest
from datetime import datetime

from backtest_harness import (
    _filter_prices_by_date,
    _max_drawdown,
    _rsi,
    _slippage_multiplier,
)


class BacktestHarnessTests(unittest.TestCase):
    def test_rsi_returns_neutral_when_insufficient_data(self):
        self.assertEqual(_rsi([100, 101, 102], period=14), 50.0)

    def test_slippage_multiplier_long_entry(self):
        mult = _slippage_multiplier("LONG", True, 10.0)
        self.assertAlmostEqual(mult, 1.001)

    def test_max_drawdown(self):
        curve = [1000, 1100, 900, 950, 1200, 1000]
        dd = _max_drawdown(curve)
        self.assertAlmostEqual(dd, -18.1818, places=3)

    def test_filter_prices_by_date(self):
        prices = [
            ("2025-01-01T00:00:00", 100.0),
            ("2025-02-01T00:00:00", 110.0),
            ("2025-03-01T00:00:00", 120.0),
        ]
        out = _filter_prices_by_date(
            prices,
            start_dt=datetime(2025, 2, 1),
            end_dt=datetime(2025, 3, 1),
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], "2025-02-01T00:00:00")


if __name__ == "__main__":
    unittest.main()
