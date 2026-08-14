import unittest
from datetime import datetime

from backtest_harness import (
    _filter_prices_by_date,
    _max_drawdown,
    _rsi,
    _simulate_symbol,
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

    def test_no_look_ahead_bias_in_signal(self):
        """Look-ahead regression: the signal at bar N must NOT see bar N's close.

        Series design: a steady up-zigzag (+0.5/-0.2) keeps RSI ~71 (blocks
        LONG's rsi<70) and trend ~+2.5% (blocks SHORT's trend<-1%), so no
        entry is ever valid on data through bar N-1. The final bar then drops
        sharply — a signal only visible if the current bar's close leaks into
        `history`. A correct harness produces ZERO trades; a look-ahead
        harness enters SHORT on the final bar.
        """
        closes = [100.0]
        for i in range(1, 70):
            closes.append(closes[-1] + (0.5 if i % 2 == 1 else -0.2))
        closes.append(closes[-1] - 5.0)  # final-bar crash

        prices = [(f"2025-01-01T00:{i:02d}:00", c) for i, c in enumerate(closes)]
        result = _simulate_symbol(
            prices,
            initial_balance=10_000.0,
            max_position_pct=0.10,
            stop_loss_pct=-5.0,
            take_profit_pct=5.0,
            trailing_stop_pct=2.0,
            fee_rate=0.001,
            slippage_bps=10.0,
        )
        self.assertEqual(
            result["trades"], 0,
            "Signal acted on the current bar's close — look-ahead bias reintroduced",
        )


if __name__ == "__main__":
    unittest.main()
