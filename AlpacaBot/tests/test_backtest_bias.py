"""
Regression guards for backtest-bias audit findings in AlpacaBot/tools.

These are source-level tripwires: the backtest files are run-as-script tools
whose logic lives inside a monolithic run loop, so we guard the exact
patterns that were fixed rather than importing them.

Findings guarded:
  - Look-ahead bias: indicator window included the current bar
    (`prices[day - LOOKBACK:day + 1]`) while filling at that same bar's close.
  - Missing costs: no options commission was modeled (~$0.65/contract/side).
"""
import os
import re
import unittest

TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")


def _read(name):
    with open(os.path.join(TOOLS, name), encoding="utf-8") as f:
        return f.read()


class TestBacktestLookAheadGuard(unittest.TestCase):
    """The signal window must exclude the bar the trade fills on."""

    BIASED_SLICE = re.compile(
        r"prices\[\s*day(?:_idx)?\s*-\s*LOOKBACK\s*:\s*day(?:_idx)?\s*\+\s*1\s*\]"
    )
    FIXED_SLICE = re.compile(
        r"prices\[\s*day(?:_idx)?\s*-\s*LOOKBACK\s*:\s*day(?:_idx)?\s*\]"
    )

    def test_backtest_signal_window_excludes_current_bar(self):
        src = _read("backtest.py")
        self.assertIsNone(
            self.BIASED_SLICE.search(src),
            "backtest.py: indicator chunk includes the current bar again "
            "(look-ahead bias reintroduced)",
        )
        self.assertIsNotNone(self.FIXED_SLICE.search(src))

    def test_backtest_v2_signal_window_excludes_current_bar(self):
        src = _read("backtest_v2.py")
        self.assertIsNone(
            self.BIASED_SLICE.search(src),
            "backtest_v2.py: indicator chunk includes the current bar again "
            "(look-ahead bias reintroduced)",
        )
        self.assertIsNotNone(self.FIXED_SLICE.search(src))

    def test_iv_estimate_excludes_current_bar(self):
        for name in ("backtest.py", "backtest_v2.py"):
            src = _read(name)
            self.assertNotIn(
                "prices[:day_idx + 1]", src,
                f"{name}: IV estimated with the current bar's close",
            )
            self.assertNotIn(
                "prices[:day + 1]", src,
                f"{name}: IV estimated with the current bar's close",
            )


class TestBacktestFeesGuard(unittest.TestCase):
    """Options commissions must be modeled on both entry and exit."""

    def test_fee_constant_defined_and_applied(self):
        for name in ("backtest.py", "backtest_v2.py"):
            src = _read(name)
            self.assertIn("FEE_PER_CONTRACT", src,
                          f"{name}: FEE_PER_CONTRACT constant removed")
            self.assertRegex(
                src, r"2\s*\*\s*FEE_PER_CONTRACT",
                f"{name}: round-trip (entry+exit) fee no longer deducted",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
