"""
Unit tests for tools/momentum_backtest.py (issue #32 research spike).

Covers: momentum signal known-answers, ranking/top-N, vol-scaled sizing caps,
no-look-ahead execution timing (signal at close T fills at close T+1), the
pre-registered criteria evaluator, and an end-to-end run on the synthetic
fixture in tests/fixtures/momentum_prices_fixture.csv.

The fixture is synthetic (seeded numpy, regime-switching drift + Gaussian
noise for 6 Alpaca-style pairs over 420 days, plus a 60-day "NEW/USD" listing
that must fail the history screen). It exists only to exercise the pipeline;
its results say nothing about the strategy.
"""
import os
import sys
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

CRYPTOBOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CRYPTOBOT_DIR / "tools"))

import momentum_backtest as mb  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "momentum_prices_fixture.csv"


def _cfg(**kw):
    defaults = dict(lookback_weeks=1, top_n=3, rebalance_days=7, vol_window=20,
                    risk_pct=0.01, notional_cap_pct=0.10, starting_equity=10_000.0,
                    slippage_bps=10.0, fee_rate=0.001)
    defaults.update(kw)
    return mb.MomentumConfig(**defaults)


# ============================================================
# Momentum signal
# ============================================================
class TestMomentumSignal:
    def test_known_answer_constant_growth(self):
        prices = [100.0 * 1.1 ** i for i in range(6)]
        out = mb.momentum_returns(prices, 2)
        assert np.isnan(out[0]) and np.isnan(out[1])
        np.testing.assert_allclose(out[2:], 0.21, rtol=1e-12)

    def test_known_answer_mixed(self):
        out = mb.momentum_returns([100, 80, 120, 60], 1)
        np.testing.assert_allclose(out[1:], [-0.2, 0.5, -0.5])

    def test_lookback_84_days_default(self):
        assert mb.MomentumConfig().lookback_days == 84
        prices = np.linspace(100, 200, 85)
        out = mb.momentum_returns(prices, 84)
        assert np.all(np.isnan(out[:84]))
        assert out[84] == pytest.approx(1.0)

    def test_missing_or_nonpositive_endpoint_is_nan(self):
        out = mb.momentum_returns([100, float("nan"), 0.0, 110, 120], 2)
        assert np.isnan(out[2])          # base 100 ok, but end 0 -> invalid
        assert np.isnan(out[3])          # base NaN
        assert np.isnan(out[4])          # base 0

    def test_series_shorter_than_lookback(self):
        assert np.all(np.isnan(mb.momentum_returns([1, 2, 3], 5)))

    def test_realized_vol_known_answer(self):
        a = 0.02
        log_p = np.cumsum([0.0] + [a if i % 2 == 0 else -a for i in range(20)])
        prices = 100 * np.exp(log_p)
        vol = mb.realized_vol(prices, 20)
        assert np.all(np.isnan(vol[:20]))
        expected = np.std([a if i % 2 == 0 else -a for i in range(20)], ddof=1)
        assert vol[20] == pytest.approx(expected)

    def test_realized_vol_zero_for_constant_growth(self):
        vol = mb.realized_vol([100 * 1.01 ** i for i in range(25)], 20)
        assert vol[-1] == pytest.approx(0.0, abs=1e-12)


# ============================================================
# Ranking / top-N
# ============================================================
class TestTopNSelection:
    def test_ranks_descending_and_caps_at_n(self):
        scores = {"A": 0.10, "B": 0.50, "C": 0.30, "D": 0.20}
        assert mb.select_top_n(scores, 3) == ["B", "C", "D"]

    def test_excludes_non_positive_and_nan(self):
        scores = {"A": 0.0, "B": -0.2, "C": float("nan"), "D": 0.05, "E": None}
        assert mb.select_top_n(scores, 3) == ["D"]

    def test_all_negative_means_cash(self):
        assert mb.select_top_n({"A": -0.1, "B": -0.01}, 3) == []

    def test_ties_broken_alphabetically(self):
        assert mb.select_top_n({"Z": 0.1, "A": 0.1, "M": 0.1}, 2) == ["A", "M"]


# ============================================================
# Vol-scaled sizing
# ============================================================
class TestVolScaledSizing:
    def test_inverse_to_vol_below_cap(self):
        lo = mb.vol_scaled_notional(10_000, 100, 0.05)
        hi = mb.vol_scaled_notional(10_000, 100, 0.10)
        assert lo == pytest.approx(2 * hi, rel=1e-3)
        assert lo == pytest.approx(0.01 * 10_000 / (0.05 * math.sqrt(7)), abs=0.01)

    def test_hard_cap_10pct(self):
        assert mb.vol_scaled_notional(10_000, 100, 0.001) == pytest.approx(1_000.0)
        assert mb.vol_scaled_notional(10_000, 100, 0.001, notional_cap_pct=0.05) == pytest.approx(500.0)

    def test_extreme_vol_still_positive_and_capped(self):
        n = mb.vol_scaled_notional(10_000, 100, 0.9)
        assert 0 < n <= 1_000

    @pytest.mark.parametrize("vol", [0.0, float("nan"), None, -0.1])
    def test_unknown_vol_is_zero(self, vol):
        assert mb.vol_scaled_notional(10_000, 100, vol) == 0.0

    def test_zero_equity_is_zero(self):
        assert mb.vol_scaled_notional(0, 100, 0.05) == 0.0

    def test_targets_respect_cap(self):
        prices = _trend_matrix()
        targets = mb.compute_targets(prices, ["UP", "DOWN"], 30, 10_000, _cfg())
        assert set(targets) == {"UP"}
        assert 0 < targets["UP"] <= 1_000.0


# ============================================================
# No look-ahead / execution timing
# ============================================================
def _trend_matrix(n=80, turn=None):
    i = np.arange(n)
    wiggle = 0.004 * (-1.0) ** i
    up = 100 * np.exp(0.01 * i + wiggle)
    if turn is not None:
        up[turn:] = up[turn] * np.exp(-0.02 * (i[turn:] - turn) + wiggle[turn:])
    down = 50 * np.exp(-0.01 * i - wiggle)
    return np.column_stack([up, down])


class TestNoLookAhead:
    def test_targets_ignore_future_rows(self):
        prices = _trend_matrix()
        t = 30
        base = mb.compute_targets(prices, ["UP", "DOWN"], t, 10_000, _cfg())
        poisoned = prices.copy()
        poisoned[t + 1:, 0] = 1e-6       # UP crashes tomorrow
        poisoned[t + 1:, 1] = 1e9        # DOWN moons tomorrow
        assert mb.compute_targets(poisoned, ["UP", "DOWN"], t, 10_000, _cfg()) == base

    def test_signal_at_T_fills_at_T_plus_1_close(self):
        prices = _trend_matrix()
        prices[26, 0] *= 1.05            # T+1 close differs from T close
        cfg = _cfg()
        res = mb.run_momentum_backtest(prices, ["UP", "DOWN"], cfg, start_idx=25)
        first = res.rebalances[0]
        assert first["signal_idx"] == 25 and first["exec_idx"] == 26
        entry = res.trades[0] if res.trades else None
        assert entry is not None and entry.symbol == "UP"
        assert entry.entry_idx == 26
        assert entry.entry_price == pytest.approx(prices[26, 0] * (1 + 10 / 10_000))
        # Nothing happens on or before the signal day itself.
        assert np.all(res.equity[:26] == cfg.starting_equity)

    def test_exit_fills_day_after_signal_flips(self):
        prices = _trend_matrix(n=90, turn=40)
        cfg = _cfg()
        res = mb.run_momentum_backtest(prices, ["UP", "DOWN"], cfg, start_idx=25)
        exits = [t for t in res.trades if t.exit_reason == "MOMENTUM_EXIT"]
        assert exits, "trend reversal should trigger a momentum exit"
        ex = exits[0]
        reb = next(r for r in res.rebalances if r["exec_idx"] == ex.exit_idx)
        assert ex.exit_idx == reb["signal_idx"] + 1
        assert "UP" not in reb["targets"]
        sell = [f for f in reb["fills"] if f["symbol"] == "UP" and f["side"] == "SELL"]
        assert len(sell) == 1
        assert sell[0]["price"] == pytest.approx(prices[ex.exit_idx, 0] * (1 - 10 / 10_000))

    def test_rebalance_every_7_days_and_costs_charged(self):
        prices = _trend_matrix()
        res = mb.run_momentum_backtest(prices, ["UP", "DOWN"], _cfg(), start_idx=25)
        sig = [r["signal_idx"] for r in res.rebalances]
        assert sig == list(range(25, 79, 7))
        assert res.fees_paid > 0 and res.slippage_cost > 0
        # All positions liquidated at the end.
        assert res.trades[-1].exit_reason == "END_OF_DATA"

    def test_long_only_no_leverage(self):
        prices = _trend_matrix()
        res = mb.run_momentum_backtest(prices, ["UP", "DOWN"], _cfg(), start_idx=25)
        assert all(t.direction == "long" and t.side == "spot" for t in res.trades)
        assert "DOWN" not in {t.symbol for t in res.trades}


# ============================================================
# Pre-registered criteria
# ============================================================
def _agg(sharpe=1.0, pos=0.8, dd=20.0, trades=40):
    return {"oos_sharpe": sharpe, "pct_windows_positive": pos,
            "oos_max_drawdown_pct": dd, "num_trades": trades,
            "oos_total_return_pct": 10.0}


class TestAcceptance:
    def test_all_pass(self):
        sweep = {w: _agg() for w in (8, 12, 16, 20)}
        acc = mb.evaluate_acceptance(sweep, 12, None, None)
        assert acc["failed"] == []
        assert acc["verdict"].startswith("PASSES")

    def test_single_lookback_negative_fails_a4(self):
        sweep = {w: _agg() for w in (8, 12, 16, 20)}
        sweep[16] = _agg(sharpe=-0.1)
        acc = mb.evaluate_acceptance(sweep, 12, None, None)
        assert acc["failed"] == ["A4"]
        assert acc["verdict"] == "NOT DEPLOYABLE"

    def test_thresholds_are_strict(self):
        sweep = {12: _agg(sharpe=0.8, pos=0.69, dd=40.0, trades=29)}
        acc = mb.evaluate_acceptance(sweep, 12, None, None)
        assert set(acc["failed"]) == {"A1", "A2", "A3", "K1"}

    def test_btc_comparison_reported(self):
        btc = {"oos_sharpe": 2.0, "oos_total_return_pct": 50.0, "oos_max_drawdown_pct": 30.0}
        acc = mb.evaluate_acceptance({12: _agg()}, 12, btc, {"available": False})
        a5 = next(c for c in acc["criteria"] if c["id"] == "A5")
        assert a5["status"] == "REPORTED"
        assert a5["value"]["beats_btc_on_sharpe"] is False


# ============================================================
# Helpers + end-to-end on the synthetic fixture
# ============================================================
class TestPipeline:
    @pytest.mark.parametrize("sym", ["BTC/USD", "BTC-USD", "BTCUSD", "XBTUSD", "btc"])
    def test_find_btc_symbol(self, sym):
        assert mb.find_btc_symbol(["ETH/USD", sym]) == sym

    def test_find_btc_symbol_missing(self):
        assert mb.find_btc_symbol(["ETH/USD", "BTCX/USD"]) is None

    def test_missing_data_file_returns_error(self, tmp_path):
        assert mb.main(["--data", str(tmp_path / "nope.csv"),
                        "--output", str(tmp_path / "o.json")]) == 1

    def test_end_to_end_fixture(self, tmp_path):
        out = tmp_path / "results.json"
        rc = mb.main(["--data", str(FIXTURE), "--output", str(out), "--sims", "200",
                      "--baseline", str(tmp_path / "missing_baseline.json")])
        assert rc == 0
        res = json.loads(out.read_text())
        assert sorted(res["sweep"], key=int) == ["8", "12", "16", "20"]
        # Every lookback evaluated on the same OOS windows.
        starts = {res["sweep"][w]["aggregate"]["oos_start"] for w in res["sweep"]}
        assert len(starts) == 1
        a = res["sweep"]["12"]["aggregate"]
        assert a["total_windows"] == len(res["sweep"]["12"]["per_window"]) > 0
        assert a["universe_size"] == 6          # NEW/USD screened out (60 days)
        ids = [c["id"] for c in res["acceptance"]["criteria"]]
        assert ids == ["A1", "A2", "A3", "A4", "A5"]
        assert res["acceptance"]["kill_criteria"][0]["id"] == "K1"
        assert res["benchmarks"]["btc_buy_and_hold"]["symbol"] == "BTC/USD"
        assert res["benchmarks"]["scalper_baseline"]["available"] is False
        assert "SYNTHETIC" in res["data_warning"]
        for t in res["primary_trades"]:
            assert t["exit_idx"] > t["entry_idx"] >= a["train_window_days"]
