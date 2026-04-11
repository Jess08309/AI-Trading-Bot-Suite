#!/usr/bin/env python3
"""
Walk-Forward Backtester with Monte Carlo Simulation
====================================================
Three-part analysis tool for CryptoBot:
  Part 1: Historical Trade Monte Carlo (10,000 simulations)
  Part 2: Walk-Forward ML Validation (rolling train/test windows)
  Part 3: Output Report + JSON export

Usage:
    python tools/walk_forward.py
    python tools/walk_forward.py --sims 5000 --train-days 60 --test-days 20

Output:
    data/backtest/walk_forward_results.json
"""

import sys
import os
import json
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — mirror the codebase import pattern
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "cryptotrades"))

try:
    from cryptotrades.utils.feature_engine import FeatureEngine
    from cryptotrades.utils.market_predictor import MarketPredictor
    from cryptotrades.utils.config import config
except ImportError:
    from utils.feature_engine import FeatureEngine
    from utils.market_predictor import MarketPredictor
    from utils.config import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADES_CSV = ROOT_DIR / "data" / "trades.csv"
HISTORICAL_CSV = ROOT_DIR / "data" / "historical" / "historical_prices_2yr.csv"
ONE_MIN_DIR = ROOT_DIR / "data" / "historical" / "1min"
OUTPUT_JSON = ROOT_DIR / "data" / "backtest" / "walk_forward_results.json"

NUM_SIMS_DEFAULT = 10_000
TRAIN_DAYS_DEFAULT = 60
TEST_DAYS_DEFAULT = 20
STEP_DAYS_DEFAULT = 20
CONFIDENCE_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

SEPARATOR = "=" * 72


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Historical Trade Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════

class MonteCarloEngine:
    """Bootstrap Monte Carlo simulation over historical trades."""

    def __init__(self, trades_df: pd.DataFrame, num_sims: int = NUM_SIMS_DEFAULT):
        self.trades = trades_df
        self.num_sims = num_sims
        self.pnl_usd = trades_df["pnl_usd"].values.astype(float)
        self.pnl_pct = trades_df["pnl_pct"].values.astype(float)
        self.rng = np.random.default_rng(seed=42)

    # ----- core simulation -----
    def run(self) -> Dict:
        """Run Monte Carlo simulations; return results dict."""
        n_trades = len(self.pnl_usd)
        print(f"\n  Running {self.num_sims:,} Monte Carlo simulations "
              f"over {n_trades} historical trades …")

        # Bootstrap: resample trade P&L with replacement
        indices = self.rng.integers(0, n_trades, size=(self.num_sims, n_trades))
        sim_pnl = self.pnl_usd[indices]                       # (sims, trades)
        sim_pct = self.pnl_pct[indices]

        # Cumulative equity curves (start at 0)
        equity_curves = np.cumsum(sim_pnl, axis=1)             # (sims, trades)

        # Terminal P&L
        terminal_pnl = equity_curves[:, -1]

        # Max drawdown per simulation
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = running_max - equity_curves
        max_drawdowns = np.max(drawdowns, axis=1)

        # Win rate per sim
        wins_per_sim = np.sum(sim_pnl > 0, axis=1)
        win_rates = wins_per_sim / n_trades

        # Sharpe ratio per sim (annualized, assume ~3 trades/day, 365 d)
        trades_per_year = 3 * 365
        mean_ret = np.mean(sim_pct, axis=1)
        std_ret = np.std(sim_pct, axis=1, ddof=1)
        std_ret = np.where(std_ret == 0, 1e-9, std_ret)
        sharpe_ratios = (mean_ret / std_ret) * np.sqrt(trades_per_year)

        # Profit factor per sim
        gross_profit = np.sum(np.where(sim_pnl > 0, sim_pnl, 0), axis=1)
        gross_loss   = np.abs(np.sum(np.where(sim_pnl < 0, sim_pnl, 0), axis=1))
        gross_loss   = np.where(gross_loss == 0, 1e-9, gross_loss)
        profit_factors = gross_profit / gross_loss

        # Longest win/loss streaks (vectorised per sim)
        avg_win_streak, avg_loss_streak = self._streak_stats(sim_pnl)

        # Build confidence intervals
        ci = {}
        for level in CONFIDENCE_LEVELS:
            pct_label = f"{int(level*100)}%"
            ci[pct_label] = {
                "terminal_pnl":   round(float(np.percentile(terminal_pnl, level * 100)), 2),
                "max_drawdown":   round(float(np.percentile(max_drawdowns, level * 100)), 2),
                "win_rate":       round(float(np.percentile(win_rates, level * 100)), 4),
                "sharpe_ratio":   round(float(np.percentile(sharpe_ratios, level * 100)), 3),
                "profit_factor":  round(float(np.percentile(profit_factors, level * 100)), 3),
            }

        # Probability of profit
        prob_profit = float(np.mean(terminal_pnl > 0))

        # Expected value per trade
        ev_per_trade = float(np.mean(terminal_pnl) / n_trades)

        # Ruin probability (draw down > 50% of starting balance)
        starting_bal = (config.PAPER_BALANCE_SPOT + config.PAPER_BALANCE_FUTURES)
        ruin_threshold = starting_bal * 0.50
        prob_ruin = float(np.mean(max_drawdowns > ruin_threshold))

        # Tail risk: worst 5% average
        cvar_5 = float(np.mean(terminal_pnl[terminal_pnl <= np.percentile(terminal_pnl, 5)]))

        results = {
            "num_simulations": self.num_sims,
            "num_trades_per_sim": int(n_trades),
            "probability_of_profit": round(prob_profit, 4),
            "probability_of_ruin_50pct": round(prob_ruin, 4),
            "expected_value_per_trade_usd": round(ev_per_trade, 4),
            "cvar_5pct_usd": round(cvar_5, 2),
            "mean_terminal_pnl_usd": round(float(np.mean(terminal_pnl)), 2),
            "median_terminal_pnl_usd": round(float(np.median(terminal_pnl)), 2),
            "std_terminal_pnl_usd": round(float(np.std(terminal_pnl)), 2),
            "mean_max_drawdown_usd": round(float(np.mean(max_drawdowns)), 2),
            "mean_sharpe_ratio": round(float(np.mean(sharpe_ratios)), 3),
            "mean_profit_factor": round(float(np.mean(profit_factors)), 3),
            "avg_longest_win_streak": round(avg_win_streak, 1),
            "avg_longest_loss_streak": round(avg_loss_streak, 1),
            "confidence_intervals": ci,
        }
        return results

    @staticmethod
    def _streak_stats(sim_pnl: np.ndarray) -> Tuple[float, float]:
        """Average longest win/loss streak across simulations."""
        win_streaks = []
        loss_streaks = []
        for row in sim_pnl:
            max_w = max_l = cur_w = cur_l = 0
            for v in row:
                if v > 0:
                    cur_w += 1
                    cur_l = 0
                elif v < 0:
                    cur_l += 1
                    cur_w = 0
                else:
                    cur_w = cur_l = 0
                max_w = max(max_w, cur_w)
                max_l = max(max_l, cur_l)
            win_streaks.append(max_w)
            loss_streaks.append(max_l)
        return float(np.mean(win_streaks)), float(np.mean(loss_streaks))

    # ----- per-symbol breakdown -----
    def symbol_breakdown(self) -> Dict:
        """Per-symbol Monte Carlo stats (smaller sim count for speed)."""
        breakdown = {}
        mini_sims = min(1000, self.num_sims)
        for symbol, grp in self.trades.groupby("symbol"):
            pnl = grp["pnl_usd"].values.astype(float)
            n = len(pnl)
            if n < 5:
                continue
            idx = self.rng.integers(0, n, size=(mini_sims, n))
            terminal = np.sum(pnl[idx], axis=1)
            breakdown[symbol] = {
                "n_trades": int(n),
                "actual_total_pnl": round(float(pnl.sum()), 2),
                "mean_sim_pnl": round(float(np.mean(terminal)), 2),
                "p5_sim_pnl": round(float(np.percentile(terminal, 5)), 2),
                "p50_sim_pnl": round(float(np.percentile(terminal, 50)), 2),
                "p95_sim_pnl": round(float(np.percentile(terminal, 95)), 2),
                "prob_profit": round(float(np.mean(terminal > 0)), 4),
            }
        return breakdown

    # ----- exit-reason breakdown -----
    def exit_reason_stats(self) -> Dict:
        """Per-exit-reason P&L stats from actual trades."""
        stats = {}
        for reason, grp in self.trades.groupby("exit_reason"):
            pnl = grp["pnl_usd"].values.astype(float)
            stats[reason] = {
                "count": int(len(pnl)),
                "total_pnl": round(float(pnl.sum()), 2),
                "mean_pnl": round(float(pnl.mean()), 2),
                "win_rate": round(float(np.mean(pnl > 0)), 4),
            }
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Walk-Forward ML Validation
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """Rolling-window train/test validation of the ML ensemble."""

    def __init__(self, train_days: int = TRAIN_DAYS_DEFAULT,
                 test_days: int = TEST_DAYS_DEFAULT,
                 step_days: int = STEP_DAYS_DEFAULT):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.fe = FeatureEngine()

    # ----- load daily price data -----
    @staticmethod
    def _load_daily_prices() -> pd.DataFrame:
        """Load historical_prices_2yr.csv → DataFrame[timestamp, symbol, price]."""
        df = pd.read_csv(HISTORICAL_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Use 'price' column (= close)
        df = df[["timestamp", "symbol", "price"]].dropna()
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return df

    # ----- build ALL features for a symbol at once (vectorised) -----
    def _build_all_features(self, prices: np.ndarray,
                            lookback: int = 30,
                            prediction_horizon: int = 5
                            ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute every indicator ONCE over the full price series, then
        assemble the feature matrix by reading off values at each index.
        ~600x faster than calling compute_all_indicators per window.
        """
        from cryptotrades.utils.technical_indicators import (
            rsi, macd, stochastic, cci, rate_of_change, momentum,
            williams_r, ultimate_oscillator, trix,
            chande_momentum_oscillator, atr_approx, trend_strength,
            bollinger_bands, mean_reversion_score, volatility_ratio,
        )

        n = len(prices)
        min_idx = max(lookback, 30)  # indicators need ~28-30 warm-up
        if n < min_idx + prediction_horizon + 1:
            return None, None

        price_list = prices.tolist()

        # --- compute each indicator array ONCE ---
        rsi_arr       = rsi(price_list, 14)
        _, _, macd_h  = macd(price_list)
        stoch_k, _    = stochastic(price_list, 14)
        cci_arr       = cci(price_list, 20)
        roc_arr       = rate_of_change(price_list, 10)
        mom_arr       = momentum(price_list, 10)
        wr_arr        = williams_r(price_list, 14)
        uo_arr        = ultimate_oscillator(price_list, 7, 14, 28)
        trix_arr      = trix(price_list, 15)
        cmo_arr       = chande_momentum_oscillator(price_list, 14)
        atr_arr       = atr_approx(price_list, 14)
        ts_arr        = trend_strength(price_list, 20)
        bb_u, _, bb_l = bollinger_bands(price_list, 20)
        mr_arr        = mean_reversion_score(price_list, 20)
        vr_arr        = volatility_ratio(price_list, 5, 20)

        # --- assemble feature matrix by reading each index ---
        X_rows = []
        y_rows = []
        for i in range(min_idx, n - prediction_horizon):
            p = prices[i]
            atr_val = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0
            bb_width = bb_u[i] - bb_l[i] if (not np.isnan(bb_u[i]) and not np.isnan(bb_l[i])) else 0.0

            row = np.array([
                float(rsi_arr[i])   if not np.isnan(rsi_arr[i])   else 50.0,
                float(macd_h[i])    if not np.isnan(macd_h[i])    else 0.0,
                float(stoch_k[i])   if not np.isnan(stoch_k[i])   else 50.0,
                float(cci_arr[i]),
                float(roc_arr[i]),
                float(mom_arr[i]),
                float(wr_arr[i]),
                float(uo_arr[i]),
                float(trix_arr[i]),
                float(cmo_arr[i]),
                atr_val / p if p > 0 else 0.0,
                float(ts_arr[i]),
                (p - bb_l[i]) / bb_width if bb_width > 0 else 0.5,
                float(mr_arr[i]),
                float(vr_arr[i]),
            ], dtype=float)

            future_price = prices[i + prediction_horizon]
            label = 1 if future_price > p else 0
            X_rows.append(row)
            y_rows.append(label)

        if len(X_rows) < 10:
            return None, None
        X = np.nan_to_num(np.array(X_rows), nan=0.0, posinf=0.0, neginf=0.0)
        return X, np.array(y_rows)

    # ----- run walk-forward -----
    def run(self) -> Dict:
        """Execute walk-forward validation; return results dict."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        print(f"\n  Loading daily price data from historical_prices_2yr.csv …")
        df = self._load_daily_prices()

        sym_counts = df.groupby("symbol").size().sort_values(ascending=False)
        symbols = sym_counts.index.tolist()
        print(f"  Found {len(symbols)} symbols, processing all …")
        print(f"  Pre-computing features for each symbol (single pass) …")

        all_results = []
        per_symbol_results = {}

        for sym in symbols:
            sym_df = df[df["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
            if len(sym_df) < self.train_days + self.test_days + 40:
                continue

            dates = sym_df["timestamp"].values
            prices = sym_df["price"].values.astype(float)

            # Pre-compute ALL features once
            X_all, y_all = self._build_all_features(prices)
            if X_all is None:
                continue
            n_samples = len(X_all)

            sym_windows = []
            start = 0
            while start + self.train_days + self.test_days <= n_samples:
                train_end = start + self.train_days
                test_end  = min(train_end + self.test_days, n_samples)

                X_train = X_all[start:train_end]
                y_train = y_all[start:train_end]
                X_test  = X_all[train_end:test_end]
                y_test  = y_all[train_end:test_end]

                if len(X_train) < 20 or len(X_test) < 5:
                    start += self.step_days
                    continue

                # Check for degenerate labels (all same class)
                if len(np.unique(y_train)) < 2:
                    start += self.step_days
                    continue

                model = RandomForestClassifier(
                    n_estimators=50, max_depth=4, random_state=42, n_jobs=1
                )
                model.fit(X_train, y_train)

                y_pred_train = model.predict(X_train)
                y_pred_test  = model.predict(X_test)

                acc_train = accuracy_score(y_train, y_pred_train)
                acc_test  = accuracy_score(y_test, y_pred_test)

                # False positive rate
                fp = int(np.sum((y_pred_test == 1) & (y_test == 0)))
                actual_neg = int(np.sum(y_test == 0))
                fpr = fp / actual_neg if actual_neg > 0 else 0.0

                correct_mask = y_pred_test == y_test
                ev_proxy = float(np.mean(correct_mask) * 2 - 1)

                # Map feature indices back to approximate dates
                feat_offset = 30  # lookback used in feature build
                train_start_idx = start + feat_offset
                test_end_idx = min(train_end + self.test_days + feat_offset, len(dates) - 1)
                window_start_date = str(pd.Timestamp(dates[min(train_start_idx, len(dates)-1)]).date())
                window_end_date   = str(pd.Timestamp(dates[min(test_end_idx, len(dates)-1)]).date())

                result = {
                    "symbol": sym,
                    "window_start": window_start_date,
                    "window_end": window_end_date,
                    "train_samples": int(len(X_train)),
                    "test_samples": int(len(X_test)),
                    "is_accuracy": round(acc_train, 4),
                    "oos_accuracy": round(acc_test, 4),
                    "overfit_delta": round(acc_train - acc_test, 4),
                    "false_positive_rate": round(fpr, 4),
                    "ev_proxy": round(ev_proxy, 4),
                }
                sym_windows.append(result)
                all_results.append(result)
                start += self.step_days

            if sym_windows:
                is_accs  = [w["is_accuracy"] for w in sym_windows]
                oos_accs = [w["oos_accuracy"] for w in sym_windows]
                per_symbol_results[sym] = {
                    "num_windows": len(sym_windows),
                    "mean_is_accuracy": round(float(np.mean(is_accs)), 4),
                    "mean_oos_accuracy": round(float(np.mean(oos_accs)), 4),
                    "std_oos_accuracy": round(float(np.std(oos_accs)), 4),
                    "min_oos_accuracy": round(float(np.min(oos_accs)), 4),
                    "max_oos_accuracy": round(float(np.max(oos_accs)), 4),
                    "mean_overfit_delta": round(float(np.mean([w["overfit_delta"] for w in sym_windows])), 4),
                    "mean_fpr": round(float(np.mean([w["false_positive_rate"] for w in sym_windows])), 4),
                }
                print(f"    {sym:12s}: {len(sym_windows):3d} windows | "
                      f"IS acc {np.mean(is_accs):.1%} -> OOS acc {np.mean(oos_accs):.1%} "
                      f"(delta {np.mean(is_accs) - np.mean(oos_accs):+.1%})")

        # Aggregate
        if all_results:
            all_is  = [r["is_accuracy"] for r in all_results]
            all_oos = [r["oos_accuracy"] for r in all_results]
            all_fpr = [r["false_positive_rate"] for r in all_results]
            all_of  = [r["overfit_delta"] for r in all_results]
            aggregate = {
                "total_windows": len(all_results),
                "unique_symbols_tested": len(per_symbol_results),
                "train_window_days": self.train_days,
                "test_window_days": self.test_days,
                "step_days": self.step_days,
                "mean_is_accuracy": round(float(np.mean(all_is)), 4),
                "mean_oos_accuracy": round(float(np.mean(all_oos)), 4),
                "std_oos_accuracy": round(float(np.std(all_oos)), 4),
                "median_oos_accuracy": round(float(np.median(all_oos)), 4),
                "mean_overfit_delta": round(float(np.mean(all_of)), 4),
                "mean_false_positive_rate": round(float(np.mean(all_fpr)), 4),
                "pct_windows_above_55": round(float(np.mean(np.array(all_oos) > 0.55)), 4),
                "pct_windows_above_50": round(float(np.mean(np.array(all_oos) > 0.50)), 4),
            }
        else:
            aggregate = {"error": "No valid walk-forward windows generated"}

        return {
            "aggregate": aggregate,
            "per_symbol": per_symbol_results,
            "all_windows": all_results,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Report + JSON Export
# ═══════════════════════════════════════════════════════════════════════════

def print_monte_carlo_report(mc: Dict):
    """Pretty-print Monte Carlo results."""
    print(f"\n{SEPARATOR}")
    print("  PART 1 — MONTE CARLO TRADE SIMULATION")
    print(SEPARATOR)
    print(f"  Simulations:              {mc['num_simulations']:>10,}")
    print(f"  Trades per simulation:    {mc['num_trades_per_sim']:>10,}")
    print(f"  Probability of profit:    {mc['probability_of_profit']:>10.1%}")
    print(f"  Probability of ruin (50%): {mc['probability_of_ruin_50pct']:>9.1%}")
    print(f"  Expected value / trade:  ${mc['expected_value_per_trade_usd']:>10.4f}")
    print(f"  CVaR (5%):               ${mc['cvar_5pct_usd']:>10.2f}")
    print(f"  Mean terminal P&L:       ${mc['mean_terminal_pnl_usd']:>10.2f}")
    print(f"  Median terminal P&L:     ${mc['median_terminal_pnl_usd']:>10.2f}")
    print(f"  Std terminal P&L:        ${mc['std_terminal_pnl_usd']:>10.2f}")
    print(f"  Mean max drawdown:       ${mc['mean_max_drawdown_usd']:>10.2f}")
    print(f"  Mean Sharpe ratio:        {mc['mean_sharpe_ratio']:>10.3f}")
    print(f"  Mean profit factor:       {mc['mean_profit_factor']:>10.3f}")
    print(f"  Avg win streak:           {mc['avg_longest_win_streak']:>10.1f}")
    print(f"  Avg loss streak:          {mc['avg_longest_loss_streak']:>10.1f}")

    print(f"\n  {'Percentile':>12s} {'Term P&L':>12s} {'Max DD':>10s} "
          f"{'Win Rate':>10s} {'Sharpe':>8s} {'PF':>8s}")
    print(f"  {'-'*12:>12s} {'-'*12:>12s} {'-'*10:>10s} "
          f"{'-'*10:>10s} {'-'*8:>8s} {'-'*8:>8s}")
    ci = mc["confidence_intervals"]
    for pct_label, vals in ci.items():
        print(f"  {pct_label:>12s} ${vals['terminal_pnl']:>10.2f} "
              f"${vals['max_drawdown']:>8.2f} "
              f"{vals['win_rate']:>10.1%} "
              f"{vals['sharpe_ratio']:>8.3f} "
              f"{vals['profit_factor']:>8.3f}")


def print_symbol_breakdown(breakdown: Dict):
    """Pretty-print per-symbol Monte Carlo."""
    print(f"\n  Per-Symbol Monte Carlo Breakdown:")
    print(f"  {'Symbol':>14s} {'Trades':>7s} {'Actual':>10s} "
          f"{'Mean Sim':>10s} {'P5':>10s} {'P95':>10s} {'P(profit)':>10s}")
    print(f"  {'-'*14} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for sym in sorted(breakdown, key=lambda s: breakdown[s]["actual_total_pnl"], reverse=True):
        s = breakdown[sym]
        print(f"  {sym:>14s} {s['n_trades']:>7d} ${s['actual_total_pnl']:>8.2f} "
              f"${s['mean_sim_pnl']:>8.2f} ${s['p5_sim_pnl']:>8.2f} "
              f"${s['p95_sim_pnl']:>8.2f} {s['prob_profit']:>10.1%}")


def print_exit_reason_stats(stats: Dict):
    """Pretty-print exit reason stats."""
    print(f"\n  Exit Reason Analysis:")
    print(f"  {'Reason':>22s} {'Count':>7s} {'Total PnL':>12s} "
          f"{'Mean PnL':>10s} {'Win Rate':>10s}")
    print(f"  {'-'*22} {'-'*7} {'-'*12} {'-'*10} {'-'*10}")
    for reason in sorted(stats, key=lambda r: stats[r]["total_pnl"], reverse=True):
        s = stats[reason]
        print(f"  {reason:>22s} {s['count']:>7d} ${s['total_pnl']:>10.2f} "
              f"${s['mean_pnl']:>8.2f} {s['win_rate']:>10.1%}")


def print_walk_forward_report(wf: Dict):
    """Pretty-print walk-forward results."""
    print(f"\n{SEPARATOR}")
    print("  PART 2 — WALK-FORWARD ML VALIDATION")
    print(SEPARATOR)
    agg = wf["aggregate"]
    if "error" in agg:
        print(f"  {agg['error']}")
        return

    print(f"  Total windows evaluated:  {agg['total_windows']:>10,}")
    print(f"  Symbols tested:           {agg['unique_symbols_tested']:>10,}")
    print(f"  Train / Test / Step:      {agg['train_window_days']}d / "
          f"{agg['test_window_days']}d / {agg['step_days']}d")
    print(f"  Mean IS accuracy:         {agg['mean_is_accuracy']:>10.1%}")
    print(f"  Mean OOS accuracy:        {agg['mean_oos_accuracy']:>10.1%}")
    print(f"  Std OOS accuracy:         {agg['std_oos_accuracy']:>10.4f}")
    print(f"  Median OOS accuracy:      {agg['median_oos_accuracy']:>10.1%}")
    print(f"  Mean overfit delta:       {agg['mean_overfit_delta']:>+10.1%}")
    print(f"  Mean false positive rate: {agg['mean_false_positive_rate']:>10.1%}")
    print(f"  Windows > 55% OOS acc:    {agg['pct_windows_above_55']:>10.1%}")
    print(f"  Windows > 50% OOS acc:    {agg['pct_windows_above_50']:>10.1%}")

    # Per-symbol summary
    ps = wf.get("per_symbol", {})
    if ps:
        print(f"\n  {'Symbol':>14s} {'Windows':>8s} {'IS Acc':>8s} {'OOS Acc':>8s} "
              f"{'Overfit':>8s} {'FPR':>8s} {'OOS Std':>8s}")
        print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for sym in sorted(ps, key=lambda s: ps[s]["mean_oos_accuracy"], reverse=True):
            s = ps[sym]
            print(f"  {sym:>14s} {s['num_windows']:>8d} "
                  f"{s['mean_is_accuracy']:>8.1%} {s['mean_oos_accuracy']:>8.1%} "
                  f"{s['mean_overfit_delta']:>+8.1%} {s['mean_fpr']:>8.1%} "
                  f"{s['std_oos_accuracy']:>8.4f}")


def save_results(results: Dict, path: Path):
    """Save results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Backtester with Monte Carlo Simulation"
    )
    parser.add_argument("--sims", type=int, default=NUM_SIMS_DEFAULT,
                        help=f"Number of MC simulations (default: {NUM_SIMS_DEFAULT})")
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS_DEFAULT,
                        help=f"Walk-forward train window in days (default: {TRAIN_DAYS_DEFAULT})")
    parser.add_argument("--test-days", type=int, default=TEST_DAYS_DEFAULT,
                        help=f"Walk-forward test window in days (default: {TEST_DAYS_DEFAULT})")
    parser.add_argument("--step-days", type=int, default=STEP_DAYS_DEFAULT,
                        help=f"Walk-forward step size in days (default: {STEP_DAYS_DEFAULT})")
    parser.add_argument("--skip-wf", action="store_true",
                        help="Skip walk-forward ML validation (Part 2)")
    parser.add_argument("--skip-mc", action="store_true",
                        help="Skip Monte Carlo simulation (Part 1)")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n{'#' * 72}")
    print(f"  WALK-FORWARD BACKTESTER WITH MONTE CARLO SIMULATION")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 72}")

    results = {
        "run_timestamp": start_time.isoformat(),
        "config": {
            "num_sims": args.sims,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "paper_balance_spot": config.PAPER_BALANCE_SPOT,
            "paper_balance_futures": config.PAPER_BALANCE_FUTURES,
            "spot_fee_rate": config.SPOT_FEE_RATE,
            "futures_fee_rate": config.FUTURES_FEE_RATE,
        },
    }

    # ── Part 1: Monte Carlo ───────────────────────────────────────────────
    if not args.skip_mc:
        if not TRADES_CSV.exists():
            print(f"\n  ERROR: trades.csv not found at {TRADES_CSV}")
            return

        trades_df = pd.read_csv(TRADES_CSV)
        print(f"\n  Loaded {len(trades_df)} trades from {TRADES_CSV.name}")
        print(f"  Date range: {trades_df['timestamp'].iloc[0]} -> "
              f"{trades_df['timestamp'].iloc[-1]}")
        print(f"  Symbols: {trades_df['symbol'].nunique()} unique")
        print(f"  Directions: {dict(trades_df['direction'].value_counts())}")

        # Quick actual stats
        actual_pnl = trades_df["pnl_usd"].astype(float)
        actual_win_rate = float((actual_pnl > 0).mean())
        print(f"\n  Actual performance:")
        print(f"    Total P&L:  ${actual_pnl.sum():.2f}")
        print(f"    Win rate:   {actual_win_rate:.1%}")
        print(f"    Avg win:    ${actual_pnl[actual_pnl > 0].mean():.2f}")
        print(f"    Avg loss:   ${actual_pnl[actual_pnl < 0].mean():.2f}")

        mc_engine = MonteCarloEngine(trades_df, num_sims=args.sims)
        mc_results = mc_engine.run()

        # Breakdowns
        symbol_bk = mc_engine.symbol_breakdown()
        exit_stats = mc_engine.exit_reason_stats()

        # Report
        print_monte_carlo_report(mc_results)
        print_symbol_breakdown(symbol_bk)
        print_exit_reason_stats(exit_stats)

        results["monte_carlo"] = mc_results
        results["symbol_breakdown"] = symbol_bk
        results["exit_reason_stats"] = exit_stats
    else:
        print("\n  Skipping Monte Carlo (--skip-mc)")

    # ── Part 2: Walk-Forward ML Validation ────────────────────────────────
    if not args.skip_wf:
        if not HISTORICAL_CSV.exists():
            print(f"\n  ERROR: historical_prices_2yr.csv not found at {HISTORICAL_CSV}")
        else:
            wf = WalkForwardValidator(
                train_days=args.train_days,
                test_days=args.test_days,
                step_days=args.step_days,
            )
            wf_results = wf.run()

            print_walk_forward_report(wf_results)

            # Store (omit per-window detail from JSON to keep it manageable)
            results["walk_forward"] = {
                "aggregate": wf_results["aggregate"],
                "per_symbol": wf_results["per_symbol"],
                "sample_windows": wf_results["all_windows"][:20],  # first 20 as sample
            }
    else:
        print("\n  Skipping walk-forward ML validation (--skip-wf)")

    # ── Part 3: Save ──────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    results["elapsed_seconds"] = round(elapsed, 1)

    save_results(results, OUTPUT_JSON)

    print(f"\n{SEPARATOR}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{SEPARATOR}\n")


if __name__ == "__main__":
    main()
