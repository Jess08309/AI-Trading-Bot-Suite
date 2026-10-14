#!/usr/bin/env python3
"""
Weekly time-series momentum (TSMOM) research backtest on daily crypto closes.

Rules (v1): 84-day return > 0 -> candidate; hold top N=3 by momentum;
weekly rebalance; weight = min(10%, VOL_TARGET / 20d realized vol).
Signals use close T, execution at close T+1 (no look-ahead). Long-only spot.

Pre-registered acceptance criteria are evaluated by ``evaluate_criteria``.

Usage:
    python tools/tsmom_backtest.py
    python tools/tsmom_backtest.py --csv path/to/prices.csv --fee-bps 25 --slip-bps 10
Output: data/backtest/tsmom_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
HISTORICAL_CSV = ROOT_DIR / "data" / "historical" / "historical_prices_2yr.csv"
BASELINE_JSON = ROOT_DIR / "data" / "state" / "baseline_report.json"
OUTPUT_JSON = ROOT_DIR / "data" / "backtest" / "tsmom_results.json"

LOOKBACK_DAYS = 84
REBALANCE_DAYS = 7
TOP_N = 3
VOL_WINDOW = 20
VOL_TARGET = 0.005  # daily vol contribution per position
MAX_WEIGHT = 0.10
FEE_BPS = 25.0
SLIPPAGE_BPS = 10.0
SENSITIVITY_WEEKS = (8, 12, 16, 20)
ANN = 365


def load_daily_closes(csv_path: Path = HISTORICAL_CSV) -> pd.DataFrame:
    """Return a date x symbol DataFrame of daily closes."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp", "symbol", "price"])
    df["date"] = df["timestamp"].dt.floor("D")
    daily = df.sort_values("timestamp").groupby(["date", "symbol"])["price"].last().unstack()
    return daily.sort_index()


def target_weights(closes: pd.DataFrame, t: int, lookback: int = LOOKBACK_DAYS,
                   top_n: int = TOP_N) -> pd.Series:
    """Target weights from information available at close index ``t``."""
    w = pd.Series(0.0, index=closes.columns)
    if t < max(lookback, VOL_WINDOW):
        return w
    px = closes.iloc[: t + 1]
    mom = px.iloc[-1] / px.iloc[-1 - lookback] - 1.0
    vol = px.pct_change().iloc[-VOL_WINDOW:].std()
    mom = mom[(mom > 0) & vol.notna() & (vol > 0)]
    for sym in mom.sort_values(ascending=False).index[:top_n]:
        w[sym] = min(MAX_WEIGHT, VOL_TARGET / vol[sym])
    return w


def run_backtest(closes: pd.DataFrame, lookback: int = LOOKBACK_DAYS,
                 top_n: int = TOP_N, fee_bps: float = FEE_BPS,
                 slip_bps: float = SLIPPAGE_BPS) -> Dict:
    """Simulate; returns daily net returns and per-position trade returns."""
    closes = closes.sort_index()
    rets = closes.pct_change().fillna(0.0)
    cost = (fee_bps + slip_bps) / 1e4
    n = len(closes)
    weights = pd.Series(0.0, index=closes.columns)
    pending: Optional[pd.Series] = None
    daily = np.zeros(n)
    trades: List[float] = []
    entry: Dict[str, float] = {}
    for t in range(1, n):
        # execute orders decided on close t-1 at close t
        if pending is not None:
            turnover = (pending - weights).abs()
            daily[t] -= float((turnover * cost).sum())
            for sym in closes.columns:
                if weights[sym] == 0 and pending[sym] > 0:
                    entry[sym] = closes[sym].iloc[t]
                elif weights[sym] > 0 and pending[sym] == 0 and sym in entry:
                    trades.append(closes[sym].iloc[t] / entry.pop(sym) - 1.0 - 2 * cost)
            weights = pending
            pending = None
        daily[t] += float((weights * rets.iloc[t]).sum())
        if t % REBALANCE_DAYS == 0:
            pending = target_weights(closes, t, lookback, top_n)
    for sym, px in entry.items():
        trades.append(closes[sym].iloc[-1] / px - 1.0 - 2 * cost)
    return {"daily": pd.Series(daily, index=closes.index), "trades": trades}


def sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ANN)) if sd and sd > 0 else 0.0


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((1 - eq / eq.cummax()).max())


def walk_forward_windows(daily: pd.Series, start: int, test_days: int = 60,
                         step_days: int = 60) -> List[pd.Series]:
    """Rolling out-of-sample windows. The strategy has no fitted parameters,
    so the warm-up period (``start``) takes the place of the train window."""
    out = []
    i = start
    while i + test_days <= len(daily):
        out.append(daily.iloc[i:i + test_days])
        i += step_days
    return out


def monte_carlo(trades: List[float], sims: int = 5000, seed: int = 42) -> Dict:
    if not trades:
        return {}
    rng = np.random.default_rng(seed)
    arr = np.asarray(trades)
    tot = np.array([np.prod(1 + rng.choice(arr, len(arr))) - 1 for _ in range(sims)])
    return {"p05": float(np.percentile(tot, 5)), "median": float(np.median(tot)),
            "p95": float(np.percentile(tot, 95)), "prob_profit": float((tot > 0).mean())}


def evaluate_criteria(closes: pd.DataFrame, fee_bps: float = FEE_BPS,
                      slip_bps: float = SLIPPAGE_BPS, test_days: int = 60) -> Dict:
    res = run_backtest(closes, fee_bps=fee_bps, slip_bps=slip_bps)
    daily = res["daily"]
    start = LOOKBACK_DAYS + 1
    oos = daily.iloc[start:]
    windows = walk_forward_windows(daily, start, test_days, test_days)
    win_pos = [float((1 + w).prod() - 1) > 0 for w in windows]
    sens = {}
    for wk in SENSITIVITY_WEEKS:
        d = run_backtest(closes, lookback=wk * 7, fee_bps=fee_bps, slip_bps=slip_bps)["daily"]
        sens[wk] = sharpe(d.iloc[wk * 7 + 1:])
    btc_col = next((c for c in closes.columns if str(c).upper().startswith("BTC")), None)
    btc = closes[btc_col].pct_change().fillna(0.0).iloc[start:] if btc_col else None
    n_trades = len(res["trades"])
    out = {
        "oos_sharpe": sharpe(oos),
        "oos_total_return": float((1 + oos).prod() - 1),
        "positive_window_frac": float(np.mean(win_pos)) if win_pos else 0.0,
        "n_windows": len(windows),
        "max_drawdown": max_drawdown(oos),
        "sensitivity_sharpe": sens,
        "n_trades": n_trades,
        "btc_buy_hold": {"sharpe": sharpe(btc), "total_return": float((1 + btc).prod() - 1),
                         "max_drawdown": max_drawdown(btc)} if btc is not None else None,
        "monte_carlo": monte_carlo(res["trades"]),
    }
    checks = {
        "sharpe_gt_0.8": out["oos_sharpe"] > 0.8,
        "positive_windows_ge_70pct": out["positive_window_frac"] >= 0.70,
        "max_dd_lt_40pct": out["max_drawdown"] < 0.40,
        "sensitivity_all_positive": all(v > 0 for v in sens.values()),
    }
    out["checks"] = checks
    out["enough_trades"] = n_trades >= 30
    out["verdict"] = "PASS" if all(checks.values()) and out["enough_trades"] else "REJECT"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csv", type=Path, default=HISTORICAL_CSV)
    ap.add_argument("--fee-bps", type=float, default=FEE_BPS)
    ap.add_argument("--slip-bps", type=float, default=SLIPPAGE_BPS)
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--out", type=Path, default=OUTPUT_JSON)
    args = ap.parse_args()
    if not args.csv.exists():
        print(f"ERROR: price data not found at {args.csv}")
        return 1
    result = evaluate_criteria(load_daily_closes(args.csv), args.fee_bps,
                               args.slip_bps, args.test_days)
    if BASELINE_JSON.exists():
        result["scalper_baseline"] = json.loads(BASELINE_JSON.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "scalper_baseline"},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
