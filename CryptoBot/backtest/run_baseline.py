"""
CryptoBot backtest driver: short-vs-long baseline over real historical data.

Runs the LIVE backtest engine (cryptotrades/utils/backtester.py's
SpotBacktester / FuturesBacktester / run_full_backtest, driven by the same
MarketPredictor and live config the production TradingBot uses) against:
  - Spot 1-min candles downloaded by tools/download_6mo_candles.py
    (data/historical/1min/)
  - Futures 1-min candles downloaded by tools/download_kraken_futures_ohlc.py
    (data/historical/1min_futures/)

Reports a per-side (spot vs futures, and within futures long vs short)
breakdown: trades, win rate, P&L, avg win/loss, profit factor, funding paid,
and exit-reason distribution -- to answer "is the live short-side futures
logic actually adding value, or should CryptoBot go long-only?"

Respects SIM_REALISM_PROFILE (env var, see cryptotrades/utils/config.py) --
run with SIM_REALISM_PROFILE=strict for the most conservative (worst-case)
execution-cost assumptions (higher slippage, partial fills, funding costs,
higher fees).

NOTE ON MODEL DISCLOSURE: --model / DEFAULT_MODEL_PATH here loads a
MarketPredictor model via config.ML_MODEL_PATH (data/models/market_model.joblib
by default). The LIVE trading bot (cryptotrades/core/trading_engine.py's
TradingEngine) loads a DIFFERENT artifact by default (models/trading_model.joblib,
via a different model-loading code path) and continuously self-retrains it on a
schedule -- the two are not the same model/pipeline. The persisted report below
records model_path + model_mtime for whichever artifact THIS script actually
loaded, precisely so this discrepancy can be checked/flagged rather than
silently assumed away when interpreting gate results.

Usage:
    cd CryptoBot
    SIM_REALISM_PROFILE=strict python3 backtest/run_baseline.py
    SIM_REALISM_PROFILE=strict python3 backtest/run_baseline.py --model data/models/market_model.joblib
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

# Make cryptotrades importable whether run from CryptoBot/ or repo root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CryptoBot/
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "cryptotrades"))

from cryptotrades.utils.config import config as live_config
from cryptotrades.utils.market_predictor import MarketPredictor
from cryptotrades.utils.backtester import (
    run_full_backtest,
    print_aggregate_summary,
    BacktestResult,
)

SPOT_DIR = os.path.join(BASE_DIR, "data", "historical", "1min")
FUTURES_DIR = os.path.join(BASE_DIR, "data", "historical", "1min_futures")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, live_config.ML_MODEL_PATH)
REPORT_PATH = os.path.join(BASE_DIR, "data", "state", "baseline_report.json")


def load_csv_prices(csv_path: str) -> List[float]:
    """Load the 'close' column from a downloader-produced CSV, oldest first."""
    closes = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: int(r["timestamp"]))
        for row in rows:
            closes.append(float(row["close"]))
    return closes


def load_price_dir(directory: str, suffix: str = "_1min.csv") -> Dict[str, List[float]]:
    """Load all per-symbol CSVs in a directory into {symbol: [closes]}."""
    data = {}
    if not os.path.isdir(directory):
        print(f"WARNING: {directory} does not exist -- skipping (did you run the downloader?)")
        return data
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(suffix):
            continue
        symbol = fname[: -len(suffix)]
        path = os.path.join(directory, fname)
        prices = load_csv_prices(path)
        if len(prices) < 200:
            print(f"  {symbol}: only {len(prices)} candles, skipping (need >= 200)")
            continue
        data[symbol] = prices
        print(f"  {symbol}: loaded {len(prices):,} candles")
    return data


def split_futures_by_direction(results: Dict[str, BacktestResult]) -> Dict[str, dict]:
    """Aggregate FUTURES_* results' trades into long-only vs short-only buckets."""
    buckets = {
        "long": {"trades": 0, "wins": 0, "pnl_usd": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
                 "exit_reasons": defaultdict(int)},
        "short": {"trades": 0, "wins": 0, "pnl_usd": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
                  "exit_reasons": defaultdict(int)},
    }
    for key, result in results.items():
        if not key.startswith("FUTURES_"):
            continue
        for t in result.trades:
            b = buckets[t.direction]
            b["trades"] += 1
            b["pnl_usd"] += t.pnl_usd
            b["exit_reasons"][t.exit_reason] += 1
            if t.pnl_usd > 0:
                b["wins"] += 1
                b["gross_profit"] += t.pnl_usd
            else:
                b["gross_loss"] += abs(t.pnl_usd)
    return buckets


def _bucket_stats(b: dict) -> dict:
    trades = b["trades"]
    win_rate = (b["wins"] / trades * 100.0) if trades else 0.0
    profit_factor = (b["gross_profit"] / b["gross_loss"]) if b["gross_loss"] > 0 else (
        float("inf") if b["gross_profit"] > 0 else 0.0
    )
    return {
        "trades": trades,
        "wins": b["wins"],
        "win_rate": win_rate,
        "pnl_usd": b["pnl_usd"],
        "gross_profit": b["gross_profit"],
        "gross_loss": b["gross_loss"],
        "profit_factor": profit_factor,
        "exit_reasons": dict(b["exit_reasons"]),
    }


def print_long_vs_short(buckets: Dict[str, dict]):
    print(f"\n{'='*70}")
    print(f"{'FUTURES: LONG vs SHORT BASELINE':^70}")
    print(f"{'='*70}")
    for direction in ("long", "short"):
        stats = _bucket_stats(buckets[direction])
        print(f"\n  {direction.upper()}:")
        print(f"    Trades:        {stats['trades']}")
        print(f"    Win rate:      {stats['win_rate']:.1f}%")
        print(f"    Net P&L:       ${stats['pnl_usd']:+,.2f}")
        print(f"    Profit factor: {stats['profit_factor']:.2f}")
        if stats["exit_reasons"]:
            print(f"    Exit reasons:  {stats['exit_reasons']}")
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                         help="Path to trained MarketPredictor model (.joblib)")
    parser.add_argument("--candle-size", type=int, default=10,
                         help="Minutes per decision candle (default 10, matches live TRADE_INTERVAL)")
    args = parser.parse_args()

    print(f"SIM_REALISM_PROFILE = {live_config.SIM_REALISM_PROFILE}")
    print(live_config.summary())

    predictor = MarketPredictor(model_path=args.model)
    if not predictor.load_model():
        print(f"\nERROR: could not load model from {args.model}")
        print("Run tools/retrain_on_6mo.py first to produce a trained model on the "
              "6-month historical data, then re-run this script.")
        sys.exit(1)
    print(f"\nLoaded model from {args.model}")

    print(f"\nLoading spot candles from {SPOT_DIR} ...")
    spot_data = load_price_dir(SPOT_DIR)

    print(f"\nLoading futures candles from {FUTURES_DIR} ...")
    futures_data = load_price_dir(FUTURES_DIR)

    if not spot_data and not futures_data:
        print("\nERROR: no historical data found. Run tools/download_6mo_candles.py "
              "and tools/download_kraken_futures_ohlc.py first.")
        sys.exit(1)

    print("\nRunning backtest (this uses the live SpotBacktester/FuturesBacktester "
          "engine and live config, not a hand-duplicated copy)...")
    results = run_full_backtest(
        predictor,
        price_data=spot_data,
        futures_data=futures_data,
        candle_size=args.candle_size,
        starting_balance_spot=live_config.PAPER_BALANCE_SPOT,
        starting_balance_futures=live_config.PAPER_BALANCE_FUTURES,
        verbose=False,
    )

    print_aggregate_summary(results)

    long_short_buckets = split_futures_by_direction(results)
    print_long_vs_short(long_short_buckets)

    # Persist full report for later reference / PR writeup.
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    model_mtime = os.path.getmtime(args.model) if os.path.exists(args.model) else None
    report = {
        "sim_realism_profile": live_config.SIM_REALISM_PROFILE,
        "model_path": args.model,
        "model_mtime": (
            datetime.fromtimestamp(model_mtime, tz=timezone.utc).isoformat() if model_mtime else None
        ),
        "candle_size": args.candle_size,
        "per_symbol": {
            key: {
                "side": r.side,
                "num_trades": r.num_trades,
                "win_rate": r.win_rate,
                "total_return_pct": r.total_return_pct,
                "total_return_usd": r.total_return_usd,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "profit_factor": r.profit_factor,
            }
            for key, r in results.items()
        },
        "futures_long_vs_short": {
            direction: _bucket_stats(b)
            for direction, b in long_short_buckets.items()
        },
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
