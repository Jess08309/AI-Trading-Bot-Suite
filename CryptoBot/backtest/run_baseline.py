"""
CryptoBot backtest driver: short-vs-long baseline over real historical data.

Runs the repository backtest engine (cryptotrades/utils/backtester.py's
SpotBacktester / FuturesBacktester / run_full_backtest, driven by
MarketPredictor and utils.config, NOT TradingBot's internal MLModel/cfg) against:
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

Usage:
    cd CryptoBot
    SIM_REALISM_PROFILE=strict python3 backtest/run_baseline.py
    SIM_REALISM_PROFILE=strict python3 backtest/run_baseline.py --model data/models/market_model.joblib
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
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
EXECUTION_FIELDS = (
    "ENABLE_EXECUTION_COSTS", "SPOT_SLIPPAGE_BPS", "FUTURES_SLIPPAGE_BPS",
    "SPOT_FEE_RATE", "FUTURES_FEE_RATE", "ENABLE_PARTIAL_FILLS",
    "PARTIAL_FILL_PROB", "PARTIAL_FILL_MIN", "PARTIAL_FILL_MAX",
    "ENABLE_FUNDING_COSTS", "FUTURES_FUNDING_RATE_PER_8H",
)


def execution_settings(config=None) -> dict:
    if config is None:
        config = live_config
    return {name: getattr(config, name) for name in EXECUTION_FIELDS}


def strict_execution_enabled() -> bool:
    """Reject overrides weakening the profile, without mutating shared config."""
    if live_config.SIM_REALISM_PROFILE != "strict":
        return False
    strict = copy.copy(live_config)
    strict._apply_realism_profile()
    settings = execution_settings()
    numeric = [value for value in settings.values() if not isinstance(value, bool)]
    return (
        all(math.isfinite(value) and value >= 0 for value in numeric)
        and 0 < live_config.PARTIAL_FILL_MIN <= live_config.PARTIAL_FILL_MAX <= 1
        and 0 <= live_config.PARTIAL_FILL_PROB <= 1
        and settings == execution_settings(strict)
    )


def parse_date(value: str) -> int:
    """Parse an ISO date/time; timestamps without an offset are UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def load_csv_prices(csv_path: str, start_ts=None, end_ts=None, coverage=None) -> List[float]:
    """Load the 'close' column from a downloader-produced CSV, oldest first."""
    closes = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: int(r["timestamp"]))
        timestamps = []
        for row in rows:
            ts = int(row["timestamp"])
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts >= end_ts:
                continue
            timestamps.append(ts)
            closes.append(float(row["close"]))
    if coverage is not None:
        unique = sorted(set(timestamps))
        first = unique[0] if unique else None
        last = unique[-1] if unique else None
        lower = start_ts if start_ts is not None else first
        upper = end_ts if end_ts is not None else (last + 60 if last is not None else None)
        expected = max(0, (upper - lower + 59) // 60) if lower is not None and upper is not None else 0
        coverage.update({
            "start_timestamp": first,
            "end_timestamp": last,
            "rows": len(timestamps),
            "unique_minutes": len(unique),
            "expected_minutes": expected,
            "coverage_pct": 100.0 * len(unique) / expected if expected else None,
            "largest_gap_minutes": max(
                ((b - a) / 60 for a, b in zip(unique, unique[1:])), default=0
            ),
        })
    return closes


def load_price_dir(directory: str, suffix: str = "_1min.csv",
                   start_ts=None, end_ts=None, coverage=None) -> Dict[str, List[float]]:
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
        symbol_coverage = {}
        prices = load_csv_prices(path, start_ts, end_ts, symbol_coverage)
        if coverage is not None:
            coverage[symbol] = symbol_coverage
        if len(prices) < 200:
            print(f"  {symbol}: only {len(prices)} candles, skipping (need >= 200)")
            continue
        data[symbol] = prices
        print(f"  {symbol}: loaded {len(prices):,} candles")
    return data


def classify_window(model_meta, coverage) -> str:
    """Label temporal separation conservatively, never infer unknown training dates."""
    if not model_meta or not model_meta.get("actual_data_end"):
        return "unknown"
    training_end = parse_date(model_meta["actual_data_end"])
    training_start = model_meta.get("actual_data_start")
    windows = [
        item for side in coverage.values() for item in side.values()
        if item["rows"] >= 200 and item["start_timestamp"] is not None
    ]
    if not windows:
        return "unknown"
    if all(item["start_timestamp"] > training_end for item in windows):
        return "out_of_sample"
    if training_start is None:
        return "unknown"
    training_start = parse_date(training_start)
    if any(item["start_timestamp"] <= training_end and
           item["end_timestamp"] >= training_start for item in windows):
        return "in_sample"
    return "unknown"


def aggregate_stats(results) -> dict:
    """Pool closed trades, not independent symbol equity curves."""
    results = list(results)
    trades = [trade for result in results for trade in result.trades]
    gross_profit = sum(max(0, trade.pnl_usd) for trade in trades)
    gross_loss = sum(max(0, -trade.pnl_usd) for trade in trades)
    return {
        "num_trades": len(trades),
        "win_rate": sum(trade.pnl_usd > 0 for trade in trades) / len(trades) if trades else 0.0,
        "total_return_usd": sum(result.total_return_usd for result in results),
        "profit_factor": gross_profit / gross_loss if gross_loss else (
            float("inf") if gross_profit else 0.0
        ),
        "worst_symbol_max_drawdown_pct": max(
            (result.max_drawdown_pct for result in results), default=0.0
        ),
        "mean_symbol_sharpe_ratio": (
            sum(result.sharpe_ratio for result in results) / len(results) if results else 0.0
        ),
        "exit_reasons": dict(Counter(trade.exit_reason for trade in trades)),
        "equity_metrics_note": "Independent symbol accounts: worst-symbol drawdown and "
                              "mean-symbol Sharpe are NOT portfolio drawdown or Sharpe.",
    }


def load_coverage(directory: str) -> Dict[str, dict]:
    """Read tools/repair_gaps.py's gap_report.json (if present) for per-symbol
    post-repair coverage %, so the gate report can show data quality alongside
    trading results instead of silently assuming full coverage."""
    path = os.path.join(directory, "gap_report.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


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
    parser.add_argument("--output", default=REPORT_PATH,
                         help="Where to write the JSON report (default data/state/baseline_report.json). "
                              "Use a distinct path per run when comparing multiple models (e.g. frozen "
                              "gate model vs live trading_model.joblib) so reports don't overwrite each other.")
    parser.add_argument("--model-meta", default=None,
                         help="Path to a tools/retrain_on_6mo.py --meta-output JSON describing this "
                              "model's training window. Omit for models with no known/recorded training "
                              "window (e.g. the live continuously-retrained artifact).")
    parser.add_argument("--start-date", type=parse_date, default=None,
                        help="Inclusive test-window ISO timestamp (naive dates use UTC)")
    parser.add_argument("--end-date", type=parse_date, default=None,
                        help="Exclusive test-window ISO timestamp (naive dates use UTC)")
    parser.add_argument("--require-out-of-sample", action="store_true",
                        help="Refuse to run unless metadata proves a later test window and strict realism")
    args = parser.parse_args()
    if args.candle_size <= 0:
        parser.error("--candle-size must be positive")
    if args.start_date is not None and args.end_date is not None and args.start_date >= args.end_date:
        parser.error("--start-date must be before --end-date")
    model_training_window = None
    if args.model_meta:
        with open(args.model_meta, "r") as f:
            model_training_window = json.load(f)

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
    coverage = {"spot": {}, "futures": {}}
    spot_data = load_price_dir(SPOT_DIR, start_ts=args.start_date, end_ts=args.end_date,
                               coverage=coverage["spot"])

    print(f"\nLoading futures candles from {FUTURES_DIR} ...")
    futures_data = load_price_dir(FUTURES_DIR, start_ts=args.start_date, end_ts=args.end_date,
                                  coverage=coverage["futures"])

    if not spot_data and not futures_data:
        print("\nERROR: no historical data found. Run tools/download_6mo_candles.py "
              "and tools/download_kraken_futures_ohlc.py first.")
        sys.exit(1)

    evaluation_type = classify_window(model_training_window, coverage)
    print(f"\nTraining-window leakage label: {evaluation_type}")
    if args.require_out_of_sample and (
        evaluation_type != "out_of_sample" or not strict_execution_enabled()
    ):
        parser.error("--require-out-of-sample needs non-overlapping training metadata "
                     "and effective strict execution settings (no weakened overrides)")

    print("\nRunning repository SpotBacktester/FuturesBacktester with MarketPredictor. "
          "This does not replay TradingBot's internal MLModel/cfg; see "
          "docs/internal/BACKTEST_BASELINE_2026-10.md.")
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
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model_mtime = os.path.getmtime(args.model) if os.path.exists(args.model) else None

    model_training_window_note = (
        None if model_training_window else
        "No --model-meta supplied: training window is unknown. This run is not out-of-sample evidence."
    )

    report = {
        "sim_realism_profile": live_config.SIM_REALISM_PROFILE,
        "execution_settings": execution_settings(),
        "effective_strict_execution": strict_execution_enabled(),
        "model_path": args.model,
        "model_mtime": (
            datetime.fromtimestamp(model_mtime, tz=timezone.utc).isoformat() if model_mtime else None
        ),
        "model_training_window": model_training_window,
        "model_training_window_note": model_training_window_note,
        "evaluation_type": evaluation_type,
        "strategy_scope": "utils.backtester + MarketPredictor; not core.TradingBot",
        "test_window": {"start_inclusive": args.start_date, "end_exclusive": args.end_date},
        "data_coverage": coverage,
        "gap_repair_coverage": {"spot": load_coverage(SPOT_DIR), "futures": load_coverage(FUTURES_DIR)},
        "candle_size": args.candle_size,
        "aggregate": {
            "all": aggregate_stats(results.values()),
            "spot": aggregate_stats(r for r in results.values() if r.side == "spot"),
            "futures": aggregate_stats(r for r in results.values() if r.side == "futures"),
        },
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
                "exit_reasons": dict(Counter(t.exit_reason for t in r.trades)),
            }
            for key, r in results.items()
        },
        "futures_long_vs_short": {
            direction: _bucket_stats(b)
            for direction, b in long_short_buckets.items()
        },
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
