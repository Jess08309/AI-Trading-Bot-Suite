"""
Backtest Runner - Test the dual-timeframe strategy on historical price data.

Usage:
    python run_backtest.py                  # Run on all symbols
    python run_backtest.py --symbol BTC-USD # Run on one symbol
    python run_backtest.py --verbose        # Show every trade
    python run_backtest.py --candles 5      # Test 5-min candles instead of 10

Reads price_history.csv from the bot's data directory and replays
the trading strategy to measure performance before going live.
"""

import sys
import os
import csv
import argparse
import logging
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cryptotrades.utils.market_predictor import MarketPredictor
from cryptotrades.utils.backtester import (
    SpotBacktester, FuturesBacktester,
    run_full_backtest, print_aggregate_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_runner")


def load_price_data(csv_path: str):
    """Load price_history.csv into {symbol: [prices]} dicts for spot and futures."""
    spot_prices = defaultdict(list)
    futures_prices = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["symbol"]
            price = float(row["price"])
            ptype = row.get("type", "spot")

            if ptype == "futures":
                futures_prices[symbol].append(price)
            else:
                spot_prices[symbol].append(price)

    logger.info(f"Loaded {sum(len(v) for v in spot_prices.values())} spot prices "
                f"across {len(spot_prices)} symbols")
    logger.info(f"Loaded {sum(len(v) for v in futures_prices.values())} futures prices "
                f"across {len(futures_prices)} symbols")

    return dict(spot_prices), dict(futures_prices)


def main():
    parser = argparse.ArgumentParser(description="Backtest the crypto trading strategy")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Test a specific symbol (e.g., BTC-USD or PI_XBTUSD)")
    parser.add_argument("--candles", type=int, default=10,
                        help="Candle size in minutes (default: 10)")
    parser.add_argument("--balance", type=float, default=2500.0,
                        help="Starting balance per side (default: 2500)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show every trade detail")
    parser.add_argument("--spot-only", action="store_true",
                        help="Only run spot backtest")
    parser.add_argument("--futures-only", action="store_true",
                        help="Only run futures backtest")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to price_history.csv (auto-detected if not set)")
    args = parser.parse_args()

    # Ensure working directory is the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Find price data
    csv_path = args.csv
    if csv_path is None:
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "cryptotrades", "price_history.csv"),
            os.path.join(os.path.dirname(__file__), "price_history.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break

    if csv_path is None or not os.path.exists(csv_path):
        logger.error("Cannot find price_history.csv. Use --csv to specify path.")
        sys.exit(1)

    logger.info(f"Loading data from: {csv_path}")
    spot_data, futures_data = load_price_data(csv_path)

    # Filter to specific symbol if requested
    if args.symbol:
        sym = args.symbol
        filtered_spot = {k: v for k, v in spot_data.items() if sym.upper() in k.upper()}
        filtered_futures = {k: v for k, v in futures_data.items() if sym.upper() in k.upper()}
        if not filtered_spot and not filtered_futures:
            logger.error(f"Symbol '{sym}' not found. Available: "
                         f"Spot={list(spot_data.keys())}, "
                         f"Futures={list(futures_data.keys())}")
            sys.exit(1)
        spot_data = filtered_spot
        futures_data = filtered_futures

    # Initialize ML predictor (loads trained model)
    logger.info("Loading ML model...")
    predictor = MarketPredictor()
    predictor.load_model()

    if predictor.model is None:
        logger.warning("No trained ML model found. Training on price history CSV...")
        success = predictor.train(csv_path)
        if success:
            logger.info("Model trained successfully")
        else:
            logger.error("Failed to train model. Check price_history.csv format.")
            sys.exit(1)

    # Run backtest
    print(f"\n{'='*70}")
    print(f"{'DUAL-TIMEFRAME BACKTEST':^70}")
    print(f"{'='*70}")
    print(f"  Candle Size:    {args.candles} minutes")
    print(f"  Risk Checks:    Every 1 minute (stop-loss, trailing, take-profit)")
    print(f"  Trade Signals:  Every {args.candles} minutes (ML + ensemble scoring)")
    print(f"  Starting Bal:   ${args.balance:,.2f} per side")
    print(f"  Data Points:    {sum(len(v) for v in spot_data.values()) + sum(len(v) for v in futures_data.values()):,}")
    print(f"{'='*70}")

    spot_results = {}
    futures_results = {}

    if not args.futures_only:
        print(f"\n--- SPOT BACKTEST ---")
        spot_bt = SpotBacktester(
            predictor,
            candle_size=args.candles,
            starting_balance=args.balance,
        )
        for symbol, prices in sorted(spot_data.items()):
            if len(prices) < 200:
                logger.info(f"  {symbol}: skipped (only {len(prices)} bars)")
                continue
            logger.info(f"  Testing {symbol} ({len(prices)} bars, "
                        f"{len(prices)/60:.1f} hours)...")
            result = spot_bt.run(symbol, prices, verbose=args.verbose)
            spot_results[f"SPOT_{symbol}"] = result
            print(result.summary())

    if not args.spot_only:
        print(f"\n--- FUTURES BACKTEST ---")
        futures_bt = FuturesBacktester(
            predictor,
            candle_size=args.candles,
            starting_balance=args.balance,
        )
        for symbol, prices in sorted(futures_data.items()):
            if len(prices) < 200:
                logger.info(f"  {symbol}: skipped (only {len(prices)} bars)")
                continue
            logger.info(f"  Testing {symbol} ({len(prices)} bars, "
                        f"{len(prices)/60:.1f} hours)...")
            result = futures_bt.run(symbol, prices, verbose=args.verbose)
            futures_results[f"FUTURES_{symbol}"] = result
            print(result.summary())

    # Aggregate summary
    all_results = {**spot_results, **futures_results}
    print_aggregate_summary(all_results)

    # Recommendation
    profitable = [k for k, v in all_results.items()
                  if v.num_trades > 0 and v.total_return_pct > 0]
    losing = [k for k, v in all_results.items()
              if v.num_trades > 0 and v.total_return_pct < 0]

    print(f"\n[RECOMMENDATION]")
    if len(profitable) > len(losing):
        print(f"  [OK] Strategy is profitable on {len(profitable)}/{len(profitable)+len(losing)} symbols")
        print(f"  Consider going live with these settings")
    elif profitable:
        print(f"  [MIXED] Mixed results: {len(profitable)} profitable, {len(losing)} losing")
        print(f"  Consider: longer candles, tighter stops, or only trading profitable symbols")
    else:
        print(f"  [WARN] Strategy is losing on all symbols")
        print(f"  Try: --candles 15 (longer timeframe) or adjust thresholds")

    print(f"\n  Try different candle sizes:")
    print(f"    python run_backtest.py --candles 5   # 5-minute candles")
    print(f"    python run_backtest.py --candles 10  # 10-minute candles (current)")
    print(f"    python run_backtest.py --candles 15  # 15-minute candles")
    print(f"    python run_backtest.py --candles 30  # 30-minute candles")


if __name__ == "__main__":
    main()
