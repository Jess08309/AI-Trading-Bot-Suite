"""
Backtest candidate coins using daily candles and add winners to a watchlist.

Run:
  python backtest_new_coins.py
  python backtest_new_coins.py --apply
"""

import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from download_historical import fetch_coinbase_historical, fetch_kraken_futures_historical


DEFAULT_CANDIDATES = [
    "PEPE-USD",
    "SUI-USD",
    "APT-USD",
    "OP-USD",
    "ARB-USD",
    "INJ-USD",
    "SEI-USD",
]

WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rewrite_engine.py")
DOWNLOAD_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_historical.py")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "backtest")
RESULTS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "state", "candidate_backtest_results.json")
WINNERS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "state", "candidate_winners.json")
PRICES_CSV = os.path.join(OUTPUT_DIR, "candidate_prices_daily.csv")

FUTURES_SYMBOL_OVERRIDES = {
    "BTC-USD": "PI_XBTUSD",
}


def _parse_symbols_arg(raw: str) -> List[str]:
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return symbols or DEFAULT_CANDIDATES


def _compute_metrics(candles: List[Dict]) -> Dict[str, float]:
    candles_sorted = sorted(candles, key=lambda c: c["timestamp"])
    closes = [float(c["close"]) for c in candles_sorted if c.get("close") is not None]
    volumes = [float(c.get("volume", 0)) for c in candles_sorted]

    if len(closes) < 2:
        return {
            "data_points": len(closes),
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "annualized_volatility_pct": 0.0,
            "avg_daily_volume": 0.0,
        }

    daily_returns = [(closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes))]
    mean_return = statistics.mean(daily_returns)
    stdev_return = statistics.pstdev(daily_returns) if len(daily_returns) > 1 else 0.0

    total_return_pct = (closes[-1] / closes[0] - 1.0) * 100.0

    # Track peak-to-trough drawdown over the series
    peak = closes[0]
    max_drawdown = 0.0
    for price in closes:
        if price > peak:
            peak = price
        drawdown = (price / peak) - 1.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    sharpe = 0.0
    annualized_volatility = 0.0
    if stdev_return > 0:
        sharpe = (mean_return / stdev_return) * (365 ** 0.5)
        annualized_volatility = stdev_return * (365 ** 0.5) * 100.0

    avg_volume = statistics.mean(volumes) if volumes else 0.0

    return {
        "data_points": len(closes),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown * 100.0,
        "sharpe": sharpe,
        "annualized_volatility_pct": annualized_volatility,
        "avg_daily_volume": avg_volume,
    }


def _passes_filters(metrics: Dict[str, float], min_days: int, min_return: float,
                    min_sharpe: float, max_drawdown: float) -> bool:
    return (
        metrics["data_points"] >= min_days
        and metrics["total_return_pct"] >= min_return
        and metrics["sharpe"] >= min_sharpe
        and metrics["max_drawdown_pct"] >= max_drawdown
    )


def _ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)


def _write_prices_csv(rows: List[Dict]):
    if not rows:
        return
    with open(PRICES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "symbol", "type", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_symbol_list_from_file(path: str, list_name: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(f"{list_name} = ["):
            start = idx
            continue
        if start is not None and line.strip().startswith("]"):
            end = idx
            break

    if start is None or end is None:
        raise ValueError(f"Could not find list {list_name} in {path}")

    items = []
    for line in lines[start + 1 : end]:
        stripped = line.strip().strip(",")
        if stripped.startswith("\"") and stripped.endswith("\""):
            items.append(stripped.strip("\""))
    return items


def _replace_symbol_list_in_file(path: str, list_name: str, symbols: List[str]):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(f"{list_name} = ["):
            start = idx
            continue
        if start is not None and line.strip().startswith("]"):
            end = idx
            break

    if start is None or end is None:
        raise ValueError(f"Could not find list {list_name} in {path}")

    new_block = [lines[start]]
    for symbol in symbols:
        new_block.append(f"    \"{symbol}\",\n")
    new_block.append(lines[end])

    updated = lines[:start] + new_block + lines[end + 1 :]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(updated)


def _spot_to_futures_symbol(spot_symbol: str) -> str:
    if spot_symbol in FUTURES_SYMBOL_OVERRIDES:
        return FUTURES_SYMBOL_OVERRIDES[spot_symbol]
    base = spot_symbol.split("-")[0]
    return f"PI_{base}USD"


def _apply_winners(spot_winners: List[str], futures_winners: List[str]) -> Tuple[List[str], List[str], List[str]]:
    watchlist = _read_symbol_list_from_file(WATCHLIST_FILE, "SPOT_WATCHLIST")
    combined_watchlist = sorted(set(watchlist + spot_winners))
    _replace_symbol_list_in_file(WATCHLIST_FILE, "SPOT_WATCHLIST", combined_watchlist)

    spot_symbols = _read_symbol_list_from_file(DOWNLOAD_SCRIPT, "SPOT_SYMBOLS")
    combined_spot = sorted(set(spot_symbols + spot_winners))
    _replace_symbol_list_in_file(DOWNLOAD_SCRIPT, "SPOT_SYMBOLS", combined_spot)

    futures_symbols = _read_symbol_list_from_file(DOWNLOAD_SCRIPT, "FUTURES_SYMBOLS")
    combined_futures = sorted(set(futures_symbols + futures_winners))
    _replace_symbol_list_in_file(DOWNLOAD_SCRIPT, "FUTURES_SYMBOLS", combined_futures)

    return combined_watchlist, combined_spot, combined_futures


def main():
    parser = argparse.ArgumentParser(description="Backtest candidate coins (daily candles)")
    parser.add_argument("--symbols", help="Comma-separated list of symbols")
    parser.add_argument("--days", type=int, default=730, help="History length in days")
    parser.add_argument("--min-days", type=int, default=365, help="Minimum days of data")
    parser.add_argument("--min-return", type=float, default=0.0, help="Min total return %%")
    parser.add_argument("--min-sharpe", type=float, default=0.2, help="Min Sharpe ratio")
    parser.add_argument("--max-drawdown", type=float, default=-60.0, help="Max drawdown %% (negative)")
    parser.add_argument("--spot-only", action="store_true", help="Only backtest spot")
    parser.add_argument("--futures", action="store_true", help="Include futures backtest")
    parser.add_argument("--futures-only", action="store_true", help="Only backtest futures")
    parser.add_argument("--apply", action="store_true", help="Add winners to watchlist + download list")

    args = parser.parse_args()
    symbols = _parse_symbols_arg(args.symbols) if args.symbols else DEFAULT_CANDIDATES

    _ensure_dirs()

    results = {"spot": {}, "futures": {}}
    all_rows = []

    print("=" * 60)
    print("BACKTESTING NEW COIN CANDIDATES")
    print("=" * 60)

    run_spot = not args.futures_only
    run_futures = args.futures or args.futures_only

    for symbol in symbols:
        if run_spot:
            candles = fetch_coinbase_historical(symbol, days=args.days)
            for candle in candles:
                all_rows.append({
                    "timestamp": candle["timestamp"],
                    "symbol": symbol,
                    "type": "spot",
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle["volume"],
                })
            metrics = _compute_metrics(candles)
            results["spot"][symbol] = metrics

        if run_futures:
            futures_symbol = _spot_to_futures_symbol(symbol)
            fut_candles = fetch_kraken_futures_historical(futures_symbol, days=args.days)
            for candle in fut_candles:
                all_rows.append({
                    "timestamp": candle["timestamp"],
                    "symbol": futures_symbol,
                    "type": "futures",
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle["volume"],
                })
            metrics = _compute_metrics(fut_candles)
            results["futures"][futures_symbol] = metrics

    _write_prices_csv(all_rows)

    spot_winners = [
        symbol
        for symbol, metrics in results["spot"].items()
        if _passes_filters(metrics, args.min_days, args.min_return, args.min_sharpe, args.max_drawdown)
    ]
    futures_winners = [
        symbol
        for symbol, metrics in results["futures"].items()
        if _passes_filters(metrics, args.min_days, args.min_return, args.min_sharpe, args.max_drawdown)
    ]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "filters": {
            "min_days": args.min_days,
            "min_return_pct": args.min_return,
            "min_sharpe": args.min_sharpe,
            "max_drawdown_pct": args.max_drawdown,
        },
        "results": results,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(WINNERS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": summary["timestamp"], "spot": spot_winners, "futures": futures_winners},
            f,
            indent=2,
        )

    print(f"Results saved: {RESULTS_JSON}")
    print(f"Winners saved: {WINNERS_JSON}")
    print(f"Daily prices: {PRICES_CSV}")
    print(f"Spot winners ({len(spot_winners)}): {', '.join(spot_winners) if spot_winners else 'None'}")
    print(f"Futures winners ({len(futures_winners)}): {', '.join(futures_winners) if futures_winners else 'None'}")

    if args.apply and (spot_winners or futures_winners):
        watchlist, spot_symbols, futures_symbols = _apply_winners(spot_winners, futures_winners)
        print(f"Updated SPOT_WATCHLIST ({len(watchlist)} coins) in {WATCHLIST_FILE}")
        print(f"Updated SPOT_SYMBOLS ({len(spot_symbols)} coins) in {DOWNLOAD_SCRIPT}")
        print(f"Updated FUTURES_SYMBOLS ({len(futures_symbols)} symbols) in {DOWNLOAD_SCRIPT}")
    elif args.apply:
        print("No winners to apply.")


if __name__ == "__main__":
    main()
