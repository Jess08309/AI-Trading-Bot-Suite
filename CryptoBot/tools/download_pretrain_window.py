"""
Download a bounded date-range slice of spot 1-min OHLC candles, strictly for
use as OUT-OF-WINDOW training data for the frozen gate model (see
tools/retrain_on_6mo.py --end-date). This is intentionally separate from
tools/download_6mo_candles.py (which always fetches "the last N months from
now" and is what the BACKTEST window itself is built from) so the two data
sets can never accidentally overlap.

Writes to data/historical/1min_pretrain/{symbol}_1min.csv (spot only --
retrain_on_6mo.py's feature set is price-action-only and symbol/asset-class
agnostic, so a spot-only training set is used for both the spot and futures
backtest legs, matching retrain_on_6mo.py's existing behavior).

Usage:
    python3 tools/download_pretrain_window.py --end 2026-03-29T00:00:00+00:00 --days 60
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_6mo_candles as spot_dl

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical", "1min_pretrain")


def download_symbol_range(symbol: str, start: datetime, end: datetime) -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"{symbol}_1min.csv")

    chunk_duration = timedelta(minutes=spot_dl.CANDLES_PER_REQ)
    current_end = end
    expected = int((end - start).total_seconds() / 60)
    seen = set()
    clean = []
    fetched = 0

    print(f"  {symbol}: downloading {expected:,} expected candles ({start.date()} to {end.date()})...")
    while current_end > start:
        current_start = max(start, current_end - chunk_duration)
        candles = spot_dl.fetch_candles(symbol, current_start, current_end)
        if candles:
            for c in candles:
                ts = int(c[0])
                if ts not in seen:
                    seen.add(ts)
                    clean.append({
                        "timestamp": ts, "open": float(c[3]), "high": float(c[2]),
                        "low": float(c[1]), "close": float(c[4]), "volume": float(c[5]),
                    })
            fetched += len(candles)
            pct = min(100, (fetched / max(expected, 1)) * 100)
            sys.stdout.write(f"\r  {symbol}: {fetched:,} candles ({pct:.0f}%)   ")
            sys.stdout.flush()
        else:
            print(f"\n  WARNING: {symbol}: no data for chunk {current_start.isoformat()} -> "
                  f"{current_end.isoformat()} after retries")
        current_end = current_start
        time.sleep(spot_dl.RATE_LIMIT_SEC)
    print()

    clean.sort(key=lambda x: x["timestamp"])
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(clean)
    print(f"  {symbol}: saved {len(clean):,} candles to {outfile}")
    return len(clean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", required=True,
                         help="ISO timestamp: pretrain data ends strictly before this (the backtest window start)")
    parser.add_argument("--days", type=int, default=60, help="How many days of pretrain data to fetch (default 60)")
    args = parser.parse_args()

    end = datetime.fromisoformat(args.end)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=args.days)

    print("=" * 60)
    print(f"DOWNLOADING PRETRAIN WINDOW (spot only, {args.days} days)")
    print(f"Range:  {start.isoformat()} -> {end.isoformat()} (exclusive)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    total = 0
    for i, symbol in enumerate(spot_dl.SYMBOLS, 1):
        print(f"[{i}/{len(spot_dl.SYMBOLS)}] {symbol}")
        total += download_symbol_range(symbol, start, end)

    print("=" * 60)
    print(f"DONE. Total candles downloaded: {total:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
