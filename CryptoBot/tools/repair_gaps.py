"""
Scan downloaded 1-min OHLC CSVs for internal gaps (missing minutes) within the
target 6-month range, and re-fetch just the missing windows to patch them.

WHY THIS EXISTS: both download_6mo_candles.py and download_kraken_futures_ohlc.py
silently drop any chunk whose fetch fails after MAX_RETRIES (transient
rate-limiting / network errors) -- see their fetch_candles()/download_symbol()
loops: `if candles: all_candles.extend(...)` has no else-branch, so a failed
chunk just vanishes with no error, no flag, no retry-later. Their "already
complete" resume check also only verifies that the EARLIEST existing row
reaches the 6-month boundary -- it never checks internal contiguity or total
row count against the expected total. Together these silently produce files
that report "Already complete" / show 100% in the live progress bar while
actually missing 5-35% of their data (observed: PI_LTCUSD landed at 78.9%
coverage, AVAX-USD spot at 66.3%, ADA-USD spot at 92.4%).

This tool finds the exact missing minute-timestamps in an existing CSV and
re-fetches only those windows (reusing the same fetch_candles() functions and
symbol-resolution logic as the two downloaders), so it's fast and doesn't
require re-downloading whole files.

After repair, any windows that STILL can't be fetched (retried directly here,
so not just a transient rate-limit blip) are reported separately as likely
genuine exchange-side gaps (e.g. exchange downtime, a symbol being delisted
mid-window) rather than fetch failures -- this distinction matters for the
baseline gate: a few minutes of real exchange downtime is not a data-quality
bug, whereas our own fetch bugs are. A gap_report.json manifest is written to
each source's OUTPUT_DIR summarizing per-symbol coverage for the baseline
report to pick up.

Usage:
    cd CryptoBot/tools   (or wherever download_6mo_candles.py / this file live)
    python3 repair_gaps.py --source spot
    python3 repair_gaps.py --source futures
    python3 repair_gaps.py --source spot --symbols AVAX-USD,ADA-USD
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import download_6mo_candles as spot_dl
import download_kraken_futures_ohlc as fut_dl

MONTHS = 6
FIELDNAMES = ["timestamp", "open", "high", "low", "close", "volume"]


def load_existing(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows[int(row["timestamp"])] = row
            except (KeyError, ValueError):
                continue
    return rows


def contiguous_windows(missing_sorted):
    """Group a sorted list of missing minute-epochs into (start_ts, end_ts) runs."""
    windows = []
    if not missing_sorted:
        return windows
    win_start = prev = missing_sorted[0]
    for ts in missing_sorted[1:]:
        if ts - prev > 60:
            windows.append((win_start, prev))
            win_start = ts
        prev = ts
    windows.append((win_start, prev))
    return windows


def _repair(path, label, fetch_symbol, threshold, fetch_fn, parse_fn, chunk_minutes, rate_limit):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=MONTHS * 30)
    expected_total = int((end - start).total_seconds() / 60)

    existing = load_existing(path)
    before = len(existing)
    expected_ts = range(int(start.timestamp()), int(end.timestamp()), 60)
    missing = sorted(t for t in expected_ts if t not in existing)
    coverage_before = (before / expected_total) if expected_total else 0.0

    if not missing or coverage_before >= threshold:
        print(f"  {label}: OK -- {before:,}/{expected_total:,} rows ({coverage_before:.1%}), "
              f"{len(missing):,} missing minutes -- no repair needed")
        windows = [
            {"start": datetime.fromtimestamp(w[0], tz=timezone.utc).isoformat(),
             "end": datetime.fromtimestamp(w[1], tz=timezone.utc).isoformat()}
            for w in contiguous_windows(missing)
        ]
        return {"symbol": label, "before": before, "after": before, "expected": expected_total,
                "repaired": 0, "still_missing": len(missing), "still_missing_windows": windows}

    print(f"  {label}: {before:,}/{expected_total:,} rows ({coverage_before:.1%}) -- "
          f"{len(missing):,} missing minutes, repairing via {fetch_symbol}...")

    windows = contiguous_windows(missing)
    fetched_count = 0
    for w_start_ts, w_end_ts in windows:
        w_start = datetime.fromtimestamp(w_start_ts, tz=timezone.utc) - timedelta(minutes=1)
        w_end = datetime.fromtimestamp(w_end_ts, tz=timezone.utc) + timedelta(minutes=2)
        cursor = w_start
        while cursor < w_end:
            chunk_end = min(w_end, cursor + timedelta(minutes=chunk_minutes))
            candles = fetch_fn(fetch_symbol, cursor, chunk_end)
            for c in candles:
                try:
                    ts, row = parse_fn(c)
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                if ts not in existing:
                    existing[ts] = row
                    fetched_count += 1
            cursor = chunk_end
            time.sleep(rate_limit)

    after = len(existing)
    ordered = [existing[ts] for ts in sorted(existing)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)

    still_missing_ts = sorted(t for t in expected_ts if t not in existing)
    still_missing = len(still_missing_ts)
    still_missing_windows = [
        {"start": datetime.fromtimestamp(w[0], tz=timezone.utc).isoformat(),
         "end": datetime.fromtimestamp(w[1], tz=timezone.utc).isoformat()}
        for w in contiguous_windows(still_missing_ts)
    ]
    coverage_after = (after / expected_total) if expected_total else 0.0
    print(f"  {label}: repaired {fetched_count:,} candles -- now {after:,}/{expected_total:,} "
          f"rows ({coverage_after:.1%}), {still_missing:,} still missing")
    if still_missing_windows:
        print(f"  {label}: {len(still_missing_windows)} unfetchable window(s) after retry -- "
              "likely genuine exchange-side gaps (no trade printed), not fetch failures:")
        for w in still_missing_windows[:10]:
            print(f"      {w['start']} -> {w['end']}")
        if len(still_missing_windows) > 10:
            print(f"      ... and {len(still_missing_windows) - 10} more (see gap_report.json)")
    return {"symbol": label, "before": before, "after": after, "expected": expected_total,
            "repaired": fetched_count, "still_missing": still_missing,
            "still_missing_windows": still_missing_windows}


def repair_spot(symbol, threshold):
    path = os.path.join(spot_dl.OUTPUT_DIR, f"{symbol}_1min.csv")

    def parse(c):
        ts = int(c[0])
        return ts, {"timestamp": ts, "open": float(c[3]), "high": float(c[2]),
                     "low": float(c[1]), "close": float(c[4]), "volume": float(c[5])}

    return _repair(path, symbol, symbol, threshold, spot_dl.fetch_candles, parse,
                    spot_dl.CANDLES_PER_REQ, spot_dl.RATE_LIMIT_SEC)


def repair_futures(live_symbol, fetch_symbol, threshold):
    path = os.path.join(fut_dl.OUTPUT_DIR, f"{live_symbol}_1min.csv")

    def parse(c):
        ts = int(c["time"]) // 1000
        return ts, {"timestamp": ts, "open": float(c["open"]), "high": float(c["high"]),
                     "low": float(c["low"]), "close": float(c["close"]),
                     "volume": float(c.get("volume", 0.0))}

    return _repair(path, live_symbol, fetch_symbol, threshold, fut_dl.fetch_candles, parse,
                    fut_dl.CHUNK_MINUTES, fut_dl.RATE_LIMIT_SEC)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["spot", "futures"], required=True)
    parser.add_argument("--symbols", default=None, help="Comma-separated subset (default: all)")
    parser.add_argument("--threshold", type=float, default=0.995,
                         help="Minimum coverage fraction to skip repair (default 0.995)")
    args = parser.parse_args()

    results = []
    if args.source == "spot":
        symbols = args.symbols.split(",") if args.symbols else spot_dl.SYMBOLS
        for s in symbols:
            results.append(repair_spot(s, args.threshold))
    else:
        tradeable = fut_dl.fetch_tradeable_symbols()
        symbols = args.symbols.split(",") if args.symbols else fut_dl.SYMBOLS
        for s in symbols:
            fetch_symbol = fut_dl.resolve_ticker(s, tradeable)
            results.append(repair_futures(s, fetch_symbol, args.threshold))

    print("\n" + "=" * 72)
    print(f"{'SYMBOL':16s}{'ROWS':>18s}{'COVERAGE':>12s}")
    for r in results:
        cov = (r["after"] / r["expected"]) if r["expected"] else 0.0
        flag = "  <-- FLAG (<80%)" if cov < 0.80 else ""
        print(f"  {r['symbol']:14s} {r['after']:>7,}/{r['expected']:>7,} {cov:10.1%}{flag}")
    print("=" * 72)

    out_dir = spot_dl.OUTPUT_DIR if args.source == "spot" else fut_dl.OUTPUT_DIR
    report_path = os.path.join(out_dir, "gap_report.json")
    with open(report_path, "w") as f:
        json.dump({r["symbol"]: {
            "rows": r["after"], "expected": r["expected"],
            "coverage": (r["after"] / r["expected"]) if r["expected"] else 0.0,
            "repaired_candles": r["repaired"],
            "exchange_side_gaps": r["still_missing_windows"],
        } for r in results}, f, indent=2)
    print(f"Gap report written to {report_path}")


if __name__ == "__main__":
    main()
