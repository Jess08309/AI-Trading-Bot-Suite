"""
Download 6 months of 1-minute candle data from Coinbase.
NO API KEY NEEDED — Coinbase public REST API.

Saves to data/historical/1min/ as CSV per symbol.
Each file: timestamp, open, high, low, close, volume

Coinbase limit: 300 candles per request, so we page backwards in 5-hour chunks.

GAP HANDLING: any chunk that returns no data after MAX_RETRIES is recorded in
failed_chunks.json (per symbol + window) instead of being silently dropped.
At the end of the run, all recorded failed chunks get one more retry pass;
anything still failing after that is left in failed_chunks.json and is very
likely a genuine exchange-side gap (the market simply didn't print a trade
that minute) rather than a fetch problem -- see tools/repair_gaps.py for a
standalone tool that does the same surgical re-fetch against existing files.

Resume check: a symbol is only considered "already complete" (skipped) if its
existing data covers >=95% of the expected 6-month row count AND its largest
interior gap is <=10 minutes. Otherwise it's resumed/re-fetched normally --
this replaces an old check that only looked at whether the earliest existing
row reached the 6-month boundary, which could (and did) mark files "complete"
despite large internal gaps.
"""

import os
import sys
import time
import csv
import json
import requests
from datetime import datetime, timezone, timedelta

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
    "DOGE-USD", "LINK-USD", "XRP-USD", "LTC-USD", "UNI-USD",
    "XLM-USD", "BCH-USD", "DOT-USD", "MATIC-USD", "ATOM-USD",
    "NEAR-USD", "AAVE-USD",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical", "1min")
FAILED_CHUNKS_PATH = os.path.join(OUTPUT_DIR, "failed_chunks.json")
MONTHS = 6              # How many months back
GRANULARITY = 60        # 1-minute candles
CANDLES_PER_REQ = 300   # Coinbase max
RATE_LIMIT_SEC = 0.35   # Polite rate limit
MAX_RETRIES = 5
MIN_COVERAGE = 0.95     # resume-check: fraction of expected rows required
MAX_INTERIOR_GAP_MIN = 10  # resume-check: largest allowed gap between rows

BASE_URL = "https://api.exchange.coinbase.com"

FAILED_CHUNKS = []  # accumulated across all symbols this run


def fetch_candles(symbol: str, start: datetime, end: datetime) -> list:
    """Fetch candles from Coinbase. Returns list of [time, low, high, open, close, volume]."""
    url = f"{BASE_URL}/products/{symbol}/candles"
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "granularity": GRANULARITY,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)
    return []


def _record_failed_chunk(symbol: str, start: datetime, end: datetime):
    print(f"  WARNING: {symbol}: no data for chunk {start.isoformat()} -> {end.isoformat()} "
          f"after {MAX_RETRIES} retries -- recorded in failed_chunks.json")
    FAILED_CHUNKS.append({"symbol": symbol, "start": start.isoformat(), "end": end.isoformat()})


def _merge_candles_into_csv(symbol: str, raw_candles: list) -> int:
    """Merge freshly-fetched Coinbase candles into a symbol's existing CSV. Returns rows added."""
    outfile = os.path.join(OUTPUT_DIR, f"{symbol}_1min.csv")
    existing = {}
    if os.path.exists(outfile):
        with open(outfile, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing[int(row["timestamp"])] = row
                except (KeyError, ValueError):
                    continue
    added = 0
    for c in raw_candles:
        ts = int(c[0])
        if ts not in existing:
            existing[ts] = {
                "timestamp": ts, "open": float(c[3]), "high": float(c[2]),
                "low": float(c[1]), "close": float(c[4]), "volume": float(c[5]),
            }
            added += 1
    ordered = [existing[ts] for ts in sorted(existing)]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(ordered)
    return added


def _load_existing_timestamps(outfile: str) -> list:
    if not os.path.exists(outfile):
        return []
    ts_list = []
    with open(outfile, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                ts_list.append(int(row[0]))
            except (IndexError, ValueError):
                continue
    ts_list.sort()
    return ts_list


def _max_interior_gap_minutes(sorted_ts: list) -> float:
    if len(sorted_ts) < 2:
        return 0.0
    return max((b - a) / 60.0 for a, b in zip(sorted_ts, sorted_ts[1:]))


def download_symbol(symbol: str, months: int = MONTHS):
    """Download N months of 1-min candles for one symbol."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"{symbol}_1min.csv")

    existing_ts = _load_existing_timestamps(outfile)
    existing_rows = len(existing_ts)
    earliest_existing = datetime.fromtimestamp(existing_ts[0], tz=timezone.utc) if existing_ts else None
    if existing_rows > 0:
        print(f"  {symbol}: Found {existing_rows:,} existing rows, earliest: {earliest_existing}")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)
    full_expected = int((end - start).total_seconds() / 60)

    if existing_ts:
        max_gap = _max_interior_gap_minutes(existing_ts)
        coverage = existing_rows / full_expected if full_expected else 0.0
        if coverage >= MIN_COVERAGE and max_gap <= MAX_INTERIOR_GAP_MIN:
            print(f"  {symbol}: Already complete ({existing_rows:,}/{full_expected:,} rows, "
                  f"{coverage:.1%} coverage, max interior gap {max_gap:.0f}min)")
            return existing_rows
        print(f"  {symbol}: existing data incomplete ({existing_rows:,}/{full_expected:,} rows, "
              f"{coverage:.1%} coverage, max interior gap {max_gap:.0f}min) -- resuming")

    # If resuming, start from the earliest existing row
    if earliest_existing and earliest_existing < end:
        end_download = earliest_existing - timedelta(minutes=1)
    else:
        end_download = end

    if end_download <= start:
        # Existing data already spans the full 6-month window; any shortfall is an
        # interior gap, which the backward-resume loop below can't reach -- leave it
        # for tools/repair_gaps.py's surgical gap-fill pass.
        return existing_rows

    # Page backwards in chunks of 300 minutes (5 hours)
    chunk_duration = timedelta(minutes=CANDLES_PER_REQ)
    all_candles = []
    current_end = end_download
    pass_expected = int((end_download - start).total_seconds() / 60)
    fetched = 0

    print(f"  {symbol}: Downloading {pass_expected:,} expected candles ({start.date()} to {end_download.date()})...")

    while current_end > start:
        current_start = max(start, current_end - chunk_duration)

        candles = fetch_candles(symbol, current_start, current_end)

        if candles:
            all_candles.extend(candles)
            fetched += len(candles)

            # Progress
            pct = min(100, (fetched / max(pass_expected, 1)) * 100)
            sys.stdout.write(f"\r  {symbol}: {fetched:,} candles ({pct:.0f}%)   ")
            sys.stdout.flush()
        else:
            _record_failed_chunk(symbol, current_start, current_end)

        current_end = current_start
        time.sleep(RATE_LIMIT_SEC)

    print()  # newline after progress

    if not all_candles:
        print(f"  {symbol}: No data received!")
        return existing_rows

    # Coinbase format: [time, low, high, open, close, volume]
    # Convert to: timestamp, open, high, low, close, volume
    # Sort by time ascending, deduplicate
    seen = set()
    clean = []
    for c in all_candles:
        ts = int(c[0])
        if ts not in seen:
            seen.add(ts)
            clean.append({
                "timestamp": ts,
                "open": float(c[3]),
                "high": float(c[2]),
                "low": float(c[1]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })

    clean.sort(key=lambda x: x["timestamp"])

    # Merge with existing data if any
    if existing_rows > 0 and os.path.exists(outfile):
        with open(outfile, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = int(row["timestamp"])
                if ts not in seen:
                    seen.add(ts)
                    clean.append({
                        "timestamp": ts,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    })
        clean.sort(key=lambda x: x["timestamp"])

    # Write CSV
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(clean)

    print(f"  {symbol}: Saved {len(clean):,} candles to {outfile}")
    return len(clean)


def _retry_failed_chunks():
    """One final retry pass over every chunk recorded as failed during the main run."""
    if not FAILED_CHUNKS:
        if os.path.exists(FAILED_CHUNKS_PATH):
            os.remove(FAILED_CHUNKS_PATH)
        return

    print("\n" + "=" * 60)
    print(f"RETRYING {len(FAILED_CHUNKS)} FAILED CHUNKS (final pass)")
    print("=" * 60)
    still_failed = []
    for entry in FAILED_CHUNKS:
        symbol = entry["symbol"]
        start = datetime.fromisoformat(entry["start"])
        end = datetime.fromisoformat(entry["end"])
        candles = fetch_candles(symbol, start, end)
        if candles:
            added = _merge_candles_into_csv(symbol, candles)
            print(f"  Recovered {symbol} {start.isoformat()} -> {end.isoformat()} ({added} new rows)")
        else:
            still_failed.append(entry)
            print(f"  STILL FAILING: {symbol} {start.isoformat()} -> {end.isoformat()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FAILED_CHUNKS_PATH, "w") as f:
        json.dump(still_failed, f, indent=2)

    if still_failed:
        print(f"\nWARNING: {len(still_failed)} chunks still unrecoverable after retry -- see "
              f"{FAILED_CHUNKS_PATH}. These likely reflect genuine exchange-side gaps "
              "(no trade printed that minute), not fetch failures.")
    else:
        print("\nAll failed chunks recovered on retry.")
        if os.path.exists(FAILED_CHUNKS_PATH):
            os.remove(FAILED_CHUNKS_PATH)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"DOWNLOADING {MONTHS}-MONTH 1-MIN CANDLES")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Output:  {OUTPUT_DIR}")
    print("=" * 60)
    print()

    total = 0
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {symbol}")
        count = download_symbol(symbol, MONTHS)
        total += count
        print()

    _retry_failed_chunks()

    print("=" * 60)
    print(f"DONE. Total candles downloaded: {total:,}")
    print(f"Location: {OUTPUT_DIR}")
    print("=" * 60)
