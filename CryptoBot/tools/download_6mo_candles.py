"""
Download 6 months of 1-minute candle data from Coinbase.
NO API KEY NEEDED — Coinbase public REST API.

Saves to data/historical/1min/ as CSV per symbol.
Each file: timestamp, open, high, low, close, volume

Coinbase limit: 300 candles per request, so we page backwards in 5-hour chunks.
"""

import os
import sys
import time
import csv
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
MONTHS = 6              # How many months back
GRANULARITY = 60        # 1-minute candles
CANDLES_PER_REQ = 300   # Coinbase max
RATE_LIMIT_SEC = 0.35   # Polite rate limit
MAX_RETRIES = 5

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
BASE_URL = "https://api.exchange.coinbase.com"

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


def download_symbol(symbol: str, months: int = MONTHS):
    """Download N months of 1-min candles for one symbol."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"{symbol}_1min.csv")

    # Check if we already have a partial download
    existing_rows = 0
    earliest_existing = None
    if os.path.exists(outfile):
        with open(outfile, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
            existing_rows = len(rows)
            if rows:
                # Find earliest timestamp in existing data
                try:
                    earliest_existing = datetime.fromtimestamp(int(rows[-1][0]), tz=timezone.utc)
                except:
                    earliest_existing = None
        if existing_rows > 0:
            print(f"  {symbol}: Found {existing_rows:,} existing rows, earliest: {earliest_existing}")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)

    # If resuming, start from the earliest existing row
    if earliest_existing and earliest_existing < end:
        end_download = earliest_existing - timedelta(minutes=1)
        if end_download <= start:
            total_expected = months * 30 * 24 * 60
            print(f"  {symbol}: Already complete ({existing_rows:,} rows)")
            return existing_rows
    else:
        end_download = end

    # Page backwards in chunks of 300 minutes (5 hours)
    chunk_duration = timedelta(minutes=CANDLES_PER_REQ)
    all_candles = []
    current_end = end_download
    total_expected = int((end_download - start).total_seconds() / 60)
    fetched = 0

    print(f"  {symbol}: Downloading {total_expected:,} expected candles ({start.date()} to {end_download.date()})...")

    while current_end > start:
        current_start = max(start, current_end - chunk_duration)

        candles = fetch_candles(symbol, current_start, current_end)

        if candles:
            all_candles.extend(candles)
            fetched += len(candles)

            # Progress
            pct = min(100, (fetched / max(total_expected, 1)) * 100)
            sys.stdout.write(f"\r  {symbol}: {fetched:,} candles ({pct:.0f}%)   ")
            sys.stdout.flush()

        current_end = current_start
        time.sleep(RATE_LIMIT_SEC)

    print()  # newline after progress

    if not all_candles:
        print(f"  {symbol}: No data received!")
        return 0

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

    print("=" * 60)
    print(f"DONE. Total candles downloaded: {total:,}")
    print(f"Location: {OUTPUT_DIR}")
    print("=" * 60)
