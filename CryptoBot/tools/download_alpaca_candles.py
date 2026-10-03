"""
Download 6 months of 1-minute candle data from Alpaca Crypto.
NO API KEY NEEDED — public historical data endpoint.

Saves to data/historical/1min/ as CSV per symbol.
Each file: timestamp, open, high, low, close, volume

Alpaca uses symbol format "BTC/USD" (slash, not dash).
Chunked into ~7-day windows to stay under rate limits.

RESUME: if a symbol's existing file covers >=95% of expected rows with no
interior gap >10 minutes, it is skipped. Otherwise resumed from earliest row.

GAP HANDLING: any chunk that returns no data after MAX_RETRIES is recorded in
failed_chunks.json (per symbol + window). At end of run, one retry pass.
Anything still failing is likely a genuine gap (no trade printed that minute).
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
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "AVAX/USD",
    "DOGE/USD", "LINK/USD", "XRP/USD", "LTC/USD", "UNI/USD",
    "XLM/USD", "BCH/USD", "DOT/USD", "POL/USD", "ATOM/USD",
    "NEAR/USD", "AAVE/USD",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical", "1min")
FAILED_CHUNKS_PATH = os.path.join(OUTPUT_DIR, "failed_chunks.json")
MONTHS = 6
GRANULARITY_MIN = 1
CHUNK_DAYS = 7              # 7-day chunks = ~10k bars per request, well under limits
RATE_LIMIT_SEC = 0.35       # Alpaca is generous; be polite anyway
MAX_RETRIES = 5
MIN_COVERAGE = 0.95
MAX_INTERIOR_GAP_MIN = 10

BASE_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"

FAILED_CHUNKS = []


def _alpaca_symbol(symbol: str) -> str:
    """'BTC/USD' -> 'BTC/USD' (already correct)."""
    return symbol


def _file_symbol(symbol: str) -> str:
    """'BTC/USD' -> 'BTC-USD' for filename compatibility with existing tools."""
    return symbol.replace("/", "-")


def fetch_bars(symbol: str, start: datetime, end: datetime) -> list:
    """Fetch 1-min bars from Alpaca. Returns list of bar dicts."""
    url = BASE_URL
    headers = {"Accept": "application/json"}
    params = {
        "symbols": _alpaca_symbol(symbol),
        "timeframe": "1Min",
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10000,
        "sort": "asc",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                bars = data.get("bars", {}).get(_alpaca_symbol(symbol), [])
                return bars
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


def _merge_bars_into_csv(symbol: str, raw_bars: list) -> int:
    """Merge freshly-fetched Alpaca bars into a symbol's existing CSV. Returns rows added."""
    file_sym = _file_symbol(symbol)
    outfile = os.path.join(OUTPUT_DIR, f"{file_sym}_1min.csv")
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
    for b in raw_bars:
        # Alpaca bar format: {"t":"2026-01-01T00:00:00Z","o":...,"h":...,"l":...,"c":...,"v":...}
        ts = int(datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp())
        if ts not in existing:
            existing[ts] = {
                "timestamp": ts,
                "open": float(b["o"]),
                "high": float(b["h"]),
                "low": float(b["l"]),
                "close": float(b["c"]),
                "volume": float(b["v"]),
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
    file_sym = _file_symbol(symbol)
    outfile = os.path.join(OUTPUT_DIR, f"{file_sym}_1min.csv")

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
        return existing_rows

    # Page backwards in chunks
    chunk_duration = timedelta(days=CHUNK_DAYS)
    current_end = end_download
    pass_expected = int((end_download - start).total_seconds() / 60)
    fetched = 0

    print(f"  {symbol}: Downloading {pass_expected:,} expected candles ({start.date()} to {end_download.date()})...")

    while current_end > start:
        current_start = max(start, current_end - chunk_duration)

        bars = fetch_bars(symbol, current_start, current_end)

        if bars:
            fetched += len(bars)
            pct = min(100, (fetched / max(pass_expected, 1)) * 100)
            sys.stdout.write(f"\r  {symbol}: {fetched:,} candles ({pct:.0f}%)   ")
            sys.stdout.flush()
            _merge_bars_into_csv(symbol, bars)
        else:
            _record_failed_chunk(symbol, current_start, current_end)

        current_end = current_start
        time.sleep(RATE_LIMIT_SEC)

    print()  # newline after progress

    if fetched == 0 and existing_rows == 0:
        print(f"  {symbol}: No data received!")
        return 0

    # Re-count for accurate return
    final_ts = _load_existing_timestamps(outfile)
    print(f"  {symbol}: Total {len(final_ts):,} candles in {outfile}")
    return len(final_ts)


def _retry_failed_chunks():
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
        bars = fetch_bars(symbol, start, end)
        if bars:
            added = _merge_bars_into_csv(symbol, bars)
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


if __name__ == "__main__":
    print("=" * 60)
    print(f"DOWNLOADING {MONTHS}-MONTH 1-MIN CANDLES FROM ALPACA")
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
