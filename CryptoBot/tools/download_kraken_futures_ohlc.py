"""
Download 6 months of REAL 1-minute OHLC candle data for Kraken Futures
perpetual contracts, using Kraken's public Futures charting API.
NO API KEY NEEDED (public endpoint).

This replaces the FUTURES_TO_SPOT proxy-mapping approach used by the
now-archived tools/legacy/backtest_6mo.py (which approximated futures price
action from Alpaca spot data). With this script, CryptoBot/backtest/run_baseline.py
can run FuturesBacktester on genuine futures OHLC instead of a spot proxy.

Saves to data/historical/1min_futures/ as CSV per symbol (same layout as
download_6mo_candles.py's spot output): timestamp, open, high, low, close, volume

Files are always named after the LIVE symbol (config.KRAKEN_FUTURES_SYMBOLS,
e.g. PI_SOLUSD) even when the underlying data had to be fetched from a
different, currently-tradeable ticker (see SYMBOL RESOLUTION below), so
downstream tools (retrain_on_6mo.py, run_baseline.py) don't need to know
about the substitution.

SYMBOL RESOLUTION: Kraken has progressively delisted the original PI_*
(coin-margined inverse) perpetuals for everything except BTC/ETH/LTC/XRP,
replacing them with PF_* (USD-margined linear) perpetuals for the same
underlying. MATIC was delisted entirely and replaced by POL. At startup
this script queries the public /derivatives/api/v3/instruments endpoint
to determine, per symbol, which ticker is actually tradeable/has data:
  - if the configured PI_ symbol is still tradeable, use it as-is
  - else try the PF_ equivalent (PI_ -> PF_ prefix swap)
  - explicit overrides (e.g. MATIC -> POL) take priority over both
A manifest of symbol -> actual source ticker used, row count, and date
range is written to data/historical/1min_futures/_symbol_sources.json.

API: GET https://futures.kraken.com/api/charts/v1/trade/{symbol}/1m?from=<epoch>&to=<epoch>
Response: {"candles": [{"time": ms_epoch, "open", "high", "low", "close", "volume"}, ...]}
Paged backwards in 1-day (1440-minute) chunks to stay well under any
undocumented per-request candle cap.

Usage:
    python3 CryptoBot/tools/download_kraken_futures_ohlc.py
    # optional: override output dir (e.g. for standalone/droplet deploys)
    FUTURES_OUTPUT_DIR=/path/to/dir python3 CryptoBot/tools/download_kraken_futures_ohlc.py
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
# Mirrors config.py's KRAKEN_FUTURES_SYMBOLS (live futures watchlist).
SYMBOLS = [
    "PI_XBTUSD", "PI_ETHUSD", "PI_SOLUSD", "PI_ADAUSD", "PI_AVAXUSD",
    "PI_DOGEUSD", "PI_MATICUSD", "PI_LTCUSD", "PI_LINKUSD", "PI_SHIBUSD",
    "PI_XRPUSD", "PI_DOTUSD", "PI_UNIUSD", "PI_ATOMUSD", "PI_XLMUSD",
]

# Kraken fully delisted MATIC and relaunched the perpetual under POL.
SYMBOL_OVERRIDES = {
    "PI_MATICUSD": "PF_POLUSD",
}

OUTPUT_DIR = os.environ.get("FUTURES_OUTPUT_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "historical", "1min_futures"
)
MONTHS = 6                  # How many months back
RESOLUTION = "1m"           # 1-minute candles
CHUNK_MINUTES = 1440        # page backwards 1 day at a time
RATE_LIMIT_SEC = 0.5        # polite rate limit (public endpoint, no auth)
MAX_RETRIES = 5

BASE_URL = "https://futures.kraken.com/api/charts/v1/trade"
INSTRUMENTS_URL = "https://futures.kraken.com/derivatives/api/v3/instruments"


def fetch_tradeable_symbols() -> set:
    """Query the public instruments list to see which tickers are currently live."""
    try:
        resp = requests.get(INSTRUMENTS_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return {i["symbol"] for i in data.get("instruments", []) if i.get("tradeable", True)}
    except Exception as e:
        print(f"WARNING: could not fetch instruments list ({e}); using configured symbols as-is")
        return set()


def resolve_ticker(live_symbol: str, tradeable: set) -> str:
    """Map a configured live PI_* symbol to the ticker to actually fetch OHLC for."""
    if live_symbol in SYMBOL_OVERRIDES:
        return SYMBOL_OVERRIDES[live_symbol]
    if not tradeable or live_symbol in tradeable:
        return live_symbol
    pf_variant = live_symbol.replace("PI_", "PF_", 1)
    if pf_variant in tradeable:
        return pf_variant
    return live_symbol  # fall back as-is; will likely yield no data, reported as a gap


def fetch_candles(symbol: str, start: datetime, end: datetime) -> list:
    """Fetch 1-min candles from Kraken Futures charting API for [start, end)."""
    url = f"{BASE_URL}/{symbol}/{RESOLUTION}"
    params = {
        "from": int(start.timestamp()),
        "to": int(end.timestamp()),
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("candles", [])
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


def download_symbol(live_symbol: str, fetch_symbol: str, months: int = MONTHS) -> dict:
    """Download N months of 1-min futures candles for one symbol, resumable.

    Output file is named after live_symbol; data is fetched using fetch_symbol
    (which may differ if the original live_symbol ticker is no longer tradeable).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"{live_symbol}_1min.csv")

    existing_rows = 0
    earliest_existing = None
    if os.path.exists(outfile):
        with open(outfile, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            rows = list(reader)
            existing_rows = len(rows)
            if rows:
                try:
                    earliest_existing = datetime.fromtimestamp(int(rows[0][0]), tz=timezone.utc)
                except Exception:
                    earliest_existing = None
        if existing_rows > 0:
            print(f"  {live_symbol}: Found {existing_rows:,} existing rows, earliest: {earliest_existing}")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)

    if earliest_existing and earliest_existing < end:
        end_download = earliest_existing - timedelta(minutes=1)
        if end_download <= start:
            print(f"  {live_symbol}: Already complete ({existing_rows:,} rows)")
            return _summarize(outfile, existing_rows, fetch_symbol)
    else:
        end_download = end

    chunk_duration = timedelta(minutes=CHUNK_MINUTES)
    all_candles = []
    current_end = end_download
    total_expected = int((end_download - start).total_seconds() / 60)
    fetched = 0

    print(f"  {live_symbol}: Downloading {total_expected:,} expected candles via {fetch_symbol} ({start.date()} to {end_download.date()})...")

    while current_end > start:
        current_start = max(start, current_end - chunk_duration)

        candles = fetch_candles(fetch_symbol, current_start, current_end)

        if candles:
            all_candles.extend(candles)
            fetched += len(candles)
            pct = min(100, (fetched / max(total_expected, 1)) * 100)
            sys.stdout.write(f"\r  {live_symbol}: {fetched:,} candles ({pct:.0f}%)   ")
            sys.stdout.flush()

        current_end = current_start
        time.sleep(RATE_LIMIT_SEC)

    print()

    if not all_candles:
        print(f"  {live_symbol}: No data received via {fetch_symbol}! (symbol may not exist or have no history this far back)")
        return _summarize(outfile, existing_rows, fetch_symbol)

    # Kraken futures candle time is epoch MILLISECONDS.
    seen = set()
    clean = []
    for c in all_candles:
        try:
            ts = int(c["time"]) // 1000
        except (KeyError, TypeError, ValueError):
            continue
        if ts not in seen:
            seen.add(ts)
            clean.append({
                "timestamp": ts,
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c.get("volume", 0.0)),
            })

    clean.sort(key=lambda x: x["timestamp"])

    if existing_rows > 0 and os.path.exists(outfile):
        with open(outfile, "r") as f:
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

    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(clean)

    print(f"  {live_symbol}: Saved {len(clean):,} candles to {outfile}")
    return _summarize(outfile, len(clean), fetch_symbol)


def _summarize(outfile: str, rows: int, fetch_symbol: str) -> dict:
    first_ts = last_ts = None
    if rows > 0 and os.path.exists(outfile):
        with open(outfile, "r") as f:
            reader = csv.DictReader(f)
            data_rows = list(reader)
        if data_rows:
            first_ts = int(data_rows[0]["timestamp"])
            last_ts = int(data_rows[-1]["timestamp"])
    return {
        "source_ticker": fetch_symbol,
        "rows": rows,
        "start": datetime.fromtimestamp(first_ts, tz=timezone.utc).isoformat() if first_ts else None,
        "end": datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat() if last_ts else None,
    }


if __name__ == "__main__":
    print("=" * 60)
    print(f"DOWNLOADING {MONTHS}-MONTH 1-MIN KRAKEN FUTURES CANDLES")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Output:  {OUTPUT_DIR}")
    print("=" * 60)
    print()

    tradeable = fetch_tradeable_symbols()
    manifest = {}
    total = 0
    for i, live_symbol in enumerate(SYMBOLS, 1):
        fetch_symbol = resolve_ticker(live_symbol, tradeable)
        label = live_symbol if fetch_symbol == live_symbol else f"{live_symbol} (via {fetch_symbol})"
        print(f"[{i}/{len(SYMBOLS)}] {label}")
        summary = download_symbol(live_symbol, fetch_symbol, MONTHS)
        manifest[live_symbol] = summary
        total += summary["rows"]
        print()

    manifest_path = os.path.join(OUTPUT_DIR, "_symbol_sources.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 60)
    print(f"DONE. Total candles downloaded: {total:,}")
    print(f"Location: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print("=" * 60)
