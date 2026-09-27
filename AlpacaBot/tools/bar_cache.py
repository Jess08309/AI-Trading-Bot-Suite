"""
Shared 1-minute bar downloader/cache for AlpacaBot research tools.
RESEARCH ONLY -- new file, does not modify any existing download/backtest
tool. Mirrors tools/download_5min.py's chunked-download pattern (rate-limit
safe) and tools/compute_indicator_correlation.py's IEX-feed fix (SIP requires
a paid data subscription this account doesn't have; IEX works on every plan).

All coarser timeframes used by tools/backtest_tuned.py (2-min, 10-min,
15-min, 1-hour, daily) are derived locally from this single 1-min dataset by
resampling -- guarantees every config trades over an identical calendar
window instead of stitching together separately-fetched series.
"""
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

load_dotenv()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_ROOT, "data", "historical")
os.makedirs(CACHE_DIR, exist_ok=True)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = StockHistoricalDataClient(
            os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
        )
    return _client


def cache_path(symbol: str, label: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{label}.csv")


def fetch_1min_cached(symbol: str, days: int = 200, chunk_days: int = 15,
                       force: bool = False) -> pd.DataFrame:
    """
    Fetch 1-minute bars for `symbol` covering the trailing `days` days,
    downloaded in `chunk_days`-day windows (same chunking pattern as
    download_5min.py) to respect Alpaca rate limits. Cached to
    data/historical/{symbol}_1min_research.csv -- reruns load from cache
    unless force=True, so the API is never re-hit for data already on disk.
    """
    path = cache_path(symbol, "1min_research")
    if os.path.exists(path) and not force:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    client = _get_client()
    end = datetime.now()
    n_chunks = max(1, -(-days // chunk_days))
    all_bars = []

    for chunk in range(n_chunks):
        chunk_end = end - timedelta(days=chunk * chunk_days)
        chunk_start = chunk_end - timedelta(days=chunk_days)
        print(f"  {symbol} 1min chunk {chunk + 1}/{n_chunks}: "
              f"{chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=chunk_start,
                end=chunk_end,
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(req)
            symbol_bars = bars.data.get(symbol, [])
            for b in symbol_bars:
                all_bars.append({
                    "timestamp": b.timestamp, "open": float(b.open),
                    "high": float(b.high), "low": float(b.low),
                    "close": float(b.close), "volume": float(b.volume),
                })
        except Exception as e:
            print(f"    {symbol} chunk {chunk + 1}/{n_chunks} error: {e}")
        time.sleep(0.3)

    if not all_bars:
        raise RuntimeError(f"No 1-min bars downloaded for {symbol}")

    df = pd.DataFrame(all_bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df
