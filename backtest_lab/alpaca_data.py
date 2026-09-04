"""Real historical data from Alpaca's free market-data API: underlying minute bars
and options daily trade bars. Credentials read from backtest_lab/.env (gitignored,
same values already used by the live PutSeller bot -- read-only here).
"""
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

API_KEY = os.environ["ALPACA_API_KEY"]
API_SECRET = os.environ["ALPACA_API_SECRET"]
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

DATA_URL = "https://data.alpaca.markets"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_alpaca")
os.makedirs(CACHE_DIR, exist_ok=True)


def _get(url: str, params: dict) -> dict:
    for attempt in range(5):
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def get_minute_bars(ticker: str, start: str, end: str, timeframe: str = "15Min") -> pd.DataFrame:
    """Real intraday underlying bars (IEX feed, free tier). Cached to disk."""
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{timeframe}_{start}_{end}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    all_bars = []
    params = {
        "timeframe": timeframe,
        "start": f"{start}T00:00:00Z",
        "end": f"{end}T23:59:59Z",
        "feed": "iex",
        "limit": 10000,
        "adjustment": "split",
    }
    url = f"{DATA_URL}/v2/stocks/{ticker}/bars"
    page_token = None
    while True:
        if page_token:
            params["page_token"] = page_token
        r = _get(url, params)
        bars = r.get("bars") or []
        all_bars.extend(bars)
        page_token = r.get("next_page_token")
        if not page_token:
            break

    if not all_bars:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    else:
        df = pd.DataFrame(all_bars)
        df["t"] = pd.to_datetime(df["t"])
        df = df.set_index("t").rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
    df.to_csv(cache_path)
    return df


def get_option_daily_bars(symbols: list, start: str, end: str) -> dict:
    """Real historical option daily trade bars (OHLC of actual trades). Cached per-symbol.
    Returns {symbol: DataFrame}. Symbols with no trades in range are omitted.
    """
    result = {}
    to_fetch = []
    for sym in symbols:
        cache_path = os.path.join(CACHE_DIR, f"OPT_{sym}_{start}_{end}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty:
                result[sym] = df
        else:
            to_fetch.append(sym)

    for i in range(0, len(to_fetch), 30):  # batch requests, comma-separated symbols
        batch = to_fetch[i:i + 30]
        params = {"symbols": ",".join(batch), "timeframe": "1Day", "start": start, "end": end, "limit": 10000}
        r = _get(f"{DATA_URL}/v1beta1/options/bars", params)
        bars_by_symbol = r.get("bars") or {}
        for sym in batch:
            bars = bars_by_symbol.get(sym, [])
            cache_path = os.path.join(CACHE_DIR, f"OPT_{sym}_{start}_{end}.csv")
            if bars:
                df = pd.DataFrame(bars)
                df["t"] = pd.to_datetime(df["t"])
                df = df.set_index("t").rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                df = df[["open", "high", "low", "close", "volume"]]
                result[sym] = df
            else:
                df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            df.to_csv(cache_path)

    return result
