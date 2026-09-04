"""Free historical data fetchers: underlying prices, VIX (IV proxy), risk-free rate.

Standalone from the production bots -- reads nothing from them, writes nothing to them.
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(ticker: str, start: str, end: str) -> str:
    safe = ticker.replace("^", "IDX_")
    return os.path.join(CACHE_DIR, f"{safe}_{start}_{end}.csv")


def get_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV for a ticker, cached to disk so repeated backtests don't re-hit the network."""
    path = _cache_path(ticker, start, end)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.to_csv(path)
    return df


def get_vix(start: str, end: str) -> pd.Series:
    """CBOE VIX close, used as the IV proxy for index-level underlyings (SPY/QQQ/IWM)."""
    df = get_bars("^VIX", start, end)
    return df["Close"] / 100.0


def get_risk_free_rate(start: str, end: str) -> pd.Series:
    """13-week T-bill yield (^IRX), used as the risk-free rate input to Black-Scholes."""
    df = get_bars("^IRX", start, end)
    return df["Close"] / 100.0


def realized_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Annualized rolling realized volatility from daily log returns -- IV proxy for single stocks."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)
