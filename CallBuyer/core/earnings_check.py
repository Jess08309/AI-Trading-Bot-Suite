"""
Earnings date checker — avoids buying calls across earnings announcements.
Uses yfinance to look up next earnings date. If yfinance is not installed,
the check fails closed (blocks trade for safety).
"""
import logging
from datetime import datetime, timedelta
from functools import lru_cache

log = logging.getLogger("CallBuyer.earnings")

_ETFS = {
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "EFA", "TLT", "GLD", "DIA",
    "SLV", "IBIT", "SOXL", "SOXS", "BITO", "ETHA", "TQQQ", "SQQQ",
    "HYG", "JNK", "LQD", "EMB", "USHY", "VCIT", "VCLT", "SPIB",
    "EEM", "EWZ", "FXI", "KWEB", "VEA",
    "GDX", "KRE", "IGV", "SMH", "RSP", "SCHD", "SCHX",
    "XLP", "XLV", "XLU", "XLRE", "IJH",
    "TSLL", "TSDD", "SPDN", "TZA", "RWM", "NVD",
}


def _is_etf(symbol: str) -> bool:
    return symbol.upper() in _ETFS


@lru_cache(maxsize=64)
def _fetch_earnings_date(symbol: str) -> datetime | None:
    """Fetch next earnings date for a symbol. Cached for the process lifetime."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        cal = t.get_calendar()
        if cal is not None and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if dates:
                return min(dates)
        return None
    except Exception as e:
        log.debug(f"Could not get earnings for {symbol}: {e}")
        return None


def has_earnings_within(symbol: str, days: int) -> bool:
    """Return True if symbol has earnings within `days` calendar days.
    ETFs always return False. If yfinance unavailable, returns True (fail-closed).
    """
    if _is_etf(symbol):
        return False

    try:
        import yfinance  # noqa: F401
    except ImportError:
        log.warning("yfinance not installed — earnings check BLOCKED (fail-closed for safety)")
        return True

    earn_date = _fetch_earnings_date(symbol.upper())
    if earn_date is None:
        return False

    now = datetime.now()
    if isinstance(earn_date, datetime):
        delta = (earn_date - now).days
    else:
        delta = (earn_date - now.date()).days

    if 0 <= delta <= days:
        log.info(f"{symbol}: earnings in {delta}d — BLOCKING call buy")
        return True

    return False
