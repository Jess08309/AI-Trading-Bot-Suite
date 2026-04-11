"""
Technical Indicators for underlying price analysis.
Ported from CryptoBot + added options-specific indicators.
All functions take numpy arrays and return numpy arrays / floats.
"""
import warnings
import numpy as np
from typing import Dict, Optional

# Suppress numpy division warnings (handled by np.where guards)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)

    if len(gains) < period:
        return np.full_like(prices, 50.0)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    result = 100.0 - (100.0 / (1.0 + rs))
    result[:period] = 50.0
    return result


def macd(prices: np.ndarray, fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple:
    """MACD line, signal line, histogram."""
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(prices: np.ndarray, period: int = 20,
                    num_std: float = 2.0) -> tuple:
    """Upper, middle, lower bands + %B position."""
    middle = _sma(prices, period)
    std = _rolling_std(prices, period)
    upper = middle + num_std * std
    lower = middle - num_std * std
    # %B: where is price relative to bands (0 = lower, 1 = upper)
    band_width = upper - lower
    pct_b = np.where(band_width > 0, (prices - lower) / band_width, 0.5)
    return upper, middle, lower, pct_b


def stochastic(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """%K stochastic oscillator."""
    result = np.full_like(prices, 50.0)
    for i in range(period, len(prices)):
        window = prices[i - period + 1:i + 1]
        low = np.min(window)
        high = np.max(window)
        if high > low:
            result[i] = 100.0 * (prices[i] - low) / (high - low)
    return result


def atr(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range (simplified, close-only)."""
    tr = np.abs(np.diff(prices))
    tr = np.insert(tr, 0, 0.0)
    result = np.zeros_like(prices)
    if len(tr) < period:
        return result
    result[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(prices)):
        result[i] = (result[i-1] * (period - 1) + tr[i]) / period
    return result


def cci(prices: np.ndarray, period: int = 20) -> np.ndarray:
    """Commodity Channel Index."""
    tp = prices  # simplified: using close as typical price
    sma = _sma(tp, period)
    mad = _rolling_mad(tp, period)
    result = np.where(mad > 0, (tp - sma) / (0.015 * mad), 0.0)
    return result


def roc(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of Change."""
    result = np.zeros_like(prices)
    for i in range(period, len(prices)):
        if prices[i - period] > 0:
            result[i] = (prices[i] - prices[i - period]) / prices[i - period] * 100
    return result


def williams_r(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Williams %R."""
    result = np.full_like(prices, -50.0)
    for i in range(period, len(prices)):
        window = prices[i - period + 1:i + 1]
        high = np.max(window)
        low = np.min(window)
        if high > low:
            result[i] = -100.0 * (high - prices[i]) / (high - low)
    return result


def volatility_ratio(prices: np.ndarray, short: int = 5,
                     long: int = 20) -> np.ndarray:
    """Ratio of short-term to long-term volatility."""
    short_std = _rolling_std(prices, short)
    long_std = _rolling_std(prices, long)
    return np.where(long_std > 0, short_std / long_std, 1.0)


def mean_reversion_zscore(prices: np.ndarray, period: int = 20) -> np.ndarray:
    """Z-score of price relative to rolling mean."""
    sma = _sma(prices, period)
    std = _rolling_std(prices, period)
    return np.where(std > 0, (prices - sma) / std, 0.0)


def trend_strength(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX-like trend strength (0-100). Simplified using directional movement."""
    result = np.zeros_like(prices)
    if len(prices) < period + 1:
        return result

    ups = np.zeros_like(prices)
    downs = np.zeros_like(prices)

    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            ups[i] = diff
        else:
            downs[i] = -diff

    smooth_up = _ema(ups, period)
    smooth_down = _ema(downs, period)
    total = smooth_up + smooth_down
    result = np.where(total > 0, np.abs(smooth_up - smooth_down) / total * 100, 0)
    return result


def iv_rank(iv_current: float, iv_history: np.ndarray) -> float:
    """
    IV Rank: where current IV sits in its 52-week range (0-100).
    iv_history should be daily IV readings over past year.
    """
    if len(iv_history) == 0:
        return 50.0
    iv_min = np.min(iv_history)
    iv_max = np.max(iv_history)
    if iv_max == iv_min:
        return 50.0
    return float((iv_current - iv_min) / (iv_max - iv_min) * 100)


def iv_percentile(iv_current: float, iv_history: np.ndarray) -> float:
    """
    IV Percentile: % of days in history where IV was lower than current.
    """
    if len(iv_history) == 0:
        return 50.0
    return float(np.sum(iv_history < iv_current) / len(iv_history) * 100)


# ── Composite Feature Generator ─────────────────────────

def compute_all_indicators(prices: np.ndarray) -> Dict[str, float]:
    """
    Compute all indicators from a price array.
    Returns dict of latest values for ML features.
    """
    if len(prices) < 30:
        return _empty_features()

    rsi_vals = rsi(prices)
    macd_line, macd_sig, macd_hist = macd(prices)
    _, _, _, bb_pct = bollinger_bands(prices)
    stoch_vals = stochastic(prices)
    atr_vals = atr(prices)
    cci_vals = cci(prices)
    roc_vals = roc(prices)
    wr_vals = williams_r(prices)
    vol_ratio = volatility_ratio(prices)
    zscore = mean_reversion_zscore(prices)
    trend = trend_strength(prices)

    # Normalize ATR by price
    atr_norm = atr_vals[-1] / prices[-1] if prices[-1] > 0 else 0

    return {
        "rsi": rsi_vals[-1],
        "macd_hist": macd_hist[-1],
        "bb_position": bb_pct[-1],
        "stochastic": stoch_vals[-1],
        "atr_normalized": atr_norm,
        "cci": cci_vals[-1],
        "roc": roc_vals[-1],
        "williams_r": wr_vals[-1],
        "volatility_ratio": vol_ratio[-1],
        "zscore": zscore[-1],
        "trend_strength": trend[-1],
        # Price-derived
        "price_change_1": (prices[-1] / prices[-2] - 1) if len(prices) > 1 else 0,
        "price_change_5": (prices[-1] / prices[-6] - 1) if len(prices) > 5 else 0,
        "price_change_20": (prices[-1] / prices[-21] - 1) if len(prices) > 20 else 0,
    }


def _empty_features() -> Dict[str, float]:
    """Return neutral feature values when not enough data."""
    return {
        "rsi": 50.0, "macd_hist": 0.0, "bb_position": 0.5,
        "stochastic": 50.0, "atr_normalized": 0.0, "cci": 0.0,
        "roc": 0.0, "williams_r": -50.0, "volatility_ratio": 1.0,
        "zscore": 0.0, "trend_strength": 0.0,
        "price_change_1": 0.0, "price_change_5": 0.0, "price_change_20": 0.0,
    }


# ── Helpers ──────────────────────────────────────────────

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    result = np.zeros_like(data)
    if len(data) < period:
        return result
    multiplier = 2.0 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]
    return result


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    return result


def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1:i + 1])
    return result


def _rolling_mad(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean absolute deviation."""
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        result[i] = np.mean(np.abs(window - np.mean(window)))
    return result
