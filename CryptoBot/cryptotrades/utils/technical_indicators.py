"""
Technical Analysis Indicators Library.
Pure NumPy implementation of 25+ indicators for ML feature engineering.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict


def sma(prices: List[float], period: int = 20) -> np.ndarray:
    """Simple Moving Average."""
    arr = np.array(prices, dtype=float)
    if len(arr) < period:
        return np.full(len(arr), np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.mean(arr[i - period + 1:i + 1])
    return result


def ema(prices: List[float], period: int = 12) -> np.ndarray:
    """Exponential Moving Average."""
    arr = np.array(prices, dtype=float)
    if len(arr) < period:
        return np.full(len(arr), np.nan)
    result = np.full(len(arr), np.nan)
    multiplier = 2.0 / (period + 1)
    result[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        result[i] = (arr[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def rsi(prices: List[float], period: int = 14) -> np.ndarray:
    """Relative Strength Index (0-100)."""
    arr = np.array(prices, dtype=float)
    if len(arr) < period + 1:
        return np.full(len(arr), 50.0)
    deltas = np.diff(arr)
    result = np.full(len(arr), 50.0)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return result


def macd(prices: List[float], fast: int = 12, slow: int = 26,
         signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD: Returns (macd_line, signal_line, histogram)."""
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = fast_ema - slow_ema

    # Build signal line from valid MACD values
    valid = ~np.isnan(macd_line)
    valid_vals = macd_line[valid].tolist()
    signal_line_partial = ema(valid_vals, signal_period)

    signal_full = np.full(len(prices), np.nan)
    valid_idx = np.where(valid)[0]
    for i, idx in enumerate(valid_idx):
        if i < len(signal_line_partial):
            signal_full[idx] = signal_line_partial[i]

    histogram = macd_line - signal_full
    return macd_line, signal_full, histogram


def bollinger_bands(prices: List[float], period: int = 20,
                    num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: Returns (upper, middle, lower)."""
    arr = np.array(prices, dtype=float)
    middle = sma(prices, period)
    upper = np.full(len(arr), np.nan)
    lower = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        std = np.std(arr[i - period + 1:i + 1])
        upper[i] = middle[i] + num_std * std
        lower[i] = middle[i] - num_std * std
    return upper, middle, lower


def atr_approx(prices: List[float], period: int = 14) -> np.ndarray:
    """Average True Range approximation (close-to-close, no high/low data)."""
    arr = np.array(prices, dtype=float)
    if len(arr) < 2:
        return np.full(len(arr), 0.0)
    tr = np.full(len(arr), 0.0)
    for i in range(1, len(arr)):
        tr[i] = abs(arr[i] - arr[i - 1])
    result = np.full(len(arr), np.nan)
    if len(arr) >= period + 1:
        result[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, len(arr)):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def stochastic(prices: List[float], period: int = 14,
               smooth_k: int = 3, smooth_d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator: Returns (%K, %D). Range 0-100."""
    arr = np.array(prices, dtype=float)
    raw_k = np.full(len(arr), 50.0)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1:i + 1]
        high, low = np.max(window), np.min(window)
        if high != low:
            raw_k[i] = ((arr[i] - low) / (high - low)) * 100
    k_vals = sma(raw_k.tolist(), smooth_k)
    d_vals = sma(k_vals.tolist(), smooth_d)
    return k_vals, d_vals


def rate_of_change(prices: List[float], period: int = 10) -> np.ndarray:
    """Rate of Change (percentage)."""
    arr = np.array(prices, dtype=float)
    result = np.full(len(arr), 0.0)
    for i in range(period, len(arr)):
        if arr[i - period] != 0:
            result[i] = ((arr[i] - arr[i - period]) / arr[i - period]) * 100
    return result


def momentum(prices: List[float], period: int = 10) -> np.ndarray:
    """Price momentum (absolute difference)."""
    arr = np.array(prices, dtype=float)
    result = np.full(len(arr), 0.0)
    for i in range(period, len(arr)):
        result[i] = arr[i] - arr[i - period]
    return result


def williams_r(prices: List[float], period: int = 14) -> np.ndarray:
    """Williams %R. Range -100 to 0."""
    arr = np.array(prices, dtype=float)
    result = np.full(len(arr), -50.0)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1:i + 1]
        high, low = np.max(window), np.min(window)
        if high != low:
            result[i] = ((high - arr[i]) / (high - low)) * -100
    return result


def obv_direction(prices: List[float]) -> np.ndarray:
    """On-Balance Volume direction proxy (uses price direction since we lack volume)."""
    arr = np.array(prices, dtype=float)
    result = np.zeros(len(arr))
    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            result[i] = result[i - 1] + 1
        elif arr[i] < arr[i - 1]:
            result[i] = result[i - 1] - 1
        else:
            result[i] = result[i - 1]
    return result


def price_position(prices: List[float], period: int = 20) -> np.ndarray:
    """Where is current price within N-period range (0 to 1)."""
    arr = np.array(prices, dtype=float)
    result = np.full(len(arr), 0.5)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1:i + 1]
        high, low = np.max(window), np.min(window)
        if high != low:
            result[i] = (arr[i] - low) / (high - low)
    return result


def trend_strength(prices: List[float], period: int = 20) -> np.ndarray:
    """ADX approximation using directional movement. Range 0-100."""
    arr = np.array(prices, dtype=float)
    if len(arr) < period + 1:
        return np.full(len(arr), 0.0)
    result = np.full(len(arr), 0.0)
    for i in range(period, len(arr)):
        window = arr[i - period:i + 1]
        ups = sum(1 for j in range(1, len(window)) if window[j] > window[j - 1])
        downs = sum(1 for j in range(1, len(window)) if window[j] < window[j - 1])
        total = ups + downs
        if total > 0:
            directional = abs(ups - downs) / total
            price_range = np.max(window) - np.min(window)
            norm = (price_range / np.min(window) * 100) if np.min(window) > 0 else 0
            result[i] = directional * min(norm * 10, 100)
    return result


def consecutive_direction(prices: List[float]) -> np.ndarray:
    """Count consecutive up/down periods. Positive=up streak, negative=down."""
    arr = np.array(prices, dtype=float)
    result = np.zeros(len(arr))
    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            result[i] = max(1, result[i - 1] + 1) if result[i - 1] > 0 else 1
        elif arr[i] < arr[i - 1]:
            result[i] = min(-1, result[i - 1] - 1) if result[i - 1] < 0 else -1
    return result


def price_acceleration(prices: List[float], period: int = 5) -> np.ndarray:
    """Rate of change of rate of change (second derivative)."""
    roc = rate_of_change(prices, period)
    roc2 = np.full(len(prices), 0.0)
    for i in range(period, len(roc)):
        roc2[i] = roc[i] - roc[i - period]
    return roc2


def volatility_ratio(prices: List[float], short_period: int = 5,
                     long_period: int = 20) -> np.ndarray:
    """Short-term volatility divided by long-term volatility."""
    arr = np.array(prices, dtype=float)
    result = np.full(len(arr), 1.0)
    for i in range(long_period, len(arr)):
        short_std = np.std(arr[i - short_period + 1:i + 1])
        long_std = np.std(arr[i - long_period + 1:i + 1])
        if long_std > 0:
            result[i] = short_std / long_std
    return result


def mean_reversion_score(prices: List[float], period: int = 20) -> np.ndarray:
    """Z-score of current price vs moving average."""
    arr = np.array(prices, dtype=float)
    ma = sma(prices, period)
    result = np.full(len(arr), 0.0)
    for i in range(period - 1, len(arr)):
        std = np.std(arr[i - period + 1:i + 1])
        if std > 0 and not np.isnan(ma[i]):
            result[i] = (arr[i] - ma[i]) / std
    return result


def higher_highs_lower_lows(prices: List[float], period: int = 5) -> Tuple[float, float]:
    """Detect higher highs / lower lows pattern. Returns (hh_score, ll_score)."""
    if len(prices) < period * 2:
        return 0.0, 0.0
    recent = prices[-period:]
    prior = prices[-period * 2:-period]
    hh = 1.0 if max(recent) > max(prior) else (-1.0 if max(recent) < max(prior) else 0.0)
    ll = 1.0 if min(recent) > min(prior) else (-1.0 if min(recent) < min(prior) else 0.0)
    return hh, ll


# ============================================================
# NEW INDICATORS — Momentum Oscillator Suite
# ============================================================

def cci(prices: List[float], period: int = 20) -> np.ndarray:
    """Commodity Channel Index.
    Measures deviation from statistical mean.
    CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
    Using close price as proxy for typical price (no H/L data).
    """
    arr = np.array(prices, dtype=float)
    result = np.full(len(arr), 0.0)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1:i + 1]
        mean = np.mean(window)
        mean_dev = np.mean(np.abs(window - mean))
        if mean_dev > 0:
            result[i] = (arr[i] - mean) / (0.015 * mean_dev)
    return result


def ultimate_oscillator(prices: List[float], p1: int = 7, p2: int = 14, p3: int = 28) -> np.ndarray:
    """Ultimate Oscillator — combines three timeframes to reduce false signals.
    BP = Close - min(Close, Prior Close)
    TR = abs(Close - Prior Close)  (approximation without H/L data)
    UO = 100 * (4*Avg7 + 2*Avg14 + Avg28) / 7
    Range: 0-100.
    """
    arr = np.array(prices, dtype=float)
    n = len(arr)
    result = np.full(n, 50.0)
    if n < p3 + 1:
        return result

    bp = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        bp[i] = arr[i] - min(arr[i], arr[i - 1])
        tr[i] = abs(arr[i] - arr[i - 1])

    for i in range(p3, n):
        tr_sum1 = np.sum(tr[i - p1 + 1:i + 1])
        tr_sum2 = np.sum(tr[i - p2 + 1:i + 1])
        tr_sum3 = np.sum(tr[i - p3 + 1:i + 1])

        avg1 = np.sum(bp[i - p1 + 1:i + 1]) / tr_sum1 if tr_sum1 > 0 else 0.5
        avg2 = np.sum(bp[i - p2 + 1:i + 1]) / tr_sum2 if tr_sum2 > 0 else 0.5
        avg3 = np.sum(bp[i - p3 + 1:i + 1]) / tr_sum3 if tr_sum3 > 0 else 0.5

        result[i] = 100.0 * (4 * avg1 + 2 * avg2 + avg3) / 7.0

    return result


def trix(prices: List[float], period: int = 15) -> np.ndarray:
    """TRIX — Triple Exponential Average oscillator.
    1-period ROC of a triple-smoothed EMA.
    Filters noise extremely well; great for trend-following.
    """
    ema1 = ema(prices, period)
    # EMA of EMA (second smooth)
    valid1 = [float(v) for v in ema1 if not np.isnan(v)]
    ema2_partial = ema(valid1, period) if len(valid1) >= period else np.full(len(valid1), np.nan)
    # EMA of EMA of EMA (third smooth)
    valid2 = [float(v) for v in ema2_partial if not np.isnan(v)]
    ema3_partial = ema(valid2, period) if len(valid2) >= period else np.full(len(valid2), np.nan)

    result = np.full(len(prices), 0.0)
    valid3 = [float(v) for v in ema3_partial if not np.isnan(v)]
    if len(valid3) >= 2:
        # Map back to original array indices
        start_idx = len(prices) - len(valid3)
        for i in range(1, len(valid3)):
            idx = start_idx + i
            if idx < len(prices) and valid3[i - 1] != 0:
                result[idx] = (valid3[i] - valid3[i - 1]) / valid3[i - 1] * 100.0
    return result


def chande_momentum_oscillator(prices: List[float], period: int = 14) -> np.ndarray:
    """Chande Momentum Oscillator (CMO).
    CMO = (sum_up - sum_down) / (sum_up + sum_down) * 100
    Range: -100 to +100. More responsive than RSI.
    """
    arr = np.array(prices, dtype=float)
    result = np.full(len(arr), 0.0)
    if len(arr) < period + 1:
        return result

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    for i in range(period, len(deltas)):
        sum_up = np.sum(gains[i - period + 1:i + 1])
        sum_down = np.sum(losses[i - period + 1:i + 1])
        total = sum_up + sum_down
        if total > 0:
            result[i + 1] = (sum_up - sum_down) / total * 100.0
    return result


def compute_all_indicators(prices: List[float]) -> Dict[str, float]:
    """Compute 15 curated technical indicators for ML feature engineering.

    10 momentum oscillators (user-selected):
        RSI, MACD, Stochastic, CCI, ROC, Momentum, Williams %R,
        Ultimate Oscillator, TRIX, CMO
    5 complementary indicators:
        ATR, Trend Strength (ADX proxy), Bollinger Band position,
        Mean Reversion Z-score, Volatility Ratio

    Requires at least 5 price points. Best with 30+.
    """
    n = len(prices)
    if n < 5:
        return {}

    features = {}

    # ── 1. RSI (14) ─────────────────────────────────────────
    rsi_vals = rsi(prices, 14)
    features["rsi_14"] = float(rsi_vals[-1]) if not np.isnan(rsi_vals[-1]) else 50.0

    # ── 2. MACD histogram ───────────────────────────────────
    macd_line_v, _signal, histogram = macd(prices)
    features["macd_histogram"] = float(histogram[-1]) if not np.isnan(histogram[-1]) else 0.0

    # ── 3. Stochastic %K ────────────────────────────────────
    k, _d = stochastic(prices, 14)
    features["stoch_k"] = float(k[-1]) if not np.isnan(k[-1]) else 50.0

    # ── 4. CCI (20) ─────────────────────────────────────────
    cci_vals = cci(prices, 20)
    features["cci_20"] = float(cci_vals[-1])

    # ── 5. Rate of Change (10) ──────────────────────────────
    features["roc_10"] = float(rate_of_change(prices, min(10, n - 1))[-1])

    # ── 6. Momentum (10) ────────────────────────────────────
    features["momentum_10"] = float(momentum(prices, min(10, n - 1))[-1])

    # ── 7. Williams %R (14) ─────────────────────────────────
    features["williams_r"] = float(williams_r(prices, 14)[-1])

    # ── 8. Ultimate Oscillator (7/14/28) ────────────────────
    uo_vals = ultimate_oscillator(prices, 7, 14, min(28, n - 1))
    features["ultimate_osc"] = float(uo_vals[-1])

    # ── 9. TRIX (15) ────────────────────────────────────────
    trix_vals = trix(prices, min(15, max(3, n // 4)))
    features["trix_15"] = float(trix_vals[-1])

    # ── 10. Chande Momentum Oscillator (14) ─────────────────
    cmo_vals = chande_momentum_oscillator(prices, 14)
    features["cmo_14"] = float(cmo_vals[-1])

    # ── 11. ATR (14) — normalized by price ──────────────────
    atr_vals = atr_approx(prices, 14)
    atr_val = float(atr_vals[-1]) if not np.isnan(atr_vals[-1]) else 0.0
    features["atr_14"] = atr_val / prices[-1] if prices[-1] > 0 else 0.0

    # ── 12. Trend Strength (ADX proxy, 20) ──────────────────
    features["trend_strength"] = float(trend_strength(prices, min(20, n))[-1])

    # ── 13. Bollinger Band position (20) ────────────────────
    bb_upper, _bb_mid, bb_lower = bollinger_bands(prices, min(20, n))
    if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
        bb_width = bb_upper[-1] - bb_lower[-1]
        features["bb_position"] = (prices[-1] - bb_lower[-1]) / bb_width if bb_width > 0 else 0.5
    else:
        features["bb_position"] = 0.5

    # ── 14. Mean Reversion Z-score (20) ─────────────────────
    features["mean_reversion"] = float(mean_reversion_score(prices, min(20, n))[-1])

    # ── 15. Volatility Ratio (5 vs 20) ──────────────────────
    features["vol_ratio"] = float(volatility_ratio(prices, 5, min(20, n))[-1])

    return features
