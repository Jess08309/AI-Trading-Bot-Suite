"""
Market Regime Detector — Shared Module for Multi-Bot Trading System.

Classifies the current market into one of four regimes:
  TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY

Uses technical indicators (SMA alignment, ADX, ATR, Bollinger Bands, etc.)
computed from price bars. Self-contained — only requires numpy.

Compatible bots:
  - CryptoBot   (C:\\Bot)          — crypto spot + futures
  - AlpacaBot   (C:\\AlpacaBot)    — options scalping
  - SpreadBot   (C:\\SpreadBot)    — credit put + call spreads
  - CallBuyer   (C:\\CallBuyer)    — momentum call buying
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger("regime_detector")

# ── Regime Constants ─────────────────────────────────────────────────────────
TRENDING_UP = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"
RANGING = "RANGING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"

ALL_REGIMES = (TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY)

# ── Default thresholds (tunable) ───────────────────────────────────────
_DEFAULTS = dict(
    adx_trend_threshold=25.0,       # ADX above this ⇒ trending
    atr_vol_threshold=1.50,         # ATR ratio above this ⇒ high-vol
    sma_slope_threshold=0.002,      # |slope| of SMA-20 normalised by price
    bb_squeeze_threshold=0.03,      # BB width / price below this ⇒ squeeze
    consec_bar_threshold=4,         # consecutive same-direction bars
    lookback_sma_short=20,
    lookback_sma_mid=50,
    lookback_sma_long=200,
    lookback_adx=14,
    lookback_atr=14,
    lookback_atr_avg=50,
    lookback_bb=20,
    bb_std_mult=2.0,
)


# ── Helpers ───────────────────────────────────────────────────────────────────────────────

def _to_arrays(bars: Union[List, Any]) -> Dict[str, np.ndarray]:
    """Convert a list of bar objects / dicts into numpy arrays."""
    closes, highs, lows, volumes = [], [], [], []
    for b in bars:
        if isinstance(b, dict):
            closes.append(float(b["close"]))
            highs.append(float(b["high"]))
            lows.append(float(b["low"]))
            volumes.append(float(b.get("volume", 0)))
        else:
            closes.append(float(b.close))
            highs.append(float(b.high))
            lows.append(float(b.low))
            volumes.append(float(getattr(b, "volume", 0)))
    return {
        "close": np.array(closes, dtype=np.float64),
        "high": np.array(highs, dtype=np.float64),
        "low": np.array(lows, dtype=np.float64),
        "volume": np.array(volumes, dtype=np.float64),
    }


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average (NaN-padded)."""
    out = np.full_like(arr, np.nan)
    if len(arr) < period:
        return out
    cumsum = np.cumsum(arr)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    out[period - 1:] = cumsum[period - 1:] / period
    return out


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    out = np.full_like(arr, np.nan)
    if len(arr) < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range array (length = len - 1, first element dropped)."""
    prev_close = close[:-1]
    h = high[1:]
    lo = low[1:]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_close), np.abs(lo - prev_close)))
    return tr


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> float:
    """Average Directional Index (scalar — latest value)."""
    n = len(close)
    if n < period * 2 + 1:
        return 0.0

    tr = _true_range(high, low, close)

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_smooth = _ema(tr, period)
    plus_di = 100.0 * _ema(plus_dm, period) / np.where(atr_smooth == 0, 1, atr_smooth)
    minus_di = 100.0 * _ema(minus_dm, period) / np.where(atr_smooth == 0, 1, atr_smooth)

    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = 100.0 * di_diff / np.where(di_sum == 0, 1, di_sum)

    adx_val = _ema(dx, period)
    last = adx_val[~np.isnan(adx_val)]
    return float(last[-1]) if len(last) > 0 else 0.0


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> float:
    """Average True Range (latest scalar)."""
    tr = _true_range(high, low, close)
    atr_vals = _ema(tr, period)
    last = atr_vals[~np.isnan(atr_vals)]
    return float(last[-1]) if len(last) > 0 else 0.0


def _atr_ratio(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               short_period: int = 14, long_period: int = 50) -> float:
    """Current ATR / average ATR over *long_period* days."""
    tr = _true_range(high, low, close)
    atr_short = _ema(tr, short_period)
    atr_long = _sma(tr, long_period)

    valid_short = atr_short[~np.isnan(atr_short)]
    valid_long = atr_long[~np.isnan(atr_long)]

    if len(valid_short) == 0 or len(valid_long) == 0 or valid_long[-1] == 0:
        return 1.0
    return float(valid_short[-1] / valid_long[-1])


def _bollinger_position(close: np.ndarray, period: int = 20,
                        std_mult: float = 2.0) -> Dict[str, float]:
    """Return BB width (normalised) and price position within bands."""
    if len(close) < period:
        return {"width": 0.0, "position": 0.5}
    mid = _sma(close, period)
    valid_start = period - 1

    # Rolling std
    std = np.full_like(close, np.nan)
    for i in range(valid_start, len(close)):
        std[i] = np.std(close[i - period + 1: i + 1], ddof=0)

    upper = mid + std_mult * std
    lower = mid - std_mult * std

    last_upper = upper[-1]
    last_lower = lower[-1]
    last_mid = mid[-1]
    last_close = close[-1]

    band_w = (last_upper - last_lower) / last_mid if last_mid != 0 else 0
    pos = ((last_close - last_lower) / (last_upper - last_lower)
           if (last_upper - last_lower) != 0 else 0.5)
    return {"width": float(band_w), "position": float(np.clip(pos, 0, 1))}


def _consecutive_direction(close: np.ndarray) -> int:
    """Count consecutive same-direction bars at the end (positive = up)."""
    if len(close) < 2:
        return 0
    diffs = np.diff(close)
    last_dir = np.sign(diffs[-1])
    if last_dir == 0:
        return 0
    count = 0
    for i in range(len(diffs) - 1, -1, -1):
        if np.sign(diffs[i]) == last_dir:
            count += 1
        else:
            break
    return int(count * last_dir)


# ── Strategy-Specific Adjustment Tables ──────────────────────────────────

# ── Regime Flip Severity Matrix ────────────────────────────────────────────
# Key = (from_regime, to_regime), Value = severity 0.0–1.0
# Higher severity = more dangerous transition, requires stronger cooldown
_FLIP_SEVERITY: Dict[Tuple[str, str], float] = {
    # Full reversals — highest severity
    (TRENDING_UP, TRENDING_DOWN): 1.0,
    (TRENDING_DOWN, TRENDING_UP): 0.9,
    # Trend → high volatility
    (TRENDING_UP, HIGH_VOLATILITY): 0.85,
    (TRENDING_DOWN, HIGH_VOLATILITY): 0.75,
    # Ranging → trending (less severe — expected breakout)
    (RANGING, TRENDING_DOWN): 0.6,
    (RANGING, TRENDING_UP): 0.4,
    (RANGING, HIGH_VOLATILITY): 0.7,
    # High vol transitions
    (HIGH_VOLATILITY, TRENDING_DOWN): 0.65,
    (HIGH_VOLATILITY, TRENDING_UP): 0.5,
    (HIGH_VOLATILITY, RANGING): 0.3,
    # Any → ranging (lower severity — market calming down)
    (TRENDING_UP, RANGING): 0.35,
    (TRENDING_DOWN, RANGING): 0.3,
}

# ── Default Flip Detection Parameters ─────────────────────────────────
_FLIP_DEFAULTS = dict(
    cooldown_minutes=60,        # minutes to stay cautious after a flip
    whipsaw_window_hours=6,     # window for counting flip frequency
    whipsaw_threshold=3,        # flips in window to trigger whipsaw mode
    max_history=50,             # max entries in regime history ring buffer
    # Adjustment multipliers during cooldown (applied on top of regime adjustments)
    cooldown_position_mult=0.50,     # halve position size during cooldown
    whipsaw_position_mult=0.30,      # 30% of normal during whipsaw
    cooldown_block_severity=0.80,    # block new entries if severity >= this
)

_ADJUSTMENTS: Dict[str, Dict[str, Dict[str, float]]] = {
    TRENDING_UP: {
        "CryptoBot": {
            "position_size": 1.15, "stop_loss_width": 1.0,
            "take_profit_width": 1.20, "trade_frequency": 1.10,
            "long_bias": 1.3, "short_bias": 0.5,
        },
        "AlpacaBot": {
            "position_size": 1.10, "stop_loss_width": 1.0,
            "take_profit_width": 1.15, "trade_frequency": 1.0,
            "call_bias": 1.4, "put_bias": 0.3,
        },
        "SpreadBot": {
            "position_size": 1.10, "stop_loss_width": 1.20,
            "take_profit_width": 1.0, "trade_frequency": 1.10,
            "otm_buffer": 0.85,   # can be tighter (bullish = safe for puts)
            "credit_threshold": 1.0,
        },
        "CallBuyer": {
            "position_size": 1.20, "stop_loss_width": 1.0,
            "take_profit_width": 1.30, "trade_frequency": 1.20,
            "confidence_offset": -0.05,  # lower threshold = more aggressive
        },
    },
    TRENDING_DOWN: {
        "CryptoBot": {
            "position_size": 0.80, "stop_loss_width": 1.10,
            "take_profit_width": 0.90, "trade_frequency": 0.80,
            "long_bias": 0.4, "short_bias": 1.4,
        },
        "AlpacaBot": {
            "position_size": 0.90, "stop_loss_width": 1.10,
            "take_profit_width": 1.0, "trade_frequency": 0.90,
            "call_bias": 0.3, "put_bias": 1.3,
        },
        "SpreadBot": {
            "position_size": 0.60, "stop_loss_width": 0.80,
            "take_profit_width": 0.80, "trade_frequency": 0.50,
            "otm_buffer": 1.40,   # much wider OTM buffer — dangerous regime
            "credit_threshold": 1.30,
        },
        "CallBuyer": {
            "position_size": 0.50, "stop_loss_width": 0.80,
            "take_profit_width": 0.70, "trade_frequency": 0.40,
            "confidence_offset": 0.15,  # raise threshold = very selective
        },
    },
    RANGING: {
        "CryptoBot": {
            "position_size": 0.90, "stop_loss_width": 0.85,
            "take_profit_width": 0.80, "trade_frequency": 1.0,
            "long_bias": 1.0, "short_bias": 1.0,
            "mean_revert": True,
        },
        "AlpacaBot": {
            "position_size": 1.0, "stop_loss_width": 0.90,
            "take_profit_width": 0.90, "trade_frequency": 1.0,
            "call_bias": 1.0, "put_bias": 1.0,
        },
        "SpreadBot": {
            "position_size": 1.15, "stop_loss_width": 1.0,
            "take_profit_width": 1.0, "trade_frequency": 1.20,
            "otm_buffer": 0.90,
            "credit_threshold": 0.85,  # lower threshold — ideal for selling premium
        },
        "CallBuyer": {
            "position_size": 0.70, "stop_loss_width": 0.85,
            "take_profit_width": 0.75, "trade_frequency": 0.60,
            "confidence_offset": 0.05,  # slightly raised threshold
        },
    },
    HIGH_VOLATILITY: {
        "CryptoBot": {
            "position_size": 0.50, "stop_loss_width": 1.50,
            "take_profit_width": 1.50, "trade_frequency": 0.50,
            "long_bias": 0.7, "short_bias": 0.7,
        },
        "AlpacaBot": {
            "position_size": 0.50, "stop_loss_width": 1.50,
            "take_profit_width": 1.40, "trade_frequency": 0.40,
            "call_bias": 0.5, "put_bias": 0.5,
        },
        "SpreadBot": {
            "position_size": 0.0,  # HALT — too dangerous
            "stop_loss_width": 0.60,
            "take_profit_width": 0.60,
            "trade_frequency": 0.0,
            "otm_buffer": 2.0,
            "credit_threshold": 2.0,
        },
        "CallBuyer": {
            "position_size": 0.40, "stop_loss_width": 1.50,
            "take_profit_width": 1.60, "trade_frequency": 0.30,
            "confidence_offset": 0.10,  # raised threshold = very selective
        },
    },
}


# ── Main Class ───────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Market regime classifier using technical indicators.

    Usage::

        detector = RegimeDetector()
        result = detector.detect(bars)
        # result = {
        #     "regime": "TRENDING_UP",
        #     "confidence": 0.82,
        #     "trend_strength": 0.65,
        #     "volatility_ratio": 1.12,
        #     "suggested_adjustments": { ... },
        # }

    Parameters
    ----------
    bot_name : str, optional
        If provided, ``suggested_adjustments`` is pre-filtered for this bot.
    **kwargs
        Override any default threshold (see ``_DEFAULTS``).
    """

    def __init__(self, bot_name: Optional[str] = None, **kwargs):
        self.bot_name = bot_name
        self.params = {**_DEFAULTS, **kwargs}
        self._last_regime: Optional[Dict[str, Any]] = None

        # ── Flip detection state ─────────────────
        self._flip_params = {**_FLIP_DEFAULTS, **{k: v for k, v in kwargs.items()
                                                   if k in _FLIP_DEFAULTS}}
        # History: list of (epoch_timestamp, regime_str, confidence)
        self._regime_history: List[Tuple[float, str, float]] = []
        self._last_flip_time: Optional[float] = None
        self._last_flip_from: Optional[str] = None
        self._last_flip_to: Optional[str] = None
        self._last_flip_severity: float = 0.0
        self._current_regime_since: float = time.time()
