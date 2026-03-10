"""
Market Regime Detector — Shared Module for Multi-Bot Trading System.

Classifies the current market into one of four regimes:
  TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY

Uses technical indicators (SMA alignment, ADX, ATR, Bollinger Bands, etc.)
computed from price bars. Self-contained — only requires numpy.

Compatible bots:
  - CryptoBot   (C:\\Bot)          — crypto spot + futures
  - AlpacaBot   (C:\\AlpacaBot)    — options scalping
  - PutSeller   (C:\\PutSeller)    — credit put spreads
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

# ── Default thresholds (tunable) ─────────────────────────────────────────────
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


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Strategy-Specific Adjustment Tables ──────────────────────────────────────

# ── Regime Flip Severity Matrix ──────────────────────────────────────────────
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

# ── Default Flip Detection Parameters ───────────────────────────────────────
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
        "PutSeller": {
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
        "PutSeller": {
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
        "PutSeller": {
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
        "PutSeller": {
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


# ── Main Class ───────────────────────────────────────────────────────────────

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

        # ── Flip detection state ─────────────────────────
        self._flip_params = {**_FLIP_DEFAULTS, **{k: v for k, v in kwargs.items()
                                                   if k in _FLIP_DEFAULTS}}
        # History: list of (epoch_timestamp, regime_str, confidence)
        self._regime_history: List[Tuple[float, str, float]] = []
        self._last_flip_time: Optional[float] = None
        self._last_flip_from: Optional[str] = None
        self._last_flip_to: Optional[str] = None
        self._last_flip_severity: float = 0.0
        self._current_regime_since: float = time.time()

    # ── Public API ───────────────────────────────────────

    def detect(
        self,
        bars: Union[List, Any],
        spy_bars: Optional[Union[List, Any]] = None,
    ) -> Dict[str, Any]:
        """Classify the current market regime from price bars.

        Parameters
        ----------
        bars : list
            Price bars (objects with .close/.high/.low or dicts). Need ≥50 bars,
            ideally ≥200 for full SMA alignment.
        spy_bars : list, optional
            SPY bars for cross-market regime context. If provided and *bars*
            represent a single stock, SPY regime is blended in.

        Returns
        -------
        dict
            ``regime``, ``confidence``, ``trend_strength``,
            ``volatility_ratio``, ``signals``, ``suggested_adjustments``.
        """
        data = _to_arrays(bars)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        n = len(close)

        if n < 30:
            return self._fallback("insufficient_data")

        # ── Compute signals ──────────────────────────────
        signals: Dict[str, Any] = {}

        # 1. SMA alignment
        sma20 = _sma(close, self.params["lookback_sma_short"])
        sma50 = _sma(close, self.params["lookback_sma_mid"])
        sma200 = _sma(close, self.params["lookback_sma_long"]) if n >= 200 else None

        sma20_last = sma20[-1] if not np.isnan(sma20[-1]) else close[-1]
        sma50_last = sma50[-1] if not np.isnan(sma50[-1]) else close[-1]
        sma200_last = sma200[-1] if sma200 is not None and not np.isnan(sma200[-1]) else None

        # SMA alignment score: +1 full bullish stack, -1 full bearish stack
        alignment = 0.0
        if sma20_last > sma50_last:
            alignment += 0.5
        else:
            alignment -= 0.5
        if sma200_last is not None:
            if sma50_last > sma200_last:
                alignment += 0.3
            else:
                alignment -= 0.3
            if close[-1] > sma200_last:
                alignment += 0.2
            else:
                alignment -= 0.2
        else:
            # No 200-SMA — weight the 20/50 relationship more
            if close[-1] > sma50_last:
                alignment += 0.3
            else:
                alignment -= 0.3

        signals["sma_alignment"] = round(float(alignment), 3)

        # 2. ADX (trend strength)
        adx_val = _adx(high, low, close, self.params["lookback_adx"])
        trend_strength = float(np.clip(adx_val / 50.0, 0, 1))  # normalise to 0-1
        signals["adx"] = round(adx_val, 2)
        signals["trend_strength"] = round(trend_strength, 3)

        # 3. ATR ratio (volatility expansion)
        vol_ratio = _atr_ratio(
            high, low, close,
            self.params["lookback_atr"],
            self.params["lookback_atr_avg"],
        )
        signals["volatility_ratio"] = round(vol_ratio, 3)

        # 4. Slope of SMA-20 (normalised by price)
        if n >= self.params["lookback_sma_short"] + 5:
            slope_window = 5
            recent_sma = sma20[-slope_window:]
            if not np.any(np.isnan(recent_sma)):
                raw_slope = (recent_sma[-1] - recent_sma[0]) / slope_window
                norm_slope = raw_slope / close[-1] if close[-1] != 0 else 0
            else:
                norm_slope = 0.0
        else:
            norm_slope = 0.0
        signals["sma20_slope"] = round(float(norm_slope), 5)

        # 5. Bollinger Band position
        bb = _bollinger_position(
            close, self.params["lookback_bb"], self.params["bb_std_mult"],
        )
        signals["bb_width"] = round(bb["width"], 4)
        signals["bb_position"] = round(bb["position"], 3)

        # 6. Consecutive directional bars
        consec = _consecutive_direction(close)
        signals["consecutive_bars"] = consec

        # ── Regime classification (scoring approach) ─────
        scores = {
            TRENDING_UP: 0.0,
            TRENDING_DOWN: 0.0,
            RANGING: 0.0,
            HIGH_VOLATILITY: 0.0,
        }

        # ─ High Volatility check (overrides others when extreme) ─
        if vol_ratio >= self.params["atr_vol_threshold"]:
            scores[HIGH_VOLATILITY] += 0.40
        if vol_ratio >= 2.0:
            scores[HIGH_VOLATILITY] += 0.20
        if bb["width"] > 0.08:
            scores[HIGH_VOLATILITY] += 0.10

        # ─ Trending signals ─
        adx_trending = adx_val >= self.params["adx_trend_threshold"]

        if adx_trending:
            if alignment > 0:
                scores[TRENDING_UP] += 0.25
            elif alignment < 0:
                scores[TRENDING_DOWN] += 0.25

        # SMA alignment contribution
        if alignment >= 0.5:
            scores[TRENDING_UP] += 0.20
        elif alignment <= -0.5:
            scores[TRENDING_DOWN] += 0.20
        elif -0.2 <= alignment <= 0.2:
            scores[RANGING] += 0.15

        # Slope contribution
        slope_thresh = self.params["sma_slope_threshold"]
        if norm_slope > slope_thresh:
            scores[TRENDING_UP] += 0.15
        elif norm_slope < -slope_thresh:
            scores[TRENDING_DOWN] += 0.15
        else:
            scores[RANGING] += 0.10

        # Consecutive bars
        ct = self.params["consec_bar_threshold"]
        if consec >= ct:
            scores[TRENDING_UP] += 0.10
        elif consec <= -ct:
            scores[TRENDING_DOWN] += 0.10

        # BB position for trend confirmation
        if bb["position"] > 0.80:
            scores[TRENDING_UP] += 0.10
        elif bb["position"] < 0.20:
            scores[TRENDING_DOWN] += 0.10
        elif 0.35 <= bb["position"] <= 0.65:
            scores[RANGING] += 0.10

        # BB squeeze → ranging
        if bb["width"] < self.params["bb_squeeze_threshold"]:
            scores[RANGING] += 0.15

        # Low ADX → ranging
        if adx_val < 20:
            scores[RANGING] += 0.20
        elif adx_val < self.params["adx_trend_threshold"]:
            scores[RANGING] += 0.10

        # ── SPY cross-market blend ───────────────────────
        spy_regime_info = None
        if spy_bars is not None:
            try:
                spy_data = _to_arrays(spy_bars)
                if len(spy_data["close"]) >= 30:
                    spy_det = RegimeDetector()
                    spy_result = spy_det.detect(spy_bars, spy_bars=None)
                    spy_regime_info = spy_result
                    # Blend: 20% weight from SPY regime
                    spy_r = spy_result["regime"]
                    for r in ALL_REGIMES:
                        if r == spy_r:
                            scores[r] += 0.10
            except Exception:
                pass

        # ── Pick winner ──────────────────────────────────
        total = sum(scores.values())
        if total == 0:
            return self._fallback("no_signal")

        # Normalise
        for r in scores:
            scores[r] /= total

        regime = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = float(scores[regime])

        # Sharpen confidence: if top score barely beats second, lower conf
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
            # confidence is already normalised; adjust by margin clarity
            confidence = float(np.clip(confidence * (0.5 + margin), 0.10, 0.99))

        # ── Build result ─────────────────────────────────
        adjustments = self.get_adjustments(regime, self.bot_name)

        result: Dict[str, Any] = {
            "regime": regime,
            "confidence": round(confidence, 3),
            "trend_strength": round(trend_strength, 3),
            "volatility_ratio": round(vol_ratio, 3),
            "signals": signals,
            "regime_scores": {r: round(v, 3) for r, v in scores.items()},
            "suggested_adjustments": adjustments,
        }

        if spy_regime_info:
            result["spy_regime"] = spy_regime_info.get("regime")

        self._last_regime = result
        return result

    # ── Regime Flip Detection ────────────────────────────

    def record_regime(self, regime: str, confidence: float = 0.0) -> Dict[str, Any]:
        """Record a regime observation and detect flips.

        Call this after every ``detect()`` call. It tracks regime history,
        detects transitions, measures severity, and computes whipsaw score.

        Parameters
        ----------
        regime : str
            Current regime classification.
        confidence : float
            Confidence of the classification (0–1).

        Returns
        -------
        dict
            Flip state — see ``get_flip_state()``.
        """
        now = time.time()
        max_hist = self._flip_params["max_history"]

        # Get previous regime (if any)
        prev_regime = self._regime_history[-1][1] if self._regime_history else None

        # Append to history
        self._regime_history.append((now, regime, confidence))
        if len(self._regime_history) > max_hist:
            self._regime_history = self._regime_history[-max_hist:]

        # Detect flip
        if prev_regime is not None and regime != prev_regime:
            severity = _FLIP_SEVERITY.get((prev_regime, regime), 0.5)
            self._last_flip_time = now
            self._last_flip_from = prev_regime
            self._last_flip_to = regime
            self._last_flip_severity = severity
            self._current_regime_since = now
            log.info(
                f"REGIME FLIP: {prev_regime} → {regime} | "
                f"severity={severity:.2f} | confidence={confidence:.2f}"
            )
        elif prev_regime is None:
            # First observation
            self._current_regime_since = now

        return self.get_flip_state()

    def get_flip_state(self) -> Dict[str, Any]:
        """Get the current regime flip state.

        Returns
        -------
        dict
            - ``is_cooldown`` (bool): Within cooldown period after a flip.
            - ``cooldown_remaining_min`` (float): Minutes left in cooldown.
            - ``flip_severity`` (float): Severity of last flip (0–1).
            - ``whipsaw_score`` (float): How "whipsaw-y" the market is (0–1).
            - ``flips_in_window`` (int): Number of flips in the whipsaw window.
            - ``last_flip_from`` (str|None): Previous regime before flip.
            - ``last_flip_to`` (str|None): New regime after flip.
            - ``time_in_current_min`` (float): Minutes in current regime.
            - ``regime_stability`` (float): 0–1 composite stability score.
            - ``adjustment_multiplier`` (float): Combined position size multiplier
              (0.3–1.0) incorporating cooldown + whipsaw.
            - ``should_block_entries`` (bool): True if entries should be blocked.
        """
        now = time.time()
        cooldown_sec = self._flip_params["cooldown_minutes"] * 60
        whipsaw_sec = self._flip_params["whipsaw_window_hours"] * 3600

        # ── Cooldown ────────────────────────────────────
        if self._last_flip_time is not None:
            elapsed = now - self._last_flip_time
            is_cooldown = elapsed < cooldown_sec
            cooldown_remaining = max(0, (cooldown_sec - elapsed) / 60)
        else:
            is_cooldown = False
            cooldown_remaining = 0.0

        # ── Whipsaw detection ───────────────────────────
        # Count regime transitions in the window
        flips = 0
        cutoff = now - whipsaw_sec
        for i in range(1, len(self._regime_history)):
            ts, reg, _ = self._regime_history[i]
            if ts < cutoff:
                continue
            prev_reg = self._regime_history[i - 1][1]
            if reg != prev_reg:
                flips += 1

        whipsaw_threshold = self._flip_params["whipsaw_threshold"]
        whipsaw_score = min(1.0, flips / max(whipsaw_threshold, 1))
        is_whipsaw = flips >= whipsaw_threshold

        # ── Time in current regime ──────────────────────
        time_in_current = (now - self._current_regime_since) / 60

        # ── Regime stability score ──────────────────────
        # Combines: time in regime (longer = more stable), low whipsaw, low cooldown
        time_factor = min(1.0, time_in_current / 120)  # maxes out at 2 hours
        stability = time_factor * (1.0 - whipsaw_score * 0.7)
        if is_cooldown:
            # Degrade stability proportional to remaining cooldown
            cooldown_pct = cooldown_remaining / self._flip_params["cooldown_minutes"]
            stability *= (1.0 - cooldown_pct * 0.5)
        stability = max(0.0, min(1.0, stability))

        # ── Adjustment multiplier ───────────────────────
        # Start at 1.0, reduce based on cooldown + whipsaw
        mult = 1.0
        if is_cooldown:
            cooldown_mult = self._flip_params["cooldown_position_mult"]
            # Fade: harder reduction early, easing as cooldown expires
            fade = cooldown_remaining / self._flip_params["cooldown_minutes"]
            mult *= (cooldown_mult + (1.0 - cooldown_mult) * (1.0 - fade))
        if is_whipsaw:
            mult *= self._flip_params["whipsaw_position_mult"]
        elif whipsaw_score > 0.3:
            # Partial whipsaw — proportional reduction
            mult *= (1.0 - whipsaw_score * 0.4)
        mult = max(0.10, min(1.0, mult))

        # ── Block entries? ───────────────────────────────
        block_threshold = self._flip_params["cooldown_block_severity"]
        should_block = (
            is_cooldown
            and self._last_flip_severity >= block_threshold
            and cooldown_remaining > self._flip_params["cooldown_minutes"] * 0.5
        )
        # Also block during active whipsaw
        if is_whipsaw and whipsaw_score >= 0.8:
            should_block = True

        return {
            "is_cooldown": is_cooldown,
            "cooldown_remaining_min": round(cooldown_remaining, 1),
            "flip_severity": round(self._last_flip_severity, 2),
            "whipsaw_score": round(whipsaw_score, 2),
            "flips_in_window": flips,
            "is_whipsaw": is_whipsaw,
            "last_flip_from": self._last_flip_from,
            "last_flip_to": self._last_flip_to,
            "time_in_current_min": round(time_in_current, 1),
            "regime_stability": round(stability, 3),
            "adjustment_multiplier": round(mult, 3),
            "should_block_entries": should_block,
        }

    def get_regime_summary(self) -> str:
        """One-line summary for logging (regime + flip state)."""
        if not self._regime_history:
            return "No regime data"
        _, regime, conf = self._regime_history[-1]
        fs = self.get_flip_state()
        parts = [f"{regime} (conf={conf:.2f})"]
        if fs["is_cooldown"]:
            parts.append(f"COOLDOWN {fs['cooldown_remaining_min']:.0f}m")
        if fs["is_whipsaw"]:
            parts.append(f"WHIPSAW ({fs['flips_in_window']} flips)")
        parts.append(f"stability={fs['regime_stability']:.2f}")
        parts.append(f"mult={fs['adjustment_multiplier']:.2f}")
        return " | ".join(parts)

    # ── Bot-Specific Adjustments ─────────────────────────

    @staticmethod
    def get_adjustments(
        regime: str,
        bot_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return parameter adjustment multipliers for a given regime.

        Parameters
        ----------
        regime : str
            One of ``TRENDING_UP``, ``TRENDING_DOWN``, ``RANGING``,
            ``HIGH_VOLATILITY``.
        bot_name : str, optional
            If given, return only that bot's adjustments.  Otherwise return
            the full dict keyed by bot name.

        Returns
        -------
        dict
        """
        regime_adj = _ADJUSTMENTS.get(regime, _ADJUSTMENTS[RANGING])
        if bot_name:
            return dict(regime_adj.get(bot_name, {
                "position_size": 1.0,
                "stop_loss_width": 1.0,
                "take_profit_width": 1.0,
                "trade_frequency": 1.0,
            }))
        return {k: dict(v) for k, v in regime_adj.items()}

    # ── Convenience ──────────────────────────────────────

    @property
    def last_regime(self) -> Optional[Dict[str, Any]]:
        """Most recent detection result (cached)."""
        return self._last_regime

    def _fallback(self, reason: str) -> Dict[str, Any]:
        """Return a neutral regime when detection is not possible."""
        result = {
            "regime": RANGING,
            "confidence": 0.0,
            "trend_strength": 0.0,
            "volatility_ratio": 1.0,
            "signals": {},
            "regime_scores": {r: 0.25 for r in ALL_REGIMES},
            "suggested_adjustments": self.get_adjustments(RANGING, self.bot_name),
            "fallback_reason": reason,
        }
        self._last_regime = result
        return result

    def __repr__(self) -> str:
        regime = self._last_regime["regime"] if self._last_regime else "N/A"
        conf = self._last_regime["confidence"] if self._last_regime else 0
        return f"<RegimeDetector regime={regime} conf={conf:.2f}>"
