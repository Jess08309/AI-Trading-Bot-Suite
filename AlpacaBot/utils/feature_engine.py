"""
Feature Engine for Options ML Model.
Builds feature vectors from underlying price history + options context.

KEY DESIGN PRINCIPLES (learned from crypto bot):
  - Train on MARKET DATA (price direction), never on own trade outcomes
  - Use canonical feature order for consistency between training & prediction
  - NaN/Inf protection on every output
  - Adaptive training windows (try recent data first, fall back to longer)
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from core.indicators import (
    rsi, macd, bollinger_bands, stochastic, atr, cci, roc,
    williams_r, volatility_ratio, mean_reversion_zscore, trend_strength,
    compute_all_indicators,
)

# ── Canonical Feature Names ──────────────────────────
# 14 price indicators (same as current signal scoring uses) + 6 options-context features = 20 total
FEATURE_NAMES = [
    # --- 14 underlying price indicators ---
    "rsi_14",               # RSI — momentum exhaustion
    "macd_histogram",       # MACD histogram — momentum shifts
    "stoch_k",              # Stochastic %K — timing entries
    "bb_position",          # Bollinger %B — range context
    "atr_normalized",       # ATR / price — volatility magnitude
    "cci_20",               # CCI — deviation from mean
    "roc_10",               # Rate of Change — momentum acceleration
    "williams_r",           # Williams %R — exhaustion points
    "vol_ratio",            # Volatility ratio (5/20) — regime change
    "zscore",               # Z-score — mean reversion signal
    "trend_strength",       # ADX-like — trending vs ranging
    "price_change_1",       # 1-bar momentum
    "price_change_5",       # 5-bar momentum
    "price_change_20",      # 20-bar longer-term trend
    # --- 6 options-context features ---
    "hour_sin",             # Time-of-day (sinusoidal encoding)
    "hour_cos",             # Time-of-day (cosine pair)
    "day_of_week",          # Day 0-4 normalized to 0-1
    "intraday_range",       # Today's high-low range / price
    "gap_open",             # Overnight gap (open vs prev close)
    "vol_regime",           # Volatility regime: low=0, normal=0.5, high=1
]


class OptionsFeatureEngine:
    """Builds feature vectors for options ML model."""

    def __init__(self, lookback: int = 50, prediction_horizon: int = 6):
        """
        Args:
            lookback: Bars of price history needed for indicators.
            prediction_horizon: How many bars ahead to predict direction.
                                6 bars × 10 min = 1 hour (options scalp horizon).
        """
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon

    def build_features(self, prices: np.ndarray,
                       timestamp: Optional[datetime] = None,
                       day_open: Optional[float] = None,
                       prev_close: Optional[float] = None) -> Optional[np.ndarray]:
        """Build a single feature vector from price history + context.

        Args:
            prices: numpy array of recent close prices (at least `lookback` bars)
            timestamp: current time (for time-of-day features)
            day_open: today's opening price (for gap calculation)
            prev_close: previous day's close (for gap calculation)

        Returns:
            numpy array of shape (len(FEATURE_NAMES),) or None if insufficient data.
        """
        if len(prices) < self.lookback:
            return None

        # Compute all 14 underlying indicators (reuse existing infrastructure)
        ind = compute_all_indicators(prices)
        if not ind:
            return None

        # Map indicator dict keys to our canonical names
        features = {
            "rsi_14": ind.get("rsi", 50.0),
            "macd_histogram": ind.get("macd_hist", 0.0),
            "stoch_k": ind.get("stochastic", 50.0),
            "bb_position": ind.get("bb_position", 0.5),
            "atr_normalized": ind.get("atr_normalized", 0.0),
            "cci_20": ind.get("cci", 0.0),
            "roc_10": ind.get("roc", 0.0),
            "williams_r": ind.get("williams_r", -50.0),
            "vol_ratio": ind.get("volatility_ratio", 1.0),
            "zscore": ind.get("zscore", 0.0),
            "trend_strength": ind.get("trend_strength", 0.0),
            "price_change_1": ind.get("price_change_1", 0.0),
            "price_change_5": ind.get("price_change_5", 0.0),
            "price_change_20": ind.get("price_change_20", 0.0),
        }

        # Time-of-day encoding (sinusoidal — captures cyclic nature of market hours)
        if timestamp is None:
            timestamp = datetime.now()
        hour_frac = timestamp.hour + timestamp.minute / 60.0
        features["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
        features["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
        features["day_of_week"] = timestamp.weekday() / 4.0  # Mon=0, Fri=1.0

        # Intraday range (high - low / current price)
        if len(prices) >= 6:
            recent = prices[-6:]  # last ~1 hour of 10-min bars
            range_val = (np.max(recent) - np.min(recent)) / prices[-1] if prices[-1] > 0 else 0
            features["intraday_range"] = range_val
        else:
            features["intraday_range"] = 0.0

        # Gap open
        if day_open is not None and prev_close is not None and prev_close > 0:
            features["gap_open"] = (day_open - prev_close) / prev_close
        else:
            features["gap_open"] = 0.0

        # Volatility regime (based on ATR percentile within lookback)
        atr_n = features["atr_normalized"]
        if atr_n < 0.003:
            features["vol_regime"] = 0.0   # Low vol
        elif atr_n < 0.008:
            features["vol_regime"] = 0.5   # Normal
        else:
            features["vol_regime"] = 1.0   # High vol

        # Convert to array in canonical order
        arr = np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def build_training_data(self, price_dict: Dict[str, np.ndarray],
                            min_samples: int = 200,
                            stride: int = 1) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build labeled training data from price history for multiple symbols.

        Labels: 1 if price goes UP over next `prediction_horizon` bars, 0 if DOWN.
        Flat moves filtered out to avoid training on noise.

        Args:
            price_dict: {symbol: prices_array} — all cached bar data
            min_samples: minimum samples required for training
            stride: sample every Nth bar (higher = faster training, default=1)

        Returns:
            (X, y) arrays or (None, None)
        """
        X_all, y_all = [], []

        for symbol, prices in price_dict.items():
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices, dtype=np.float64)

            if len(prices) < self.lookback + self.prediction_horizon + 10:
                continue

            # Slide a window through the price history
            for i in range(self.lookback, len(prices) - self.prediction_horizon, stride):
                window = prices[:i + 1]
                features = self.build_features(window[-self.lookback - 20:])  # give extra buffer
                if features is None:
                    continue

                # Label: future price direction
                current_price = prices[i]
                future_price = prices[i + self.prediction_horizon]
                if current_price <= 0:
                    continue

                future_return = (future_price - current_price) / current_price

                # Filter out flat moves (< 0.05% move = noise)
                if abs(future_return) < 0.0005:
                    continue

                label = 1 if future_return > 0 else 0
                X_all.append(features)
                y_all.append(label)

        if len(X_all) < min_samples:
            return None, None

        X = np.array(X_all)
        y = np.array(y_all)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y
