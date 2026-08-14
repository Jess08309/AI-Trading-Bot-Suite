"""
Feature Engineering for ML model training and prediction.
KEY CHANGE: Trains on MARKET DATA (price history) to predict price DIRECTION,
instead of training on past trade outcomes (which was circular).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

try:
    from utils.technical_indicators import compute_all_indicators
except ImportError:
    from .technical_indicators import compute_all_indicators


# Canonical feature order - must be consistent between training and prediction
# 7 curated indicators, trimmed from 15 based on live feature-importance data
# (trix_15/atr_14/vol_ratio/rsi_14/mean_reversion consistently rank highest;
# macd_histogram and bb_position kept for non-redundant trend/range context).
# Dropped for near-zero importance: stoch_k, cci_20, roc_10, momentum_10,
# williams_r, ultimate_osc, cmo_14, trend_strength.
FEATURE_NAMES = [
    "rsi_14",           # RSI — momentum exhaustion / overbought-oversold
    "macd_histogram",   # MACD — momentum + trend shifts
    "trix_15",          # TRIX — triple-smoothed trend momentum
    "atr_14",           # ATR — volatility (normalized by price)
    "bb_position",      # Bollinger Band position — range context
    "mean_reversion",   # Z-score — overextended move detection
    "vol_ratio",        # Volatility ratio — regime detection
]


class FeatureEngine:
    """Builds feature vectors from price history for ML training and prediction."""

    def __init__(self, lookback: int = 30, prediction_horizon: int = 5):
        """
        Args:
            lookback: Number of price points needed to compute features.
            prediction_horizon: How many periods ahead to predict direction.
        """
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon

    def build_features_from_prices(self, prices: List[float]) -> Optional[Dict[str, float]]:
        """Build feature vector from a price history list.
        Returns dict of named features, or None if insufficient data.
        """
        if len(prices) < max(self.lookback, 15):
            return None
        return compute_all_indicators(prices)

    def features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in canonical order."""
        return np.array([features.get(name, 0.0) for name in FEATURE_NAMES])

    def build_training_data_from_csv(self, price_csv_path: str,
                                     min_samples: int = 100,
                                     days: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build training data from price history CSV.

        THIS IS THE KEY IMPROVEMENT:
        - Loads ALL price snapshots from CSV
        - For each point, computes 30 technical indicators
        - Labels: 1 if price goes UP over next N periods, 0 if DOWN
        - This trains the model on MARKET PATTERNS, not our own trade outcomes

        Returns:
            (X, y) arrays or (None, None) if insufficient data
        """
        try:
            df = pd.read_csv(price_csv_path, names=["timestamp", "pair", "price"])
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna()

            # Apply rolling window if specified
            if days is not None:
                cutoff = datetime.now() - timedelta(days=days)
                df = df[df["timestamp"] >= cutoff]

            if len(df) < min_samples:
                return None, None

            all_features = []
            all_labels = []

            # Process each coin separately
            for pair in df["pair"].unique():
                pair_df = df[df["pair"] == pair].sort_values("timestamp")
                prices = pair_df["price"].values.tolist()

                if len(prices) < self.lookback + self.prediction_horizon:
                    continue

                # Slide window through price history
                for i in range(self.lookback, len(prices) - self.prediction_horizon):
                    window = prices[i - self.lookback:i + 1]
                    features = compute_all_indicators(window)

                    if not features:
                        continue

                    # Label: price direction over next N periods
                    current_price = prices[i]
                    future_price = prices[i + self.prediction_horizon]

                    if current_price <= 0:
                        continue

                    # 1 = price goes up, 0 = price goes down
                    label = 1 if future_price > current_price else 0

                    feature_array = self.features_to_array(features)
                    all_features.append(feature_array)
                    all_labels.append(label)

            if len(all_features) < min_samples:
                return None, None

            X = np.array(all_features)
            y = np.array(all_labels)

            # Replace NaN/Inf with 0
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            return X, y

        except Exception as e:
            print(f"Feature engine error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def build_training_data_adaptive(self, price_csv_path: str,
                                     min_samples: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
        """Adaptive rolling window: try 3, 7, 14, 30 days, then all data.
        Returns (X, y, days_used) or (None, None, None).
        """
        for days in [3, 7, 14, 30]:
            X, y = self.build_training_data_from_csv(price_csv_path, min_samples, days=days)
            if X is not None:
                return X, y, days
        # Fall back to all data
        X, y = self.build_training_data_from_csv(price_csv_path, min_samples, days=None)
        return X, y, None
