"""
Market Direction Predictor.
Uses GradientBoosting trained on market data (price history + technical indicators).
Replaces the old RandomForest that was trained on its own trade outcomes.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import glob
import shutil

try:
    from utils.feature_engine import FeatureEngine, FEATURE_NAMES
    from utils.technical_indicators import compute_all_indicators
except ImportError:
    from .feature_engine import FeatureEngine, FEATURE_NAMES
    from .technical_indicators import compute_all_indicators


class MarketPredictor:
    """ML model that predicts market direction from technical indicators."""

    def __init__(self, model_path: str = "data/models/market_model.joblib",
                 lookback: int = 30, prediction_horizon: int = 5,
                 max_versions: int = 5, min_accuracy: float = 0.48):
        self.model_path = model_path
        self.model = None
        self.feature_engine = FeatureEngine(lookback=lookback,
                                            prediction_horizon=prediction_horizon)
        self.last_train_time = None
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.feature_importances = {}
        self.max_versions = max_versions
        self.min_accuracy = min_accuracy


    def rollback_model(self, log_fn=None) -> bool:
        """Roll back to previous model version if current one is worse."""
        if log_fn is None:
            log_fn = print

        model_dir = os.path.dirname(self.model_path)
        base_name = os.path.splitext(os.path.basename(self.model_path))[0]
        pattern = os.path.join(model_dir, f"{base_name}_*.joblib")
        versions = sorted(glob.glob(pattern))

        if len(versions) < 2:
            log_fn("[ML] No previous version to roll back to")
            return False

        # Current = latest, previous = second to last
        prev_path = versions[-2]
        try:
            self.model = load(prev_path)
            dump(self.model, self.model_path)
            log_fn(f"[ML] ROLLED BACK to {os.path.basename(prev_path)}")
            return True
        except Exception as e:
            log_fn(f"[ML] Rollback failed: {e}")
            return False

    def load_model(self) -> bool:
        """Load saved model. Returns True if successful."""
        try:
            if os.path.exists(self.model_path):
                self.model = load(self.model_path)
                return True
        except Exception:
            pass
        return False

    def save_model(self, version_tag: str = ""):
        """Save model with versioning. Keeps last N versions for rollback."""
        if self.model is None:
            return

        model_dir = os.path.dirname(self.model_path)
        os.makedirs(model_dir, exist_ok=True)

        # Save current version with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.model_path))[0]
        tag = f"_{version_tag}" if version_tag else ""
        versioned_path = os.path.join(
            model_dir, f"{base_name}_{timestamp}{tag}.joblib")
        dump(self.model, versioned_path)

        # Also save as the "current" model (for load_model compatibility)
        dump(self.model, self.model_path)

        # Save metadata
        meta_path = os.path.join(model_dir, f"{base_name}_meta.json")
        import json
        meta = {
            "timestamp": timestamp,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "top_features": dict(list(self.feature_importances.items())[:5]),
            "version_path": versioned_path,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Prune old versions (keep last N)
        pattern = os.path.join(model_dir, f"{base_name}_*.joblib")
        versions = sorted(glob.glob(pattern))
        while len(versions) > self.max_versions:
            old = versions.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass

    def train(self, price_csv_path: str, min_samples: int = 100,
              log_fn=None) -> bool:
        """Train model on price history data.

        KEY DIFFERENCE from old system:
        - Old: Trained on own trade outcomes (circular, learned noise)
        - New: Trained on price data to predict FUTURE DIRECTION (learns market patterns)

        Uses TimeSeriesSplit (no look-ahead bias).
        Uses GradientBoosting (better than RandomForest for sequential data).
        """
        if log_fn is None:
            log_fn = print

        # Build training data with adaptive window
        X, y, days_used = self.feature_engine.build_training_data_adaptive(
            price_csv_path, min_samples=min_samples
        )

        if X is None or y is None:
            log_fn(f"[ML] Insufficient data for training")
            return False

        window_str = f"{days_used}-day window" if days_used else "all available data"
        log_fn(f"[ML] Training on {len(X)} samples ({window_str}), {len(FEATURE_NAMES)} features")

        # Class balance check
        pos_ratio = np.mean(y)
        log_fn(f"[ML] Class balance: {pos_ratio:.1%} UP / {1-pos_ratio:.1%} DOWN")

        # Time-series aware cross-validation (no look-ahead bias)
        n_splits = min(5, max(2, len(X) // 100))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        best_score = 0
        best_model = None
        cv_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)

            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            log_fn("[ML] Training failed - no valid folds")
            return False

        prev_model = self.model  # Save for rollback if new model is worse
        self.model = best_model

        # Final evaluation
        self.train_accuracy = float(np.mean(cv_scores))
        self.test_accuracy = best_score

        # Feature importances
        importances = self.model.feature_importances_
        self.feature_importances = {
            name: float(imp) for name, imp in
            sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
        }

        log_fn(f"[ML] MODEL TRAINED:")
        log_fn(f"[ML]   CV Accuracy: {self.train_accuracy:.1%} (avg {n_splits} folds)")
        log_fn(f"[ML]   Best Fold:   {self.test_accuracy:.1%}")

        # Top 5 features
        top5 = list(self.feature_importances.items())[:5]
        log_fn(f"[ML]   Top features: {', '.join(f'{n}({v:.3f})' for n, v in top5)}")

        # Check if new model meets minimum accuracy
        if self.test_accuracy < self.min_accuracy:
            log_fn(f"[ML] WARNING: Model accuracy {self.test_accuracy:.1%} below "
                   f"minimum {self.min_accuracy:.1%} -- keeping old model")
            self.model = prev_model  # Restore previous model
            return False

        self.save_model(version_tag=f"acc{self.test_accuracy:.0%}")
        self.last_train_time = datetime.now()

        return True

    def predict(self, prices: List[float]) -> Dict[str, float]:
        """Predict market direction from price history.

        Returns dict with:
            - confidence: probability price goes UP (0.0 to 1.0)
            - direction: 1.0 (bullish) or -1.0 (bearish)
            - strength: how far from 50% (0.0 to 0.5)
            - volatility: current volatility from features
            - rsi: current RSI value
            - trend: current trend strength
        """
        result = {
            "confidence": 0.5,
            "direction": 0.0,
            "strength": 0.0,
            "volatility": 0.0,
            "rsi": 50.0,
            "trend": 0.0,
        }

        if self.model is None or len(prices) < 15:
            return result

        try:
            features = self.feature_engine.build_features_from_prices(prices)
            if features is None:
                return result

            feature_array = self.feature_engine.features_to_array(features)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Get probability prediction
            proba = self.model.predict_proba(feature_array.reshape(1, -1))[0]
            confidence = float(proba[1])  # P(price goes up)

            result["confidence"] = confidence
            result["direction"] = 1.0 if confidence > 0.5 else -1.0
            result["strength"] = abs(confidence - 0.5)
            result["volatility"] = features.get("volatility", 0.0)
            result["rsi"] = features.get("rsi_14", 50.0)
            result["trend"] = features.get("trend_strength", 0.0)
            result["bb_position"] = features.get("bb_position", 0.5)
            result["macd_crossover"] = features.get("macd_crossover", 0.0)
            result["mean_reversion"] = features.get("mean_reversion", 0.0)

            return result

        except Exception as e:
            return result

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importances from trained model."""
        return self.feature_importances
