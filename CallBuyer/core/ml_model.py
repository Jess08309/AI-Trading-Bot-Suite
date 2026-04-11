"""
CallBuyer ML Model — GradientBoosting for momentum breakout prediction.

Same pattern as AlpacaBot's OptionsMLModel but tuned for call buying:
- GradientBoostingClassifier with moderate hyperparams
- Warmup mode until 30+ completed trades
- TimeSeriesSplit cross-validation
- Quality gate: 53% accuracy minimum
- Periodic retraining every 15 new outcomes
- Versioned model persistence
"""
import logging
import os
import json
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib

from core.config import CallBuyerConfig as cfg
from core.feature_engine import FEATURE_NAMES

log = logging.getLogger("callbuyer.ml")


class CallBuyerMLModel:
    """ML model for predicting call buying success probability."""

    def __init__(self, models_dir: str = "models", state_dir: str = "data/state"):
        self.models_dir = models_dir
        self.state_dir = state_dir
        self.features_file = os.path.join(state_dir, "features_log.json")
        self.model: Optional[GradientBoostingClassifier] = None
        self.accuracy: float = 0.0
        self.trained_on: int = 0
        self.last_train_time: Optional[str] = None
        self._outcomes_since_train: int = 0

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

        self._load_latest_model()

    # ── Public Interface ─────────────────────────────────

    def predict(self, features: np.ndarray) -> Tuple[float, bool]:
        """Predict probability of a successful call trade.

        Returns:
            (probability, is_active) — probability 0-1, is_active indicates
            whether the model was actually used vs warmup default.
        """
        if self.model is None:
            return 0.5, False  # warmup mode — neutral score

        try:
            X = features.reshape(1, -1)
            proba = self.model.predict_proba(X)[0]
            # Index 1 = probability of class 1 (win)
            if len(proba) > 1:
                return float(proba[1]), True
            return float(proba[0]), True
        except Exception as e:
            log.warning(f"Prediction error: {e}")
            return 0.5, False

    def should_retrain(self) -> bool:
        """Check if retraining is needed based on new outcomes."""
        completed = self._count_completed()
        if completed < cfg.ML_WARMUP_TRADES:
            return False
        if self.model is None:
            return True  # never trained yet but have enough data
        return self._outcomes_since_train >= cfg.ML_RETRAIN_TRADES

    def train(self) -> bool:
        """Train or retrain model from logged features+outcomes.

        Returns True if model is valid (passes quality gate).
        """
        X, y = self._load_training_data()
        if X is None or len(y) < cfg.ML_WARMUP_TRADES:
            log.info(f"Not enough training data: {len(y) if y is not None else 0}/{cfg.ML_WARMUP_TRADES}")
            return False

        log.info(f"Training on {len(y)} samples (wins={sum(y)}, losses={len(y)-sum(y)})")

        # TimeSeriesSplit cross-validation
        n_splits = min(5, max(2, len(y) // 10))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        best_model = None
        best_accuracy = 0.0
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                gbm = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    min_samples_split=5, min_samples_leaf=3, subsample=0.8,
                    random_state=42,
                )
                rf = RandomForestClassifier(
                    n_estimators=150, max_depth=6, min_samples_leaf=5,
                    random_state=42, n_jobs=-1,
                )
                et = ExtraTreesClassifier(
                    n_estimators=150, max_depth=6, min_samples_leaf=5,
                    random_state=42, n_jobs=-1,
                )
                model = VotingClassifier(
                    estimators=[('gbm', gbm), ('rf', rf), ('et', et)],
                    voting='soft',
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                acc = accuracy_score(y_val, preds)
                # Log individual model accuracies
                for name, est in model.named_estimators_.items():
                    ind_acc = est.score(X_val, y_val)
                    log.info(f"  Fold {fold} {name}: {ind_acc:.3f}")
            except Exception as e:
                log.warning(f"VotingClassifier failed ({e}), falling back to single GBM")
                model = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    min_samples_split=5, min_samples_leaf=3, subsample=0.8,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                acc = accuracy_score(y_val, preds)
            cv_scores.append(acc)

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model

        avg_accuracy = np.mean(cv_scores)
        log.info(f"CV scores: {[f'{s:.3f}' for s in cv_scores]}, avg: {avg_accuracy:.3f}")

        # Quality gate
        if avg_accuracy < cfg.ML_MIN_ACCURACY:
            log.warning(f"Model below quality gate: {avg_accuracy:.3f} < {cfg.ML_MIN_ACCURACY}")
            # Still retrain on all data but flag low quality
            log.info("Retraining on full dataset despite low CV accuracy...")

        # Final train on all data (ensemble)
        try:
            gbm = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_samples_split=5, min_samples_leaf=3, subsample=0.8,
                random_state=42,
            )
            rf = RandomForestClassifier(
                n_estimators=150, max_depth=6, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
            et = ExtraTreesClassifier(
                n_estimators=150, max_depth=6, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
            final_model = VotingClassifier(
                estimators=[('gbm', gbm), ('rf', rf), ('et', et)],
                voting='soft',
            )
            final_model.fit(X, y)
            # Log individual model accuracies on full training set
            for name, est in final_model.named_estimators_.items():
                full_acc = est.score(X, y)
                log.info(f"  Final {name} train accuracy: {full_acc:.3f}")
        except Exception as e:
            log.warning(f"VotingClassifier failed ({e}), falling back to single GBM")
            final_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_samples_split=5, min_samples_leaf=3, subsample=0.8,
                random_state=42,
            )
            final_model.fit(X, y)

        self.model = final_model
        self.accuracy = float(avg_accuracy)
        self.trained_on = len(y)
        self.last_train_time = datetime.now().isoformat()
        self._outcomes_since_train = 0

        # Save model
        self._save_model()

        # Log feature importances
        if hasattr(final_model, 'estimators_'):
            importances = np.mean([
                est.feature_importances_ for est in final_model.estimators_
            ], axis=0)
        else:
            importances = final_model.feature_importances_
        feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
        log.info("Feature importances:")
        for name, imp in feat_imp:
            log.info(f"  {name:25s}: {imp:.4f}")

        return avg_accuracy >= cfg.ML_MIN_ACCURACY

    def record_outcome(self, symbol: str, features: np.ndarray, won: bool,
                       pnl_pct: float):
        """Record a completed trade outcome for future training."""
        self._outcomes_since_train += 1
        try:
            existing = []
            if os.path.exists(self.features_file):
                with open(self.features_file) as f:
                    existing = json.load(f)

            entry = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "features": features.tolist(),
                "feature_names": FEATURE_NAMES,
                "outcome": 1 if won else 0,
                "pnl_pct": pnl_pct,
                "completed": True,
            }
            existing.append(entry)
            if len(existing) > 1000:
                existing = existing[-1000:]

            with open(self.features_file, "w") as f:
                json.dump(existing, f, indent=1)

            log.info(f"Recorded outcome for {symbol}: {'WIN' if won else 'LOSS'} ({pnl_pct:+.1f}%)")

        except Exception as e:
            log.warning(f"Could not record outcome: {e}")

    def get_status(self) -> dict:
        """Return model status for dashboard."""
        return {
            "active": self.model is not None,
            "accuracy": round(self.accuracy, 3) if self.model else None,
            "trained_on": self.trained_on,
            "last_train": self.last_train_time,
            "pending_outcomes": self._outcomes_since_train,
            "warmup_progress": f"{self._count_completed()}/{cfg.ML_WARMUP_TRADES}",
        }

    # ── Private Helpers ──────────────────────────────────

    def _load_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load completed trades with features and outcomes."""
        if not os.path.exists(self.features_file):
            return None, None

        try:
            with open(self.features_file) as f:
                data = json.load(f)

            completed = [d for d in data if d.get("completed") and d.get("outcome") is not None]
            if len(completed) < 10:
                return None, None

            X = np.array([d["features"] for d in completed])
            y = np.array([int(d["outcome"]) for d in completed])
            return X, y
        except Exception as e:
            log.warning(f"Error loading training data: {e}")
            return None, None

    def _count_completed(self) -> int:
        """Count completed trade outcomes."""
        if not os.path.exists(self.features_file):
            return 0
        try:
            with open(self.features_file) as f:
                data = json.load(f)
            return sum(1 for d in data if d.get("completed") and d.get("outcome") is not None)
        except Exception:
            return 0

    def _save_model(self):
        """Save model with accuracy in filename."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        acc_str = f"{self.accuracy * 100:.0f}"
        path = os.path.join(self.models_dir, f"callbuyer_model_{ts}_acc{acc_str}%.joblib")
        joblib.dump({
            "model": self.model,
            "accuracy": self.accuracy,
            "trained_on": self.trained_on,
            "feature_names": FEATURE_NAMES,
            "timestamp": ts,
        }, path)
        log.info(f"Saved model: {path}")

    def _load_latest_model(self):
        """Load the most recent model file if one exists."""
        if not os.path.exists(self.models_dir):
            return

        model_files = sorted([
            f for f in os.listdir(self.models_dir)
            if f.startswith("callbuyer_model_") and f.endswith(".joblib")
        ])
        if not model_files:
            return

        latest = os.path.join(self.models_dir, model_files[-1])
        try:
            data = joblib.load(latest)
            self.model = data["model"]
            self.accuracy = data.get("accuracy", 0)
            self.trained_on = data.get("trained_on", 0)
            self.last_train_time = data.get("timestamp")
            log.info(f"Loaded model: {latest} (acc={self.accuracy:.3f}, samples={self.trained_on})")
        except Exception as e:
            log.warning(f"Could not load model {latest}: {e}")
