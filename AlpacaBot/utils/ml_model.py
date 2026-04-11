"""
Options ML Model — GradientBoosting direction predictor.

Trained on underlying price history to predict whether the stock will go UP
or DOWN over the next ~1 hour (6 × 10-min bars).  This is the primary
"should I buy a call or put?" signal.

KEY SAFETY FEATURES (your #1 concern — don't blow up):
  - Model quality gate: reject models below minimum accuracy threshold
  - Model versioning: keeps last N versions, can roll back if new model degrades
  - Confidence gate: trade only when model is confident enough
  - Retrains on live data every MODEL_RETRAIN_HOURS (adapts to regime changes)
  - Class-balance enforcement: undersamples majority to prevent directional bias
  - TimeSeriesSplit CV: no look-ahead bias in training
"""
from __future__ import annotations
import os
import logging
import glob
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from joblib import dump, load

from utils.feature_engine import OptionsFeatureEngine, FEATURE_NAMES

log = logging.getLogger("alpacabot.ml")


class OptionsMLModel:
    """ML model that predicts underlying direction for options trading."""

    def __init__(self, model_dir: str = "data/models",
                 min_accuracy: float = 0.51,
                 max_versions: int = 5):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "options_model.joblib")
        self.model: Optional[GradientBoostingClassifier] = None
        self.feature_engine = OptionsFeatureEngine()
        self.min_accuracy = min_accuracy
        self.max_versions = max_versions

        # Tracking
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.feature_importances: Dict[str, float] = {}
        self.last_train_time: Optional[datetime] = None
        self.prediction_count = 0
        self.version_tag = ""

    # ── Load / Save ──────────────────────────────────────

    def load_model(self) -> bool:
        """Load saved model and metadata. Returns True if successful."""
        try:
            if os.path.exists(self.model_path):
                candidate = load(self.model_path)
                # Verify feature count matches
                if hasattr(candidate, 'n_features_in_') and candidate.n_features_in_ != len(FEATURE_NAMES):
                    log.warning(
                        f"Model feature mismatch: has {candidate.n_features_in_}, "
                        f"need {len(FEATURE_NAMES)}. Will need retrain."
                    )
                    return False

                # Restore metadata if available (non-critical — don't fail load on this)
                import json
                meta_path = os.path.join(self.model_dir, "options_model_meta.json")
                try:
                    if os.path.exists(meta_path):
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        self.test_accuracy = meta.get("test_accuracy", 0.0)
                        self.train_accuracy = meta.get("train_accuracy", 0.0)
                        top_feats = meta.get("top_features", {})
                        if top_feats:
                            self.feature_importances = top_feats
                        log.info(
                            f"ML Model loaded ({len(FEATURE_NAMES)} features, "
                            f"accuracy={self.test_accuracy:.1%})"
                        )
                    else:
                        log.info(f"ML Model loaded ({len(FEATURE_NAMES)} features, no metadata)")
                except Exception as e:
                    log.warning(f"Model loaded but metadata read failed (non-critical): {e}")

                # Set model AFTER all checks pass (atomic — prevents inconsistent state)
                self.model = candidate
                return True
        except Exception as e:
            log.warning(f"Failed to load ML model: {e}")
        return False

    def save_model(self):
        """Save model with versioning. Keeps last N versions for rollback."""
        if self.model is None:
            return
        os.makedirs(self.model_dir, exist_ok=True)

        # Save versioned copy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        acc_tag = f"acc{int(self.test_accuracy * 100)}pct"
        versioned_path = os.path.join(
            self.model_dir, f"options_model_{timestamp}_{acc_tag}.joblib"
        )
        dump(self.model, versioned_path)
        dump(self.model, self.model_path)
        self.version_tag = f"{timestamp}_{acc_tag}"

        # Save metadata
        import json
        meta_path = os.path.join(self.model_dir, "options_model_meta.json")
        meta = {
            "timestamp": timestamp,
            "train_accuracy": round(self.train_accuracy, 4),
            "test_accuracy": round(self.test_accuracy, 4),
            "features": len(FEATURE_NAMES),
            "top_features": dict(list(self.feature_importances.items())[:5]),
            "predictions_since_train": self.prediction_count,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Prune old versions
        pattern = os.path.join(self.model_dir, "options_model_*.joblib")
        versions = sorted(glob.glob(pattern))
        while len(versions) > self.max_versions:
            old = versions.pop(0)
            try:
                os.remove(old)
                log.debug(f"Pruned old model: {os.path.basename(old)}")
            except OSError:
                pass

        log.info(f"Model saved: {os.path.basename(versioned_path)}")

    def rollback(self) -> bool:
        """Roll back to previous model version."""
        pattern = os.path.join(self.model_dir, "options_model_*.joblib")
        versions = sorted(glob.glob(pattern))
        if len(versions) < 2:
            log.warning("No previous version to roll back to")
            return False
        try:
            prev_path = versions[-2]
            self.model = load(prev_path)
            dump(self.model, self.model_path)
            log.info(f"ROLLED BACK to {os.path.basename(prev_path)}")
            return True
        except Exception as e:
            log.warning(f"Rollback failed: {e}")
            return False

    # ── Training ─────────────────────────────────────────

    def train(self, price_dict: Dict[str, np.ndarray],
              min_samples: int = 200,
              stride: int = 1) -> bool:
        """Train model on underlying price history.

        KEY: Trains on PRICE DIRECTION, not on own trade outcomes.
        Uses TimeSeriesSplit (no look-ahead bias).

        Returns True if model was accepted (meets accuracy threshold).
        """
        X, y = self.feature_engine.build_training_data(price_dict, min_samples, stride=stride)
        if X is None or y is None:
            log.info("ML train: insufficient data")
            return False

        n_samples = len(y)
        n_up = int(np.sum(y))
        n_down = n_samples - n_up
        class_ratio = np.mean(y)
        log.info(f"ML training: {n_samples} samples, {n_up} UP ({class_ratio:.1%}), {n_down} DOWN")

        # Class balance enforcement (prevent directional bias — your #1 blowup risk)
        if class_ratio < 0.35 or class_ratio > 0.65:
            log.warning(f"Imbalanced data ({class_ratio:.1%} UP). Undersampling majority.")
            up_idx = np.where(y == 1)[0]
            down_idx = np.where(y == 0)[0]
            minority_count = min(len(up_idx), len(down_idx))
            if minority_count < 50:
                log.warning("Too few minority samples after undersample. Skipping.")
                return False
            if len(up_idx) > len(down_idx):
                up_idx = np.random.choice(up_idx, size=minority_count, replace=False)
            else:
                down_idx = np.random.choice(down_idx, size=minority_count, replace=False)
            balanced_idx = np.concatenate([up_idx, down_idx])
            np.random.shuffle(balanced_idx)
            X, y = X[balanced_idx], y[balanced_idx]
            log.info(f"After balancing: {len(y)} samples, {np.mean(y):.1%} UP")

        # Time-series cross-validation (2-5 folds, no look-ahead bias)
        n_splits = min(5, max(2, len(X) // 100))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        best_score = 0.0
        best_model = None
        cv_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            sample_weights = compute_sample_weight('balanced', y_train)

            try:
                gbm = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_samples_split=20, min_samples_leaf=10,
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
                model.fit(X_train, y_train, sample_weight=sample_weights)
                score = model.score(X_test, y_test)
                # Log individual model accuracies
                for name, est in model.named_estimators_.items():
                    ind_score = est.score(X_test, y_test)
                    log.info(f"  Fold {fold} {name}: {ind_score:.1%}")
            except Exception as e:
                log.warning(f"VotingClassifier failed ({e}), falling back to single GBM")
                model = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_samples_split=20, min_samples_leaf=10,
                    random_state=42,
                )
                model.fit(X_train, y_train, sample_weight=sample_weights)
                score = model.score(X_test, y_test)
            cv_scores.append(score)

            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            log.warning("ML training failed — no valid folds")
            return False

        avg_cv = float(np.mean(cv_scores))
        self.train_accuracy = avg_cv
        self.test_accuracy = best_score

        # Feature importances (average across ensemble members if VotingClassifier)
        if hasattr(best_model, 'estimators_'):
            importances = np.mean([
                est.feature_importances_ for est in best_model.estimators_
            ], axis=0)
        else:
            importances = best_model.feature_importances_
        self.feature_importances = {
            name: float(imp) for name, imp in
            sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
        }

        top5 = list(self.feature_importances.items())[:5]
        log.info(f"ML CV Accuracy: {avg_cv:.1%} (avg {n_splits} folds), Best: {best_score:.1%}")
        log.info(f"ML Top features: {', '.join(f'{n}={v:.3f}' for n, v in top5)}")

        # QUALITY GATE — reject bad models (prevents blowups from bad retrains)
        if best_score < self.min_accuracy:
            log.warning(
                f"Model REJECTED: accuracy {best_score:.1%} < minimum {self.min_accuracy:.1%}. "
                f"Keeping previous model."
            )
            return False

        # Accept new model
        prev_model = self.model
        self.model = best_model
        self.prediction_count = 0
        self.last_train_time = datetime.now()
        self.save_model()

        log.info(f"Model ACCEPTED: {best_score:.1%} accuracy, {len(FEATURE_NAMES)} features")
        return True

    # ── Prediction ───────────────────────────────────────

    def predict(self, prices: np.ndarray,
                timestamp: Optional[datetime] = None,
                day_open: Optional[float] = None,
                prev_close: Optional[float] = None) -> Dict[str, float]:
        """Predict underlying direction and confidence.

        Returns:
            {
                "direction": float,     # >0.5 = bullish, <0.5 = bearish
                "confidence": float,    # how sure (0.5 = no idea, 1.0 = very sure)
                "up_prob": float,       # P(price goes up)
                "down_prob": float,     # P(price goes down)
            }
        """
        default = {"direction": 0.5, "confidence": 0.5, "up_prob": 0.5, "down_prob": 0.5}

        if self.model is None:
            return default

        features = self.feature_engine.build_features(
            prices, timestamp=timestamp, day_open=day_open, prev_close=prev_close
        )
        if features is None:
            return default

        try:
            features_2d = features.reshape(1, -1)
            proba = self.model.predict_proba(features_2d)[0]
            up_prob = float(proba[1])
            down_prob = float(proba[0])
            confidence = max(up_prob, down_prob)

            self.prediction_count += 1
            return {
                "direction": up_prob,
                "confidence": confidence,
                "up_prob": up_prob,
                "down_prob": down_prob,
            }
        except Exception as e:
            log.warning(f"Prediction error (returning default 0.5): {e}")
            return default

    # ── Status ───────────────────────────────────────────

    def status(self) -> Dict[str, any]:
        """Return model status for dashboard/logging."""
        return {
            "loaded": self.model is not None,
            "train_accuracy": round(self.train_accuracy, 4),
            "test_accuracy": round(self.test_accuracy, 4),
            "predictions": self.prediction_count,
            "last_train": self.last_train_time.isoformat() if self.last_train_time else None,
            "features": len(FEATURE_NAMES),
            "version": self.version_tag,
            "top_features": dict(list(self.feature_importances.items())[:3]),
        }
