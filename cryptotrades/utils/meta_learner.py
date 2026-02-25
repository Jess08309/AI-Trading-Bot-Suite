"""
Enhanced Meta-Learning Ensemble.
Tracks per-model accuracy and dynamically weights predictions.
ML + Sentiment + RL (RL re-enabled with confidence threshold safeguard).
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import os
import numpy as np


class MetaLearner:
    """
    Meta-learner that:
    1. Tracks accuracy of ML model, Sentiment, and RL agent
    2. Weights their predictions by recent performance
    3. Adjusts buy/sell thresholds dynamically

    RL agent re-enabled with safeguard: only included in ensemble
    if |prediction - 0.5| > 0.1 (i.e., RL is expressing a real opinion).
    """

    # Minimum confidence threshold for RL to be included in ensemble
    RL_CONFIDENCE_THRESHOLD = 0.1

    def __init__(self, initial_buy_threshold: float = 0.55,
                 initial_sell_threshold: float = 0.45):
        self.buy_threshold = initial_buy_threshold
        self.sell_threshold = initial_sell_threshold

        # Per-model tracking (ML + Sentiment + RL)
        self.model_history: Dict[str, List[Dict]] = {
            "ml_model": [],
            "sentiment": [],
            "rl_agent": [],
        }

        # Learned weights (RL starts low, earns weight through accuracy)
        self.model_weights: Dict[str, float] = {
            "ml_model": 0.50,
            "sentiment": 0.35,
            "rl_agent": 0.15,
        }

        # Threshold history
        self.threshold_history: List[Dict] = []
        self.max_history = 200

    def record_prediction(self, predictions: Dict[str, float], actual_outcome: float):
        """Record predictions from each model and the actual outcome.

        Args:
            predictions: {"ml_model": 0.7, "sentiment": 0.3}
            actual_outcome: 1.0 for profitable trade, 0.0 for loss
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        for model_name, prediction in predictions.items():
            if model_name in self.model_history:
                predicted_up = prediction > 0.5
                actual_up = actual_outcome > 0.5
                correct = predicted_up == actual_up

                self.model_history[model_name].append({
                    "prediction": prediction,
                    "outcome": actual_outcome,
                    "correct": correct,
                    "timestamp": timestamp,
                })

                if len(self.model_history[model_name]) > self.max_history:
                    self.model_history[model_name] = self.model_history[model_name][-self.max_history:]

        self._update_weights()

    def _update_weights(self):
        """Recalculate model weights based on recent accuracy."""
        accuracies = {}
        for model_name, history in self.model_history.items():
            if len(history) >= 5:
                recent = history[-50:]
                correct = sum(1 for h in recent if h["correct"])
                accuracies[model_name] = correct / len(recent)
            else:
                accuracies[model_name] = 0.5

        total = sum(max(0.1, acc) for acc in accuracies.values())
        if total > 0:
            for model_name in self.model_weights:
                acc = accuracies.get(model_name, 0.5)
                self.model_weights[model_name] = max(0.1, acc) / total

    def get_ensemble_prediction(self, predictions: Dict[str, float]) -> float:
        """Get weighted ensemble prediction from active models.

        Args:
            predictions: {"ml_model": 0.7, "sentiment": 0.6, "rl_agent": 0.52}

        Returns:
            Weighted prediction (0.0 to 1.0)
        """
        weighted_sum = 0.0
        weight_total = 0.0

        for model_name, prediction in predictions.items():
            # RL safeguard: only include if expressing a real opinion
            if model_name == "rl_agent":
                if abs(prediction - 0.5) < self.RL_CONFIDENCE_THRESHOLD:
                    continue  # Skip RL if too close to 0.5

            weight = self.model_weights.get(model_name, 0.1)
            weighted_sum += prediction * weight
            weight_total += weight

        if weight_total > 0:
            return weighted_sum / weight_total
        return 0.5

    def get_buy_threshold(self, sentiment: float = 0.0) -> float:
        """Dynamic buy threshold with sentiment modulation."""
        sentiment_adj = sentiment * 0.05
        return max(0.40, min(0.80, self.buy_threshold - sentiment_adj))

    def get_sell_threshold(self, sentiment: float = 0.0) -> float:
        """Dynamic sell threshold with sentiment modulation."""
        sentiment_adj = -sentiment * 0.05
        return max(0.20, min(0.60, self.sell_threshold - sentiment_adj))

    def update_thresholds(self, win_rate: float, sharpe_ratio: float = 0.0,
                          max_drawdown: float = 0.0):
        """Adjust thresholds based on recent trading performance."""
        wr_signal = (win_rate - 50) / 100
        adjustment = wr_signal * 0.02

        self.buy_threshold = max(0.40, min(0.80, self.buy_threshold - adjustment))
        self.sell_threshold = max(0.20, min(0.60, self.sell_threshold + adjustment))

        self.threshold_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "win_rate": win_rate,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        })
        if len(self.threshold_history) > self.max_history:
            self.threshold_history = self.threshold_history[-self.max_history:]

    def get_model_stats(self) -> Dict[str, Dict]:
        """Get stats for each model in the ensemble."""
        stats = {}
        for model_name in self.model_weights:
            history = self.model_history.get(model_name, [])
            if history:
                recent = history[-50:]
                accuracy = sum(1 for h in recent if h["correct"]) / len(recent)
                stats[model_name] = {
                    "weight": self.model_weights[model_name],
                    "accuracy": accuracy,
                    "predictions": len(history),
                }
            else:
                stats[model_name] = {
                    "weight": self.model_weights[model_name],
                    "accuracy": 0.5,
                    "predictions": 0,
                }
        return stats

    def save_learner(self, filepath: str = "data/state/meta_learner.json"):
        """Save learner state."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump({
                    "buy_threshold": self.buy_threshold,
                    "sell_threshold": self.sell_threshold,
                    "model_weights": self.model_weights,
                    "model_history": {k: v[-50:] for k, v in self.model_history.items()},
                    "threshold_history": self.threshold_history[-20:],
                }, f, indent=2)
        except Exception:
            pass

    def load_learner(self, filepath: str = "data/state/meta_learner.json"):
        """Load learner state."""
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.buy_threshold = data.get("buy_threshold", 0.55)
            self.sell_threshold = data.get("sell_threshold", 0.45)

            # Load weights including rl_agent
            loaded_weights = data.get("model_weights", self.model_weights)
            valid_models = ("ml_model", "sentiment", "rl_agent")
            if "ml_model" in loaded_weights:
                total = sum(v for k, v in loaded_weights.items() if k in valid_models)
                if total > 0:
                    self.model_weights = {
                        k: v / total for k, v in loaded_weights.items()
                        if k in valid_models
                    }
                    # Ensure rl_agent exists even if not in saved data
                    if "rl_agent" not in self.model_weights:
                        self.model_weights["rl_agent"] = 0.15
                        total = sum(self.model_weights.values())
                        self.model_weights = {k: v / total for k, v in self.model_weights.items()}
                else:
                    self.model_weights = {"ml_model": 0.50, "sentiment": 0.35, "rl_agent": 0.15}

            # Load history including rl_agent
            loaded_history = data.get("model_history", self.model_history)
            self.model_history = {
                "ml_model": loaded_history.get("ml_model", []),
                "sentiment": loaded_history.get("sentiment", []),
                "rl_agent": loaded_history.get("rl_agent", []),
            }
            self.threshold_history = data.get("threshold_history", [])
        except Exception:
            pass
