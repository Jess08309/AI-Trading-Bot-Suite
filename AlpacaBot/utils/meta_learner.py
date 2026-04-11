"""
Meta-Learner Ensemble for Options Trading.

Dynamically weights the three signal sources:
  1. ML Model (price direction prediction)
  2. Market Sentiment (Fear & Greed, VIX, SPY trend)
  3. Rule-Based Score (the existing 14-indicator bull/bear scoring)

Tracks per-source accuracy over a rolling window and adjusts weights
so the best-performing source gets the most influence.

Also manages dynamic buy/sell thresholds — tightens thresholds when
the bot is on a losing streak (more selective), loosens when winning.

This is the "brain" that prevents blowups: if ML starts giving bad
predictions, its weight drops automatically and the rule-based system
takes over (which is what's been working already).
"""
from __future__ import annotations
import json
import logging
import os
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional

log = logging.getLogger("alpacabot.meta")


class MetaLearner:
    """Ensemble meta-learner that weights ML, Sentiment, and Rule-based signals."""

    def __init__(self,
                 state_file: str = "data/state/meta_learner.json"):
        self.state_file = state_file

        # Per-source tracking
        self.source_history: Dict[str, List[Dict]] = {
            "ml_model": [],
            "sentiment": [],
            "rule_score": [],
        }

        # Learned weights (rule_score starts highest — it's proven)
        self.source_weights: Dict[str, float] = {
            "ml_model": 0.35,
            "sentiment": 0.15,
            "rule_score": 0.50,
        }

        # Dynamic thresholds (raised for tighter quality)
        self.confidence_threshold = 0.68     # Don't trade below this ensemble score (raised from 0.62 — 18% WR diagnostic)
        self.min_rule_score = 6              # Minimum bull/bear score (raised from 4 — score has no predictive value below 6)
        self.threshold_history: List[Dict] = []
        self.max_history = 200

        self._load_state()

    def record_prediction(self, predictions: Dict[str, float], actual_outcome: float):
        """Record what each source predicted vs actual trade outcome.

        Args:
            predictions: {"ml_model": 0.72, "sentiment": 0.3, "rule_score": 0.8}
                         Each value is 0-1 where >0.5 = bullish prediction
            actual_outcome: 1.0 if trade was profitable, 0.0 if loss
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        for source, prediction in predictions.items():
            if source not in self.source_history:
                continue

            predicted_up = prediction > 0.5
            actual_up = actual_outcome > 0.5
            correct = predicted_up == actual_up

            self.source_history[source].append({
                "prediction": prediction,
                "outcome": actual_outcome,
                "correct": correct,
                "timestamp": timestamp,
            })

            # Cap history
            if len(self.source_history[source]) > self.max_history:
                self.source_history[source] = self.source_history[source][-self.max_history:]

        self._update_weights()
        self._save_state()

    def _update_weights(self):
        """Recalculate source weights based on recent accuracy.

        Sources with better accuracy get more weight.
        Minimum weight of 0.10 ensures no source is completely ignored.
        """
        accuracies = {}
        for source, history in self.source_history.items():
            if len(history) >= 10:
                recent = history[-50:]  # Last 50 predictions
                correct = sum(1 for h in recent if h["correct"])
                accuracies[source] = correct / len(recent)
            else:
                # Default: assume 50% accuracy until proven
                accuracies[source] = 0.5

        # Normalize to weights with minimum floor
        total = sum(max(0.10, acc) for acc in accuracies.values())
        if total > 0:
            for source in self.source_weights:
                acc = accuracies.get(source, 0.5)
                self.source_weights[source] = max(0.10, acc) / total

        log.debug(
            f"Meta weights updated: " +
            " | ".join(f"{k}={v:.2f}(acc={accuracies.get(k, 0.5):.1%})"
                       for k, v in self.source_weights.items())
        )

    def get_ensemble_score(self, predictions: Dict[str, float]) -> float:
        """Get weighted ensemble score from all sources.

        Args:
            predictions: {"ml_model": 0.72, "sentiment": 0.6, "rule_score": 0.85}

        Returns:
            float: 0.0-1.0 weighted score. >0.5 = bullish, <0.5 = bearish.
        """
        weighted_sum = 0.0
        weight_total = 0.0

        for source, prediction in predictions.items():
            weight = self.source_weights.get(source, 0.10)
            weighted_sum += prediction * weight
            weight_total += weight

        if weight_total > 0:
            return weighted_sum / weight_total
        return 0.5

    def should_trade(self, ensemble_score: float, rule_score: int,
                     conf_adjust: float = 0.0, rule_adjust: int = 0) -> bool:
        """Final gate: should we take this trade?

        Combines ensemble confidence with rule-based minimum score.
        Both must pass for a trade to execute.

        Args:
            ensemble_score: 0-1 from get_ensemble_score()
            rule_score: integer bull/bear score from indicator scoring (3-16)
            conf_adjust: optional offset to confidence_threshold (negative = looser, positive = stricter)
            rule_adjust: optional offset to min_rule_score (negative = looser, positive = stricter)

        Returns:
            True if trade qualifies
        """
        # Ensemble confidence must exceed threshold
        effective_conf_threshold = max(0.50, min(0.85, self.confidence_threshold + conf_adjust))
        effective_min_rule = max(3, min(10, self.min_rule_score + rule_adjust))

        ensemble_conf = abs(ensemble_score - 0.5) * 2.0  # Convert to 0-1 confidence
        if ensemble_conf < (effective_conf_threshold - 0.5) * 2.0:
            return False

        # Rule score must be at least minimum
        if rule_score < effective_min_rule:
            return False

        return True

    def update_thresholds(self, recent_win_rate: float,
                          consecutive_losses: int,
                          rolling_loss_rate: float = 0.0):
        """Dynamically adjust thresholds based on performance.

        Two independent triggers (either can tighten thresholds):
          - Option 4: consecutive_losses decays on wins (streak pressure persists)
          - Option 1: rolling_loss_rate over last 10 trades catches spread-out losing

        WINNING → slightly loosen thresholds (capture more opportunities)
        LOSING → tighten thresholds (be more selective, prevent blowup)
        """
        # Option 1: Rolling window trigger (>=60% losses in last 10 trades)
        if rolling_loss_rate >= 0.60:
            self.confidence_threshold = min(0.80, self.confidence_threshold + 0.02)
            self.min_rule_score = min(8, self.min_rule_score + 1)
            log.info(f"Meta: TIGHTENING thresholds (rolling loss rate={rolling_loss_rate:.0%}): "
                     f"conf≥{self.confidence_threshold:.2f}, score≥{self.min_rule_score}")

        # Option 4: Streak trigger (consecutive_losses decays slowly, builds fast)
        elif consecutive_losses >= 3:
            # Emergency tightening — getting hit, be very selective
            self.confidence_threshold = min(0.80, self.confidence_threshold + 0.02)
            self.min_rule_score = min(8, self.min_rule_score + 1)
            log.info(f"Meta: TIGHTENING thresholds (streak={consecutive_losses}): "
                     f"conf≥{self.confidence_threshold:.2f}, score≥{self.min_rule_score}")

        elif recent_win_rate > 60 and consecutive_losses == 0 and rolling_loss_rate < 0.40:
            # Winning on both measures — can afford to be slightly less selective
            self.confidence_threshold = max(0.60, self.confidence_threshold - 0.005)
            self.min_rule_score = max(6, self.min_rule_score)  # Never go below baseline of 6

        # Gradual mean reversion when things normalize
        elif consecutive_losses < 2 and 45 < recent_win_rate < 60 and rolling_loss_rate < 0.50:
            self.confidence_threshold = 0.68  # Reset to new default (was 0.62)
            self.min_rule_score = 6              # Reset to new baseline (was 3)

        self.threshold_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "win_rate": recent_win_rate,
            "consecutive_losses": consecutive_losses,
            "rolling_loss_rate": round(rolling_loss_rate, 3),
            "confidence_threshold": self.confidence_threshold,
            "min_rule_score": self.min_rule_score,
        })
        if len(self.threshold_history) > self.max_history:
            self.threshold_history = self.threshold_history[-self.max_history:]

    def get_source_stats(self) -> Dict[str, Dict]:
        """Get accuracy/weight stats for each source."""
        stats = {}
        for source in self.source_weights:
            history = self.source_history.get(source, [])
            if history:
                recent = history[-50:]
                accuracy = sum(1 for h in recent if h["correct"]) / len(recent)
                stats[source] = {
                    "weight": round(self.source_weights[source], 3),
                    "accuracy": round(accuracy, 3),
                    "predictions": len(history),
                }
            else:
                stats[source] = {
                    "weight": round(self.source_weights[source], 3),
                    "accuracy": 0.5,
                    "predictions": 0,
                }
        return stats

    # ── Persistence ──────────────────────────────────────

    def _save_state(self):
        """Save learner state to disk."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({
                    "source_weights": self.source_weights,
                    "source_history": {k: v[-50:] for k, v in self.source_history.items()},
                    "confidence_threshold": self.confidence_threshold,
                    "min_rule_score": self.min_rule_score,
                    "threshold_history": self.threshold_history[-20:],
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception as e:
            log.debug(f"Meta save failed: {e}")

    def _load_state(self):
        """Load learner state from disk."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)

            loaded_weights = data.get("source_weights", {})
            valid_sources = ("ml_model", "sentiment", "rule_score")
            if any(k in loaded_weights for k in valid_sources):
                total = sum(v for k, v in loaded_weights.items() if k in valid_sources)
                if total > 0:
                    self.source_weights = {
                        k: v / total for k, v in loaded_weights.items()
                        if k in valid_sources
                    }

            loaded_history = data.get("source_history", {})
            for source in self.source_history:
                if source in loaded_history:
                    self.source_history[source] = loaded_history[source]

            self.confidence_threshold = data.get("confidence_threshold", 0.58)
            self.min_rule_score = data.get("min_rule_score", 3)
            self.threshold_history = data.get("threshold_history", [])

            log.info(
                f"Meta learner loaded: weights=" +
                " ".join(f"{k}={v:.2f}" for k, v in self.source_weights.items())
            )
        except Exception as e:
            log.warning(f"Meta load failed: {e}")

    def status(self) -> Dict:
        """Return status for dashboard."""
        return {
            "weights": {k: round(v, 3) for k, v in self.source_weights.items()},
            "stats": self.get_source_stats(),
            "confidence_threshold": self.confidence_threshold,
            "min_rule_score": self.min_rule_score,
        }
