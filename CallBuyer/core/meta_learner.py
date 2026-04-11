"""
CallBuyer Meta-Learner — Adaptive Ensemble Decision Engine

Combines rules-based score + ML probability into a final confidence score.
Dynamically adjusts thresholds based on recent performance (win/loss streaks).

Ensemble weights (configurable via config):
  - Rules score:  0.60 (default)
  - ML model:     0.40 (when active; otherwise 100% rules)

Adaptation logic mirrors AlpacaBot's MetaLearner:
  - Win streak (3+):  loosen thresholds slightly (more aggressive)
  - Loss streak (3+): tighten thresholds (more selective)
  - Neutral:          drift back to defaults
"""
import logging
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional

from core.config import CallBuyerConfig as cfg

log = logging.getLogger("callbuyer.meta")


class MetaLearner:
    """Adaptive decision engine combining rules + ML for call buying."""

    def __init__(self, state_dir: str = "data/state"):
        self.state_file = os.path.join(state_dir, "meta_state.json")

        # Current operating thresholds (start at defaults)
        self.confidence_threshold: float = cfg.META_CONFIDENCE_THRESHOLD
        self.min_rule_score: float = cfg.META_MIN_RULE_SCORE

        # Weights
        self.rule_weight: float = 1.0 - cfg.ML_WEIGHT
        self.ml_weight: float = cfg.ML_WEIGHT

        # Rolling performance
        self.recent_trades: list = []
        self.win_streak: int = 0
        self.loss_streak: int = 0
        self.total_trades: int = 0

        self._load_state()

    def evaluate(self, rule_score: float, ml_proba: float,
                 ml_active: bool,
                 conf_adjust: float = 0.0,
                 rule_adjust: float = 0.0) -> Tuple[float, bool, str]:
        """Evaluate whether to take a trade.

        Args:
            rule_score: Rules-based momentum score (0-10)
            ml_proba: ML model win probability (0-1)
            ml_active: Whether ML model is active (vs warmup)
            conf_adjust: offset to confidence_threshold (negative = looser)
            rule_adjust: offset to min_rule_score (negative = looser)

        Returns:
            (confidence, should_trade, reason)
        """
        # Normalize rule score to 0-1
        rule_norm = min(rule_score / 10.0, 1.0)

        if ml_active:
            confidence = self.rule_weight * rule_norm + self.ml_weight * ml_proba
        else:
            confidence = rule_norm  # 100% rules during warmup

        # Apply thresholds with time-of-day adjustments
        effective_min_rule = max(1.5, self.min_rule_score + rule_adjust)
        effective_conf = max(0.25, min(0.85, self.confidence_threshold + conf_adjust))

        if rule_score < effective_min_rule:
            reason = f"rule_score {rule_score:.1f} < min {effective_min_rule:.1f}"
            return confidence, False, reason

        if confidence < effective_conf:
            reason = f"confidence {confidence:.3f} < threshold {effective_conf:.3f}"
            return confidence, False, reason

        ml_str = f"ml={ml_proba:.2f}" if ml_active else "ml=warmup"
        reason = f"PASS rules={rule_score:.1f} {ml_str} conf={confidence:.3f}"
        return confidence, True, reason

    def record_result(self, won: bool, pnl_pct: float):
        """Record a trade result and adapt thresholds."""
        self.total_trades += 1
        self.recent_trades.append({
            "won": won,
            "pnl_pct": pnl_pct,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep rolling window
        window = cfg.META_WINDOW
        if len(self.recent_trades) > window:
            self.recent_trades = self.recent_trades[-window:]

        # Update streaks
        if won:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        # Adapt thresholds
        self._adapt_thresholds()
        self._save_state()

    def get_status(self) -> dict:
        """Return meta-learner status for dashboard."""
        recent_wins = sum(1 for t in self.recent_trades[-20:] if t["won"]) if self.recent_trades else 0
        recent_total = min(len(self.recent_trades), 20)
        win_rate = recent_wins / recent_total if recent_total > 0 else 0

        return {
            "confidence_threshold": round(self.confidence_threshold, 3),
            "min_rule_score": round(self.min_rule_score, 1),
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "total_trades": self.total_trades,
            "recent_win_rate": round(win_rate, 3),
            "mode": self._get_mode_label(),
        }

    def _adapt_thresholds(self):
        """Adjust thresholds based on streaks.

        Win streak 3+:  loosen slightly → take more trades
        Loss streak 3+: tighten → be more selective
        Neutral:         drift back to config defaults
        """
        base_conf = cfg.META_CONFIDENCE_THRESHOLD
        base_rule = cfg.META_MIN_RULE_SCORE

        if self.loss_streak >= 5:
            # Heavy tightening — 5+ consecutive losses
            self.confidence_threshold = min(base_conf + 0.10, 0.85)
            self.min_rule_score = min(base_rule + 2.0, 8.0)
            log.warning(f"META: Heavy tightening (loss streak {self.loss_streak}): "
                        f"conf={self.confidence_threshold:.2f}, min_rule={self.min_rule_score:.1f}")
        elif self.loss_streak >= 3:
            # Moderate tightening
            self.confidence_threshold = min(base_conf + 0.05, 0.80)
            self.min_rule_score = min(base_rule + 1.0, 7.0)
            log.info(f"META: Tightening (loss streak {self.loss_streak}): "
                     f"conf={self.confidence_threshold:.2f}, min_rule={self.min_rule_score:.1f}")
        elif self.win_streak >= 5:
            # Moderate loosening — riding hot streak
            self.confidence_threshold = max(base_conf - 0.05, 0.45)
            self.min_rule_score = max(base_rule - 1.0, 2.0)
            log.info(f"META: Loosening (win streak {self.win_streak}): "
                     f"conf={self.confidence_threshold:.2f}, min_rule={self.min_rule_score:.1f}")
        elif self.win_streak >= 3:
            # Slight loosening
            self.confidence_threshold = max(base_conf - 0.02, 0.50)
            self.min_rule_score = max(base_rule - 0.5, 2.5)
        else:
            # Drift back to defaults
            self.confidence_threshold = base_conf
            self.min_rule_score = base_rule

    def _get_mode_label(self) -> str:
        if self.loss_streak >= 5:
            return "DEFENSIVE"
        elif self.loss_streak >= 3:
            return "CAUTIOUS"
        elif self.win_streak >= 5:
            return "AGGRESSIVE"
        elif self.win_streak >= 3:
            return "PRESSING"
        return "NORMAL"

    def _save_state(self):
        """Persist meta-learner state."""
        try:
            state = {
                "confidence_threshold": self.confidence_threshold,
                "min_rule_score": self.min_rule_score,
                "win_streak": self.win_streak,
                "loss_streak": self.loss_streak,
                "total_trades": self.total_trades,
                "recent_trades": self.recent_trades[-cfg.META_WINDOW:],
                "updated": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save meta state: {e}")

    def _load_state(self):
        """Load persisted state if available.

        Only loads trade history and total_trades — thresholds always come
        from config so that config changes take effect on restart.  Streaks
        are re-derived from the loaded trade history.
        """
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            # Load trade history (NOT thresholds — those come from config)
            self.total_trades = state.get("total_trades", 0)
            self.recent_trades = state.get("recent_trades", [])

            # Re-derive streaks from the tail of recent_trades
            self.win_streak = 0
            self.loss_streak = 0
            for t in reversed(self.recent_trades):
                if t.get("won"):
                    if self.loss_streak > 0:
                        break
                    self.win_streak += 1
                else:
                    if self.win_streak > 0:
                        break
                    self.loss_streak += 1

            # Let adaptation logic adjust thresholds from config baseline
            self._adapt_thresholds()

            log.info(f"Loaded meta state: {self.total_trades} trades, "
                     f"streak W{self.win_streak}/L{self.loss_streak}, "
                     f"conf={self.confidence_threshold:.2f}, "
                     f"min_rule={self.min_rule_score:.1f}, mode={self._get_mode_label()}")
        except Exception as e:
            log.warning(f"Could not load meta state: {e}")
