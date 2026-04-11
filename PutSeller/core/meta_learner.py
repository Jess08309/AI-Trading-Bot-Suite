"""
PutSeller Meta-Learner — Adaptive Ensemble for Credit Spread Quality

Combines rules-based score + ML probability into a final confidence score.
Dynamically adjusts thresholds based on recent performance.

Weights: rules 0.65, ML 0.35 (rules are the backbone for income strategies)
"""
import logging
import json
import os
from datetime import datetime
from typing import Tuple

log = logging.getLogger("putseller.meta")

# Config defaults
META_CONFIDENCE_THRESHOLD = 0.55  # Lower than CallBuyer — put selling has edge
META_MIN_RULE_SCORE = 3.0
META_WINDOW = 40
ML_WEIGHT = 0.35


class PutSellerMetaLearner:
    """Adaptive decision engine for credit spread quality assessment."""

    def __init__(self, state_dir: str = "data/state"):
        self.state_file = os.path.join(state_dir, "meta_state.json")
        self.confidence_threshold: float = META_CONFIDENCE_THRESHOLD
        self.min_rule_score: float = META_MIN_RULE_SCORE
        self.rule_weight: float = 1.0 - ML_WEIGHT
        self.ml_weight: float = ML_WEIGHT

        self.recent_trades: list = []
        self.win_streak: int = 0
        self.loss_streak: int = 0
        self.total_trades: int = 0

        self._load_state()

    def evaluate(self, rule_score: float, ml_proba: float,
                 ml_active: bool) -> Tuple[float, bool, str]:
        """Evaluate whether to open a spread.

        Returns (confidence, should_trade, reason).
        """
        rule_norm = min(rule_score / 10.0, 1.0)

        if ml_active:
            confidence = self.rule_weight * rule_norm + self.ml_weight * ml_proba
        else:
            confidence = rule_norm

        if rule_score < self.min_rule_score:
            return confidence, False, f"rule_score {rule_score:.1f} < {self.min_rule_score:.1f}"

        if confidence < self.confidence_threshold:
            return confidence, False, f"conf {confidence:.3f} < {self.confidence_threshold:.3f}"

        ml_str = f"ml={ml_proba:.2f}" if ml_active else "ml=warmup"
        return confidence, True, f"PASS rules={rule_score:.1f} {ml_str} conf={confidence:.3f}"

    def record_result(self, won: bool, pnl_pct: float):
        """Record a trade result and adapt thresholds."""
        self.total_trades += 1
        self.recent_trades.append({
            "won": won,
            "pnl_pct": pnl_pct,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.recent_trades) > META_WINDOW:
            self.recent_trades = self.recent_trades[-META_WINDOW:]

        if won:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        self._adapt()
        self._save_state()

    def get_status(self) -> dict:
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
            "mode": self._mode_label(),
        }

    def _adapt(self):
        base_conf = META_CONFIDENCE_THRESHOLD
        base_rule = META_MIN_RULE_SCORE

        if self.loss_streak >= 5:
            self.confidence_threshold = min(base_conf + 0.10, 0.80)
            self.min_rule_score = min(base_rule + 2.0, 8.0)
            log.warning(f"DEFENSIVE (loss streak {self.loss_streak})")
        elif self.loss_streak >= 3:
            self.confidence_threshold = min(base_conf + 0.05, 0.75)
            self.min_rule_score = min(base_rule + 1.0, 7.0)
            log.info(f"CAUTIOUS (loss streak {self.loss_streak})")
        elif self.win_streak >= 5:
            self.confidence_threshold = max(base_conf - 0.05, 0.40)
            self.min_rule_score = max(base_rule - 1.0, 2.0)
        elif self.win_streak >= 3:
            self.confidence_threshold = max(base_conf - 0.02, 0.45)
            self.min_rule_score = max(base_rule - 0.5, 2.5)
        else:
            self.confidence_threshold = base_conf
            self.min_rule_score = base_rule

    def _mode_label(self) -> str:
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
        try:
            state = {
                "confidence_threshold": self.confidence_threshold,
                "min_rule_score": self.min_rule_score,
                "win_streak": self.win_streak,
                "loss_streak": self.loss_streak,
                "total_trades": self.total_trades,
                "recent_trades": self.recent_trades[-META_WINDOW:],
                "updated": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save meta state: {e}")

    def _load_state(self):
        """Load trade history only — thresholds always come from config
        constants so that changes take effect on restart."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            # Load trade history (NOT thresholds)
            self.total_trades = state.get("total_trades", 0)
            self.recent_trades = state.get("recent_trades", [])

            # Re-derive streaks from tail of recent_trades
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

            # Let adaptation logic adjust from config baseline
            self._adapt()

            log.info(f"Meta loaded: {self.total_trades} trades, "
                     f"streak W{self.win_streak}/L{self.loss_streak}, "
                     f"conf={self.confidence_threshold:.2f}, mode={self._mode_label()}")
        except Exception as e:
            log.warning(f"Could not load meta state: {e}")
