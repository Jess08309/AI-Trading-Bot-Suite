"""Adaptive Parameter Tuning — agent-suggested parameter adjustments.

The orchestrator can suggest bounded parameter changes based on recent
performance data. Changes are logged and applied only when conditions
are met (minimum sample size, bounded ranges, cooldown periods).

Safety:
    - All parameters have hard min/max bounds
    - Changes are capped at ±20% per adjustment
    - Minimum 10 resolved trades before any changes
    - Cooldown period between adjustments (1 hour)
    - In "suggest" mode (default): logs suggestions but doesn't apply
    - In "auto" mode: applies suggestions within bounds

Storage: C:/Bot/data/state/tuning_state.json
"""
from __future__ import annotations
import json, os, logging, time
from typing import Dict, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger("agent_advisor")

TUNING_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "state", "tuning_state.json"
)

# ── Tunable Parameters & Bounds ──────────────────────────
# Each entry: (current_config_attr, min_val, max_val, description)
TUNABLE_PARAMS = {
    "STOP_LOSS_PCT": {
        "attr": "STOP_LOSS_PCT",
        "min": -3.0,
        "max": -0.5,
        "description": "Spot stop-loss percentage (negative, e.g. -1.5%)",
        "step": 0.25,
    },
    "TAKE_PROFIT_PCT": {
        "attr": "TAKE_PROFIT_PCT",
        "min": 1.0,
        "max": 6.0,
        "description": "Spot take-profit percentage",
        "step": 0.5,
    },
    "TRAILING_STOP_PCT": {
        "attr": "TRAILING_STOP_PCT",
        "min": 0.3,
        "max": 2.0,
        "description": "Trailing stop percentage",
        "step": 0.1,
    },
    "TRAILING_ACTIVATE_PCT": {
        "attr": "TRAILING_ACTIVATE_PCT",
        "min": 0.5,
        "max": 3.0,
        "description": "Trailing stop activation threshold",
        "step": 0.1,
    },
    "MIN_ML_CONFIDENCE": {
        "attr": "MIN_ML_CONFIDENCE",
        "min": 0.52,
        "max": 0.70,
        "description": "Minimum ML confidence to enter a trade",
        "step": 0.02,
    },
    "MIN_ENSEMBLE_SCORE": {
        "attr": "MIN_ENSEMBLE_SCORE",
        "min": 0.45,
        "max": 0.70,
        "description": "Minimum ensemble score to enter",
        "step": 0.02,
    },
    "MAX_HOLD_HOURS_SPOT": {
        "attr": "MAX_HOLD_HOURS_SPOT",
        "min": 1.0,
        "max": 6.0,
        "description": "Max hold hours for flat spot positions",
        "step": 0.5,
    },
    "MAX_HOLD_HOURS_FUTURES": {
        "attr": "MAX_HOLD_HOURS_FUTURES",
        "min": 0.5,
        "max": 4.0,
        "description": "Max hold hours for flat futures positions",
        "step": 0.5,
    },
}

# Max change per adjustment (±20% of current value)
MAX_CHANGE_PCT = 0.20
MIN_TRADES_BEFORE_TUNING = 10
COOLDOWN_SECONDS = 3600  # 1 hour between adjustments


class AdaptiveTuner:
    """Manages parameter suggestions from agents and applies them safely."""

    def __init__(self, mode: str = "suggest"):
        """
        Args:
            mode: "suggest" = log-only (default), "auto" = apply within bounds
        """
        self.mode = mode
        self.suggestions: List[Dict] = []
        self.applied_changes: List[Dict] = []
        self.last_adjustment_time = 0.0
        self._load()

    def _load(self):
        if not os.path.exists(TUNING_FILE):
            return
        try:
            with open(TUNING_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.suggestions = data.get("suggestions", [])[-50:]  # keep last 50
            self.applied_changes = data.get("applied_changes", [])[-50:]
            self.last_adjustment_time = data.get("last_adjustment_time", 0.0)
            logger.info(
                f"AdaptiveTuner loaded: {len(self.suggestions)} suggestions, "
                f"{len(self.applied_changes)} applied changes"
            )
        except Exception as e:
            logger.warning(f"AdaptiveTuner load failed: {e}")

    def save(self):
        os.makedirs(os.path.dirname(TUNING_FILE), exist_ok=True)
        data = {
            "suggestions": self.suggestions[-50:],
            "applied_changes": self.applied_changes[-50:],
            "last_adjustment_time": self.last_adjustment_time,
            "mode": self.mode,
        }
        try:
            with open(TUNING_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"AdaptiveTuner save failed: {e}")

    def record_suggestion(self, param_name: str, direction: str, reasoning: str,
                          agent: str = "orchestrator"):
        """Record a parameter change suggestion from an agent.

        Args:
            param_name: One of the TUNABLE_PARAMS keys
            direction: "increase" or "decrease"
            reasoning: Agent's reasoning for the change
            agent: Which agent made the suggestion
        """
        if param_name not in TUNABLE_PARAMS:
            logger.debug(f"Tuner: unknown param '{param_name}', ignoring")
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "param": param_name,
            "direction": direction,
            "reasoning": reasoning,
            "agent": agent,
            "applied": False,
        }
        self.suggestions.append(entry)
        logger.info(
            f"TUNING SUGGESTION: {agent} recommends {direction} {param_name} — {reasoning}"
        )
        self.save()

    def process_suggestions(self, config_obj, resolved_trades: int) -> List[Dict]:
        """Process pending suggestions and optionally apply them.

        Args:
            config_obj: The TradingConfig instance to modify
            resolved_trades: Number of trades with known outcomes

        Returns:
            List of changes that were applied (or would be in suggest mode)
        """
        now = time.time()

        # Safety checks
        if resolved_trades < MIN_TRADES_BEFORE_TUNING:
            return []
        if now - self.last_adjustment_time < COOLDOWN_SECONDS:
            return []

        # Count recent suggestions per parameter (last hour)
        recent = [s for s in self.suggestions
                  if not s.get("applied") and
                  (now - self._parse_ts(s.get("timestamp", ""))) < 3600]

        if not recent:
            return []

        # Tally votes per parameter
        votes: Dict[str, Dict] = {}  # param -> {increase: n, decrease: n}
        for s in recent:
            param = s["param"]
            d = s["direction"]
            if param not in votes:
                votes[param] = {"increase": 0, "decrease": 0, "reasons": []}
            votes[param][d] = votes[param].get(d, 0) + 1
            votes[param]["reasons"].append(s["reasoning"])

        changes = []
        for param, v in votes.items():
            spec = TUNABLE_PARAMS[param]
            if v["increase"] > v["decrease"] and v["increase"] >= 2:
                direction = "increase"
            elif v["decrease"] > v["increase"] and v["decrease"] >= 2:
                direction = "decrease"
            else:
                continue  # No consensus

            current_val = getattr(config_obj, spec["attr"], None)
            if current_val is None:
                continue

            step = spec["step"]
            if direction == "increase":
                new_val = current_val + step
            else:
                new_val = current_val - step

            # Clamp to bounds
            new_val = max(spec["min"], min(spec["max"], new_val))

            # Cap change at ±20%
            if abs(current_val) > 0.001:
                max_delta = abs(current_val) * MAX_CHANGE_PCT
                delta = new_val - current_val
                if abs(delta) > max_delta:
                    new_val = current_val + (max_delta if delta > 0 else -max_delta)

            if abs(new_val - current_val) < 0.001:
                continue  # No meaningful change

            change = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "param": param,
                "old_value": current_val,
                "new_value": round(new_val, 4),
                "direction": direction,
                "votes": {"increase": v["increase"], "decrease": v["decrease"]},
                "applied": self.mode == "auto",
            }

            if self.mode == "auto":
                setattr(config_obj, spec["attr"], round(new_val, 4))
                logger.info(
                    f"TUNING APPLIED: {param} {current_val} → {new_val:.4f} "
                    f"({direction}, votes={v['increase']}↑/{v['decrease']}↓)"
                )
            else:
                logger.info(
                    f"TUNING SUGGESTION (not applied): {param} {current_val} → {new_val:.4f} "
                    f"({direction}, votes={v['increase']}↑/{v['decrease']}↓)"
                )

            self.applied_changes.append(change)
            changes.append(change)

        # Mark recent suggestions as processed
        for s in recent:
            s["applied"] = True

        if changes:
            self.last_adjustment_time = now
            self.save()

        return changes

    def get_tuning_prompt(self) -> str:
        """Generate a prompt section describing tunable parameters for the orchestrator."""
        lines = [
            "═══ ADAPTIVE PARAMETER TUNING ═══",
            "You can suggest parameter adjustments based on observed performance.",
            "Add a 'tuning_suggestions' array to your response JSON if you have recommendations.",
            "Format: [{\"param\": \"PARAM_NAME\", \"direction\": \"increase|decrease\", \"reasoning\": \"why\"}]",
            "",
            "Tunable parameters (current values):",
        ]
        return "\n".join(lines)

    def get_current_values_str(self, config_obj) -> str:
        """Get current parameter values for the orchestrator prompt."""
        lines = []
        for name, spec in TUNABLE_PARAMS.items():
            val = getattr(config_obj, spec["attr"], "?")
            lines.append(
                f"  {name} = {val} (range: {spec['min']} to {spec['max']}, "
                f"step: {spec['step']}) — {spec['description']}"
            )
        # Recent changes
        recent_changes = self.applied_changes[-3:]
        if recent_changes:
            lines.append("")
            lines.append("Recent changes:")
            for c in recent_changes:
                lines.append(
                    f"  {c['param']}: {c['old_value']} → {c['new_value']} "
                    f"({'applied' if c.get('applied') else 'suggested'})"
                )
        return "\n".join(lines)

    @staticmethod
    def _parse_ts(ts_str: str) -> float:
        """Parse ISO timestamp to epoch seconds."""
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return 0.0
