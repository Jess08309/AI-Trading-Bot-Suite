"""
Reinforcement Learning Shadow Agent for Options Trading.

Runs in SHADOW MODE by default — observes the same signals the bot generates,
makes its own hypothetical decisions, and tracks how it would have performed
WITHOUT touching real trades.

Once shadow performance proves better than baseline over a configurable
window, it can be promoted to influence real sizing (never entry/exit directly).

Architecture:
  - State: (RSI bucket, trend bucket, vol_regime, sentiment bucket, ML confidence bucket)
  - Actions: SKIP, SMALL, MEDIUM, LARGE (position sizing multiplier)
  - Reward: trade PnL (clipped to prevent single outlier from dominating)
  - Q-table (tabular) + optional DQN upgrade path

WHY SHADOW FIRST:
  The crypto bot's RL performed WORSE than baseline (+$39 vs +$49 over 200 trades).
  We start shadow-only and require proof before graduating to live influence.
"""
from __future__ import annotations
import json
import logging
import os
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

log = logging.getLogger("alpacabot.rl")

# Actions: sizing multipliers
ACTIONS = {
    0: ("SKIP", 0.0),       # Don't take this trade
    1: ("SMALL", 0.5),      # Half size
    2: ("MEDIUM", 1.0),     # Normal size
    3: ("LARGE", 1.5),      # 1.5x size
}


class RLShadowAgent:
    """Reinforcement Learning agent that runs in shadow mode alongside the bot."""

    def __init__(self,
                 state_file: str = "data/state/rl_agent.json",
                 shadow_report_file: str = "data/state/rl_shadow_report.json",
                 shadow_events_file: str = "data/state/rl_shadow_events.jsonl",
                 learning_rate: float = 0.1,
                 discount: float = 0.95,
                 epsilon: float = 0.15):
        self.state_file = state_file
        self.shadow_report_file = shadow_report_file
        self.shadow_events_file = shadow_events_file

        # Q-learning parameters
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05

        # Q-table: state_key → [Q(skip), Q(small), Q(medium), Q(large)]
        self.q_table: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        # Shadow tracking
        self.shadow_equity = 100000.0  # Match AlpacaBot starting balance
        self.baseline_equity = 100000.0
        self.shadow_trades: List[Dict] = []
        self.baseline_trades: List[Dict] = []
        self.shadow_open: Dict[str, Dict] = {}  # hypothetical open positions
        self.total_episodes = 0

        # Promotion tracking
        self.shadow_better_streak = 0
        self.PROMOTION_THRESHOLD = 50  # Must outperform baseline for 50 trades

        self._load_state()

    # ── State Discretization ─────────────────────────────

    def _discretize_state(self, features: Dict[str, float]) -> str:
        """Convert continuous features into discrete state key for Q-table.

        Buckets:
          - RSI: oversold(0), neutral(1), overbought(2)
          - Trend: down(0), side(1), up(2)
          - Vol regime: low(0), normal(1), high(2)
          - Sentiment: bearish(0), neutral(1), bullish(2)
          - ML confidence: low(0), medium(1), high(2)
        """
        rsi = features.get("rsi", 50)
        if rsi < 35:
            rsi_bucket = 0
        elif rsi > 65:
            rsi_bucket = 2
        else:
            rsi_bucket = 1

        trend = features.get("trend_strength", 0)
        price_change = features.get("price_change_5", 0)
        if price_change > 0.003 and trend > 20:
            trend_bucket = 2  # UP
        elif price_change < -0.003 and trend > 20:
            trend_bucket = 0  # DOWN
        else:
            trend_bucket = 1  # SIDE

        vol = features.get("atr_normalized", 0)
        if vol < 0.003:
            vol_bucket = 0
        elif vol > 0.008:
            vol_bucket = 2
        else:
            vol_bucket = 1

        sentiment = features.get("sentiment", 0)
        if sentiment < -0.2:
            sent_bucket = 0
        elif sentiment > 0.2:
            sent_bucket = 2
        else:
            sent_bucket = 1

        ml_conf = features.get("ml_confidence", 0.5)
        if ml_conf < 0.55:
            ml_bucket = 0
        elif ml_conf > 0.70:
            ml_bucket = 2
        else:
            ml_bucket = 1

        return f"{rsi_bucket}_{trend_bucket}_{vol_bucket}_{sent_bucket}_{ml_bucket}"

    # ── Shadow Mode Core ─────────────────────────────────

    def shadow_evaluate(self, signal: Dict, features: Dict[str, float]) -> Dict:
        """Evaluate a signal in shadow mode — decide what RL would do.

        Called for every signal the bot generates. Records the RL's hypothetical
        decision alongside what the bot actually does.

        Args:
            signal: {"symbol": ..., "direction": ..., "score": ..., "price": ...}
            features: indicator values + sentiment + ML confidence

        Returns:
            {"action": str, "sizing_mult": float, "state_key": str}
        """
        state_key = self._discretize_state(features)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(len(ACTIONS))
        else:
            q_values = self.q_table[state_key]
            action = int(np.argmax(q_values))

        action_name, sizing_mult = ACTIONS[action]

        # Record shadow position if RL says to trade
        symbol = signal.get("underlying", signal.get("symbol", "?"))
        if action > 0:  # Not SKIP
            self.shadow_open[symbol] = {
                "direction": signal.get("direction", "call"),
                "entry_price": signal.get("price", 0),
                "sizing_mult": sizing_mult,
                "state_key": state_key,
                "action": action,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "action": action_name,
            "sizing_mult": sizing_mult,
            "state_key": state_key,
        }

    def shadow_record_outcome(self, symbol: str, pnl: float, pnl_pct: float):
        """Record the outcome of a trade for both baseline and RL shadow.

        Called whenever the bot closes a real trade. The RL agent learns
        from its hypothetical sizing decision.

        Args:
            symbol: underlying symbol
            pnl: actual dollar PnL of the bot's trade
            pnl_pct: percentage PnL
        """
        # Baseline always tracks the actual bot performance
        self.baseline_equity += pnl
        self.baseline_trades.append({
            "symbol": symbol,
            "pnl": pnl,
            "equity": self.baseline_equity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # RL shadow: did it have a position?
        shadow_pos = self.shadow_open.pop(symbol, None)
        if shadow_pos:
            sizing_mult = shadow_pos.get("sizing_mult", 1.0)
            state_key = shadow_pos.get("state_key", "")
            action = shadow_pos.get("action", 2)

            # RL's hypothetical PnL (scaled by its sizing decision)
            rl_pnl = pnl * sizing_mult

            self.shadow_equity += rl_pnl
            self.shadow_trades.append({
                "symbol": symbol,
                "pnl": rl_pnl,
                "sizing_mult": sizing_mult,
                "action": ACTIONS[action][0],
                "equity": self.shadow_equity,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Reward: PnL clipped to prevent outliers from dominating learning
            reward = np.clip(pnl_pct * sizing_mult * 100, -5.0, 5.0)

            # Q-learning update
            old_q = self.q_table[state_key][action]
            # For terminal state (trade closed), no future value
            self.q_table[state_key][action] = old_q + self.lr * (reward - old_q)

            self.total_episodes += 1

            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Log shadow event
            self._log_shadow_event(symbol, pnl, rl_pnl, sizing_mult, state_key, action)

        else:
            # RL chose to SKIP this trade — was that smart?
            # If bot lost money, RL was right to skip (positive reward for skip)
            # If bot made money, RL missed out (small negative reward for skip)
            # We don't know the state key, so we can't update Q-table
            # This is intentional — SKIP rewards come through not losing money
            pass

        # Track if RL is outperforming
        if len(self.shadow_trades) > 10 and len(self.baseline_trades) > 10:
            if self.shadow_equity > self.baseline_equity:
                self.shadow_better_streak += 1
            else:
                self.shadow_better_streak = 0

        self._save_state()

    def _log_shadow_event(self, symbol: str, bot_pnl: float, rl_pnl: float,
                          sizing_mult: float, state_key: str, action: int):
        """Append shadow event to JSONL log."""
        try:
            os.makedirs(os.path.dirname(self.shadow_events_file), exist_ok=True)
            event = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "bot_pnl": round(bot_pnl, 2),
                "rl_pnl": round(rl_pnl, 2),
                "sizing": sizing_mult,
                "action": ACTIONS[action][0],
                "state": state_key,
                "rl_equity": round(self.shadow_equity, 2),
                "baseline_equity": round(self.baseline_equity, 2),
            }
            with open(self.shadow_events_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    # ── Promotion Check ──────────────────────────────────

    def is_ready_for_promotion(self) -> bool:
        """Check if RL has proven itself enough for live influence.

        Requires:
          - At least 50 shadow trades
          - Outperforming baseline for last 50 consecutive outcomes
          - Shadow equity > baseline equity
        """
        if len(self.shadow_trades) < self.PROMOTION_THRESHOLD:
            return False
        if self.shadow_better_streak < self.PROMOTION_THRESHOLD:
            return False
        return self.shadow_equity > self.baseline_equity

    def get_sizing_recommendation(self, signal: Dict,
                                  features: Dict[str, float]) -> float:
        """Get RL's sizing recommendation (only used if promoted to live).

        Returns: sizing multiplier (0.0 = skip, 0.5 = small, 1.0 = normal, 1.5 = large)
        """
        state_key = self._discretize_state(features)
        q_values = self.q_table[state_key]
        action = int(np.argmax(q_values))
        _, sizing_mult = ACTIONS[action]
        return sizing_mult

    # ── State Persistence ────────────────────────────────

    def _save_state(self):
        """Save Q-table and shadow tracking to disk."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            state = {
                "q_table": dict(self.q_table),
                "epsilon": self.epsilon,
                "total_episodes": self.total_episodes,
                "shadow_equity": self.shadow_equity,
                "baseline_equity": self.baseline_equity,
                "shadow_better_streak": self.shadow_better_streak,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Save shadow report
            report = self.get_shadow_report()
            with open(self.shadow_report_file, "w") as f:
                json.dump(report, f, indent=2)

        except Exception as e:
            log.debug(f"RL save failed: {e}")

    def _load_state(self):
        """Load Q-table and shadow tracking from disk."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            for key, values in state.get("q_table", {}).items():
                self.q_table[key] = values
            self.epsilon = state.get("epsilon", self.epsilon)
            self.total_episodes = state.get("total_episodes", 0)
            self.shadow_equity = state.get("shadow_equity", 100000.0)
            self.baseline_equity = state.get("baseline_equity", 100000.0)
            self.shadow_better_streak = state.get("shadow_better_streak", 0)
            log.info(f"RL agent loaded: {self.total_episodes} episodes, "
                     f"ε={self.epsilon:.3f}, shadow={self.shadow_equity:,.2f}, "
                     f"baseline={self.baseline_equity:,.2f}")
        except Exception as e:
            log.warning(f"RL load failed: {e}")

    # ── Reporting ────────────────────────────────────────

    def get_shadow_report(self) -> Dict:
        """Generate shadow mode performance report."""
        n_shadow = len(self.shadow_trades)
        n_baseline = len(self.baseline_trades)

        shadow_wins = sum(1 for t in self.shadow_trades if t.get("pnl", 0) > 0)
        baseline_wins = sum(1 for t in self.baseline_trades if t.get("pnl", 0) > 0)

        return {
            "mode": "shadow",
            "ready_for_promotion": self.is_ready_for_promotion(),
            "total_episodes": self.total_episodes,
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "shadow": {
                "equity": round(self.shadow_equity, 2),
                "pnl": round(self.shadow_equity - 100000.0, 2),
                "trades": n_shadow,
                "win_rate": round(shadow_wins / n_shadow * 100, 1) if n_shadow > 0 else 0,
            },
            "baseline": {
                "equity": round(self.baseline_equity, 2),
                "pnl": round(self.baseline_equity - 100000.0, 2),
                "trades": n_baseline,
                "win_rate": round(baseline_wins / n_baseline * 100, 1) if n_baseline > 0 else 0,
            },
            "outperform_streak": self.shadow_better_streak,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
