"""
Deep Q-Network (DQN) Reinforcement Learning Trading Agent.
Replaces the Q-table approach with a neural network that generalizes
across continuous state space — no more 243-state limitation.

Drop-in replacement: same class name, same method signatures.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from collections import deque
import json
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Neural network for Q-value approximation
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    Small feedforward network: 5 continuous inputs → 5 action Q-values.
    Two hidden layers with layer normalization for stable training.
    """

    def __init__(self, state_dim: int = 5, action_dim: int = 5,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)

    def to_list(self) -> List:
        """Serialize for JSON save."""
        return [
            [s.tolist() if hasattr(s, 'tolist') else list(s),
             int(a), float(r),
             ns.tolist() if hasattr(ns, 'tolist') else list(ns),
             float(d)]
            for s, a, r, ns, d in self.buffer
        ]

    def from_list(self, data: List):
        """Load from JSON data."""
        self.buffer.clear()
        for item in data:
            s, a, r, ns, d = item
            self.buffer.append((
                np.array(s, dtype=np.float32),
                int(a),
                float(r),
                np.array(ns, dtype=np.float32),
                float(d),
            ))


# ---------------------------------------------------------------------------
# DQN Trading Agent (drop-in replacement for Q-table version)
# ---------------------------------------------------------------------------
class RLTradingAgent:
    """
    DQN agent with continuous 5-dimensional state space.
    State: [sentiment, volatility, trend, rsi_normalized, ml_confidence]
    Actions: position size multipliers [0.25, 0.5, 1.0, 1.5, 2.0]

    Key advantages over Q-table:
    - Generalizes across similar states (no need to visit every state)
    - Handles continuous inputs (no discretization loss)
    - Experience replay for stable, sample-efficient learning
    - Target network to prevent oscillation
    """

    def __init__(self, learning_rate: float = 0.001, discount_factor: float = 0.95,
                 exploration_rate: float = 0.15):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.999
        self.min_exploration = 0.05

        # Action space: position size multipliers
        self.actions = [0.25, 0.5, 1.0, 1.5, 2.0]
        self.state_dim = 5
        self.action_dim = len(self.actions)

        # Networks
        self.policy_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=self.learning_rate)

        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=10_000)
        self.batch_size = 32
        self.target_update_freq = 50  # Sync target net every N updates

        # Trade history for learning
        self.recent_trades: List[Dict] = []
        self.max_recent = 200

        # Stats
        self.cumulative_reward = 0.0
        self.episode_count = 0
        self.total_updates = 0

        # Per-coin tracking for exploration bonus and performance analysis
        self.coin_stats: Dict[str, Dict] = {}
        self.exploration_bonus_trades = 10  # Bonus for coins with < N trades

        # Track the last state for building transitions
        self._last_state_vec: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # State encoding — continuous, no discretization
    # ------------------------------------------------------------------
    def _encode_state(self, sentiment: float, volatility: float,
                      trend: float, rsi: float,
                      ml_confidence: float) -> np.ndarray:
        """Encode raw market features into a normalized state vector.

        Each dimension is scaled to roughly [-1, 1] or [0, 1]:
        - sentiment: already in [-1, 1]
        - volatility: clamped and scaled [0, 0.1] → [0, 1]
        - trend: clamped [-50, 50] → [-1, 1]
        - rsi: [0, 100] → [0, 1]
        - ml_confidence: already in [0, 1]
        """
        return np.array([
            np.clip(sentiment, -1.0, 1.0),
            np.clip(volatility / 0.1, 0.0, 1.0),
            np.clip(trend / 50.0, -1.0, 1.0),
            np.clip(rsi / 100.0, 0.0, 1.0),
            np.clip(ml_confidence, 0.0, 1.0),
        ], dtype=np.float32)

    def get_state(self, sentiment: float = 0.0, volatility: float = 0.0,
                  ml_confidence: float = 0.5, rsi: float = 50.0,
                  trend: float = 0.0) -> str:
        """Build state and return string key (for interface compatibility).

        Also caches the continuous vector for use by get_action/get_confidence.
        The returned string is for logging — the network uses the vector.
        """
        self._last_state_vec = self._encode_state(
            sentiment, volatility, trend, rsi, ml_confidence
        )
        # Return a readable string (for logs and backward compat)
        return (f"{sentiment:.2f}_{volatility:.4f}_{trend:.1f}"
                f"_{rsi:.1f}_{ml_confidence:.2f}")

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def get_action(self, state: str) -> float:
        """Get position size multiplier using epsilon-greedy DQN policy."""
        if self._last_state_vec is None:
            return 1.0  # Default

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            action_idx = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(self._last_state_vec).unsqueeze(0)
                q_values = self.policy_net(state_t)
                action_idx = q_values.argmax(dim=1).item()

        return self.actions[action_idx]

    def get_confidence(self, state: str, action_type: str = "BUY") -> float:
        """Get confidence for an action in current state (0 to 1).

        Uses the spread of Q-values: if one action dominates, confidence is high.
        If all actions have similar Q-values, confidence is ~0.5 (uncertain).
        """
        if self._last_state_vec is None:
            return 0.5

        with torch.no_grad():
            state_t = torch.FloatTensor(self._last_state_vec).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze()

            # Softmax to get action probabilities
            probs = torch.softmax(q_values * 2.0, dim=0)  # Temperature=0.5
            max_prob = probs.max().item()

            # Scale: uniform (0.2) → 0.5, dominant (1.0) → 1.0
            confidence = 0.5 + (max_prob - 1.0 / self.action_dim) * 0.625
            return float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update_q_value(self, state, action, reward: float, next_state,
                       done: bool = False, symbol: str = None):
        """Store transition and train the DQN.

        Accepts both old string-format and new vector states for compatibility.
        If symbol is provided, applies exploration bonus for less-traded coins.
        """
        # Parse state vectors
        state_vec = self._parse_state(state)
        next_state_vec = self._parse_state(next_state)

        # Map action to index
        action_idx = self._action_to_idx(action)

        # Reward shaping: amplify significant outcomes
        shaped_reward = reward
        if abs(reward) > 0.5:
            shaped_reward = reward * 1.5
        elif abs(reward) < 0.1:
            shaped_reward = reward * 0.5

        # Exploration bonus for less-traded coins (incentivize learning new coins)
        if symbol:
            coin_trade_count = self.coin_stats.get(symbol, {}).get("trades", 0)
            if coin_trade_count < self.exploration_bonus_trades:
                # Small bonus that decays as we trade more of this coin
                exploration_bonus = 0.1 * (1 - coin_trade_count / self.exploration_bonus_trades)
                shaped_reward += exploration_bonus

        # Store experience
        self.replay_buffer.push(
            state_vec, action_idx, shaped_reward, next_state_vec, float(done)
        )

        # Train if we have enough experience
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()

        # Track stats
        self.cumulative_reward += reward
        self.total_updates += 1

        # Decay exploration
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

        # Sync target network periodically
        if self.total_updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_step(self):
        """One gradient step on a batch from replay buffer."""
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Current Q-values for chosen actions
        q_values = self.policy_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN: use policy net to select, target to evaluate)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states)
            next_q_selected = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.discount_factor * next_q_selected * (1 - dones)

        # Huber loss (smooth L1) — more robust to outliers than MSE
        loss = nn.SmoothL1Loss()(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _parse_state(self, state) -> np.ndarray:
        """Convert various state formats to numpy vector."""
        if isinstance(state, np.ndarray):
            return state
        if isinstance(state, (list, tuple)):
            return np.array(state, dtype=np.float32)
        if isinstance(state, str):
            # Try to parse "val_val_val_val_val" format
            try:
                parts = state.split("_")
                if len(parts) == 5:
                    vals = [float(p) for p in parts]
                    # Check if these are the old discretized values (-1,0,1 etc.)
                    if all(abs(v) <= 2 for v in vals):
                        # Old format: re-normalize from discrete buckets
                        return np.array([
                            vals[0] / 1.0,      # regime: -1/0/1
                            vals[1] / 2.0,       # vol: 0/1/2
                            vals[2] / 1.0,       # trend: -1/0/1
                            vals[3] / 2.0,       # rsi: 0/1/2
                            vals[4] / 2.0,       # ml: 0/1/2
                        ], dtype=np.float32)
                    else:
                        # New continuous format — already normalized
                        return self._encode_state(vals[0], vals[1], vals[2],
                                                  vals[3], vals[4])
            except (ValueError, IndexError):
                pass
        # Fallback: neutral state
        return np.array([0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float32)

    def _action_to_idx(self, action) -> int:
        """Map action value to index."""
        if isinstance(action, str):
            return 2  # Default to 1.0 multiplier
        try:
            action_f = float(action)
            closest_idx = min(range(len(self.actions)),
                              key=lambda i: abs(self.actions[i] - action_f))
            return closest_idx
        except (ValueError, TypeError):
            return 2

    # ------------------------------------------------------------------
    # Trade logging & stats (unchanged interface)
    # ------------------------------------------------------------------
    def log_trade(self, symbol: str, side: str, pnl: float,
                  position_multiplier: float = 1.0):
        """Log trade result for analysis."""
        self.recent_trades.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "pnl": pnl,
            "multiplier": position_multiplier,
        })
        if len(self.recent_trades) > self.max_recent:
            self.recent_trades.pop(0)

        # Update per-coin stats
        self._update_coin_stats(symbol, pnl)

    def _update_coin_stats(self, symbol: str, pnl: float):
        """Track per-coin cumulative reward and trade count."""
        if symbol not in self.coin_stats:
            self.coin_stats[symbol] = {"cumulative_pnl": 0.0, "trades": 0, "wins": 0}
        
        self.coin_stats[symbol]["cumulative_pnl"] += pnl
        self.coin_stats[symbol]["trades"] += 1
        if pnl > 0:
            self.coin_stats[symbol]["wins"] += 1

    def get_coin_stats(self, symbol: str = None) -> Dict:
        """Get per-coin performance stats. If symbol is None, returns all."""
        if symbol:
            stats = self.coin_stats.get(symbol, {"cumulative_pnl": 0.0, "trades": 0, "wins": 0})
            stats["win_rate"] = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 50.0
            return stats
        # Return all with computed win rates
        result = {}
        for sym, stats in self.coin_stats.items():
            result[sym] = {
                **stats,
                "win_rate": (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 50.0
            }
        return result

    def get_win_rate(self) -> float:
        """Calculate win rate from recent trades."""
        if not self.recent_trades:
            return 50.0
        profitable = sum(1 for t in self.recent_trades if t.get("pnl", 0) > 0)
        return profitable / len(self.recent_trades) * 100

    def get_avg_win_loss(self) -> Tuple[float, float]:
        """Get average win % and average loss %."""
        wins = [t["pnl"] for t in self.recent_trades if t.get("pnl", 0) > 0]
        losses = [abs(t["pnl"]) for t in self.recent_trades
                  if t.get("pnl", 0) < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.02
        avg_loss = sum(losses) / len(losses) if losses else 0.02
        return avg_win, avg_loss

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            "type": "DQN",
            "replay_buffer_size": len(self.replay_buffer),
            "total_updates": self.total_updates,
            "exploration_rate": round(self.exploration_rate, 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "recent_trades": len(self.recent_trades),
            "win_rate": round(self.get_win_rate(), 1),
            "coins_tracked": len(self.coin_stats),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_agent(self, filepath: str = "data/state/rl_agent.json"):
        """Save DQN agent state: network weights + replay buffer + stats."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save network weights as a .pt file alongside the JSON
            pt_path = filepath.replace('.json', '_dqn.pt')
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, pt_path)

            # Save metadata + replay buffer as JSON
            data = {
                "agent_type": "DQN",
                "cumulative_reward": self.cumulative_reward,
                "episode_count": self.episode_count,
                "total_updates": self.total_updates,
                "exploration_rate": self.exploration_rate,
                "recent_trades": self.recent_trades[-50:],
                "replay_buffer": self.replay_buffer.to_list()[-5000:],
                "coin_stats": self.coin_stats,
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            pass

    def load_agent(self, filepath: str = "data/state/rl_agent.json"):
        """Load DQN agent state from disk. Handles both old Q-table and new DQN format."""
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.cumulative_reward = data.get("cumulative_reward", 0.0)
            self.episode_count = data.get("episode_count", 0)
            self.total_updates = data.get("total_updates", 0)
            self.exploration_rate = data.get("exploration_rate", 0.15)
            self.recent_trades = data.get("recent_trades", [])
            self.coin_stats = data.get("coin_stats", {})

            if data.get("agent_type") == "DQN":
                # Load DQN weights
                pt_path = filepath.replace('.json', '_dqn.pt')
                if os.path.exists(pt_path):
                    checkpoint = torch.load(pt_path, weights_only=True)
                    self.policy_net.load_state_dict(checkpoint['policy_net'])
                    self.target_net.load_state_dict(checkpoint['target_net'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])

                # Load replay buffer
                if "replay_buffer" in data:
                    self.replay_buffer.from_list(data["replay_buffer"])

            else:
                # Migrating from old Q-table agent
                # Seed replay buffer with synthetic experiences from Q-table
                q_table = data.get("q_table", {})
                self._migrate_q_table(q_table)

        except Exception:
            pass

    def _migrate_q_table(self, q_table: Dict):
        """Convert old Q-table knowledge into replay buffer experiences.

        For each state-action pair with a non-zero Q-value, create a
        synthetic experience so the DQN can learn from the Q-table's
        accumulated knowledge.
        """
        count = 0
        for state_key, action_values in q_table.items():
            state_vec = self._parse_state(state_key)
            for action_str, q_val in action_values.items():
                if abs(float(q_val)) < 0.001:
                    continue
                action_idx = self._action_to_idx(float(action_str))
                # Use Q-value as reward signal for migration
                self.replay_buffer.push(
                    state_vec, action_idx, float(q_val),
                    state_vec, 1.0  # done=True (terminal, no chaining)
                )
                count += 1

        if count > 0:
            # Do a few training passes to absorb the migrated knowledge
            for _ in range(min(count, 100)):
                if len(self.replay_buffer) >= self.batch_size:
                    self._train_step()
            self.target_net.load_state_dict(self.policy_net.state_dict())
