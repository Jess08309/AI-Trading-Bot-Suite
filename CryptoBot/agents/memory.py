"""Agent Memory System — rolling history of recommendations vs. outcomes.

Gives each agent access to its own track record so it can learn from
recent accuracy. A decay-weighted scoring system emphasises the most
recent cycles.

Storage: C:/Bot/data/state/agent_memory.json
Format:
{
  "cycles": [                     # rolling deque, max MEMORY_CYCLES entries
    {
      "run_id": "abc12345",
      "timestamp": "2026-04-08T14:00:00Z",
      "signals": [
        {
          "symbol": "BTC/USD",
          "direction": "LONG",
          "ml_confidence": 0.72,
          "technical": {"verdict": "BUY", "confidence": 0.8},
          "sentiment": {"verdict": "HOLD", "confidence": 0.5},
          "risk":      {"verdict": "BUY", "confidence": 0.7},
          "orchestrator": {"action": "TAKE", "confidence": 0.75, "size_modifier": 1.0},
          "outcome": null | {"pnl_pct": 1.2, "result": "WIN", "hold_minutes": 45}
        }
      ]
    }
  ],
  "agent_accuracy": {             # auto-computed summary stats
    "technical":   {"correct": 12, "total": 18, "accuracy": 0.667},
    "sentiment":   {"correct": 9,  "total": 18, "accuracy": 0.500},
    "risk":        {"correct": 14, "total": 18, "accuracy": 0.778},
    "orchestrator": {"correct": 11, "total": 18, "accuracy": 0.611}
  },
  "symbol_history": {             # per-symbol hit rate
    "BTC/USD": {"wins": 3, "losses": 2, "total": 5}
  }
}
"""
from __future__ import annotations
import json, os, logging, time
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger("agent_advisor")

# ── Config ────────────────────────────────────────────────
MEMORY_CYCLES = 20           # Rolling window size
DECAY_HALF_LIFE = 8          # Half the weight after this many cycles
MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "state", "agent_memory.json"
)


def _decay_weight(age: int) -> float:
    """Exponential decay weight: newest=1.0, halves every DECAY_HALF_LIFE cycles."""
    return 2.0 ** (-age / DECAY_HALF_LIFE)


class AgentMemory:
    """Manages rolling cycle memory with outcome tracking and accuracy stats."""

    def __init__(self):
        self.cycles: List[Dict] = []
        self.agent_accuracy: Dict[str, Dict] = {}
        self.symbol_history: Dict[str, Dict] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────

    def _load(self):
        """Load memory from disk."""
        if not os.path.exists(MEMORY_FILE):
            logger.info("Agent memory: no existing file, starting fresh")
            return
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.cycles = data.get("cycles", [])[-MEMORY_CYCLES:]
            self.agent_accuracy = data.get("agent_accuracy", {})
            self.symbol_history = data.get("symbol_history", {})
            logger.info(
                f"Agent memory loaded: {len(self.cycles)} cycles, "
                f"{sum(sh.get('total', 0) for sh in self.symbol_history.values())} tracked trades"
            )
        except Exception as e:
            logger.warning(f"Agent memory load failed: {e}")

    def save(self):
        """Persist memory to disk."""
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        data = {
            "cycles": self.cycles[-MEMORY_CYCLES:],
            "agent_accuracy": self.agent_accuracy,
            "symbol_history": self.symbol_history,
        }
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Agent memory save failed: {e}")

    # ── Record a cycle ────────────────────────────────────

    def record_cycle(
        self,
        run_id: str,
        signals: List[Dict],
        technical_analyses: List[Dict],
        sentiment_analyses: List[Dict],
        risk_analyses: List[Dict],
        recommendations: List[Dict],
    ):
        """Record one agent evaluation cycle (before outcome is known)."""
        cycle_signals = []
        for i, sig in enumerate(signals):
            entry = {
                "symbol": sig.get("symbol", "?"),
                "direction": sig.get("direction", "?"),
                "ml_confidence": sig.get("confidence", 0),
            }
            # Agent verdicts
            if i < len(technical_analyses):
                ta = technical_analyses[i]
                entry["technical"] = {
                    "verdict": ta.get("verdict", "HOLD"),
                    "confidence": ta.get("confidence", 0.5),
                }
            if i < len(sentiment_analyses):
                sa = sentiment_analyses[i]
                entry["sentiment"] = {
                    "verdict": sa.get("verdict", "HOLD"),
                    "confidence": sa.get("confidence", 0.5),
                }
            if i < len(risk_analyses):
                ra = risk_analyses[i]
                entry["risk"] = {
                    "verdict": ra.get("verdict", "HOLD"),
                    "confidence": ra.get("confidence", 0.5),
                }
            # Orchestrator recommendation
            rec = next(
                (r for r in recommendations if r.get("symbol") == sig.get("symbol")),
                None,
            )
            if rec:
                entry["orchestrator"] = {
                    "action": rec.get("action", "SKIP"),
                    "confidence": rec.get("confidence", 0),
                    "size_modifier": rec.get("size_modifier", 1.0),
                }
            entry["outcome"] = None  # filled in later by record_outcome
            cycle_signals.append(entry)

        cycle_entry = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": cycle_signals,
        }
        self.cycles.append(cycle_entry)
        # Trim to rolling window
        if len(self.cycles) > MEMORY_CYCLES:
            self.cycles = self.cycles[-MEMORY_CYCLES:]
        self.save()

    # ── Record trade outcomes ─────────────────────────────

    def record_outcome(self, symbol: str, pnl_pct: float, hold_minutes: float = 0):
        """Record the outcome of a closed trade, matched to the most recent
        cycle entry for that symbol that has no outcome yet."""
        result = "WIN" if pnl_pct > 0 else "LOSS"

        # Find most recent unresolved entry for this symbol
        for cycle in reversed(self.cycles):
            for sig_entry in cycle.get("signals", []):
                if sig_entry.get("symbol") == symbol and sig_entry.get("outcome") is None:
                    sig_entry["outcome"] = {
                        "pnl_pct": round(pnl_pct, 4),
                        "result": result,
                        "hold_minutes": round(hold_minutes, 1),
                    }
                    # Update agent accuracy stats
                    self._update_accuracy(sig_entry, result)
                    # Update symbol history
                    self._update_symbol_history(symbol, result)
                    self.save()
                    logger.info(
                        f"Agent memory: recorded {result} for {symbol} "
                        f"(pnl={pnl_pct:+.2f}%, hold={hold_minutes:.0f}m)"
                    )
                    return

    def _update_accuracy(self, sig_entry: Dict, result: str):
        """Update per-agent accuracy based on whether the agent's verdict
        aligned with the actual outcome."""
        is_win = result == "WIN"
        direction = sig_entry.get("direction", "LONG")

        for agent_name in ("technical", "sentiment", "risk"):
            agent_data = sig_entry.get(agent_name)
            if not agent_data:
                continue
            verdict = agent_data.get("verdict", "HOLD")
            # Did the agent correctly predict the outcome?
            bullish_verdicts = {"STRONG_BUY", "BUY"}
            bearish_verdicts = {"STRONG_SELL", "SELL"}

            if direction == "LONG":
                correct = (verdict in bullish_verdicts and is_win) or \
                          (verdict in bearish_verdicts and not is_win)
            else:  # SHORT
                correct = (verdict in bearish_verdicts and is_win) or \
                          (verdict in bullish_verdicts and not is_win)
            # HOLD is neutral — count as correct if trade was marginal (|pnl| < 0.5%)
            if verdict == "HOLD":
                pnl = abs(sig_entry.get("outcome", {}).get("pnl_pct", 0))
                correct = pnl < 0.5

            stats = self.agent_accuracy.setdefault(
                agent_name, {"correct": 0, "total": 0, "accuracy": 0.0}
            )
            stats["total"] += 1
            if correct:
                stats["correct"] += 1
            stats["accuracy"] = round(stats["correct"] / max(stats["total"], 1), 3)

        # Orchestrator accuracy: did TAKE result in WIN, did SKIP avoid a LOSS?
        orch = sig_entry.get("orchestrator")
        if orch:
            action = orch.get("action", "SKIP")
            orch_correct = (action == "TAKE" and is_win) or (action == "SKIP" and not is_win)
            stats = self.agent_accuracy.setdefault(
                "orchestrator", {"correct": 0, "total": 0, "accuracy": 0.0}
            )
            stats["total"] += 1
            if orch_correct:
                stats["correct"] += 1
            stats["accuracy"] = round(stats["correct"] / max(stats["total"], 1), 3)

    def _update_symbol_history(self, symbol: str, result: str):
        """Track per-symbol win/loss history."""
        sh = self.symbol_history.setdefault(symbol, {"wins": 0, "losses": 0, "total": 0})
        sh["total"] += 1
        if result == "WIN":
            sh["wins"] += 1
        else:
            sh["losses"] += 1

    # ── Generate context for agent prompts ─────────────────

    def get_agent_context(self, agent_name: str) -> str:
        """Build a concise memory summary for injection into an agent's prompt.

        Returns a formatted string like:
        AGENT MEMORY (last 20 cycles):
        - Your accuracy: 66.7% (12/18 correct)
        - Recent calls: BUY BTC ✓, SELL ETH ✗, BUY SOL ✓ ...
        - Best symbol: SOL (80% WR), Worst: DOGE (20% WR)
        """
        if not self.cycles:
            return "AGENT MEMORY: No history yet — this is your first cycle."

        lines = [f"AGENT MEMORY (last {len(self.cycles)} cycles):"]

        # Agent accuracy
        acc = self.agent_accuracy.get(agent_name, {})
        if acc.get("total", 0) > 0:
            lines.append(
                f"- Your accuracy: {acc['accuracy']:.0%} "
                f"({acc['correct']}/{acc['total']} correct)"
            )
        else:
            lines.append("- Your accuracy: no resolved trades yet")

        # Recent calls with outcomes (decay-weighted)
        recent_calls = []
        for idx, cycle in enumerate(reversed(self.cycles)):
            age = idx
            weight = _decay_weight(age)
            for sig in cycle.get("signals", []):
                agent_data = sig.get(agent_name, sig.get("orchestrator") if agent_name == "orchestrator" else None)
                if not agent_data:
                    continue
                outcome = sig.get("outcome")
                sym = sig.get("symbol", "?")
                if agent_name == "orchestrator":
                    verdict_str = agent_data.get("action", "?")
                else:
                    verdict_str = agent_data.get("verdict", "?")

                if outcome:
                    result_icon = "✓" if outcome["result"] == "WIN" else "✗"
                    pnl_str = f"{outcome['pnl_pct']:+.1f}%"
                    recent_calls.append(f"{verdict_str} {sym} {result_icon} ({pnl_str}, wt={weight:.2f})")
                else:
                    recent_calls.append(f"{verdict_str} {sym} ⏳ (pending, wt={weight:.2f})")

        if recent_calls:
            lines.append(f"- Recent calls: {', '.join(recent_calls[:10])}")

        # Symbol performance summary
        if self.symbol_history:
            sorted_syms = sorted(
                self.symbol_history.items(),
                key=lambda x: x[1].get("wins", 0) / max(x[1].get("total", 1), 1),
                reverse=True,
            )
            if len(sorted_syms) >= 2:
                best = sorted_syms[0]
                worst = sorted_syms[-1]
                best_wr = best[1]["wins"] / max(best[1]["total"], 1)
                worst_wr = worst[1]["wins"] / max(worst[1]["total"], 1)
                lines.append(
                    f"- Best: {best[0]} ({best_wr:.0%} WR, {best[1]['total']} trades) | "
                    f"Worst: {worst[0]} ({worst_wr:.0%} WR, {worst[1]['total']} trades)"
                )

        # Decay-weighted win rate
        weighted_wins = 0.0
        weighted_total = 0.0
        for idx, cycle in enumerate(reversed(self.cycles)):
            age = idx
            weight = _decay_weight(age)
            for sig in cycle.get("signals", []):
                outcome = sig.get("outcome")
                if outcome:
                    weighted_total += weight
                    if outcome["result"] == "WIN":
                        weighted_wins += weight
        if weighted_total > 0:
            weighted_wr = weighted_wins / weighted_total
            lines.append(f"- Decay-weighted win rate: {weighted_wr:.0%} (recent cycles weighted higher)")

        return "\n".join(lines)

    def get_symbol_context(self, symbol: str) -> str:
        """Get memory context for a specific symbol."""
        sh = self.symbol_history.get(symbol)
        if not sh or sh.get("total", 0) == 0:
            return f"No agent memory for {symbol} yet."

        wr = sh["wins"] / max(sh["total"], 1)
        # Recent trades for this symbol
        recent = []
        for cycle in reversed(self.cycles[-10:]):
            for sig in cycle.get("signals", []):
                if sig.get("symbol") == symbol and sig.get("outcome"):
                    o = sig["outcome"]
                    recent.append(f"{o['result']} ({o['pnl_pct']:+.1f}%)")

        line = f"{symbol}: {wr:.0%} WR ({sh['wins']}W/{sh['losses']}L)"
        if recent:
            line += f" | Recent: {', '.join(recent[:5])}"
        return line
