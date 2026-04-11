"""
PutSeller Risk Manager — Position sizing and risk limits.
Isolated from AlpacaBot's risk_manager.
"""
import json
import logging
import os
from datetime import datetime, date
from typing import Dict, Any, Optional

from core.config import PutSellerConfig

log = logging.getLogger("putseller.risk")


class RiskManager:
    """Manages capital allocation, position sizing, and risk tracking."""

    def __init__(self, config: PutSellerConfig):
        self.config = config
        self.state = {
            "current_balance": config.ALLOCATION_PCT * 100000,  # default, updated on connect
            "peak_balance": config.ALLOCATION_PCT * 100000,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "last_trade_date": None,
            "consecutive_losses": 0,
        }
        self._load_state()

    def _load_state(self):
        """Load risk state from disk."""
        try:
            if os.path.exists(self.config.STATE_FILE):
                with open(self.config.STATE_FILE) as f:
                    saved = json.load(f)
                self.state.update(saved)

                # Reset daily PnL if new day
                today = date.today().isoformat()
                if self.state.get("last_trade_date") != today:
                    self.state["daily_pnl"] = 0.0

                log.info(f"Loaded risk state: balance=${self.state['current_balance']:,.2f}, "
                         f"daily=${self.state['daily_pnl']:+,.2f}, "
                         f"streak={self.state['consecutive_losses']}")
        except Exception as e:
            log.warning(f"Could not load risk state: {e}")

    def save_state(self):
        """Persist risk state to disk."""
        try:
            os.makedirs(os.path.dirname(self.config.STATE_FILE), exist_ok=True)
            with open(self.config.STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Failed to save risk state: {e}")

    def update_allocation(self, account_equity: float):
        """Update balance based on current account equity and allocation %."""
        # Guard against API returning $0 equity (transient error)
        if account_equity < 100:
            log.warning(f"Ignoring suspicious account equity ${account_equity:.2f} (too low, likely API error)")
            return
        allocation = account_equity * self.config.ALLOCATION_PCT
        self.state["current_balance"] = allocation
        if allocation > self.state.get("peak_balance", 0):
            self.state["peak_balance"] = allocation

    def get_max_risk_per_trade(self) -> float:
        """Max capital at risk for a single spread."""
        return self.state["current_balance"] * self.config.MAX_POSITION_RISK_PCT

    def can_open_position(self, current_positions: int,
                          underlying: str,
                          positions: Dict[str, Any],
                          spread_type: str = "put") -> tuple:
        """Check if we can open a new position.

        Checks put and call limits INDEPENDENTLY per tastylive iron condor
        mechanics — each side is managed separately so one side's limits
        don't block the other.

        Returns (can_open: bool, reason: str)
        """
        # Reset daily counters if new day (runtime reset)
        today = date.today().isoformat()
        if self.state.get("last_trade_date") and self.state["last_trade_date"] != today:
            self.state["daily_pnl"] = 0.0

        # ── Risk utilization cap: block if total risk > 85% of balance ──
        total_risk = sum(p.get("max_loss_total", 0) for p in positions.values())
        max_allowed_risk = self.state["current_balance"] * 0.85
        if total_risk >= max_allowed_risk:
            return False, f"risk utilization {total_risk/self.state['current_balance']*100:.0f}% >= 85% cap"

        # Separate put/call position counts
        put_count = sum(1 for p in positions.values() if p.get("spread_type") != "call")
        call_count = sum(1 for p in positions.values() if p.get("spread_type") == "call")

        if spread_type == "call":
            if call_count >= self.config.MAX_CALL_POSITIONS:
                return False, f"max call positions reached ({self.config.MAX_CALL_POSITIONS})"
        else:
            if put_count >= self.config.MAX_POSITIONS:
                return False, f"max put positions reached ({self.config.MAX_POSITIONS})"

        # Total position hard cap (puts + calls)
        total_cap = self.config.MAX_POSITIONS + self.config.MAX_CALL_POSITIONS
        if current_positions >= total_cap:
            return False, f"max total positions reached ({total_cap})"

        # Leveraged/inverse ETF exposure cap
        if underlying in self.config.LEVERAGED_ETFS:
            lev_count = sum(
                1 for p in positions.values()
                if p.get("underlying") in self.config.LEVERAGED_ETFS
            )
            if lev_count >= self.config.MAX_LEVERAGED_POSITIONS:
                return False, f"max leveraged ETF positions reached ({self.config.MAX_LEVERAGED_POSITIONS})"

        # Per-underlying limit
        underlying_count = sum(
            1 for p in positions.values()
            if p.get("underlying") == underlying
        )
        if underlying_count >= self.config.MAX_PER_UNDERLYING:
            return False, f"max {self.config.MAX_PER_UNDERLYING} positions per underlying ({underlying})"

        # Daily loss limit: -3% of allocation
        # Guard: if balance is near-zero (API glitch), skip this check
        if self.state["current_balance"] < 100:
            return False, "balance too low — possible API error, blocking new positions"
        daily_limit = self.state["current_balance"] * -0.03
        if self.state["daily_pnl"] <= daily_limit:
            return False, f"daily loss limit hit (${self.state['daily_pnl']:+,.2f})"

        # Consecutive losses: check per-side so put losses don't freeze calls
        side_key = f"consecutive_losses_{spread_type}"
        side_losses = self.state.get(side_key, self.state["consecutive_losses"])
        if side_losses >= 4:
            return False, f"consecutive {spread_type} losses: {side_losses} — cooling off"

        return True, "OK"

    def size_position(self, max_loss_per_contract: float,
                      current_capital_in_use: float) -> int:
        """Calculate number of contracts for a spread.

        max_loss_per_contract: spread_width - credit (per share) × 100
        Returns number of contracts (spread sets).
        """
        if max_loss_per_contract <= 0:
            return 0

        available = self.state["current_balance"] - current_capital_in_use
        max_risk = min(
            self.get_max_risk_per_trade(),
            available * 0.25,  # never more than 25% of remaining
        )

        contracts = int(max_risk / max_loss_per_contract)
        return max(contracts, 0)

    def record_trade(self, pnl: float, symbol: str, spread_type: str = "put"):
        """Record a completed trade with per-side streak tracking."""
        today = date.today().isoformat()
        self.state["total_trades"] += 1
        self.state["total_pnl"] += pnl
        self.state["daily_pnl"] += pnl
        self.state["last_trade_date"] = today

        side_key = f"consecutive_losses_{spread_type}"

        if pnl >= 0:
            self.state["wins"] += 1
            self.state["consecutive_losses"] = 0
            self.state[side_key] = 0
        else:
            self.state["losses"] += 1
            self.state["consecutive_losses"] += 1
            self.state[side_key] = self.state.get(side_key, 0) + 1

        win_rate = (self.state["wins"] / self.state["total_trades"] * 100
                    if self.state["total_trades"] > 0 else 0)

        log.info(f"Trade recorded: {symbol} ({spread_type}) PnL ${pnl:+,.2f} | "
                 f"Daily ${self.state['daily_pnl']:+,.2f} | "
                 f"Streak: {self.state['consecutive_losses']} ({spread_type}: {self.state.get(side_key, 0)}) | "
                 f"WR: {win_rate:.0f}%")

        self.save_state()

    def get_stats(self) -> Dict[str, Any]:
        """Return current risk/performance stats."""
        total = self.state["total_trades"]
        return {
            "balance": self.state["current_balance"],
            "peak_balance": self.state["peak_balance"],
            "total_trades": total,
            "wins": self.state["wins"],
            "losses": self.state["losses"],
            "win_rate": (self.state["wins"] / total * 100) if total > 0 else 0,
            "total_pnl": self.state["total_pnl"],
            "daily_pnl": self.state["daily_pnl"],
            "consecutive_losses": self.state["consecutive_losses"],
        }
