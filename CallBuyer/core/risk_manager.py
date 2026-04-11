"""
CallBuyer Risk Manager — Position sizing for call buying.
Small bets, convex payoff (lose small, win big).
"""
import json
import logging
import os
from datetime import date
from typing import Dict, Any

from core.config import CallBuyerConfig

log = logging.getLogger("callbuyer.risk")


class RiskManager:
    """Manages capital allocation and risk for call buying."""

    def __init__(self, config: CallBuyerConfig):
        self.config = config
        self.state = {
            "current_balance": config.ALLOCATION_PCT * 100000,
            "peak_balance": config.ALLOCATION_PCT * 100000,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "last_trade_date": None,
            "consecutive_losses": 0,
            "consecutive_wins": 0,
        }
        self._load_state()

    def _load_state(self):
        state_file = getattr(self.config, 'RISK_STATE_FILE', self.config.STATE_FILE)
        try:
            if os.path.exists(state_file):
                with open(state_file) as f:
                    saved = json.load(f)
                self.state.update(saved)
                today = date.today().isoformat()
                if self.state.get("last_trade_date") != today:
                    self.state["daily_pnl"] = 0.0
                    self.state["daily_trades"] = 0
                log.info(f"Loaded risk state: balance=${self.state['current_balance']:,.2f}")
        except Exception as e:
            log.warning(f"Could not load risk state: {e}")

    def save_state(self):
        state_file = getattr(self.config, 'RISK_STATE_FILE', self.config.STATE_FILE)
        try:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Failed to save risk state: {e}")

    def update_allocation(self, account_equity: float):
        # Guard against API returning $0 equity (transient error)
        if account_equity < 100:
            log.warning(f"Ignoring suspicious account equity ${account_equity:.2f} (too low, likely API error)")
            return
        allocation = account_equity * self.config.ALLOCATION_PCT
        self.state["current_balance"] = allocation
        if allocation > self.state.get("peak_balance", 0):
            self.state["peak_balance"] = allocation

    def get_max_risk_per_trade(self) -> float:
        """Max capital for a single call purchase (premium cost)."""
        return self.state["current_balance"] * self.config.MAX_POSITION_RISK_PCT

    def can_open_position(self, current_positions: int,
                          underlying: str,
                          positions: Dict[str, Any]) -> tuple:
        if current_positions >= self.config.MAX_POSITIONS:
            return False, f"max positions reached ({self.config.MAX_POSITIONS})"

        underlying_count = sum(
            1 for p in positions.values()
            if p.get("underlying") == underlying
        )
        if underlying_count >= self.config.MAX_PER_UNDERLYING:
            return False, f"max {self.config.MAX_PER_UNDERLYING} per underlying ({underlying})"

        # Daily loss limit: -5% of allocation
        daily_limit = self.state["current_balance"] * -0.05
        if self.state["daily_pnl"] <= daily_limit:
            return False, f"daily loss limit hit (${self.state['daily_pnl']:+,.2f})"

        # Consecutive losses: pause after 5
        if self.state["consecutive_losses"] >= 5:
            return False, f"consecutive losses: {self.state['consecutive_losses']} — cooling off"

        # Max 8 trades per day
        if self.state.get("daily_trades", 0) >= 8:
            return False, "max daily trades reached (8)"

        return True, "OK"

    def size_position(self, premium_per_contract: float = 0,
                      current_capital_in_use: float = 0,
                      allocation: float = None,
                      option_price: float = None) -> int:
        """Calculate number of contracts to buy.
        For call buying, risk = premium paid (max loss = premium).
        Accepts either positional args or keyword allocation/option_price.
        """
        # Support keyword args from call_engine
        if option_price is not None:
            premium_per_contract = option_price * 100  # per-contract cost
        if allocation is not None:
            current_capital_in_use = max(0, self.state["current_balance"] - allocation)
        if premium_per_contract <= 0:
            return 0

        available = self.state["current_balance"] - current_capital_in_use
        max_risk = min(
            self.get_max_risk_per_trade(),
            available * 0.20,  # max 20% of remaining in one trade
        )

        contracts = int(max_risk / premium_per_contract)
        # Call buying: keep positions small (1-5 contracts typically)
        return min(max(contracts, 0), 5)

    def can_trade(self) -> bool:
        """Quick check: can we open any new trades right now?"""
        daily_limit = self.state["current_balance"] * -0.05
        if self.state["daily_pnl"] <= daily_limit:
            return False
        if self.state["consecutive_losses"] >= 5:
            return False
        if self.state.get("daily_trades", 0) >= 8:
            return False
        return True

    @property
    def daily_trades(self) -> int:
        return self.state.get("daily_trades", 0)

    @property
    def daily_pnl_pct(self) -> float:
        bal = self.state.get("current_balance", 1)
        return (self.state.get("daily_pnl", 0) / bal * 100) if bal else 0.0

    def record_trade(self, pnl: float = 0.0, symbol: str = "",
                     won: bool = None):
        """Record a trade outcome.
        Accepts either pnl/symbol (PutSeller style) or won= (CallBuyer style).
        """
        today = date.today().isoformat()
        self.state["last_trade_date"] = today

        # If won= passed but no pnl, just count the result
        if won is None and pnl == 0.0:
            # Trade opened, outcome pending — just bump daily count
            self.state["daily_trades"] = self.state.get("daily_trades", 0) + 1
            self.save_state()
            return

        self.state["total_trades"] += 1
        self.state["total_pnl"] += pnl
        self.state["daily_pnl"] += pnl
        self.state["daily_trades"] = self.state.get("daily_trades", 0) + 1

        # Determine outcome
        trade_won = won if won is not None else (pnl >= 0)
        if trade_won:
            self.state["wins"] += 1
            self.state["consecutive_losses"] = 0
            self.state["consecutive_wins"] = self.state.get("consecutive_wins", 0) + 1
        else:
            self.state["losses"] += 1
            self.state["consecutive_losses"] += 1
            self.state["consecutive_wins"] = 0

        win_rate = (self.state["wins"] / self.state["total_trades"] * 100
                    if self.state["total_trades"] > 0 else 0)

        log.info(f"Trade recorded: {symbol or 'CALL'} {'WIN' if trade_won else 'LOSS'} | "
                 f"PnL ${pnl:+,.2f} | Daily ${self.state['daily_pnl']:+,.2f} | "
                 f"WR: {win_rate:.0f}% | "
                 f"Streak: {'W' if trade_won else 'L'}{max(self.state['consecutive_wins'], self.state['consecutive_losses'])}")
        self.save_state()

    @property
    def consecutive_losses(self) -> int:
        return self.state.get("consecutive_losses", 0)

    def rolling_loss_rate(self) -> float:
        """Approximate rolling loss rate."""
        total = self.state["total_trades"]
        if total < 5:
            return 0.0
        return self.state["losses"] / total

    def get_stats(self) -> Dict[str, Any]:
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
            "daily_trades": self.state.get("daily_trades", 0),
            "consecutive_losses": self.state["consecutive_losses"],
        }
