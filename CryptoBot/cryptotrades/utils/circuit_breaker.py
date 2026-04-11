"""
Circuit Breaker - Safety mechanism to pause trading after consecutive losses.

Tracks:
- Consecutive losses (spot + futures separately)
- Daily P&L limits
- Portfolio drawdown from peak
- Cooldown periods after triggering
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple
import json
import os


class CircuitBreaker:
    """
    Prevents catastrophic loss streaks by pausing trading.

    Three independent triggers:
    1. Consecutive losses: N losses in a row → pause
    2. Daily loss limit: total daily P&L exceeds threshold → pause for the day
    3. Drawdown limit: portfolio drops X% from peak → pause until recovery

    Each trigger has its own cooldown period.
    """

    def __init__(
        self,
        max_consecutive_losses: int = 5,
        daily_loss_limit_pct: float = -5.0,
        max_drawdown_pct: float = -10.0,
        cooldown_minutes: int = 60,
        save_path: str = "data/state/circuit_breaker.json",
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_minutes = cooldown_minutes
        self.save_path = save_path

        # --- State ---
        # Consecutive loss tracking (separate for spot/futures)
        self.consecutive_losses: Dict[str, int] = {"spot": 0, "futures": 0}
        self.consecutive_wins: Dict[str, int] = {"spot": 0, "futures": 0}

        # Daily P&L
        self.daily_pnl: float = 0.0
        self.daily_trade_count: int = 0
        self.current_date: str = ""

        # Drawdown tracking (watermark)
        self.peak_balance: float = 0.0
        self.current_balance: float = 0.0

        # Cooldown state
        self.paused_until: Dict[str, Optional[str]] = {
            "spot": None,
            "futures": None,
        }
        self.pause_reason: Dict[str, str] = {"spot": "", "futures": ""}

        # History for analysis
        self.trigger_history: list = []

    def record_trade(self, market: str, profit_pct: float,
                     current_balance: float) -> Tuple[bool, str]:
        """Record a trade result and check if circuit breaker should trigger.

        Args:
            market: "spot" or "futures"
            profit_pct: Trade P&L as percentage
            current_balance: Current total portfolio balance

        Returns:
            (triggered, reason) - whether this trade caused a circuit break
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # Reset daily counters on new day
        if today != self.current_date:
            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trade_count = 0

        # Update daily stats
        self.daily_pnl += profit_pct
        self.daily_trade_count += 1

        # Update consecutive counters
        if profit_pct > 0:
            self.consecutive_losses[market] = 0
            self.consecutive_wins[market] = self.consecutive_wins.get(market, 0) + 1
        elif profit_pct < 0:
            self.consecutive_wins[market] = 0
            self.consecutive_losses[market] = self.consecutive_losses.get(market, 0) + 1

        # Update drawdown watermark
        self.current_balance = current_balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # --- CHECK TRIGGERS ---
        triggered = False
        reason = ""

        # 1. Consecutive losses
        if self.consecutive_losses[market] >= self.max_consecutive_losses:
            triggered = True
            reason = (f"CONSECUTIVE_LOSSES: {self.consecutive_losses[market]} "
                      f"{market} losses in a row (limit: {self.max_consecutive_losses})")

        # 2. Daily loss limit
        if self.daily_pnl <= self.daily_loss_limit_pct:
            triggered = True
            reason = (f"DAILY_LOSS_LIMIT: {self.daily_pnl:+.2f}% today "
                      f"(limit: {self.daily_loss_limit_pct}%)")

        # 3. Drawdown from peak
        if self.peak_balance > 0:
            drawdown = ((current_balance - self.peak_balance) / self.peak_balance) * 100
            if drawdown <= self.max_drawdown_pct:
                triggered = True
                reason = (f"MAX_DRAWDOWN: {drawdown:.1f}% from peak "
                          f"${self.peak_balance:,.2f} -> ${current_balance:,.2f} "
                          f"(limit: {self.max_drawdown_pct}%)")

        if triggered:
            pause_until = now + timedelta(minutes=self.cooldown_minutes)
            self.paused_until[market] = pause_until.isoformat()
            self.pause_reason[market] = reason
            self.trigger_history.append({
                "timestamp": now.isoformat(),
                "market": market,
                "reason": reason,
                "daily_pnl": self.daily_pnl,
                "consecutive_losses": self.consecutive_losses[market],
                "balance": current_balance,
            })
            # Keep last 50 triggers
            self.trigger_history = self.trigger_history[-50:]

        return triggered, reason

    def can_trade(self, market: str) -> Tuple[bool, str]:
        """Check if trading is allowed for a market.

        Returns:
            (allowed, reason) - whether trading is permitted
        """
        now = datetime.now(timezone.utc)

        # Check cooldown
        pause_str = self.paused_until.get(market)
        if pause_str:
            try:
                pause_time = datetime.fromisoformat(pause_str)
                if now < pause_time:
                    remaining = (pause_time - now).total_seconds() / 60
                    return False, (f"PAUSED ({remaining:.0f}min remaining): "
                                   f"{self.pause_reason.get(market, 'unknown')}")
                else:
                    # Cooldown expired — clear pause
                    self.paused_until[market] = None
                    self.pause_reason[market] = ""
            except (ValueError, TypeError):
                self.paused_until[market] = None

        # Check daily loss even outside cooldown

        # Reset daily counters if new day
        today = now.strftime("%Y-%m-%d")
        if today != self.current_date:
            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
        if self.daily_pnl <= self.daily_loss_limit_pct:
            return False, (f"DAILY_LIMIT_REACHED: {self.daily_pnl:+.2f}% "
                           f"(limit: {self.daily_loss_limit_pct}%)")

        return True, "ok"

    def get_status(self) -> Dict:
        """Get full circuit breaker status."""
        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = ((self.current_balance - self.peak_balance) /
                        self.peak_balance) * 100

        return {
            "consecutive_losses_spot": self.consecutive_losses.get("spot", 0),
            "consecutive_losses_futures": self.consecutive_losses.get("futures", 0),
            "consecutive_wins_spot": self.consecutive_wins.get("spot", 0),
            "consecutive_wins_futures": self.consecutive_wins.get("futures", 0),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trade_count,
            "peak_balance": round(self.peak_balance, 2),
            "current_balance": round(self.current_balance, 2),
            "drawdown_pct": round(drawdown, 2),
            "spot_paused": self.paused_until.get("spot") is not None,
            "futures_paused": self.paused_until.get("futures") is not None,
            "spot_pause_reason": self.pause_reason.get("spot", ""),
            "futures_pause_reason": self.pause_reason.get("futures", ""),
            "total_triggers": len(self.trigger_history),
        }

    def save_state(self):
        """Persist circuit breaker state."""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            data = {
                "consecutive_losses": self.consecutive_losses,
                "consecutive_wins": self.consecutive_wins,
                "daily_pnl": self.daily_pnl,
                "daily_trade_count": self.daily_trade_count,
                "current_date": self.current_date,
                "peak_balance": self.peak_balance,
                "current_balance": self.current_balance,
                "paused_until": self.paused_until,
                "pause_reason": self.pause_reason,
                "trigger_history": self.trigger_history[-20:],
            }
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load_state(self):
        """Load persisted circuit breaker state."""
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            self.consecutive_losses = data.get("consecutive_losses",
                                                {"spot": 0, "futures": 0})
            self.consecutive_wins = data.get("consecutive_wins",
                                              {"spot": 0, "futures": 0})
            self.daily_pnl = data.get("daily_pnl", 0.0)
            self.daily_trade_count = data.get("daily_trade_count", 0)
            self.current_date = data.get("current_date", "")
            self.peak_balance = data.get("peak_balance", 0.0)
            self.current_balance = data.get("current_balance", 0.0)
            self.paused_until = data.get("paused_until",
                                          {"spot": None, "futures": None})
            self.pause_reason = data.get("pause_reason",
                                          {"spot": "", "futures": ""})
            self.trigger_history = data.get("trigger_history", [])
        except Exception:
            pass

