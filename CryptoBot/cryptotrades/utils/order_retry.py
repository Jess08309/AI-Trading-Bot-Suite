"""
Order Retry Manager - exponential backoff + halt for order cancel/reject storms.

Addresses the "8 HYPEUSD market buys cancelled in 13 min" failure mode: the
bot retried blindly every ~2 minutes with no backoff and no circuit-breaker
on the retry loop itself. This tracks consecutive cancel/reject events per
symbol, hands back an exponentially increasing delay before the next retry,
and halts (cooldown) the symbol entirely after `max_retries` consecutive
failures.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Optional


class OrderRetryManager:
    """Tracks per-symbol order failure streaks and enforces backoff/halt."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        cooldown_minutes: float = 30.0,
        clock=datetime.now,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.cooldown_minutes = cooldown_minutes
        self._clock = clock

        self._attempts: Dict[str, int] = {}
        self._halted_until: Dict[str, datetime] = {}

    def backoff_delay(self, attempt: int) -> float:
        """Delay (seconds) to wait before the given (1-indexed) retry attempt."""
        if attempt <= 0:
            return 0.0
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)

    def is_halted(self, symbol: str) -> bool:
        """Whether a symbol is currently in its post-failure cooldown."""
        halted_until = self._halted_until.get(symbol)
        if halted_until is None:
            return False
        if self._clock() >= halted_until:
            # Cooldown expired — clear it and reset the streak.
            del self._halted_until[symbol]
            self._attempts[symbol] = 0
            return False
        return True

    def record_failure(self, symbol: str) -> Dict[str, object]:
        """Record a cancel/reject for `symbol`.

        Returns a dict with:
            attempts: consecutive failures so far (including this one)
            halted: True if this failure just tripped the halt
            delay: backoff delay (s) before the NEXT retry, 0 if halted
        """
        attempts = self._attempts.get(symbol, 0) + 1
        self._attempts[symbol] = attempts

        if attempts >= self.max_retries:
            self._halted_until[symbol] = self._clock() + timedelta(minutes=self.cooldown_minutes)
            return {"attempts": attempts, "halted": True, "delay": 0.0}

        return {"attempts": attempts, "halted": False, "delay": self.backoff_delay(attempts)}

    def record_success(self, symbol: str) -> None:
        """Reset the failure streak for `symbol` after a confirmed fill."""
        self._attempts[symbol] = 0
        self._halted_until.pop(symbol, None)

    def attempts(self, symbol: str) -> int:
        return self._attempts.get(symbol, 0)

    def halted_until(self, symbol: str) -> Optional[datetime]:
        return self._halted_until.get(symbol)
