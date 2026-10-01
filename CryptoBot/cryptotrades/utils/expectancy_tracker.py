"""
Symbol Expectancy Tracker.

Tracks round-trip P&L per symbol and auto-disables a symbol from new
entries when either:
  - It has 3 consecutive losing round trips, or
  - Its trailing 20-trade expectancy (average P&L %) is negative.

This targets the LINK (3 losses of 4 trips) and LTC (3 of 3 losses)
patterns called out in the problem statement, where the bot kept
re-entering a symbol with a clearly negative edge.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Deque, Dict, Optional, Tuple
from collections import deque


class SymbolExpectancyTracker:
    def __init__(
        self,
        max_consecutive_losses: int = 3,
        expectancy_window: int = 20,
        save_path: str = "data/state/symbol_expectancy.json",
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.expectancy_window = expectancy_window
        self.save_path = save_path

        self._history: Dict[str, Deque[float]] = {}
        self._consecutive_losses: Dict[str, int] = {}
        self._disabled: Dict[str, str] = {}  # symbol -> reason

    def record_round_trip(self, symbol: str, pnl_pct: float, timestamp: Optional[str] = None) -> Tuple[bool, str]:
        """Record a closed round trip's P&L percentage for `symbol`.

        Returns (just_disabled, reason). `reason` is "" if not disabled.
        """
        history = self._history.setdefault(symbol, deque(maxlen=self.expectancy_window))
        history.append(pnl_pct)

        if pnl_pct < 0:
            self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
        elif pnl_pct > 0:
            self._consecutive_losses[symbol] = 0
        # pnl_pct == 0 (breakeven) does not reset or extend the loss streak.

        reason = ""
        if self._consecutive_losses.get(symbol, 0) >= self.max_consecutive_losses:
            reason = (
                f"CONSECUTIVE_LOSSES: {self._consecutive_losses[symbol]} losing round "
                f"trips in a row (limit {self.max_consecutive_losses})"
            )
        elif len(history) >= self.expectancy_window:
            expectancy = sum(history) / len(history)
            if expectancy < 0:
                reason = (
                    f"NEGATIVE_EXPECTANCY: {expectancy:.3f}% avg P&L over last "
                    f"{len(history)} trades"
                )

        just_disabled = bool(reason) and symbol not in self._disabled
        if reason:
            self._disabled[symbol] = reason
        return just_disabled, reason

    def is_disabled(self, symbol: str) -> Tuple[bool, str]:
        reason = self._disabled.get(symbol, "")
        return bool(reason), reason

    def re_enable(self, symbol: str) -> None:
        """Manually clear a symbol's disabled state (e.g. after a cooldown)."""
        self._disabled.pop(symbol, None)
        self._consecutive_losses[symbol] = 0

    def get_expectancy(self, symbol: str) -> Optional[float]:
        history = self._history.get(symbol)
        if not history:
            return None
        return sum(history) / len(history)

    def get_consecutive_losses(self, symbol: str) -> int:
        return self._consecutive_losses.get(symbol, 0)

    def save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            data = {
                "history": {s: list(h) for s, h in self._history.items()},
                "consecutive_losses": self._consecutive_losses,
                "disabled": self._disabled,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load_state(self) -> None:
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)
            self._history = {
                s: deque(v, maxlen=self.expectancy_window)
                for s, v in data.get("history", {}).items()
            }
            self._consecutive_losses = data.get("consecutive_losses", {})
            self._disabled = data.get("disabled", {})
        except Exception:
            pass
