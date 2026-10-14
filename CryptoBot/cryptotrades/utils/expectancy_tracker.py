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
from datetime import timedelta
from typing import Deque, Dict, Optional, Tuple
from collections import deque


_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAVE_PATH = os.path.join(
    os.environ.get("CRYPTOBOT_DATA_DIR") or os.path.join(_PACKAGE_ROOT, "data"),
    "state", "symbol_expectancy.json",
)


class SymbolExpectancyTracker:
    def __init__(
        self,
        max_consecutive_losses: int = 3,
        expectancy_window: int = 20,
        save_path: Optional[str] = None,
        reenable_after_hours: Optional[float] = 24.0,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.expectancy_window = expectancy_window
        self.save_path = save_path or DEFAULT_SAVE_PATH
        # Time-based self-heal: a disabled symbol is put back on probation
        # (fresh history) after this many hours. None disables auto re-enable.
        self.reenable_after_hours = reenable_after_hours

        self._history: Dict[str, Deque[float]] = {}
        self._consecutive_losses: Dict[str, int] = {}
        self._disabled: Dict[str, str] = {}  # symbol -> reason
        self._disabled_at: Dict[str, str] = {}  # symbol -> ISO timestamp

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
            if symbol not in self._disabled:
                self._disabled_at[symbol] = datetime.now(timezone.utc).isoformat()
            self._disabled[symbol] = reason
        return just_disabled, reason

    def is_disabled(self, symbol: str) -> Tuple[bool, str]:
        if symbol in self._disabled and self._cooldown_elapsed(symbol):
            self.re_enable(symbol)
            self.save_state()
        reason = self._disabled.get(symbol, "")
        return bool(reason), reason

    def _cooldown_elapsed(self, symbol: str) -> bool:
        if not self.reenable_after_hours:
            return False
        try:
            disabled_at = datetime.fromisoformat(self._disabled_at[symbol])
            if disabled_at.tzinfo is None:
                disabled_at = disabled_at.replace(tzinfo=timezone.utc)
        except Exception:
            # Unknown disable time (legacy state): start the cooldown now.
            self._disabled_at[symbol] = datetime.now(timezone.utc).isoformat()
            return False
        return datetime.now(timezone.utc) - disabled_at >= timedelta(hours=self.reenable_after_hours)

    def re_enable(self, symbol: str) -> None:
        """Clear a symbol's disabled state and restart its track record."""
        self._disabled.pop(symbol, None)
        self._disabled_at.pop(symbol, None)
        self._consecutive_losses[symbol] = 0
        self._history.pop(symbol, None)

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
                "disabled_at": self._disabled_at,
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
            self._disabled_at = data.get("disabled_at", {})
        except Exception:
            pass
