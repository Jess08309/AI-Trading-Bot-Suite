"""
Risk Manager - Circuit breakers, position limits, and exposure tracking.
"""
import logging
import json
import os
from collections import deque
from datetime import datetime, date
from typing import Dict, List, Any, Optional

from core.config import Config

log = logging.getLogger("alpacabot.risk")


class RiskManager:
    """
    Manages all risk controls:
    - Position count limits
    - Portfolio exposure limits
    - Graduated response (throttle, not halt) after losses
    - Direction-specific loss tracking (block losing direction)
    - Rapid-fire loss detection
    - Hard daily loss cap (-5%)
    - Per-symbol pause after losses
    """

    # ── Graduated Response Tiers ──────────────────────────
    # Instead of binary ON/OFF circuit breaker, losses trigger
    # progressively tighter controls. Trading never fully stops
    # (except hard daily cap), so you don't miss opportunities.
    #
    #  Streak   Size     Min Score  Cooldown   Direction Lock
    #  0-2      100%     normal(3)  none       none
    #  3-4       50%     raised(7)  5 min      block losing dir
    #  5-7       25%     high(9)    10 min     block losing dir
    #  8+        10%     max(10)    15 min     block losing dir
    #  Daily -5%  HARD STOP  —       rest of day    —

    GRADUATED_TIERS = [
        # (min_losses, size_mult, min_score, cooldown_sec)
        (0,  1.00,  3,    0),    # Normal: full size, standard score
        (3,  0.50,  7,  300),    # Caution: half size, 5-min cooldown
        (5,  0.25,  9,  600),    # Defensive: quarter size, 10-min cooldown
        (8,  0.10, 10,  900),    # Probe: tiny size, 15-min cooldown
    ]

    HARD_DAILY_LOSS_PCT = -0.05   # -5% daily = hard stop, no bypass
    RAPID_FIRE_WINDOW = 1200     # 20 minutes
    RAPID_FIRE_THRESHOLD = 3     # 3 losses in 20 min = extra 10-min cooldown

    def __init__(self, config: Config):
        self.config = config

        # Tracking state
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.consecutive_losses: int = 0
        self.peak_balance: float = config.INITIAL_BALANCE
        self.current_balance: float = config.INITIAL_BALANCE
        self.today: str = date.today().isoformat()

        # Per-symbol tracking
        self.symbol_losses: Dict[str, int] = {}  # consecutive losses per symbol

        # Direction-specific loss tracking
        self.direction_losses: Dict[str, int] = {"call": 0, "put": 0}
        self.direction_locked: Dict[str, bool] = {"call": False, "put": False}
        self.DIRECTION_LOCK_THRESHOLD = 3  # 3 consecutive same-direction losses = lock

        # Graduated response state (replaces binary circuit breaker)
        self.breaker_active: bool = False  # only True for hard daily cap
        self.breaker_until: Optional[datetime] = None
        self.breaker_reason: str = ""
        self.last_trade_time: Optional[datetime] = None
        self.recent_loss_times: deque = deque(maxlen=20)  # timestamps of recent losses

        # Trade history for today
        self.today_trades: List[Dict[str, Any]] = []

        # Rolling outcome window (last 10 trades, cross-symbol, cross-day)
        self.rolling_outcomes: deque = deque(maxlen=10)  # 1=win, 0=loss

    # ── Pre-Trade Checks ─────────────────────────────────

    def can_trade(self) -> tuple:
        """
        Check if trading is allowed.
        Returns (allowed: bool, reason: str)

        Graduated response: never fully stops (except hard daily cap).
        Instead returns OK and lets get_throttle() control size/score.
        """
        # Reset daily stats if new day
        self._check_new_day()

        # Hard daily cap — the ONLY thing that fully stops trading
        if self.breaker_active:
            if self.breaker_until and datetime.now() > self.breaker_until:
                self._reset_breaker()
            else:
                return False, f"HARD STOP: {self.breaker_reason}"

        if self.current_balance > 0:
            daily_loss_pct = self.daily_pnl / self.current_balance
            if daily_loss_pct <= self.HARD_DAILY_LOSS_PCT:
                self._trip_breaker(f"Daily loss {daily_loss_pct:.1%} hit -5% hard cap")
                return False, f"HARD STOP: Daily loss {daily_loss_pct:.1%}"

        # Cooldown check — graduated response adds time between entries
        tier = self._get_current_tier()
        cooldown_sec = tier[3]

        # Rapid-fire detection adds extra cooldown
        if self._is_rapid_fire():
            cooldown_sec = max(cooldown_sec, 600)  # at least 10 min after rapid losses
            log.warning(f"Rapid-fire losses detected ({self.RAPID_FIRE_THRESHOLD} in "
                        f"{self.RAPID_FIRE_WINDOW//60}min) — enforcing {cooldown_sec//60}min cooldown")

        if cooldown_sec > 0 and self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < cooldown_sec:
                remaining = int(cooldown_sec - elapsed)
                return False, (f"Cooldown: {remaining}s remaining "
                               f"(streak={self.consecutive_losses}, "
                               f"tier={tier[0]}+ losses)")

        # Log graduated state if throttled
        if self.consecutive_losses >= 3:
            log.info(f"Graduated response: {self.consecutive_losses} losses → "
                     f"size={tier[1]:.0%}, min_score={tier[2]}, "
                     f"cooldown={tier[3]//60}min")

        return True, "OK"

    def can_open_position(self, num_open: int) -> tuple:
        """Check if we can open another position."""
        if num_open >= self.config.MAX_POSITIONS:
            return False, f"Max positions ({self.config.MAX_POSITIONS}) reached"

        can, reason = self.can_trade()
        if not can:
            return False, reason

        return True, "OK"

    def can_trade_symbol(self, symbol: str) -> tuple:
        """Check if a specific underlying is allowed (not paused)."""
        losses = self.symbol_losses.get(symbol, 0)
        if losses >= 3:  # 3 consecutive losses on same symbol = pause
            return False, f"{symbol} paused ({losses} consecutive losses)"
        return True, "OK"

    def can_trade_direction(self, direction: str) -> tuple:
        """Check if a direction (call/put) is allowed.
        Blocks a direction after 3 consecutive losses in that direction.
        The OTHER direction stays open."""
        d = direction.lower()
        if self.direction_locked.get(d, False):
            other = "call" if d == "put" else "put"
            return False, (f"{d.upper()}s locked ({self.direction_losses.get(d, 0)} "
                           f"consecutive {d} losses) — {other.upper()}s still allowed")
        return True, "OK"

    # ── Position Sizing ──────────────────────────────────

    def calculate_size(self, balance: float, confidence: float,
                       premium: float, num_open: int) -> int:
        """
        Calculate number of contracts to buy.

        Uses confidence-weighted sizing:
        - Higher ML confidence → larger position
        - More open positions → smaller new positions
        - Never exceed MAX_POSITION_PCT of balance

        Returns number of contracts (0 = skip).
        """
        if premium <= 0 or balance <= 0:
            return 0

        # Base allocation
        max_spend = balance * self.config.MAX_POSITION_PCT
        min_spend = balance * self.config.MIN_POSITION_PCT

        # Confidence scaling (0.5-1.0 confidence → 0.5-1.0x multiplier)
        conf_factor = max(0.5, min(1.0, confidence))

        # Position count scaling (more positions → smaller new ones)
        count_factor = max(0.4, 1.0 - num_open * 0.12)

        # Final spend
        spend = max_spend * conf_factor * count_factor
        spend = max(min_spend, min(spend, max_spend))

        # Check portfolio risk limit
        total_at_risk = sum(
            t.get("cost", 0) for t in self.today_trades if t.get("status") == "open"
        )
        remaining_risk = (balance * self.config.MAX_PORTFOLIO_RISK_PCT) - total_at_risk
        spend = min(spend, remaining_risk)

        if spend <= 0:
            return 0

        # Convert to contracts (each contract = premium × 100)
        cost_per = premium * 100
        contracts = int(spend / cost_per)

        # Floor: buy at least 1 if we can afford it and it's within limits
        if contracts == 0 and cost_per <= spend * 1.5:
            contracts = 1

        return contracts

    # ── Trade Recording ──────────────────────────────────

    def record_trade(self, trade: Dict[str, Any]):
        """Record a completed trade and update risk state."""
        pnl = trade.get("pnl", 0)
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.today_trades.append(trade)
        self.last_trade_time = datetime.now()

        # Update balance
        self.current_balance += pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Direction tracking
        direction = trade.get("direction", trade.get("option_type", "")).lower()
        symbol = trade.get("underlying", "")

        if pnl < 0:
            # ── LOSS ──
            self.consecutive_losses += 1
            self.symbol_losses[symbol] = self.symbol_losses.get(symbol, 0) + 1
            self.rolling_outcomes.append(0)
            self.recent_loss_times.append(datetime.now())

            # Direction-specific tracking
            if direction in self.direction_losses:
                self.direction_losses[direction] += 1
                if self.direction_losses[direction] >= self.DIRECTION_LOCK_THRESHOLD:
                    self.direction_locked[direction] = True
                    log.warning(f"Direction LOCKED: {direction.upper()}s blocked after "
                                f"{self.direction_losses[direction]} consecutive {direction} losses")
        else:
            # ── WIN ──
            self.consecutive_losses = 0  # full reset on win (not decay)
            self.symbol_losses[symbol] = 0
            self.rolling_outcomes.append(1)

            # Unlock direction on win
            if direction in self.direction_losses:
                self.direction_losses[direction] = 0
                if self.direction_locked.get(direction, False):
                    self.direction_locked[direction] = False
                    log.info(f"Direction UNLOCKED: {direction.upper()}s re-enabled after win")

        tier = self._get_current_tier()
        locked_dirs = [d for d, v in self.direction_locked.items() if v]
        lock_str = f" | LOCKED: {','.join(locked_dirs)}" if locked_dirs else ""
        log.info(f"Trade recorded: {trade.get('symbol', '?')} PnL ${pnl:+.2f} "
                 f"| Daily ${self.daily_pnl:+.2f} | Streak: {self.consecutive_losses} "
                 f"| Tier: size={tier[1]:.0%} score>={tier[2]}{lock_str}")

    def rolling_loss_rate(self) -> float:
        """Loss rate over the last 10 trades (cross-symbol, cross-day)."""
        if len(self.rolling_outcomes) < 3:  # not enough data yet
            return 0.0
        return 1.0 - (sum(self.rolling_outcomes) / len(self.rolling_outcomes))

    # ── Graduated Response Engine ─────────────────────────

    def _get_current_tier(self) -> tuple:
        """Return the current graduated response tier based on consecutive losses.
        Returns (min_losses, size_mult, min_score, cooldown_sec)."""
        tier = self.GRADUATED_TIERS[0]
        for t in self.GRADUATED_TIERS:
            if self.consecutive_losses >= t[0]:
                tier = t
        return tier

    def get_throttle(self) -> Dict[str, Any]:
        """Get current throttle settings for the trading engine.
        Called by the engine to adjust position sizing and signal filtering.

        Returns dict with:
            size_multiplier: float (1.0 = normal, 0.5 = half, etc)
            min_score: int (minimum signal score to accept)
            locked_directions: list of locked directions (e.g. ['put'])
            tier_name: str description of current tier
        """
        tier = self._get_current_tier()
        locked = [d for d, v in self.direction_locked.items() if v]

        if self.consecutive_losses < 3:
            name = "NORMAL"
        elif self.consecutive_losses < 5:
            name = "CAUTION"
        elif self.consecutive_losses < 8:
            name = "DEFENSIVE"
        else:
            name = "PROBE"

        return {
            "size_multiplier": tier[1],
            "min_score": tier[2],
            "cooldown_sec": tier[3],
            "locked_directions": locked,
            "tier_name": name,
            "consecutive_losses": self.consecutive_losses,
        }

    def _is_rapid_fire(self) -> bool:
        """Detect rapid-fire losses — 3+ losses in 20 minutes."""
        if len(self.recent_loss_times) < self.RAPID_FIRE_THRESHOLD:
            return False
        now = datetime.now()
        recent = [
            t for t in self.recent_loss_times
            if (now - t).total_seconds() <= self.RAPID_FIRE_WINDOW
        ]
        return len(recent) >= self.RAPID_FIRE_THRESHOLD

    def _trip_breaker(self, reason: str):
        """Hard daily cap breaker — only used for -5% daily loss."""
        self.breaker_active = True
        self.breaker_until = None  # rest of day — no auto-reset
        self.breaker_reason = reason
        log.warning(f"HARD STOP ACTIVATED: {reason} — no more trades today")

    def _reset_breaker(self):
        """Reset hard stop (only happens on new day)."""
        log.info(f"Hard stop reset (was: {self.breaker_reason})")
        self.breaker_active = False
        self.breaker_until = None
        self.breaker_reason = ""

    def _check_new_day(self):
        """Reset daily stats and graduated response on new day."""
        today = date.today().isoformat()
        if today != self.today:
            log.info(f"New trading day: {today} | Yesterday P&L: ${self.daily_pnl:+.2f}")
            self.today = today
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.today_trades = []
            self.consecutive_losses = 0
            self.direction_losses = {"call": 0, "put": 0}
            self.direction_locked = {"call": False, "put": False}
            self.recent_loss_times.clear()
            self._reset_breaker()  # clear hard stop from yesterday

    # ── Exit Checks ──────────────────────────────────────

    def should_exit(self, position: Dict[str, Any]) -> Optional[str]:
        """
        Check if a position should be exited.

        Args:
            position: dict with entry_price, current_price, entry_time,
                      expiration, peak_price

        Returns:
            Exit reason string, or None if should hold.
        """
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", 0)
        peak_price = position.get("peak_price", entry_price)

        if entry_price <= 0:
            return None

        pnl_pct = (current_price - entry_price) / entry_price

        # Stop loss
        if pnl_pct <= self.config.STOP_LOSS_PCT:
            return f"STOP_LOSS ({pnl_pct:.0%})"

        # Take profit
        if pnl_pct >= self.config.TAKE_PROFIT_PCT:
            return f"TAKE_PROFIT ({pnl_pct:.0%})"

        # Trailing stop (from peak)
        if peak_price > entry_price:
            drop_from_peak = (current_price - peak_price) / peak_price
            if drop_from_peak <= -self.config.TRAILING_STOP_PCT:
                return f"TRAILING_STOP ({drop_from_peak:.0%} from peak)"

        # DTE exit (close before expiry to avoid assignment risk)
        expiration = position.get("expiration", "")
        if expiration:
            try:
                exp_date = datetime.strptime(expiration, "%Y-%m-%d")
                dte = (exp_date - datetime.now()).days
                if dte <= self.config.MIN_DTE_EXIT:
                    return f"DTE_EXIT ({dte}d to expiry)"
            except ValueError:
                pass

        # Max hold time
        entry_time = position.get("entry_time")
        if entry_time:
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time)
                except ValueError:
                    entry_time = None
            if entry_time:
                hold_days = (datetime.now() - entry_time).total_seconds() / 86400
                if hold_days >= self.config.MAX_HOLD_DAYS:
                    return f"MAX_HOLD ({hold_days:.1f}d)"

        return None

    # ── State Persistence ────────────────────────────────

    def save_state(self, filepath: str = ""):
        """Save risk state to disk."""
        if not filepath:
            filepath = self.config.STATE_FILE
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "peak_balance": self.peak_balance,
            "current_balance": self.current_balance,
            "today": self.today,
            "symbol_losses": self.symbol_losses,
            "direction_losses": self.direction_losses,
            "direction_locked": self.direction_locked,
            "breaker_active": self.breaker_active,
            "breaker_reason": self.breaker_reason,
            "graduated_tier": self._get_current_tier()[0],
            "saved_at": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str = ""):
        """Load risk state from disk."""
        if not filepath:
            filepath = self.config.STATE_FILE
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath) as f:
                state = json.load(f)
            self.daily_pnl = state.get("daily_pnl", 0)
            self.daily_trades = state.get("daily_trades", 0)
            self.consecutive_losses = state.get("consecutive_losses", 0)
            self.peak_balance = state.get("peak_balance", self.config.INITIAL_BALANCE)
            self.current_balance = state.get("current_balance", self.config.INITIAL_BALANCE)
            self.today = state.get("today", date.today().isoformat())
            self.symbol_losses = state.get("symbol_losses", {})
            self.direction_losses = state.get("direction_losses", {"call": 0, "put": 0})
            self.direction_locked = state.get("direction_locked", {"call": False, "put": False})
            log.info(f"Loaded risk state: balance=${self.current_balance:,.2f}, "
                     f"daily=${self.daily_pnl:+.2f}, streak={self.consecutive_losses}")
        except Exception as e:
            log.warning(f"Failed to load risk state: {e}")

    def status(self) -> str:
        """Human-readable risk status."""
        dd = 0
        if self.peak_balance > 0:
            dd = (self.current_balance - self.peak_balance) / self.peak_balance

        tier = self._get_current_tier()
        throttle = self.get_throttle()
        locked = throttle["locked_directions"]

        lines = [
            f"Balance: ${self.current_balance:,.2f} (peak ${self.peak_balance:,.2f})",
            f"Drawdown: {dd:.1%}",
            f"Daily P&L: ${self.daily_pnl:+.2f} ({self.daily_trades} trades)",
            f"Consecutive losses: {self.consecutive_losses}",
            f"Response tier: {throttle['tier_name']} (size={tier[1]:.0%}, min_score={tier[2]}, cooldown={tier[3]//60}min)",
        ]

        if locked:
            lines.append(f"Direction locked: {', '.join(d.upper() for d in locked)}")
        if self.breaker_active:
            lines.append(f"HARD STOP: {self.breaker_reason}")
        if self.symbol_losses:
            paused = [f"{s}({n})" for s, n in self.symbol_losses.items() if n >= 3]
            if paused:
                lines.append(f"Paused symbols: {', '.join(paused)}")

        return "\n".join(lines)
