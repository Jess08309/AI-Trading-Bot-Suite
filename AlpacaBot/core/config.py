"""
AlpacaBot Configuration - SCALP OPTIONS STRATEGY
Per-symbol DTE optimization via 1-year backtest + DTE sweep (150 tests, 61 symbols).
2DTE is the sweet spot for most symbols. Per-symbol DTE map overrides global defaults.
Override any value via environment variable with ALPACA_ prefix.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List


def _env(key: str, default, cast=str):
    """Read env var with ALPACA_ prefix, cast to type."""
    val = os.getenv(f"ALPACA_{key}", None)
    if val is None:
        return default
    if cast == bool:
        return val.lower() in ("1", "true", "yes")
    return cast(val)


# ═══════════════════════════════════════════════════════════
#  PER-SYMBOL OPTIMAL DTE — from DTE sweep backtest
#  Each symbol tested at every available expiration (1-30 DTE)
#  Best DTE chosen by highest profit factor (PF >= 1.0)
#  Validated: 2025-02-17 | 365d of 10-min bars | 150 backtests
# ═══════════════════════════════════════════════════════════
SYMBOL_DTE_MAP: Dict[str, int] = {
    # KEEP tier (PF >= 1.2) — proven winners
    "AVGO":  2,   # +$29,066 PF 1.34 | 184 trades
    "PYPL":  2,   # +$26,185 PF 2.13 |  50 trades
    "NFLX":  2,   # +$25,598 PF 1.57 | 122 trades
    "NKE":   2,   # +$23,410 PF 1.27 | 243 trades
    "SBUX":  9,   # +$19,509 PF 1.24 | 349 trades
    "F":    30,   # +$12,067 PF 1.39 | 199 trades
    "COST":  2,   # +$6,947  PF 1.28 |  82 trades
    "SHOP":  2,   # +$6,318  PF 1.27 |  93 trades
    "IWM":   1,   # +$4,679  PF 1.38 |  29 trades (daily exp available)
    "META":  2,   # +$4,051  PF 1.20 |  70 trades
    # MAYBE tier (PF 1.0-1.2) — included for scanner diversification
    "JPM":   2,   # +$3,872  PF 1.17 |  75 trades
    "AMZN":  5,   # +$3,807  PF 1.14 | 120 trades
    "DIS":  16,   # +$3,786  PF 1.07 | 336 trades
    "SMH":   2,   # +$3,609  PF 1.20 |  58 trades
    "RIVN": 16,   # +$3,143  PF 1.10 | 167 trades
    "XLK":  16,   # +$3,018  PF 1.09 | 172 trades
    "GM":    2,   # +$2,953  PF 1.10 | 115 trades
    "SOFI":  9,   # +$2,623  PF 1.08 | 142 trades
    "QCOM":  2,   # +$2,549  PF 1.06 | 161 trades
    "MARA": 23,   # +$2,400  PF 1.07 | 200 trades
    "MA":    2,   # +$2,332  PF 1.16 |  43 trades
    "AAPL":  2,   # +$2,040  PF 1.16 |  46 trades
    "WMT":   2,   # +$1,923  PF 1.12 |  66 trades
    "V":     9,   # +$1,828  PF 1.10 | 110 trades
    "LLY":   2,   # +$1,778  PF 1.16 |  39 trades
    "INTC":  9,   # +$1,732  PF 1.12 |  61 trades
    "ABBV":  2,   # +$1,148  PF 1.11 |  38 trades
    "COIN":  2,   # +$1,136  PF 1.05 |  91 trades
    "JNJ":   2,   # +$1,010  PF 1.04 |  93 trades
    "AMD":   2,   # +$1,003  PF 1.18 |  11 trades
    "MU":    2,   # +$381    PF 1.05 |  28 trades
    "UBER":  9,   # +$370    PF 1.02 |  85 trades
    "NVDA":  5,   # +$175    PF 1.02 |  38 trades
}

# Default DTE for symbols not in the map
DEFAULT_DTE = 2


@dataclass
class Config:
    # -- API --
    API_KEY: str = _env("API_KEY", "")
    API_SECRET: str = _env("API_SECRET", "")
    PAPER: bool = _env("PAPER", True, bool)
    BASE_URL: str = ""  # set in __post_init__

    # -- Capital --
    INITIAL_BALANCE: float = _env("INITIAL_BALANCE", 50000.0, float)
    ALLOCATION_PCT: float = _env("ALLOCATION_PCT", 0.00, float)   # 0% — bot lost 89%, capital redirected to IronCondor

    # -- Symbols (top KEEP winners from DTE sweep backtest) --
    WATCHLIST: List[str] = field(default_factory=lambda: [
        # Ranked by combined call+put backtest edge. Gates handle direction: SPY regime + sentiment.
        # Source: scanner_universe_results.json — prior runs excluded (broken crypto sentiment)
        "MA",    # #1 call P&L $13,741 | PF 1.91
        "NFLX",  # #2 call P&L  $9,992 | PF 1.73
        "GM",    # #3 call P&L $10,270 | PF 1.25
        "IWM",   # #4 call P&L  $6,668 | PF 1.38
        "LLY",   # #5 call P&L  $6,500 | PF 1.49
        "SNOW",  # #6 call P&L  $2,827 | PF 1.44
        # AVGO removed — call P&L -$1,725, PF 0.24
    ])

    # -- Options Parameters (SCALP) --
    MIN_DTE: int = _env("MIN_DTE", 1, int)           # NEVER 0DTE
    MAX_DTE: int = _env("MAX_DTE", 31, int)           # F needs 30DTE
    TARGET_DTE: int = _env("TARGET_DTE", 2, int)      # 2DTE = proven sweet spot (fallback)

    # Strike selection — ITM preferred (higher delta ~0.65-0.75, follows underlying better)
    MAX_OTM_PCT: float = _env("MAX_OTM_PCT", 0.01, float)    # allow up to 1% OTM (was 4% OTM)
    TARGET_ITM_PCT: float = _env("TARGET_ITM_PCT", 0.03, float)  # target 3% ITM (delta ~0.65-0.75)
    PREFER_ATM: bool = _env("PREFER_ATM", False, bool)       # ITM scoring used instead

    # Minimum open interest / volume for liquidity
    MIN_OPEN_INTEREST: int = _env("MIN_OPEN_INTEREST", 50, int)
    MIN_VOLUME: int = _env("MIN_VOLUME", 10, int)
    MAX_BID_ASK_SPREAD_PCT: float = _env("MAX_BID_ASK_SPREAD", 0.15, float)

    # -- Position Sizing (SCALP) --
    MAX_POSITION_PCT: float = _env("MAX_POSITION_PCT", 0.04, float)   # 4% per trade (reduced from 5%)
    MIN_POSITION_PCT: float = _env("MIN_POSITION_PCT", 0.02, float)   # 2% floor
    MAX_POSITIONS: int = _env("MAX_POSITIONS", 5, int)                 # max concurrent
    MAX_OPENS_PER_CYCLE: int = _env("MAX_OPENS_PER_CYCLE", 1, int)     # 1 new open per scan cycle (reduced from 2)
    MAX_PORTFOLIO_RISK_PCT: float = _env("MAX_PORTFOLIO_RISK", 0.50, float)

    # -- Signal Parameters --
    MIN_SIGNAL_SCORE: int = _env("MIN_SIGNAL_SCORE", 4, int)
    SIGNAL_CHECK_BARS: int = _env("SIGNAL_CHECK_BARS", 2, int)  # check every 2 bars (20 min)
    COOLDOWN_BARS: int = _env("COOLDOWN_BARS", 6, int)  # 1 hour cooldown after exit

    # -- Direction Filter (diagnostic: disable PUT signals temporarily) --
    PUTS_ENABLED: bool = _env("PUTS_ENABLED", False, bool)   # Disabled — 18% WR (12W/54L), catastrophic performance
    CALLS_ENABLED: bool = _env("CALLS_ENABLED", True, bool)

    # -- Level 3: Multi-Leg Strategies --
    SPREADS_ENABLED: bool = _env("SPREADS_ENABLED", True, bool)       # Vertical spreads (capital efficient, defined risk)
    IRON_CONDOR_ENABLED: bool = _env("IRON_CONDOR", False, bool)      # Iron condors in neutral regime (future)

    # -- Exit Rules (SCALP - tight, for single-leg options) --
    STOP_LOSS_PCT: float = _env("STOP_LOSS_PCT", -0.12, float)       # -12% of premium — tighter cut
    TAKE_PROFIT_PCT: float = _env("TAKE_PROFIT_PCT", 0.30, float)    # +30% of premium — let winners run (2.5:1 R:R)
    TRAILING_STOP_PCT: float = _env("TRAILING_STOP_PCT", 0.08, float)  # 8% from peak (tighter trail)
    TRAILING_TRIGGER: float = _env("TRAILING_TRIGGER", 0.10, float)  # trigger at +10%
    MIN_DTE_EXIT: int = _env("MIN_DTE_EXIT", 0, int)                  # exit on expiry day
    MAX_HOLD_DAYS: int = _env("MAX_HOLD_DAYS", 3, int)                # 3 day default (DTE-dependent)

    # -- Exit Rules (SPREAD - wider, ride winners to max profit) --
    SPREAD_TP_PCT_OF_MAX: float = _env("SPREAD_TP_PCT", 0.65, float)         # Take profit at 65% of max profit
    SPREAD_STOP_LOSS_PCT: float = _env("SPREAD_SL_PCT", -0.50, float)        # Stop loss at -50% of debit
    SPREAD_TRAILING_STOP_PCT: float = _env("SPREAD_TRAIL_PCT", 0.20, float)  # 20% drop from peak spread value
    SPREAD_TRAILING_TRIGGER: float = _env("SPREAD_TRAIL_TRIG", 0.30, float)  # Trigger trailing at 30% of max profit
    SPREAD_NEAREXP_TRAIL_PCT: float = _env("SPREAD_NE_TRAIL", 0.10, float)   # Tighten to 10% near expiry (≤1 DTE)

    # -- Circuit Breakers --
    MAX_DAILY_LOSS_PCT: float = _env("MAX_DAILY_LOSS", -0.03, float)   # -3% daily (tighter)
    MAX_DRAWDOWN_PCT: float = _env("MAX_DRAWDOWN", -0.15, float)
    MAX_CONSECUTIVE_LOSSES: int = _env("MAX_CONSEC_LOSSES", 5, int)    # 5 — gives room to recover
    PAUSE_AFTER_BREAKER_MIN: int = _env("PAUSE_MINUTES", 60, int)     # 1 hour pause (was 30m)

    # -- Timing (10-min bars = 600s) --
    BAR_INTERVAL_SEC: int = _env("BAR_INTERVAL", 600, int)    # 10-min candles
    BARS_PER_DAY: int = 39     # 9:30-16:00 = 390 min / 10 = 39
    LOOKBACK_BARS: int = _env("LOOKBACK_BARS", 50, int)       # indicator lookback
    RISK_CHECK_INTERVAL: int = _env("RISK_CHECK_SEC", 15, int)  # check exits every 15s
    MODEL_RETRAIN_HOURS: int = _env("RETRAIN_HOURS", 24, int)

    # -- Scanner --
    SCANNER_ENABLED: bool = _env("SCANNER_ENABLED", True, bool)
    SCANNER_MAX_SYMBOLS: int = _env("SCANNER_MAX", 50, int)        # scan top N symbols
    SCANNER_INTERVAL_MIN: int = _env("SCANNER_INTERVAL", 10, int)  # scan every N minutes

    # -- Morning Window (time-of-day aggression) --
    # Based on user's edge: volatility + momentum are strongest at open.
    # Morning (9:30-11:00 ET): lower thresholds → more willing to enter.
    # Afternoon (after 11:00 ET): raise thresholds → only strong signals.
    # Exits are NOT affected — stop loss / take profit / trailing stop remain rule-based.
    MORNING_WINDOW_ENABLED: bool = _env("MORNING_ENABLED", True, bool)
    MORNING_WINDOW_END_MIN: int = 110         # minutes after 9:30 → 11:00 AM ET
    MORNING_SCORE_REDUCTION: int = 1          # MIN_SIGNAL_SCORE lowered by 1 in morning
    MORNING_META_CONF_REDUCTION: float = 0.05 # meta confidence_threshold reduced 0.05
    MORNING_META_RULE_REDUCTION: int = 1      # meta min_rule_score reduced by 1
    AFTERNOON_SCORE_INCREASE: int = 1         # MIN_SIGNAL_SCORE raised by 1 in afternoon
    AFTERNOON_META_CONF_INCREASE: float = 0.03  # meta confidence_threshold raised 0.03
    AFTERNOON_META_RULE_INCREASE: int = 1     # meta min_rule_score raised by 1

    # -- Dashboard --
    DASHBOARD_PORT: int = _env("DASHBOARD_PORT", 5555, int)
    DASHBOARD_HOST: str = _env("DASHBOARD_HOST", "127.0.0.1")

    # -- Logging --
    LOG_DIR: str = _env("LOG_DIR", "logs")
    TRADE_LOG: str = _env("TRADE_LOG", "data/trades.csv")
    STATE_FILE: str = _env("STATE_FILE", "data/state/bot_state.json")
    VERBOSE: bool = _env("VERBOSE", True, bool)

    def __post_init__(self):
        if not self.BASE_URL:
            self.BASE_URL = (
                "https://paper-api.alpaca.markets"
                if self.PAPER
                else "https://api.alpaca.markets"
            )

        # Load from .env file if keys not set via env vars
        if not self.API_KEY or not self.API_SECRET:
            self._load_dotenv()

    def _load_dotenv(self):
        """Try loading API keys from .env file."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            if not self.API_KEY:
                self.API_KEY = os.getenv("ALPACA_API_KEY", "")
            if not self.API_SECRET:
                self.API_SECRET = os.getenv("ALPACA_API_SECRET", "")
        except ImportError:
            pass

    @property
    def has_keys(self) -> bool:
        return bool(self.API_KEY and self.API_SECRET)

    def get_target_dte(self, symbol: str) -> int:
        """Return optimal DTE for a symbol from the sweep-validated map."""
        return SYMBOL_DTE_MAP.get(symbol, DEFAULT_DTE)

    def get_max_hold_days(self, symbol: str) -> int:
        """Return max hold days based on symbol's target DTE.
        Short DTE (1-2) = 1 day max, medium (5-9) = 3 days, long (16+) = 7 days.
        """
        dte = self.get_target_dte(symbol)
        if dte <= 2:
            return 1
        elif dte <= 9:
            return 3
        else:
            return 7

    def summary(self) -> str:
        mode = "PAPER" if self.PAPER else "*** LIVE ***"
        dte_info = ", ".join(f"{s}={d}" for s, d in list(SYMBOL_DTE_MAP.items())[:5])
        return (
            f"AlpacaBot SCALP Config [{mode}]\n"
            f"  Balance: ${self.INITIAL_BALANCE:,.0f}\n"
            f"  Watchlist: {', '.join(self.WATCHLIST)}\n"
            f"  Scanner: {'ON' if self.SCANNER_ENABLED else 'OFF'} ({self.SCANNER_MAX_SYMBOLS} symbols)\n"
            f"  Per-symbol DTE: {dte_info}... ({len(SYMBOL_DTE_MAP)} symbols mapped)\n"
            f"  Strike: ITM {self.TARGET_ITM_PCT:.1%} target | OTM max {self.MAX_OTM_PCT:.1%} | Bars: 10-min\n"
            f"  Position: {self.MIN_POSITION_PCT:.0%}-{self.MAX_POSITION_PCT:.0%}, max {self.MAX_POSITIONS}\n"
            f"  SL: {self.STOP_LOSS_PCT:.0%} / TP: +{self.TAKE_PROFIT_PCT:.0%} / Trail: {self.TRAILING_STOP_PCT:.0%}\n"
            f"  Circuit: {self.MAX_CONSECUTIVE_LOSSES} losses, {self.MAX_DAILY_LOSS_PCT:.0%} daily, {self.MAX_DRAWDOWN_PCT:.0%} DD\n"
            f"  Dashboard: http://localhost:{self.DASHBOARD_PORT}\n"
            f"  API Keys: {'SET' if self.has_keys else 'MISSING'}"
        )
