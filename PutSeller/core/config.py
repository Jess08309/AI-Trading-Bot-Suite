"""
IronCondor Configuration - Credit Put & Call Spread Strategy (Right Leg)
Sells bull put spreads AND bear call spreads on quality large-cap stocks.
Combines both sides for iron condor risk profile.
Same Alpaca account, separate capital allocation.
"""
import os
from dataclasses import dataclass, field
from typing import List

# Load .env BEFORE dataclass fields evaluate (fixes ALLOCATION_PCT reading default 1.00)
try:
    from dotenv import load_dotenv as _load_dotenv_early
    _load_dotenv_early()
except ImportError:
    pass


def _env(key: str, default, cast=str):
    """Read env var, cast to type."""
    val = os.getenv(key, None)
    if val is None:
        return default
    if cast == bool:
        return val.lower() in ("1", "true", "yes")
    return cast(val)


@dataclass
class PutSellerConfig:
    # ── API ──────────────────────────────────────────────
    API_KEY: str = ""
    API_SECRET: str = ""
    PAPER: bool = True
    BASE_URL: str = ""

    # ── Capital Allocation ───────────────────────────────
    # 100% of Alpaca equity — primary strategy
    ALLOCATION_PCT: float = _env("ALLOCATION_PCT", 1.00, float)
    MAX_POSITION_RISK_PCT: float = 0.03     # 3% of allocation per spread (live-realistic)
    MAX_POSITIONS: int = 12                  # max concurrent put spreads (was 37 — reduced for $53K account)
    MAX_PER_UNDERLYING: int = 2              # max 2 spreads per stock (live-realistic)

    # ── Strategy: Bull Put Spreads (Credit) ──────────────
    # DTE targeting
    MIN_DTE: int = 30                        # earliest expiration
    MAX_DTE: int = 60                        # latest expiration
    TARGET_DTE: int = 45                     # sweet spot for theta (tastytrade proven)

    # Strike selection (PUTS)
    SHORT_DELTA_MIN: float = 0.10            # min delta magnitude (furthest OTM)
    SHORT_DELTA_MAX: float = 0.25            # max delta magnitude (tighter = higher POP)
    TARGET_SHORT_DELTA: float = 0.16         # ~16 delta (1 SD, 84% POP)

    # Spread width (based on stock price)
    SPREAD_WIDTH_SMALL: float = 5.0          # stocks < $200
    SPREAD_WIDTH_MED: float = 10.0           # stocks $200-500
    SPREAD_WIDTH_LARGE: float = 25.0         # stocks > $500

    # Credit quality
    MIN_CREDIT_PCT: float = 0.15             # min credit as % of spread width (15% — original)
    MIN_ROC_ANNUAL: float = 0.12             # min 12% annualized return on capital

    # ── Strategy: Bear Call Spreads (Credit) ─────────────
    CALL_SPREADS_ENABLED: bool = True        # enable call credit spread side
    CALL_SHORT_DELTA_MIN: float = 0.10       # min delta (furthest OTM)
    CALL_SHORT_DELTA_MAX: float = 0.25       # max delta (tighter = higher POP)
    CALL_TARGET_DELTA: float = 0.16          # ~16 delta (1 SD, matches put side)
    CALL_MIN_CREDIT_PCT: float = 0.15        # min credit as % of width (matches put side)
    CALL_MIN_ROC_ANNUAL: float = 0.12        # min annualized ROC (matches put side)
    MAX_CALL_POSITIONS: int = 8              # max concurrent call spreads (was 20 — reduced for $53K account)
    # Uses same DTE, spread width, IV filter as put side

    # ── IV Filter ────────────────────────────────────────
    MIN_IV_PREMIUM: float = 1.20             # IV/HV20 ratio — require meaningfully elevated IV
    HV_LOOKBACK_DAYS: int = 30               # historical vol lookback

    # ── Exit Rules ───────────────────────────────────────
    TAKE_PROFIT_PCT: float = 0.50            # close at 50% of max profit
    STOP_LOSS_MULT: float = 2.0              # close if loss >= 2x credit (1.0 caused cascading liquidation)
    MIN_DTE_EXIT: int = 21                   # close at 21 DTE (avoid gamma risk zone)
    EMERGENCY_BUFFER_PCT: float = 0.05       # close if underlying within 5% of short strike (was 2% — caused premature exits on normal intraday vol)

    # ── Smart Hold Monitoring ────────────────────────────
    DELTA_EXIT_THRESHOLD: float = 0.40       # close if short leg |delta| crosses this (moving ITM)
    IV_SPIKE_EXIT_MULT: float = 1.50         # close if current IV > 1.5x entry IV (risk expanded)
    REGIME_BEAR_BUFFER_MULT: float = 1.5     # widen emergency buffer 1.5x in TRENDING_DOWN regime

    # ── Timing ───────────────────────────────────────────
    SCAN_INTERVAL_SEC: int = 900             # 15 min between full opportunity scans
    CHECK_INTERVAL_SEC: int = 300            # 5 min between position checks (exits)
    MARKET_OPEN_HOUR: int = 7               # 7:30 AM MT = 9:30 AM ET
    MARKET_OPEN_MIN: int = 30
    MARKET_CLOSE_HOUR: int = 14              # 2:00 PM MT = 4:00 PM ET
    MARKET_CLOSE_MIN: int = 0

    # Avoid opening in first/last 30 minutes (volatile)
    NO_OPEN_FIRST_MIN: int = 30              # skip first 30 min
    NO_OPEN_LAST_MIN: int = 30               # skip last 30 min

    # ── Watchlist — Large-cap, liquid, quality names ─────
    WATCHLIST: List[str] = field(default_factory=lambda: [
        # ETFs — broad market, no earnings risk, most liquid options
        "SPY", "QQQ", "IWM",
        # Mega-cap only (liquid, stable, less event risk)
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        # Large-cap diversified
        "JPM", "V", "HD", "UNH",
    ])

    # ── Leveraged / Inverse ETF Guardrails ─────────────
    LEVERAGED_ETFS: frozenset = frozenset({
        "TQQQ", "SQQQ", "SOXL", "SOXS", "TSLL", "TSDD",
        "SPDN", "TZA", "RWM", "NVD", "UVXY", "SVXY",
        "LABU", "LABD", "SPXL", "SPXS", "FAS", "FAZ",
        "NUGT", "DUST", "JNUG", "JDST", "TECL", "TECS",
        "UPRO", "SDOW", "UDOW", "SDS", "QLD", "QID",
    })
    MAX_LEVERAGED_POSITIONS: int = 3         # max total leveraged/inverse ETF positions
    LEVERAGED_QTY_CAP: int = 1               # max 1 contract per spread on leveraged
    LEVERAGED_OTM_MULT: float = 1.5          # require 1.5x wider OTM distance

    # ── Order Tagging ────────────────────────────────────
    ORDER_PREFIX: str = "ic_"                # prefix for client_order_id
    BOT_NAME: str = "IronCondor"

    # ── Dashboard ────────────────────────────────────────
    DASHBOARD_PORT: int = 5556
    DASHBOARD_HOST: str = "127.0.0.1"

    # ── Paths ────────────────────────────────────────────
    LOG_DIR: str = "logs"
    STATE_FILE: str = "data/state/bot_state.json"
    POSITIONS_FILE: str = "data/state/positions.json"
    TRADE_LOG: str = "data/trades.csv"

    # ── Logging ──────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    VERBOSE: bool = True

    def __post_init__(self):
        if not self.BASE_URL:
            self.BASE_URL = (
                "https://paper-api.alpaca.markets"
                if self.PAPER
                else "https://api.alpaca.markets"
            )
        # Load .env
        self._load_dotenv()

    def _load_dotenv(self):
        """Load API keys from .env file."""
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

    def get_spread_width(self, price: float) -> float:
        """Return appropriate spread width for a stock price."""
        if price < 200:
            return self.SPREAD_WIDTH_SMALL
        elif price < 500:
            return self.SPREAD_WIDTH_MED
        else:
            return self.SPREAD_WIDTH_LARGE

    def summary(self) -> str:
        mode = "PAPER" if self.PAPER else "*** LIVE ***"
        call_status = "ENABLED" if self.CALL_SPREADS_ENABLED else "DISABLED"
        return (
            f"IronCondor Config [{mode}]\n"
            f"  Allocation: {self.ALLOCATION_PCT:.0%} of account equity\n"
            f"  Strategy: Credit put + call spreads (iron condor)\n"
            f"  Call spreads: {call_status} (max {self.MAX_CALL_POSITIONS})\n"
            f"  Watchlist: {', '.join(self.WATCHLIST[:6])}... ({len(self.WATCHLIST)} symbols)\n"
            f"  DTE: {self.MIN_DTE}-{self.MAX_DTE}d (target {self.TARGET_DTE}d)\n"
            f"  Put delta: {self.SHORT_DELTA_MIN:.0%}-{self.SHORT_DELTA_MAX:.0%}\n"
            f"  Call delta: {self.CALL_SHORT_DELTA_MIN:.0%}-{self.CALL_SHORT_DELTA_MAX:.0%}\n"
            f"  Min credit: {self.MIN_CREDIT_PCT:.0%} of width | Min ROC: {self.MIN_ROC_ANNUAL:.0%}/yr\n"
            f"  Exit: TP {self.TAKE_PROFIT_PCT:.0%} | SL {self.STOP_LOSS_MULT}x | DTE {self.MIN_DTE_EXIT}\n"
            f"  Max positions: {self.MAX_POSITIONS} (puts) + {self.MAX_CALL_POSITIONS} (calls)\n"
            f"  Per-underlying: {self.MAX_PER_UNDERLYING}\n"
            f"  Scan interval: {self.SCAN_INTERVAL_SEC // 60}min | Check: {self.CHECK_INTERVAL_SEC // 60}min\n"
            f"  Dashboard: http://localhost:{self.DASHBOARD_PORT}\n"
            f"  API Keys: {'SET' if self.has_keys else 'MISSING'}"
        )


# Singleton
CFG = PutSellerConfig()
