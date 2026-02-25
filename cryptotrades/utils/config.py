"""
Centralized configuration for the crypto trading bot.

All tunable parameters in one place. Values can be overridden by environment
variables (loaded from .env file).

Usage:
    from utils.config import config
    
    if config.PAPER_TRADING:
        ...
    
    print(config.TRADE_INTERVAL)
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ('true', '1', 'yes', 'on')


def _env_csv_list(key: str, default: List[str]) -> List[str]:
    val = os.getenv(key)
    if val is None:
        return default
    items = [x.strip() for x in val.split(',') if x.strip()]
    return items if items else default


@dataclass
class TradingConfig:
    """All bot configuration in one place."""

    # ── Symbol Lists ──────────────────────────────────────
    POPULAR_PAIRS: List[str] = field(default_factory=lambda: [
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
        "DOGE-USD", "MATIC-USD", "LTC-USD", "LINK-USD", "SHIB-USD",
        "XRP-USD", "DOT-USD", "UNI-USD", "ATOM-USD", "XLM-USD",
    ])

    SPOT_WATCHLIST: List[str] = field(default_factory=lambda: [
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
        "DOGE-USD", "MATIC-USD", "LTC-USD", "LINK-USD", "SHIB-USD",
        "XRP-USD", "DOT-USD", "UNI-USD", "ATOM-USD", "XLM-USD", "AAVE-USD",
        "OP-USD", "ARB-USD", "SUI-USD",  # Added for AI learning
    ])

    KRAKEN_FUTURES_SYMBOLS: List[str] = field(default_factory=lambda: [
        "PI_XBTUSD", "PI_ETHUSD", "PI_SOLUSD", "PI_ADAUSD", "PI_AVAXUSD",
        "PI_DOGEUSD", "PI_MATICUSD", "PI_LTCUSD", "PI_LINKUSD", "PI_SHIBUSD",
        "PI_XRPUSD", "PI_DOTUSD", "PI_UNIUSD", "PI_ATOMUSD", "PI_XLMUSD",
    ])

    FUTURES_CORE_SYMBOLS: List[str] = field(default_factory=lambda: [
        "PI_ETHUSD", "PI_XRPUSD",
    ])
    FUTURES_PROBATION_CANDIDATES: List[str] = field(default_factory=lambda: [
        "PI_XBTUSD", "PI_LTCUSD", "PI_LINKUSD", "PI_SOLUSD",
    ])
    ENABLE_FUTURES_PROBATION: bool = True
    FUTURES_PROBATION_COHORT_SIZE: int = 6   # run 5-8 at a time (default 6)
    FUTURES_PROBATION_ROTATE_HOURS: int = 24
    FUTURES_PROBATION_MIN_CLOSED: int = 20
    FUTURES_PROBATION_PROMOTE_MIN_WIN_RATE: float = 0.55
    FUTURES_PROBATION_PROMOTE_MIN_AVG_PCT: float = 0.20
    FUTURES_PROBATION_DEMOTE_MAX_WIN_RATE: float = 0.45
    FUTURES_PROBATION_DEMOTE_MAX_AVG_PCT: float = -0.25

    # ── Feature Toggles ───────────────────────────────────
    ENABLE_SPOT: bool = True          # spot trading/prices (Coinbase)
    ENABLE_FUTURES: bool = True       # futures trading/prices (Kraken)
    ENABLE_COINBASE: bool = True      # connect to Coinbase for spot data
    ENABLE_KRAKEN: bool = True        # connect to Kraken for futures data
    ENABLE_COINBASE_FUTURES_DATA: bool = True  # use Coinbase futures market data (fallback to Kraken)

    # ── Timing ────────────────────────────────────────────
    CHECK_INTERVAL: int = 60          # seconds between risk monitoring cycles
    TRADE_INTERVAL: int = 10          # trade decisions every N cycles (10 = 10 min)
    ROTATION_CHECK_INTERVAL: int = 168 * 3600  # 7 days
    INACTIVE_SAMPLE_SIZE: int = 3
    SCORE_IMPROVEMENT_THRESHOLD: float = 1.20
    MODEL_RETRAIN_INTERVAL: int = 21600  # 6 hours

    # ── Paper Trading ─────────────────────────────────────
    PAPER_TRADING: bool = True
    PAPER_BALANCE_SPOT: float = 2500.0
    PAPER_BALANCE_FUTURES: float = 2500.0
    SIM_REALISM_PROFILE: str = "strict"   # off|normal|strict

    # ── Spot Trading Thresholds ───────────────────────────
    SPOT_MIN_SCORE: int = 6           # minimum AI score out of 15 to buy
    SPOT_MAX_POSITIONS: int = 15
    SPOT_STOP_LOSS: float = -3.0      # % (risk phase, 1-min)
    SPOT_TAKE_PROFIT: float = 5.0     # % (risk phase, 1-min)
    SPOT_TRAILING_STOP: float = 2.0   # % from peak

    # ── Futures Trading Thresholds ────────────────────────
    FUTURES_LEVERAGE: int = 2
    FUTURES_MAX_POSITIONS: int = 10
    FUTURES_LONG_CONFIDENCE: float = 0.60   # v3.1
    FUTURES_SHORT_CONFIDENCE: float = 0.40  # v3.1
    FUTURES_STOP_LOSS: float = -2.5         # v3.1
    FUTURES_TAKE_PROFIT: float = 4.0        # v3.1
    FUTURES_EXIT_BREACH_CONFIRM: int = 2    # require N consecutive breach checks before exit

    # ── Circuit Breaker ───────────────────────────────────
    CB_MAX_CONSECUTIVE_LOSSES: int = 5
    CB_DAILY_LOSS_LIMIT_PCT: float = -5.0
    CB_MAX_DRAWDOWN_PCT: float = -10.0
    CB_COOLDOWN_MINUTES: int = 60

    # ── Rapid Streak/Drawdown Guards (short-horizon) ──
    CB_FAST_LOSSES: int = 3                # pause entries after N consecutive losses (short horizon)
    CB_FAST_WINDOW_TRADES: int = 5         # window size to check recent P/L
    CB_FAST_PNL_THRESHOLD: float = -2.0    # % P/L over window to trigger pause
    CB_FAST_COOLDOWN_MIN: int = 30         # minutes to pause when fast guard trips

    # ── Volatility Scaling ─────────────────────────────
    VOL_WINDOW: int = 20                   # trades to measure realized vol
    VOL_MIN: float = 0.5                   # min vol scaler
    VOL_MAX: float = 1.6                   # max vol scaler
    VOL_BASE_STOP: float = 1.2             # base stop for spot (abs %)
    VOL_BASE_TP: float = 2.0               # base take-profit for spot (abs %)
    VOL_BASE_TRAIL: float = 1.2            # base trailing stop from max (abs %)

    # ── Fees (simulation) ───────────────────────────────
    SPOT_FEE_RATE: float = 0.0005          # 0.05% taker
    FUTURES_FEE_RATE: float = 0.0005       # 0.05% per side on notional

    # ── Execution Realism (paper/backtest) ───────────────
    ENABLE_EXECUTION_COSTS: bool = True
    SPOT_SLIPPAGE_BPS: float = 5.0         # bps impact per side around mid
    FUTURES_SLIPPAGE_BPS: float = 8.0      # bps impact per side around mid

    ENABLE_PARTIAL_FILLS: bool = True
    PARTIAL_FILL_PROB: float = 0.10        # probability a market order partially fills
    PARTIAL_FILL_MIN: float = 0.50         # min fill ratio when partial
    PARTIAL_FILL_MAX: float = 1.00         # max fill ratio when partial
    EXECUTION_RANDOM_SEED: int = 1337

    ENABLE_FUNDING_COSTS: bool = True
    FUTURES_FUNDING_RATE_PER_8H: float = 0.0001  # 0.01% per 8h (conservative approx)

    # ── Minimum Order Sizes ─────────────────────────────
    MIN_TRADE_USD: float = 75.0            # Minimum spot notional
    MIN_FUTURES_TRADE_USD: float = 75.0    # Minimum futures notional

    # ── Regime Detection ───────────────────────────────
    REGIME_TREND_LOOKBACK: int = 30        # prices for slope (10m candles)
    REGIME_VOL_LOOKBACK: int = 30          # prices for volatility
    REGIME_TREND_THRESH: float = 0.0005    # slope threshold for flat vs trending
    REGIME_VOL_MULT: float = 1.35          # vol spike multiple to flag regime change
    REGIME_COOLDOWN_MIN: int = 20          # minutes to pause entries after flip

    # ── Stress Correlation Tightening ──────────────────
    STRESS_WINDOW_TRADES: int = 8
    STRESS_PNL_THRESHOLD: float = -1.5     # % over window
    STRESS_CORR_MAX: float = 0.70          # tighter corr cap under stress
    STRESS_MAX_POS: int = 8                # cap open positions when stressed

    # ── RL Adaptive Exploration ────────────────────────
    RL_BASE_EXPLORATION: float = 0.15
    RL_MAX_EXPLORATION: float = 0.30
    RL_DRAW_PNL_THRESH: float = -2.0       # % over last window to boost epsilon
    RL_RECOVER_PNL_THRESH: float = 1.0     # % to revert epsilon
    RL_WINDOW_TRADES: int = 10

    # ── Time-of-Day Filter ─────────────────────────────
    QUIET_HOURS_START: int = 0             # 0-23 (UTC)
    QUIET_HOURS_END: int = 3               # block/limit entries in this window
    QUIET_SIZE_REDUCTION: float = 0.5      # scale position size during quiet window

    # ── API Retry ─────────────────────────────────────────
    COINBASE_RETRY_ATTEMPTS: int = 3
    COINBASE_RETRY_BASE_DELAY: float = 0.5
    KRAKEN_RETRY_ATTEMPTS: int = 3
    KRAKEN_RETRY_BASE_DELAY: float = 0.5

    # ── ML Model ──────────────────────────────────────────
    ML_MODEL_PATH: str = "data/models/market_model.joblib"
    ML_LOOKBACK: int = 30
    ML_PREDICTION_HORIZON: int = 5
    ML_MIN_TRAIN_SAMPLES: int = 100
    ML_MAX_VERSIONS: int = 5

    # ── RL Agent ──────────────────────────────────────────
    RL_LEARNING_RATE: float = 0.001
    RL_DISCOUNT_FACTOR: float = 0.95
    RL_EXPLORATION_RATE: float = 0.15

    # ── Correlation Tracker ───────────────────────────────
    CORR_WINDOW: int = 30
    CORR_MAX_CORRELATION: float = 0.85

    # ── Position Sizing Scaling ───────────────────────────
    ENABLE_SCALED_SIZING: bool = True      # Enable 1-3x position multipliers for strong setups
    SCALED_SIZING_MIN_SCORE: int = 3       # Min score for 2x sizing (out of 6)
    SCALED_SIZING_PERFECT_SCORE: int = 5   # Min score for 3x sizing (out of 6)

    # ── Alerting (Discord) ────────────────────────────────
    DISCORD_WEBHOOK_URL: str = ""
    ALERT_ON_CIRCUIT_BREAKER: bool = True
    ALERT_ON_LARGE_LOSS_PCT: float = -3.0

    # ── Data Paths ────────────────────────────────────────
    STATE_DIR: str = "data/state"
    MODELS_DIR: str = "data/models"
    LOGS_DIR: str = "logs"

    def __post_init__(self):
        """Override defaults with environment variables if set."""
        self.PAPER_TRADING = _env_bool("PAPER_TRADING", self.PAPER_TRADING)
        self.PAPER_BALANCE_SPOT = _env_float("PAPER_BALANCE_SPOT", self.PAPER_BALANCE_SPOT)
        self.PAPER_BALANCE_FUTURES = _env_float("PAPER_BALANCE_FUTURES", self.PAPER_BALANCE_FUTURES)
        self.SIM_REALISM_PROFILE = os.getenv("SIM_REALISM_PROFILE", self.SIM_REALISM_PROFILE).strip().lower()
        self._apply_realism_profile()

        self.ENABLE_SPOT = _env_bool("ENABLE_SPOT", self.ENABLE_SPOT)
        self.ENABLE_FUTURES = _env_bool("ENABLE_FUTURES", self.ENABLE_FUTURES)
        self.ENABLE_COINBASE = _env_bool("ENABLE_COINBASE", self.ENABLE_COINBASE)
        self.ENABLE_KRAKEN = _env_bool("ENABLE_KRAKEN", self.ENABLE_KRAKEN)
        self.ENABLE_COINBASE_FUTURES_DATA = _env_bool(
            "ENABLE_COINBASE_FUTURES_DATA", self.ENABLE_COINBASE_FUTURES_DATA
        )

        self.CHECK_INTERVAL = _env_int("CHECK_INTERVAL", self.CHECK_INTERVAL)
        self.TRADE_INTERVAL = _env_int("TRADE_INTERVAL", self.TRADE_INTERVAL)

        self.COINBASE_RETRY_ATTEMPTS = _env_int("COINBASE_RETRY_ATTEMPTS", self.COINBASE_RETRY_ATTEMPTS)
        self.COINBASE_RETRY_BASE_DELAY = _env_float("COINBASE_RETRY_BASE_DELAY", self.COINBASE_RETRY_BASE_DELAY)
        self.KRAKEN_RETRY_ATTEMPTS = _env_int("KRAKEN_RETRY_ATTEMPTS", self.KRAKEN_RETRY_ATTEMPTS)
        self.KRAKEN_RETRY_BASE_DELAY = _env_float("KRAKEN_RETRY_BASE_DELAY", self.KRAKEN_RETRY_BASE_DELAY)

        self.MIN_TRADE_USD = _env_float("MIN_TRADE_USD", self.MIN_TRADE_USD)
        self.MIN_FUTURES_TRADE_USD = _env_float("MIN_FUTURES_TRADE_USD", self.MIN_FUTURES_TRADE_USD)
        self.FUTURES_EXIT_BREACH_CONFIRM = _env_int(
            "FUTURES_EXIT_BREACH_CONFIRM", self.FUTURES_EXIT_BREACH_CONFIRM
        )

        self.SPOT_FEE_RATE = _env_float("SPOT_FEE_RATE", self.SPOT_FEE_RATE)
        self.FUTURES_FEE_RATE = _env_float("FUTURES_FEE_RATE", self.FUTURES_FEE_RATE)

        self.ENABLE_EXECUTION_COSTS = _env_bool("ENABLE_EXECUTION_COSTS", self.ENABLE_EXECUTION_COSTS)
        self.SPOT_SLIPPAGE_BPS = _env_float("SPOT_SLIPPAGE_BPS", self.SPOT_SLIPPAGE_BPS)
        self.FUTURES_SLIPPAGE_BPS = _env_float("FUTURES_SLIPPAGE_BPS", self.FUTURES_SLIPPAGE_BPS)

        self.ENABLE_PARTIAL_FILLS = _env_bool("ENABLE_PARTIAL_FILLS", self.ENABLE_PARTIAL_FILLS)
        self.PARTIAL_FILL_PROB = _env_float("PARTIAL_FILL_PROB", self.PARTIAL_FILL_PROB)
        self.PARTIAL_FILL_MIN = _env_float("PARTIAL_FILL_MIN", self.PARTIAL_FILL_MIN)
        self.PARTIAL_FILL_MAX = _env_float("PARTIAL_FILL_MAX", self.PARTIAL_FILL_MAX)
        self.EXECUTION_RANDOM_SEED = _env_int("EXECUTION_RANDOM_SEED", self.EXECUTION_RANDOM_SEED)

        self.ENABLE_FUNDING_COSTS = _env_bool("ENABLE_FUNDING_COSTS", self.ENABLE_FUNDING_COSTS)
        self.FUTURES_FUNDING_RATE_PER_8H = _env_float(
            "FUTURES_FUNDING_RATE_PER_8H", self.FUTURES_FUNDING_RATE_PER_8H
        )

        self.FUTURES_CORE_SYMBOLS = _env_csv_list("FUTURES_CORE_SYMBOLS", self.FUTURES_CORE_SYMBOLS)
        self.FUTURES_PROBATION_CANDIDATES = _env_csv_list(
            "FUTURES_PROBATION_CANDIDATES", self.FUTURES_PROBATION_CANDIDATES
        )
        self.ENABLE_FUTURES_PROBATION = _env_bool(
            "ENABLE_FUTURES_PROBATION", self.ENABLE_FUTURES_PROBATION
        )
        self.FUTURES_PROBATION_COHORT_SIZE = _env_int(
            "FUTURES_PROBATION_COHORT_SIZE", self.FUTURES_PROBATION_COHORT_SIZE
        )
        self.FUTURES_PROBATION_ROTATE_HOURS = _env_int(
            "FUTURES_PROBATION_ROTATE_HOURS", self.FUTURES_PROBATION_ROTATE_HOURS
        )
        self.FUTURES_PROBATION_MIN_CLOSED = _env_int(
            "FUTURES_PROBATION_MIN_CLOSED", self.FUTURES_PROBATION_MIN_CLOSED
        )
        self.FUTURES_PROBATION_PROMOTE_MIN_WIN_RATE = _env_float(
            "FUTURES_PROBATION_PROMOTE_MIN_WIN_RATE", self.FUTURES_PROBATION_PROMOTE_MIN_WIN_RATE
        )
        self.FUTURES_PROBATION_PROMOTE_MIN_AVG_PCT = _env_float(
            "FUTURES_PROBATION_PROMOTE_MIN_AVG_PCT", self.FUTURES_PROBATION_PROMOTE_MIN_AVG_PCT
        )
        self.FUTURES_PROBATION_DEMOTE_MAX_WIN_RATE = _env_float(
            "FUTURES_PROBATION_DEMOTE_MAX_WIN_RATE", self.FUTURES_PROBATION_DEMOTE_MAX_WIN_RATE
        )
        self.FUTURES_PROBATION_DEMOTE_MAX_AVG_PCT = _env_float(
            "FUTURES_PROBATION_DEMOTE_MAX_AVG_PCT", self.FUTURES_PROBATION_DEMOTE_MAX_AVG_PCT
        )

        self.CB_MAX_CONSECUTIVE_LOSSES = _env_int("CB_MAX_CONSECUTIVE_LOSSES", self.CB_MAX_CONSECUTIVE_LOSSES)
        self.CB_DAILY_LOSS_LIMIT_PCT = _env_float("CB_DAILY_LOSS_LIMIT_PCT", self.CB_DAILY_LOSS_LIMIT_PCT)
        self.CB_MAX_DRAWDOWN_PCT = _env_float("MAX_DRAWDOWN_THRESHOLD", self.CB_MAX_DRAWDOWN_PCT)
        self.CB_COOLDOWN_MINUTES = _env_int("CB_COOLDOWN_MINUTES", self.CB_COOLDOWN_MINUTES)

        self.RL_LEARNING_RATE = _env_float("RL_LEARNING_RATE", self.RL_LEARNING_RATE)
        self.RL_EXPLORATION_RATE = _env_float("RL_EXPLORATION_RATE", self.RL_EXPLORATION_RATE)

        self.DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", self.DISCORD_WEBHOOK_URL)
        self.ALERT_ON_LARGE_LOSS_PCT = _env_float("ALERT_ON_LARGE_LOSS_PCT", self.ALERT_ON_LARGE_LOSS_PCT)

    def _apply_realism_profile(self):
        profile = (self.SIM_REALISM_PROFILE or "").strip().lower()
        if profile in ("off", "none"):
            self.ENABLE_EXECUTION_COSTS = False
            self.ENABLE_PARTIAL_FILLS = False
            self.ENABLE_FUNDING_COSTS = False
            return
        if profile == "strict":
            self.ENABLE_EXECUTION_COSTS = True
            self.SPOT_SLIPPAGE_BPS = max(self.SPOT_SLIPPAGE_BPS, 12.0)
            self.FUTURES_SLIPPAGE_BPS = max(self.FUTURES_SLIPPAGE_BPS, 20.0)
            self.ENABLE_PARTIAL_FILLS = True
            self.PARTIAL_FILL_PROB = max(self.PARTIAL_FILL_PROB, 0.30)
            self.PARTIAL_FILL_MIN = min(self.PARTIAL_FILL_MIN, 0.60)
            self.PARTIAL_FILL_MAX = min(self.PARTIAL_FILL_MAX, 0.95)
            self.ENABLE_FUNDING_COSTS = True
            self.FUTURES_FUNDING_RATE_PER_8H = max(self.FUTURES_FUNDING_RATE_PER_8H, 0.0002)
            self.SPOT_FEE_RATE = max(self.SPOT_FEE_RATE, 0.0007)
            self.FUTURES_FEE_RATE = max(self.FUTURES_FEE_RATE, 0.0006)
            return
        if profile == "normal":
            self.ENABLE_EXECUTION_COSTS = True
            self.ENABLE_PARTIAL_FILLS = True
            self.ENABLE_FUNDING_COSTS = True

    def summary(self) -> str:
        """Human-readable config summary for logging."""
        lines = [
            "CONFIGURATION:",
            f"  Paper Trading: {self.PAPER_TRADING}",
            f"  Sim Realism Profile: {self.SIM_REALISM_PROFILE}",
            f"  Check Interval: {self.CHECK_INTERVAL}s, Trade Interval: {self.TRADE_INTERVAL} cycles",
            f"  Spot: min_score={self.SPOT_MIN_SCORE}, max_pos={self.SPOT_MAX_POSITIONS}, "
            f"SL={self.SPOT_STOP_LOSS}%, TP={self.SPOT_TAKE_PROFIT}%",
            f"  Futures: leverage={self.FUTURES_LEVERAGE}x, max_pos={self.FUTURES_MAX_POSITIONS}, "
            f"SL={self.FUTURES_STOP_LOSS}%, TP={self.FUTURES_TAKE_PROFIT}%",
            f"  Circuit Breaker: {self.CB_MAX_CONSECUTIVE_LOSSES} losses, "
            f"daily {self.CB_DAILY_LOSS_LIMIT_PCT}%, dd {self.CB_MAX_DRAWDOWN_PCT}%",
            f"  Retry: Coinbase {self.COINBASE_RETRY_ATTEMPTS}x, Kraken {self.KRAKEN_RETRY_ATTEMPTS}x",
            f"  Discord Alerts: {'ON' if self.DISCORD_WEBHOOK_URL else 'OFF'}",
        ]
        return "\n".join(lines)


# Singleton config instance
config = TradingConfig()
