"""
CallBuyer Configuration — Momentum Call Buying Strategy (Left Leg)
Completely isolated from AlpacaBot, PutSeller, and CryptoBot.
Same Alpaca account, separate capital allocation.
"""
import os
from dataclasses import dataclass, field
from typing import List

# Load .env BEFORE dataclass fields evaluate (same fix as PutSeller)
try:
    from dotenv import load_dotenv as _load_dotenv_early
    _load_dotenv_early()
except ImportError:
    pass


def _env(key: str, default, cast=str):
    val = os.getenv(key, None)
    if val is None:
        return default
    if cast == bool:
        return val.lower() in ("1", "true", "yes")
    return cast(val)


@dataclass
class CallBuyerConfig:
    # ── API ──────────────────────────────────────────────
    API_KEY: str = ""
    API_SECRET: str = ""
    PAPER: bool = True
    BASE_URL: str = ""

    # ── Capital Allocation ───────────────────────────────
    # 15% of total Alpaca equity — small, high-risk/high-reward
    # AlpacaBot 50% + PutSeller 35% + CallBuyer 15% = 100%
    ALLOCATION_PCT: float = _env("ALLOCATION_PCT", 0.15, float)
    MAX_POSITION_RISK_PCT: float = 0.02     # 2% of allocation per trade (small bets)
    MAX_POSITIONS: int = 4                   # max concurrent calls (concentrate capital)
    MAX_PER_UNDERLYING: int = 2              # max 2 calls per stock

    # ── Strategy: Buy ITM Calls on Momentum Breakout ────
    # DTE targeting — enough time for move, avoid theta crush
    MIN_DTE: int = 21                        # minimum days to expiration
    MAX_DTE: int = 60                        # maximum days to expiration
    TARGET_DTE: int = 45                     # sweet spot (30-60d proven best for call buying)
    EARNINGS_BUFFER_DAYS: int = 14           # skip symbols with earnings within 14d (was MAX_DTE=60 — blocked everything)

    # Strike selection — ITM for higher delta and better directional momentum capture
    MIN_OTM_PCT: float = -0.05              # allow up to 5% ITM (negative = ITM)
    MAX_OTM_PCT: float = 0.02               # max 2% OTM buffer
    TARGET_OTM_PCT: float = -0.02           # target 2% ITM (~delta 0.65)

    # ── Momentum Signal Thresholds ───────────────────────
    MIN_RSI: float = 40.0                    # RSI 40+ = bullish (buy pullbacks, not tops)
    MAX_RSI: float = 85.0                    # allow momentum breakouts (was 70 — blocked ALL scanner candidates)
    MIN_VOLUME_SURGE: float = 1.3            # 30% above avg volume
    MIN_BREAKOUT_SCORE: float = 0.3          # composite breakout signal threshold
    MIN_SMA_ALIGNMENT: float = 0.5           # trend alignment score

    # ── ML Model ─────────────────────────────────────────
    ML_ENABLED: bool = True                  # use ML scoring when model is trained
    ML_WARMUP_TRADES: int = 15               # trades before first training (lower to bootstrap faster)
    ML_RETRAIN_TRADES: int = 15              # retrain every N new trades
    ML_MIN_ACCURACY: float = 0.53            # min accuracy to accept model
    ML_WEIGHT: float = 0.40                  # ML weight in ensemble (rules get 0.60)
    MODEL_DIR: str = "models"
    MODEL_FILE: str = "callbuyer_model.joblib"
    FEATURES_LOG: str = "data/state/features_log.json"

    # ── Meta-Learner ─────────────────────────────────────
    META_CONFIDENCE_THRESHOLD: float = 0.30  # min ensemble score to trade (was 0.45 — impossible during ML warmup where conf=score/10)
    META_MIN_RULE_SCORE: float = 2           # min raw rule score (out of 10) (was 3 — typical stocks score 2-4)
    META_WINDOW: int = 40                    # rolling window for accuracy
    META_UPDATE_CYCLES: int = 20             # update thresholds every N cycles

    # ── Exit Rules ───────────────────────────────────────
    TAKE_PROFIT_PCT: float = 0.50            # close at 50% gain
    STOP_LOSS_PCT: float = -0.25             # close at 25% loss (cut fast — 2:1 R:R)
    MIN_DTE_EXIT: int = 7                    # close at 7 DTE (avoid theta acceleration zone)
    TRAILING_STOP_PCT: float = 0.15          # 15% trailing stop (tighter to lock gains)

    # ── Morning Window (mild time-of-day adjustment) ────
    # Momentum breakouts are strongest at market open.
    # Morning (9:30-11:00 ET): small confidence boost → slightly more entries.
    # Afternoon (after 11:00 ET): small confidence penalty → only strong setups.
    # Exits are NOT affected — they remain purely rule-based.
    MORNING_WINDOW_ENABLED: bool = True
    MORNING_WINDOW_END_MIN: int = 90          # minutes after 9:30 → 11:00 AM ET
    MORNING_CONF_BOOST: float = 0.03          # confidence boosted by 0.03 in morning
    MORNING_RULE_REDUCTION: float = 0.5       # min_rule_score reduced by 0.5
    AFTERNOON_CONF_PENALTY: float = 0.02      # confidence penalized by 0.02 in afternoon
    AFTERNOON_RULE_INCREASE: float = 0.5      # min_rule_score raised by 0.5

    # ── Timing ───────────────────────────────────────────
    SCAN_INTERVAL_SEC: int = 600             # 10 min between opportunity scans
    CHECK_INTERVAL_SEC: int = 180            # 3 min between position checks
    MARKET_OPEN_HOUR: int = 7               # 7:30 AM MT = 9:30 AM ET
    MARKET_OPEN_MIN: int = 30
    MARKET_CLOSE_HOUR: int = 14              # 2:00 PM MT = 4:00 PM ET
    MARKET_CLOSE_MIN: int = 0
    NO_OPEN_FIRST_MIN: int = 0               # scan immediately at open
    NO_OPEN_LAST_MIN: int = 30               # skip last 30 min

    # ── Watchlist — Momentum-friendly, high-beta names ───
    WATCHLIST: List[str] = field(default_factory=lambda: [
        # ETFs — broadest liquidity, best fills
        "SPY", "QQQ", "IWM",
        # Mega-cap liquid movers (proven momentum names)
        "NVDA", "TSLA", "AMD", "META", "AMZN",
        # High-beta with good options liquidity
        "NFLX", "AVGO",
    ])

    # ── Order Tagging ────────────────────────────────────
    ORDER_PREFIX: str = "cb_"                # prefix for client_order_id
    BOT_NAME: str = "CallBuyer"

    # ── Paths ────────────────────────────────────────────
    LOG_DIR: str = "logs"
    STATE_FILE: str = "data/state/bot_state.json"
    RISK_STATE_FILE: str = "data/state/risk_state.json"
    POSITIONS_FILE: str = "data/state/positions.json"
    TRADE_LOG: str = "data/trades.csv"

    LOG_LEVEL: str = "INFO"
    VERBOSE: bool = True

    def __post_init__(self):
        if not self.BASE_URL:
            self.BASE_URL = (
                "https://paper-api.alpaca.markets"
                if self.PAPER
                else "https://api.alpaca.markets"
            )
        self._load_dotenv()

    def _load_dotenv(self):
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

    def summary(self) -> str:
        mode = "PAPER" if self.PAPER else "*** LIVE ***"
        return (
            f"CallBuyer Config [{mode}]\n"
            f"  Allocation: {self.ALLOCATION_PCT:.0%} of account equity\n"
            f"  Strategy: Buy ITM calls on momentum breakouts\n"
            f"  Watchlist: {', '.join(self.WATCHLIST[:6])}... ({len(self.WATCHLIST)} symbols)\n"
            f"  DTE: {self.MIN_DTE}-{self.MAX_DTE}d (target {self.TARGET_DTE}d)\n"
            f"  OTM: {self.MIN_OTM_PCT:.0%}-{self.MAX_OTM_PCT:.0%}\n"
            f"  ML: {'ON' if self.ML_ENABLED else 'OFF'} | weight {self.ML_WEIGHT:.0%}\n"
            f"  Exit: TP {self.TAKE_PROFIT_PCT:.0%} | SL {self.STOP_LOSS_PCT:.0%} | DTE {self.MIN_DTE_EXIT}\n"
            f"  Max positions: {self.MAX_POSITIONS}\n"
            f"  API Keys: {'SET' if self.has_keys else 'MISSING'}"
        )


CFG = CallBuyerConfig()
