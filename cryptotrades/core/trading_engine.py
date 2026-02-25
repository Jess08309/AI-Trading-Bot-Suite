from __future__ import annotations
import sys
import os
import logging
import time
import signal
import json
import csv
import smtplib
import hashlib
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

# Configuration module
from utils.config import config
try:
    from utils.rl_agent import RLTradingAgent
except Exception:
    RLTradingAgent = None

# Optional imports with fallbacks
try:
    from coinbase.wallet.client import Client
    COINBASE_AVAILABLE = True
except ImportError:
    COINBASE_AVAILABLE = False
    Client = None

# ============================================================
# PRODUCTION CONFIGURATION - Centralized & Simplified
# ============================================================

@dataclass
class TradingConfig:
    """Production trading configuration."""
    # Mode
    PAPER_TRADING: bool = True
    LIVE_TRADING_WARNING: bool = False
    DRY_RUN: bool = False

    # Capital
    INITIAL_SPOT_BALANCE: float = 2500.0
    INITIAL_FUTURES_BALANCE: float = 2500.0

    # Timing
    RISK_CHECK_INTERVAL: int = 60          # 1 minute
    TRADE_CYCLE_INTERVAL: int = 5          # 5 minutes (5 * 60 seconds)
    MODEL_RETRAIN_HOURS: int = 8           # Retrain every 8h with new indicators

    # Data/API robustness
    PRICE_FETCH_RETRIES: int = 3
    PRICE_FETCH_RETRY_DELAY_SEC: float = 1.0
    PRICE_FETCH_RATE_LIMIT_SEC: float = 0.2

    # Direction mode: "both", "short_only", "long_only"
    DIRECTION_MODE: str = os.getenv("DIRECTION_MODE", "both")

    # Symbol blacklist — comma-separated, env-overridable
    SYMBOL_BLACKLIST: Tuple[str, ...] = tuple(
        s.strip() for s in os.getenv(
            "SYMBOL_BLACKLIST",
            ""
        ).split(",") if s.strip()
    )

    # Symbols
    SPOT_SYMBOLS: Tuple[str, ...] = (
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
        "DOGE-USD", "LINK-USD", "XRP-USD", "LTC-USD", "UNI-USD",
        "XLM-USD", "BCH-USD", "DOT-USD", "MATIC-USD", "ATOM-USD",
        "NEAR-USD", "AAVE-USD",
        "PAXG-USD",  # Pax Gold — tokenized physical gold
    )
    FUTURES_SYMBOLS: Tuple[str, ...] = (
        "PI_XBTUSD", "PI_ETHUSD", "PI_SOLUSD", "PI_ADAUSD", "PI_DOGEUSD",
        "PI_LINKUSD", "PI_AVAXUSD", "PI_DOTUSD", "PI_BCHUSD", "PI_LTCUSD",
        "PI_XRPUSD", "PI_ATOMUSD"
    )

    # Risk Limits - STRICT for production
    MAX_POSITIONS_SPOT: int = 12
    MAX_POSITIONS_FUTURES: int = 8
    MAX_POSITIONS_PER_SYMBOL_SPOT: int = 1
    MAX_POSITIONS_PER_SYMBOL_FUTURES: int = 2
    MAX_POSITION_PCT: float = 0.08         # 8% max per position
    MIN_POSITION_PCT: float = 0.02         # 2% min

    # Stop Loss / Take Profit  (1.25:1 R:R — needs >44% WR)
    STOP_LOSS_PCT: float = -2.0            # -2% stop (cut losers fast)
    TAKE_PROFIT_PCT: float = 2.5           # 2.5% target (more achievable)
    TRAILING_STOP_PCT: float = 1.5         # 1.5% trailing (lock profits earlier)

    # Futures specific
    FUTURES_LEVERAGE: int = 2
    FUTURES_STOP_LOSS: float = -3.0        # -3% for leveraged positions

    # Stale profit-decay exit
    STALE_EXIT_PROFIT_ARM_PCT: float = 1.5
    STALE_EXIT_DECAY_PCT: float = 0.8
    STALE_EXIT_STALE_MIN_SPOT: int = 45
    STALE_EXIT_STALE_MIN_FUTURES: int = 25

    # Max hold time — close flat positions that go nowhere
    MAX_HOLD_HOURS_SPOT: int = 4           # Close spot after 4h if flat
    MAX_HOLD_HOURS_FUTURES: int = 3        # Close futures after 3h if flat
    MAX_HOLD_FLAT_BAND_PCT: float = 0.5    # |P/L| < 0.5% counts as "flat"

    # Circuit Breaker
    CB_MAX_CONSECUTIVE_LOSSES: int = 5     # Stop after 5 losses in a row
    CB_DAILY_LOSS_LIMIT_PCT: float = -4.0  # Stop if down 4% daily
    CB_MAX_DRAWDOWN_PCT: float = -8.0      # Stop if down 8% from peak

    # Entry Thresholds - QUALITY GATES
    # Confidence = max(up_prob, down_prob), range 0.50-1.0
    MIN_ML_CONFIDENCE: float = 0.66        # Model must predict >=66% one direction
    MIN_ENSEMBLE_SCORE: float = 0.62       # Composite ML+trend quality score
    MAX_RSI_LONG: float = 62.0             # Don't buy if RSI > 62 (tighter)
    MIN_RSI_SHORT: float = 38.0            # Don't short if RSI < 38 (tighter)

    # Signal quality filters
    SIDE_MARKET_FILTER: bool = True        # Skip SIDE-trend signals
    SIDE_MARKET_ML_OVERRIDE: float = 0.75  # ML conf above this can override SIDE filter
    COUNTER_TREND_ML_OVERRIDE: float = 0.92  # ML conf above this can trade against trend (nearly disabled)
    MIN_MODEL_TEST_ACCURACY: float = 0.58  # Reject models below this OOS accuracy
    SYMBOL_PAUSE_CONSECUTIVE_LOSSES: int = 3  # Pause symbol after 3 straight losses

    # Direction Performance Tracker — auto-pause losing direction
    DIRECTION_PAUSE_LOOKBACK: int = 20     # Rolling window of recent trades per direction
    DIRECTION_PAUSE_MIN_WR: float = 0.30   # Pause direction if WR drops below this
    DIRECTION_PAUSE_HOURS: float = 1.0     # How long to pause a failing direction
    DIRECTION_PAUSE_MIN_TRADES: int = 10   # Need at least this many trades to judge

    # Sentiment
    USE_SENTIMENT: bool = True
    SENTIMENT_THRESHOLD: float = 0.15      # Minimum |sentiment| to consider

    # Trend
    TREND_LOOKBACK: int = 20               # 20 periods for trend detection
    MIN_TREND_SLOPE: float = 0.0005        # Minimum slope to consider trending

    # Correlation
    MAX_CORRELATION: float = 0.75          # Tight correlation limit

    # Execution quality
    SLIPPAGE_BPS_SPOT: float = 5.0         # 0.05%
    SLIPPAGE_BPS_FUTURES: float = 8.0      # 0.08%

    # Liquidity gate
    USE_LIQUIDITY_GATE: bool = False
    ORDERBOOK_LEVELS: int = 10
    MIN_DEPTH_USD_SPOT: float = 10000.0
    MIN_DEPTH_USD_FUTURES: float = 30000.0
    DEPTH_TO_TRADE_RATIO: float = 2.0
    LIQUIDITY_GATE_FAIL_OPEN: bool = True

    # Alerts
    ALERT_DRAWDOWN_PCT: float = -7.0
    ALERT_DAILY_LOSS_PCT: float = -4.0
    ALERT_COOLDOWN_MIN: int = 60

    def __post_init__(self):
        """Apply env overrides and validate configuration for safe runtime defaults."""
        def _env_bool(name: str, default: bool) -> bool:
            value = os.getenv(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        def _env_int(name: str, current: int) -> int:
            value = os.getenv(name)
            if value is None:
                return current
            try:
                return int(value)
            except ValueError:
                return current

        def _env_float(name: str, current: float) -> float:
            value = os.getenv(name)
            if value is None:
                return current
            try:
                return float(value)
            except ValueError:
                return current

        aggressive_burst = _env_bool("AGGRESSIVE_FUTURES_BURST", False)
        if aggressive_burst:
            # Burst mode only controls speed and position limits.
            # It NO LONGER disables quality gates (ML confidence, correlation, circuit breakers).
            self.TRADE_CYCLE_INTERVAL = 2

        self.TRADE_CYCLE_INTERVAL = _env_int("TRADE_CYCLE_INTERVAL_OVERRIDE", self.TRADE_CYCLE_INTERVAL)
        self.MODEL_RETRAIN_HOURS = _env_int("MODEL_RETRAIN_HOURS_OVERRIDE", self.MODEL_RETRAIN_HOURS)
        self.MAX_POSITIONS_SPOT = _env_int("MAX_POSITIONS_SPOT_OVERRIDE", self.MAX_POSITIONS_SPOT)
        self.MAX_POSITIONS_FUTURES = _env_int("MAX_POSITIONS_FUTURES_OVERRIDE", self.MAX_POSITIONS_FUTURES)
        self.MAX_POSITIONS_PER_SYMBOL_SPOT = _env_int("MAX_POSITIONS_PER_SYMBOL_SPOT_OVERRIDE", self.MAX_POSITIONS_PER_SYMBOL_SPOT)
        self.MAX_POSITIONS_PER_SYMBOL_FUTURES = _env_int("MAX_POSITIONS_PER_SYMBOL_FUTURES_OVERRIDE", self.MAX_POSITIONS_PER_SYMBOL_FUTURES)
        self.FUTURES_LEVERAGE = _env_int("FUTURES_LEVERAGE_OVERRIDE", self.FUTURES_LEVERAGE)
        self.STALE_EXIT_STALE_MIN_SPOT = _env_int("STALE_EXIT_STALE_MIN_SPOT_OVERRIDE", self.STALE_EXIT_STALE_MIN_SPOT)
        self.STALE_EXIT_STALE_MIN_FUTURES = _env_int("STALE_EXIT_STALE_MIN_FUTURES_OVERRIDE", self.STALE_EXIT_STALE_MIN_FUTURES)
        self.CB_MAX_CONSECUTIVE_LOSSES = _env_int("CB_MAX_CONSECUTIVE_LOSSES_OVERRIDE", self.CB_MAX_CONSECUTIVE_LOSSES)

        self.MIN_ML_CONFIDENCE = _env_float("MIN_ML_CONFIDENCE_OVERRIDE", self.MIN_ML_CONFIDENCE)
        self.MIN_ENSEMBLE_SCORE = _env_float("MIN_ENSEMBLE_SCORE_OVERRIDE", self.MIN_ENSEMBLE_SCORE)
        self.MAX_CORRELATION = _env_float("MAX_CORRELATION_OVERRIDE", self.MAX_CORRELATION)
        self.CB_DAILY_LOSS_LIMIT_PCT = _env_float("CB_DAILY_LOSS_LIMIT_PCT_OVERRIDE", self.CB_DAILY_LOSS_LIMIT_PCT)
        self.CB_MAX_DRAWDOWN_PCT = _env_float("CB_MAX_DRAWDOWN_PCT_OVERRIDE", self.CB_MAX_DRAWDOWN_PCT)
        self.STALE_EXIT_PROFIT_ARM_PCT = _env_float("STALE_EXIT_PROFIT_ARM_PCT_OVERRIDE", self.STALE_EXIT_PROFIT_ARM_PCT)
        self.STALE_EXIT_DECAY_PCT = _env_float("STALE_EXIT_DECAY_PCT_OVERRIDE", self.STALE_EXIT_DECAY_PCT)

        # New quality filter overrides
        self.SIDE_MARKET_FILTER = _env_bool("SIDE_MARKET_FILTER_OVERRIDE", self.SIDE_MARKET_FILTER)
        self.SIDE_MARKET_ML_OVERRIDE = _env_float("SIDE_MARKET_ML_OVERRIDE_OVERRIDE", self.SIDE_MARKET_ML_OVERRIDE)
        self.COUNTER_TREND_ML_OVERRIDE = _env_float("COUNTER_TREND_ML_OVERRIDE_OVERRIDE", self.COUNTER_TREND_ML_OVERRIDE)
        self.MAX_HOLD_HOURS_SPOT = _env_int("MAX_HOLD_HOURS_SPOT_OVERRIDE", self.MAX_HOLD_HOURS_SPOT)
        self.MAX_HOLD_HOURS_FUTURES = _env_int("MAX_HOLD_HOURS_FUTURES_OVERRIDE", self.MAX_HOLD_HOURS_FUTURES)
        self.SYMBOL_PAUSE_CONSECUTIVE_LOSSES = _env_int("SYMBOL_PAUSE_CONSECUTIVE_LOSSES_OVERRIDE", self.SYMBOL_PAUSE_CONSECUTIVE_LOSSES)
        self.MAX_POSITION_PCT = _env_float("MAX_POSITION_PCT_OVERRIDE", self.MAX_POSITION_PCT)
        self.MIN_POSITION_PCT = _env_float("MIN_POSITION_PCT_OVERRIDE", self.MIN_POSITION_PCT)
        self.STOP_LOSS_PCT = _env_float("STOP_LOSS_PCT_OVERRIDE", self.STOP_LOSS_PCT)
        self.TAKE_PROFIT_PCT = _env_float("TAKE_PROFIT_PCT_OVERRIDE", self.TAKE_PROFIT_PCT)
        self.TRAILING_STOP_PCT = _env_float("TRAILING_STOP_PCT_OVERRIDE", self.TRAILING_STOP_PCT)
        self.FUTURES_STOP_LOSS = _env_float("FUTURES_STOP_LOSS_OVERRIDE", self.FUTURES_STOP_LOSS)

        if not 0 < self.MAX_POSITION_PCT <= 1.0:
            raise ValueError("MAX_POSITION_PCT must be in (0, 1]")
        if not 0 < self.MIN_POSITION_PCT <= self.MAX_POSITION_PCT:
            raise ValueError("MIN_POSITION_PCT must be in (0, MAX_POSITION_PCT]")
        if self.STOP_LOSS_PCT >= 0 or self.FUTURES_STOP_LOSS >= 0:
            raise ValueError("STOP_LOSS_PCT and FUTURES_STOP_LOSS must be negative")
        if self.FUTURES_LEVERAGE < 1:
            raise ValueError("FUTURES_LEVERAGE must be >= 1")
        if self.MAX_POSITIONS_PER_SYMBOL_SPOT < 1 or self.MAX_POSITIONS_PER_SYMBOL_FUTURES < 1:
            raise ValueError("Per-symbol position limits must be >= 1")
        if self.STALE_EXIT_PROFIT_ARM_PCT <= 0 or self.STALE_EXIT_DECAY_PCT <= 0:
            raise ValueError("Stale-exit profit arm and decay values must be > 0")
        if self.STALE_EXIT_STALE_MIN_SPOT <= 0 or self.STALE_EXIT_STALE_MIN_FUTURES <= 0:
            raise ValueError("Stale-exit stale minutes must be > 0")
        if not 0 <= self.MIN_ML_CONFIDENCE <= 1:
            raise ValueError("MIN_ML_CONFIDENCE must be between 0 and 1")
        if not 0 <= self.MIN_ENSEMBLE_SCORE <= 1:
            raise ValueError("MIN_ENSEMBLE_SCORE must be between 0 and 1")
        if self.MAX_CORRELATION <= 0 or self.MAX_CORRELATION > 1:
            raise ValueError("MAX_CORRELATION must be in (0, 1]")
        if self.SLIPPAGE_BPS_SPOT < 0 or self.SLIPPAGE_BPS_FUTURES < 0:
            raise ValueError("Slippage bps must be >= 0")
        if self.ORDERBOOK_LEVELS <= 0:
            raise ValueError("ORDERBOOK_LEVELS must be > 0")
        if self.MIN_DEPTH_USD_SPOT < 0 or self.MIN_DEPTH_USD_FUTURES < 0:
            raise ValueError("Min depth must be >= 0")
        if self.DEPTH_TO_TRADE_RATIO <= 0:
            raise ValueError("DEPTH_TO_TRADE_RATIO must be > 0")


# Global config instance
cfg = TradingConfig()

# ============================================================
# LOGGING SETUP - ASCII Safe for Windows
# ============================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup ASCII-safe logging for Windows compatibility."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers
    logger.handlers = []
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers = []
    root_logger.setLevel(logging.WARNING)

    # Format without unicode
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    fh = logging.FileHandler(
        os.path.join(log_dir, f'trading_{datetime.now().strftime("%Y%m%d")}.log'),
        encoding='utf-8'  # UTF-8 for file is fine
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler - ASCII only
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Position:
    """Unified position tracking."""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    max_price: float  # For trailing stops
    peak_pnl_pct: float = 0.0
    stale_since: Optional[datetime] = None
    ml_confidence: float = 0.0
    entry_reason: str = ""


@dataclass
class Signal:
    """Trading signal with metadata."""
    symbol: str
    direction: str
    confidence: float
    ml_score: float
    rsi: float
    trend: str  # 'UP', 'DOWN', 'SIDE'
    sentiment: float
    correlation: float
    volatility: float
    trend_slope: float
    reason: str
    timestamp: datetime


@dataclass
class ShadowPosition:
    """Shadow-mode position tracking for strategy comparison."""
    symbol: str
    direction: str
    entry_price: float
    size: float
    entry_cycle: int
    stop_loss_pct: float
    max_price: float
    peak_pnl_pct: float
    stale_since_cycle: Optional[int]
    is_futures: bool
    source: str
    rl_state: str = ""
    rl_action: float = 1.0
    rl_confidence: float = 0.5

# ============================================================
# MARKET DATA & INDICATORS
# ============================================================

class MarketData:
    """Clean market data handler."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.price_history: Dict[str, List[float]] = {}
        self.last_update: Dict[str, float] = {}

    def update_price(self, symbol: str, price: float):
        """Add price to history."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(price)
        self.last_update[symbol] = time.time()

        # Keep last 500 prices (~8 hours at 1-min)
        if len(self.price_history[symbol]) > 500:
            self.price_history[symbol].pop(0)

    def get_candles(self, symbol: str, period: int = 10) -> List[float]:
        """Aggregate 1-min prices to N-min candles."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < period:
            return prices

        candles = []
        for i in range(period - 1, len(prices), period):
            candles.append(prices[i])

        # Include latest if not already included
        if candles and prices[-1] != candles[-1]:
            candles.append(prices[-1])

        return candles

    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """Calculate RSI."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_trend(self, symbol: str, lookback: int = 20) -> Tuple[str, float]:
        """Determine trend direction and strength."""
        prices = self.get_candles(symbol, 10)  # Use 10-min candles
        if len(prices) < lookback:
            return "SIDE", 0.0

        recent = prices[-lookback:]
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)

        # Normalize slope
        normalized_slope = slope / np.mean(recent)

        if normalized_slope > cfg.MIN_TREND_SLOPE:
            return "UP", normalized_slope
        elif normalized_slope < -cfg.MIN_TREND_SLOPE:
            return "DOWN", normalized_slope
        else:
            return "SIDE", normalized_slope

    def calculate_volatility(self, symbol: str, lookback: int = 20) -> float:
        """Calculate recent volatility."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < lookback:
            return 0.01

        returns = np.diff(prices[-lookback:]) / prices[-lookback:-1]
        return float(np.std(returns))

# ============================================================
# ML MODEL — 15-Indicator Momentum Suite
# ============================================================

# Import the curated indicator library
try:
    from utils.technical_indicators import compute_all_indicators as _compute_indicators
except ImportError:
    try:
        from cryptotrades.utils.technical_indicators import compute_all_indicators as _compute_indicators
    except ImportError:
        _compute_indicators = None

# Canonical feature order — must match technical_indicators.compute_all_indicators keys
_ML_FEATURE_NAMES = [
    "rsi_14", "macd_histogram", "stoch_k", "cci_20", "roc_10",
    "momentum_10", "williams_r", "ultimate_osc", "trix_15", "cmo_14",
    "atr_14", "trend_strength", "bb_position", "mean_reversion", "vol_ratio",
]


class MLModel:
    """ML predictor using 15 curated momentum + complementary indicators."""

    def __init__(self, model_path: str = "models/trading_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.logger = logging.getLogger("TradingBot")
        self.feature_names = list(_ML_FEATURE_NAMES)

    def load_or_train(self, price_history: Dict[str, List[float]]):
        """Load existing model or train new one."""
        needs_retrain = False
        if os.path.exists(self.model_path):
            try:
                candidate = load(self.model_path)
                # Verify feature count matches — old models had 10 features
                if hasattr(candidate, 'n_features_in_') and candidate.n_features_in_ != len(self.feature_names):
                    self.logger.warning(
                        f"Model feature mismatch: model has {candidate.n_features_in_} features, "
                        f"need {len(self.feature_names)}. Will retrain."
                    )
                    needs_retrain = True
                else:
                    self.model = candidate
                    self.logger.info("ML Model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load model: {e}")
                needs_retrain = True
        else:
            needs_retrain = True

        if needs_retrain and price_history:
            self._train(price_history)

    def _aggregate_to_hourly(self, prices: List[float], period: int = 60) -> List[float]:
        """Aggregate 1-min prices into N-minute candle closes for less noisy training."""
        if len(prices) < period:
            return prices
        candles = []
        for i in range(period - 1, len(prices), period):
            candles.append(prices[i])
        # always include the latest price
        if candles and prices[-1] != candles[-1]:
            candles.append(prices[-1])
        return candles

    def _train(self, price_history: Dict[str, List[float]]):
        """Train direction model on 15-indicator momentum suite."""
        self.logger.info("Training ML model on 15-indicator momentum suite...")

        if _compute_indicators is None:
            self.logger.error("compute_all_indicators not available — cannot train")
            return

        X, y = [], []

        for symbol, prices in price_history.items():
            # Build candidate series to train from (prefer longer timeframe)
            # Each entry: (series, target_threshold, flat_filter)
            series_list = []
            # Use most recent data only to avoid stale patterns dominating
            max_points = 2000  # ~33 hours of 1-min data — keeps model responsive

            # Trim to recent data to keep model responsive to regime changes
            recent_prices = prices[-max_points:] if len(prices) > max_points else prices

            # Aggregate to ~1-hour candles for less noisy targets
            hourly = self._aggregate_to_hourly(recent_prices, 60)
            if len(hourly) >= 60:
                series_list.append((hourly, 0.005, 0.002))  # 0.5% target, 0.2% flat filter

            # 10-min candles as middle ground
            candles_10m = self._aggregate_to_hourly(recent_prices, 10)
            if len(candles_10m) >= 60:
                series_list.append((candles_10m, 0.003, 0.001))  # 0.3% target, 0.1% flat filter

            # Raw 1-min as fallback (noisier but at least provides training data)
            if len(recent_prices) >= 120 and not series_list:
                series_list.append((recent_prices, 0.002, 0.0008))  # 0.2% target, 0.08% flat filter

            for series, target_thresh, flat_thresh in series_list:
                if len(series) < 60:
                    continue
                for i in range(50, len(series) - 10):
                    features = self._extract_features(series[:i])
                    if features is not None:
                        future_return = (series[i + 10] - series[i]) / series[i]
                        if abs(future_return) < flat_thresh:
                            continue  # Skip ambiguous flat moves
                        label = 1 if future_return > target_thresh else 0
                        X.append(features)
                        y.append(label)

        if len(X) < 100:
            self.logger.warning("Insufficient data for training")
            return

        X, y = np.array(X), np.array(y)

        # Check class balance — reject heavily skewed datasets
        class_ratio = np.mean(y)
        n_up = int(np.sum(y))
        n_down = len(y) - n_up
        self.logger.info(f"Training data: {len(y)} samples, {n_up} UP ({class_ratio:.1%}), {n_down} DOWN ({1-class_ratio:.1%})")

        if class_ratio < 0.35 or class_ratio > 0.65:
            self.logger.warning(
                f"Training data imbalanced ({class_ratio:.1%} positive). "
                f"Undersampling majority class to restore balance."
            )
            # Undersample majority class instead of giving up
            up_idx = np.where(y == 1)[0]
            down_idx = np.where(y == 0)[0]
            minority_count = min(len(up_idx), len(down_idx))
            if minority_count < 50:
                self.logger.warning("Too few samples in minority class after undersample. Skipping.")
                return
            if len(up_idx) > len(down_idx):
                up_idx = np.random.choice(up_idx, size=minority_count, replace=False)
            else:
                down_idx = np.random.choice(down_idx, size=minority_count, replace=False)
            balanced_idx = np.concatenate([up_idx, down_idx])
            np.random.shuffle(balanced_idx)
            X, y = X[balanced_idx], y[balanced_idx]
            class_ratio = np.mean(y)
            self.logger.info(f"After undersampling: {len(y)} samples, ratio={class_ratio:.1%}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, stratify=y
        )

        # Use balanced sample weights to prevent directional bias
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_train)

        candidate = GradientBoostingClassifier(
            n_estimators=200,       # More trees for 15 features
            max_depth=4,            # Slightly deeper to capture indicator interactions
            learning_rate=0.05,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        candidate.fit(X_train, y_train, sample_weight=sample_weights)

        train_score = candidate.score(X_train, y_train)
        test_score = candidate.score(X_test, y_test)

        self.logger.info(f"Model trained — Train: {train_score:.2%}, Test: {test_score:.2%}")

        # Log top feature importances
        try:
            importances = candidate.feature_importances_
            top = sorted(zip(self.feature_names, importances), key=lambda x: -x[1])[:5]
            top_str = ", ".join(f"{n}={v:.3f}" for n, v in top)
            self.logger.info(f"Top features: {top_str}")
        except Exception:
            pass

        # Quality gate: reject models that don't beat random
        if test_score < cfg.MIN_MODEL_TEST_ACCURACY:
            self.logger.warning(
                f"Model REJECTED: test accuracy {test_score:.2%} < "
                f"minimum {cfg.MIN_MODEL_TEST_ACCURACY:.2%}. Using previous model or none."
            )
            return

        self.model = candidate
        self.logger.info(f"Model ACCEPTED: test accuracy {test_score:.2%}")

        # Save (with accuracy in filename for tracking)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        dump(self.model, self.model_path)
        # Also save a versioned copy
        try:
            versioned = self.model_path.replace(
                '.joblib',
                f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}_acc{int(test_score*100)}%.joblib'
            )
            dump(self.model, versioned)
        except Exception:
            pass

    def _extract_features(self, prices: List[float]) -> Optional[List[float]]:
        """Extract 15 indicator features using compute_all_indicators."""
        if len(prices) < 30 or _compute_indicators is None:
            return None

        try:
            feat_dict = _compute_indicators(prices)
            if not feat_dict:
                return None
            return [float(feat_dict.get(name, 0.0)) for name in _ML_FEATURE_NAMES]
        except Exception:
            return None

    def predict(self, prices: List[float]) -> Dict[str, float]:
        """Predict direction and confidence."""
        if self.model is None or len(prices) < 30:
            return {"direction": 0.5, "confidence": 0.5, "up_prob": 0.5}

        features = self._extract_features(prices)
        if features is None:
            return {"direction": 0.5, "confidence": 0.5, "up_prob": 0.5}

        features_arr = np.array(features).reshape(1, -1)
        features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            proba = self.model.predict_proba(features_arr)[0]
        except Exception:
            return {"direction": 0.5, "confidence": 0.5, "up_prob": 0.5}

        up_prob = proba[1]
        confidence = max(up_prob, proba[0])

        return {
            "direction": up_prob,
            "confidence": confidence,
            "up_prob": up_prob,
            "down_prob": proba[0]
        }

# ============================================================
# SENTIMENT ANALYZER - Fixed Logic
# ============================================================

class SentimentAnalyzer:
    """Enhanced multi-source sentiment analysis.

    Blends 9+ free data sources into a single composite score:
      - Fear & Greed Index (alternative.me)
      - CoinGecko global market data
      - Kraken funding rates (crowding signal)
      - Kraken open interest
      - Dollar strength proxy
      - Bitcoin mempool congestion
      - Stablecoin supply ratio
      - Top coin 24h momentum
      - Kraken long/short ratio (basis spread)
    All sources are 100% free with no API keys required.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.cache = {}
        self.last_fetch = 0
        self.fng_url = "https://api.alternative.me/fng/?limit=1&format=json"
        self.last_source = "none"
        self.last_error = ""
        self.last_value = 0.0
        self.last_success_at: Optional[datetime] = None
        self.enhanced_signals: Dict = {}
        self._enhanced_available = False

        # Try to import enhanced sentiment module
        try:
            from utils.enhanced_sentiment import fetch_all_enhanced_signals, get_enhanced_sentiment_score
            self._fetch_enhanced = fetch_all_enhanced_signals
            self._get_enhanced_score = get_enhanced_sentiment_score
            self._enhanced_available = True
            logger.info("[SENTIMENT] Enhanced multi-source sentiment loaded (9 free sources)")
        except ImportError:
            self._fetch_enhanced = None
            self._get_enhanced_score = None
            logger.info("[SENTIMENT] Enhanced sentiment not available, using Fear & Greed only")

    def _decay_stale_sentiment(self, raw_value: float) -> float:
        """Decay sentiment toward neutral (0.0) if last success is too old.
        After 60 min without a fresh reading, linearly decay to 0 over the
        following 60 min.  This prevents a stuck extreme-fear reading from
        permanently biasing every signal."""
        if self.last_success_at is None:
            return 0.0  # Never fetched — neutral
        staleness = (datetime.now() - self.last_success_at).total_seconds()
        STALE_START_SEC = 3600   # 60 min
        STALE_FULL_SEC  = 7200   # 120 min — fully decayed
        if staleness <= STALE_START_SEC:
            return raw_value
        decay = min(1.0, (staleness - STALE_START_SEC) / (STALE_FULL_SEC - STALE_START_SEC))
        return raw_value * (1.0 - decay)

    def fetch_sentiment(self) -> Tuple[float, Dict[str, float]]:
        """
        Fetch and return sentiment from multiple free sources.
        Returns: (global_sentiment, coin_sentiment_dict)
        """
        # Rate limit - fetch every 10 minutes max
        if time.time() - self.last_fetch < 600:
            self.last_source = "cache"
            cached_global = self.cache.get('global', 0.0)
            decayed = self._decay_stale_sentiment(cached_global)
            self.last_value = decayed
            return decayed, self.cache.get('coins', {})

        sentiment = 0.0
        coin_sentiment = {}

        # --- Enhanced multi-source path ---
        if self._enhanced_available:
            try:
                self.enhanced_signals = self._fetch_enhanced()
                composite = self.enhanced_signals.get("composite", {}).get("value", None)

                if composite is not None:
                    sentiment = float(np.clip(composite, -1.0, 1.0))

                    # Extract per-coin sentiment from 24h momentum data
                    momentum = self.enhanced_signals.get("coin_momentum", {})
                    changes = momentum.get("changes_24h", {})
                    for coin_sym, change_pct in changes.items():
                        # Normalize ±5% to ±1.0
                        coin_sentiment[coin_sym.upper()] = max(-1.0, min(1.0, change_pct / 5.0))

                    # Log enhanced details
                    fg = self.enhanced_signals.get("fear_greed", {})
                    fr = self.enhanced_signals.get("funding_rates", {})
                    cm = self.enhanced_signals.get("coin_momentum", {})
                    self.logger.info(
                        f"[SENTIMENT] Enhanced: composite={sentiment:+.3f} | "
                        f"F&G={fg.get('value', '?')}/{fg.get('label', '?')} | "
                        f"funding={fr.get('avg_rate', 0):.6f} | "
                        f"momentum={cm.get('avg_change_24h', 0):+.2f}% | "
                        f"sources={self.enhanced_signals.get('composite', {}).get('components_used', 0)}"
                    )

                    self.cache = {'global': sentiment, 'coins': coin_sentiment}
                    self.last_fetch = time.time()
                    self.last_source = "live"
                    self.last_error = ""
                    self.last_value = sentiment
                    self.last_success_at = datetime.now()
                    return sentiment, coin_sentiment

            except Exception as e:
                self.logger.warning(f"[SENTIMENT] Enhanced fetch failed, falling back to F&G: {e}")

        # --- Fallback: Fear & Greed only ---
        try:
            resp = requests.get(self.fng_url, timeout=8)
            resp.raise_for_status()
            payload = resp.json()

            value = float(payload['data'][0]['value'])
            sentiment = float(np.clip((value - 50.0) / 50.0, -1.0, 1.0))

            self.cache = {'global': sentiment, 'coins': coin_sentiment}
            self.last_fetch = time.time()
            self.last_source = "live"
            self.last_error = ""
            self.last_value = sentiment
            self.last_success_at = datetime.now()

            return sentiment, coin_sentiment
        except Exception as e:
            self.logger.warning(f"Sentiment fetch failed, using decayed/neutral: {e}")

            cached_global = self.cache.get('global', 0.0)
            decayed = self._decay_stale_sentiment(cached_global)
            cached_coins = self.cache.get('coins', {})
            self.last_fetch = time.time()
            self.last_source = "cache_on_error"
            self.last_error = str(e)
            self.last_value = decayed
            return decayed, cached_coins

    def status(self) -> Dict[str, str]:
        """Return a compact status snapshot for logging/diagnostics."""
        result = {
            "source": self.last_source,
            "value": f"{self.last_value:+.3f}",
            "last_success": self.last_success_at.isoformat() if self.last_success_at else "never",
            "last_error": self.last_error,
        }
        # Add enhanced details if available
        if self.enhanced_signals:
            fg = self.enhanced_signals.get("fear_greed", {})
            result["fear_greed"] = f"{fg.get('value', '?')}"
            result["enhanced"] = "true"
        return result

    def get_signal(self, symbol: str, global_sent: float, coin_sent: Dict[str, float]) -> float:
        """
        Get sentiment signal for symbol.
        Uses coin-specific 24h momentum if available, otherwise global composite.
        Returns: -1 to 1 (negative to positive)
        """
        # Extract coin from symbol (e.g., BTC-USD -> BTC, PI_XBTUSD -> XBT)
        coin = symbol.split('-')[0].replace('PI_', '').replace('USD', '')

        # Map futures coin codes to standard tickers
        futures_coin_map = {
            "XBT": "BTC", "XBTUSD": "BTC",
        }
        coin = futures_coin_map.get(coin, coin)

        # Use coin-specific if available, else global
        sent = coin_sent.get(coin, global_sent)

        # Normalize to -1 to 1
        return np.clip(sent, -1, 1)

# ============================================================
# RISK MANAGER - Production Grade
# ============================================================

class RiskManager:
    """Centralized risk management."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.last_reset = datetime.now()
        self.trade_history: List[Dict] = []

    def update_balance(self, balance: float):
        """Update balance tracking."""
        self.current_balance = balance
        if balance > self.peak_balance:
            self.peak_balance = balance

    def record_trade(self, pnl: float):
        """Record trade P&L."""
        self.trade_history.append({
            'time': datetime.now(),
            'pnl': pnl
        })

        # Keep last 50 trades
        if len(self.trade_history) > 50:
            self.trade_history.pop(0)

        # Update daily P&L
        self.daily_pnl += pnl

        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        # Check daily reset
        if datetime.now().date() != self.last_reset.date():
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.last_reset = datetime.now()

        # Consecutive losses
        if self.consecutive_losses >= cfg.CB_MAX_CONSECUTIVE_LOSSES:
            return False, f"Circuit Breaker: {self.consecutive_losses} consecutive losses"

        # Daily loss limit (only after balance has been initialized)
        if self.current_balance > 0:
            daily_loss_limit_usd = abs(cfg.CB_DAILY_LOSS_LIMIT_PCT) * self.current_balance / 100
            if self.daily_pnl <= -daily_loss_limit_usd:
                return False, f"Circuit Breaker: Daily loss limit hit ({self.daily_pnl:.2f})"

        # Max drawdown
        if self.peak_balance > 0:
            drawdown = (self.current_balance - self.peak_balance) / self.peak_balance * 100
            if drawdown <= cfg.CB_MAX_DRAWDOWN_PCT:
                return False, f"Circuit Breaker: Max drawdown ({drawdown:.1f}%)"

        return True, "OK"

    def calculate_position_size(self,
                                balance: float,
                                confidence: float,
                                volatility: float,
                                num_positions: int) -> float:
        """Calculate position size using Kelly Criterion (simplified)."""
        # Kelly fraction: 2p - 1 where p is confidence (0.5 to 1.0)
        # Confidence already passed MIN_ML_CONFIDENCE gate in generate_signals
        kelly = max(0.01, (2 * confidence) - 1)  # Floor at 0.01 for penalized signals

        # Adjust for volatility (reduce size in high vol)
        vol_adjust = 1.0 / (1 + volatility * 10)  # Simple vol scaling

        # Position size
        size = balance * cfg.MAX_POSITION_PCT * kelly * vol_adjust

        # Adjust for existing positions (reduce as we add more)
        position_factor = max(0.3, 1 - (num_positions * 0.15))
        size *= position_factor

        # Min/Max bounds
        min_size = balance * cfg.MIN_POSITION_PCT
        max_size = balance * cfg.MAX_POSITION_PCT

        size = max(min_size, min(size, max_size))

        return size

# ============================================================
# TRADING BOT - Production Version
# ============================================================

class TradingBot:
    """Production trading bot with trend-following strategy."""

    def __init__(self):
        load_dotenv()

        # Setup
        self.logger = setup_logging()
        self.running = True
        self.cycle = 0
        self.use_locked_profile = os.getenv("USE_LOCKED_PROFILE", "true").strip().lower() == "true"
        self.locked_profile_path = os.getenv("LOCKED_PROFILE_PATH", "data/state/locked_profile.json").strip()
        self._apply_locked_profile()

        # Data & Models
        self.market_data = MarketData(self.logger)
        self.ml_model = MLModel()
        self.sentiment = SentimentAnalyzer(self.logger)
        self.risk_manager = RiskManager(self.logger)
        self.last_train_attempt_cycle = 0
        self.last_model_retrain = datetime.now()
        self.enable_futures_data = os.getenv("ENABLE_FUTURES", "true").lower() == "true"
        self.enable_coinbase_futures_data = os.getenv("ENABLE_COINBASE_FUTURES_DATA", "true").lower() == "true"
        self.enable_kraken_futures_fallback = os.getenv("ENABLE_KRAKEN_FUTURES_FALLBACK", "true").lower() == "true"
        self.coinbase_futures_product_map: Dict[str, Tuple[str, ...]] = {
            "PI_XBTUSD": ("BTC-USDC-PERP", "BTC-USD-PERP", "BTC-PERP"),
            "PI_ETHUSD": ("ETH-USDC-PERP", "ETH-USD-PERP", "ETH-PERP"),
            "PI_SOLUSD": ("SOL-USDC-PERP", "SOL-USD-PERP", "SOL-PERP"),
            "PI_ADAUSD": ("ADA-USDC-PERP", "ADA-USD-PERP", "ADA-PERP"),
            "PI_DOGEUSD": ("DOGE-USDC-PERP", "DOGE-USD-PERP", "DOGE-PERP"),
            "PI_LINKUSD": ("LINK-USDC-PERP", "LINK-USD-PERP", "LINK-PERP"),
            "PI_AVAXUSD": ("AVAX-USDC-PERP", "AVAX-USD-PERP", "AVAX-PERP"),
            "PI_DOTUSD": ("DOT-USDC-PERP", "DOT-USD-PERP", "DOT-PERP"),
            "PI_BCHUSD": ("BCH-USDC-PERP", "BCH-USD-PERP", "BCH-PERP"),
            "PI_LTCUSD": ("LTC-USDC-PERP", "LTC-USD-PERP", "LTC-PERP"),
            "PI_XRPUSD": ("XRP-USDC-PERP", "XRP-USD-PERP", "XRP-PERP"),
            "PI_ATOMUSD": ("ATOM-USDC-PERP", "ATOM-USD-PERP", "ATOM-PERP"),
        }
        self.correlation_lookback = 120
        self.direction_bias = os.getenv("DIRECTION_BIAS", "neutral").strip().lower()
        if self.direction_bias not in {"neutral", "short_lean", "long_lean"}:
            self.direction_bias = "neutral"
        try:
            self.direction_bias_strength = float(os.getenv("DIRECTION_BIAS_STRENGTH", "0.04"))
        except ValueError:
            self.direction_bias_strength = 0.04
        self.direction_bias_strength = max(0.0, min(self.direction_bias_strength, 0.15))
        self.alert_webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
        self.alert_email_to = os.getenv("ALERT_EMAIL_TO", "").strip()
        self.alert_email_from = os.getenv("ALERT_EMAIL_FROM", "").strip()
        self.alert_smtp_host = os.getenv("ALERT_SMTP_HOST", "").strip()
        self.alert_smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587") or "587")
        self.alert_smtp_user = os.getenv("ALERT_SMTP_USER", "").strip()
        self.alert_smtp_password = os.getenv("ALERT_SMTP_PASSWORD", "").strip()
        self.last_alert_at: Dict[str, datetime] = {}

        # RL shadow mode (Phase 1): RL decisions are evaluated but never placed live
        self.rl_shadow_mode = os.getenv("RL_SHADOW_MODE", "true").strip().lower() == "true"
        self.rl_shadow_min_multiplier = 0.5
        try:
            self.rl_shadow_min_multiplier = float(os.getenv("RL_SHADOW_MIN_MULTIPLIER", "0.5"))
        except ValueError:
            self.rl_shadow_min_multiplier = 0.5
        self.rl_shadow_min_multiplier = max(0.25, min(self.rl_shadow_min_multiplier, 2.0))
        self.rl_shadow_fee_rate = 0.001
        self.rl_agent: Optional[RLTradingAgent] = None
        self.rl_live_size_control = os.getenv("RL_LIVE_SIZE_CONTROL", "false").strip().lower() == "true"
        try:
            self.rl_live_size_min_mult = float(os.getenv("RL_LIVE_SIZE_MIN_MULT", "0.5"))
        except ValueError:
            self.rl_live_size_min_mult = 0.5
        try:
            self.rl_live_size_max_mult = float(os.getenv("RL_LIVE_SIZE_MAX_MULT", "1.5"))
        except ValueError:
            self.rl_live_size_max_mult = 1.5
        self.rl_live_size_min_mult = max(0.25, min(self.rl_live_size_min_mult, 2.0))
        self.rl_live_size_max_mult = max(self.rl_live_size_min_mult, min(self.rl_live_size_max_mult, 2.0))

        initial_balance = cfg.INITIAL_SPOT_BALANCE + cfg.INITIAL_FUTURES_BALANCE
        self.shadow_initial_balance = initial_balance
        self.shadow_books: Dict[str, Dict] = {
            "baseline": {
                "balance": initial_balance,
                "peak_equity": initial_balance,
                "max_drawdown_pct": 0.0,
                "positions": {},
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "realized_pnl": 0.0,
            },
            "rl": {
                "balance": initial_balance,
                "peak_equity": initial_balance,
                "max_drawdown_pct": 0.0,
                "positions": {},
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "realized_pnl": 0.0,
            },
        }
        self.rl_shadow_recent_events: List[Dict] = []
        self.rl_shadow_max_events = 400

        # State
        self.positions: Dict[str, Position] = {}
        self.balance_spot = cfg.INITIAL_SPOT_BALANCE
        self.balance_futures = cfg.INITIAL_FUTURES_BALANCE
        self.risk_manager.update_balance(self.balance_spot + self.balance_futures)

        # Per-symbol loss tracking for auto-pause
        self.symbol_consecutive_losses: Dict[str, int] = {}
        self.symbol_paused_until: Dict[str, datetime] = {}

        # Per-direction performance tracker
        self.direction_recent_results: Dict[str, List[bool]] = {"LONG": [], "SHORT": []}
        self.direction_paused_until: Dict[str, Optional[datetime]] = {"LONG": None, "SHORT": None}

        # API
        self.client = None
        self._init_api()

        # Load history if exists
        self._load_state()

        # Trim stacked positions to enforce new per-symbol limits
        self._trim_stacked_positions()

        # Train model if needed
        self.ml_model.load_or_train(self.market_data.price_history)
        force_retrain = os.getenv("FORCE_MODEL_RETRAIN_ON_START", "false").strip().lower() == "true"
        if force_retrain:
            self.logger.info("Forced ML retrain at startup enabled")
            try:
                self.ml_model._train(self.market_data.price_history)
                self.last_model_retrain = datetime.now()
                self.logger.info("Forced ML retrain at startup completed")
            except Exception as e:
                self.logger.warning(f"Forced ML retrain at startup failed: {e}")

        # Signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("=" * 60)
        self.logger.info("PRODUCTION TRADING BOT v5.0 — QUALITY-FIRST")
        self.logger.info(f"Mode: {'PAPER' if cfg.PAPER_TRADING else 'LIVE'}")
        self.logger.info(f"Risk Check: {cfg.RISK_CHECK_INTERVAL}s | Trade Cycle: {cfg.TRADE_CYCLE_INTERVAL * cfg.RISK_CHECK_INTERVAL}s")
        self.logger.info(
            f"Risk: SL_spot={cfg.STOP_LOSS_PCT}% SL_fut={cfg.FUTURES_STOP_LOSS}% "
            f"TP={cfg.TAKE_PROFIT_PCT}% Trail={cfg.TRAILING_STOP_PCT}% "
            f"MaxHold={cfg.MAX_HOLD_HOURS_SPOT}h/{cfg.MAX_HOLD_HOURS_FUTURES}h"
        )
        self.logger.info(
            f"Quality: ML_conf>={cfg.MIN_ML_CONFIDENCE} Ensemble>={cfg.MIN_ENSEMBLE_SCORE} "
            f"SIDE_filter={'ON' if cfg.SIDE_MARKET_FILTER else 'OFF'} "
            f"Symbol_pause_after={cfg.SYMBOL_PAUSE_CONSECUTIVE_LOSSES}_losses "
            f"Model_min_acc={cfg.MIN_MODEL_TEST_ACCURACY}"
        )
        self.logger.info(
            f"Direction Tracker: lookback={cfg.DIRECTION_PAUSE_LOOKBACK} "
            f"min_WR={cfg.DIRECTION_PAUSE_MIN_WR:.0%} pause={cfg.DIRECTION_PAUSE_HOURS}h "
            f"min_trades={cfg.DIRECTION_PAUSE_MIN_TRADES}"
        )
        self.logger.info(
            f"Limits: max_spot={cfg.MAX_POSITIONS_SPOT} max_futures={cfg.MAX_POSITIONS_FUTURES} | "
            f"per_symbol_spot={cfg.MAX_POSITIONS_PER_SYMBOL_SPOT} per_symbol_futures={cfg.MAX_POSITIONS_PER_SYMBOL_FUTURES} | "
            f"futures_leverage={cfg.FUTURES_LEVERAGE}x | max_corr={cfg.MAX_CORRELATION}"
        )
        self.logger.info(f"Direction bias: {self.direction_bias} (strength={self.direction_bias_strength:.2f})")
        self.logger.info(
            f"Model status: ml_ready={'yes' if self.ml_model.model is not None else 'no'} | "
            f"rl_shadow={'on' if self.rl_shadow_mode else 'off'} | "
            f"rl_live_size={'on' if self.rl_live_size_control else 'off'}"
        )
        self.logger.info("=" * 60)

        if not cfg.PAPER_TRADING:
            self.logger.error("WARNING: LIVE TRADING MODE!")
            live_confirm = os.getenv("LIVE_TRADING_CONFIRM", "")
            if live_confirm != "CONFIRM LIVE TRADING":
                raise RuntimeError(
                    "Live trading blocked. Set LIVE_TRADING_CONFIRM='CONFIRM LIVE TRADING' to proceed."
                )

        self._save_state()

        self._init_rl_shadow()
        self._save_runtime_fingerprint()

    def _coerce_config_value(self, key: str, value):
        """Coerce loaded profile value to existing config field type."""
        current = getattr(cfg, key)
        if isinstance(current, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if isinstance(current, int) and not isinstance(current, bool):
            return int(value)
        if isinstance(current, float):
            return float(value)
        if isinstance(current, tuple):
            if isinstance(value, (list, tuple)):
                return tuple(value)
            return current
        return value

    def _apply_locked_profile(self):
        """Load and apply locked config profile before bot initialization continues."""
        if not self.use_locked_profile:
            self.logger.info("Locked profile: disabled")
            return

        profile_path = self.locked_profile_path or "data/state/locked_profile.json"
        if not os.path.exists(profile_path):
            self.logger.info(f"Locked profile: not found ({profile_path})")
            return

        try:
            with open(profile_path, "r", encoding="utf-8-sig") as f:
                payload = json.load(f)

            overrides = payload.get("config_overrides", payload) if isinstance(payload, dict) else {}
            if not isinstance(overrides, dict):
                self.logger.warning("Locked profile ignored: invalid format")
                return

            applied = []
            for key, value in overrides.items():
                if not hasattr(cfg, key):
                    continue
                try:
                    coerced = self._coerce_config_value(key, value)
                    setattr(cfg, key, coerced)
                    applied.append(key)
                except Exception:
                    continue

            if applied:
                self.logger.info(
                    f"Locked profile applied from {profile_path} ({len(applied)} overrides)"
                )
            else:
                self.logger.info(f"Locked profile loaded but no valid overrides in {profile_path}")

            allow_env_overrides = os.getenv("LOCKED_PROFILE_ALLOW_ENV_OVERRIDES", "true").strip().lower() in {"1", "true", "yes", "on"}
            if allow_env_overrides:
                try:
                    cfg.__post_init__()
                    self.logger.info("Locked profile: environment overrides re-applied")
                except Exception as reapply_error:
                    self.logger.warning(f"Locked profile env re-apply failed: {reapply_error}")
        except Exception as e:
            self.logger.warning(f"Locked profile load failed: {e}")

    def _file_meta(self, path: str) -> Dict[str, object]:
        """Return basic metadata for reproducibility snapshots."""
        if not os.path.exists(path):
            return {"exists": False}
        try:
            stat = os.stat(path)
            return {
                "exists": True,
                "size": int(stat.st_size),
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except Exception:
            return {"exists": True}

    def _sha256_file(self, path: str) -> Optional[str]:
        if not os.path.exists(path):
            return None
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except Exception:
            return None

    def _save_runtime_fingerprint(self):
        """Persist startup fingerprint so strong runs can be reproduced exactly."""
        try:
            os.makedirs("data/state", exist_ok=True)
            engine_path = os.path.abspath(__file__)
            model_path = getattr(self.ml_model, "model_path", "models/trading_model.joblib")
            fingerprint = {
                "timestamp": datetime.now().isoformat(),
                "engine_path": engine_path,
                "engine_sha256": self._sha256_file(engine_path),
                "config": asdict(cfg),
                "flags": {
                    "direction_bias": self.direction_bias,
                    "direction_bias_strength": self.direction_bias_strength,
                    "enable_futures": self.enable_futures_data,
                    "enable_coinbase_futures_data": self.enable_coinbase_futures_data,
                    "enable_kraken_futures_fallback": self.enable_kraken_futures_fallback,
                    "rl_shadow_mode": self.rl_shadow_mode,
                    "rl_shadow_min_multiplier": self.rl_shadow_min_multiplier,
                    "rl_live_size_control": self.rl_live_size_control,
                    "rl_live_size_min_mult": self.rl_live_size_min_mult,
                    "rl_live_size_max_mult": self.rl_live_size_max_mult,
                    "use_locked_profile": self.use_locked_profile,
                    "locked_profile_path": self.locked_profile_path,
                },
                "model_files": {
                    "ml_model": self._file_meta(model_path),
                    "rl_agent_json": self._file_meta("data/state/rl_agent.json"),
                    "rl_agent_pt": self._file_meta("data/state/rl_agent_dqn.pt"),
                },
            }

            latest_path = "data/state/runtime_fingerprint_latest.json"
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(fingerprint, f, indent=2)

            history_path = "data/state/runtime_fingerprint_history.jsonl"
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fingerprint) + "\n")

            self.logger.info(
                f"Runtime fingerprint saved: {latest_path}"
            )
        except Exception as e:
            self.logger.warning(f"Runtime fingerprint save failed: {e}")

    def _init_rl_shadow(self):
        """Initialize RL shadow mode agent and persisted state."""
        if not self.rl_shadow_mode:
            self.logger.info("RL shadow mode: disabled")
            return

        if RLTradingAgent is None:
            self.rl_shadow_mode = False
            self.logger.warning("RL shadow mode disabled: RL agent dependencies unavailable")
            return

        try:
            self.rl_agent = RLTradingAgent()
            self.rl_agent.load_agent("data/state/rl_agent.json")
            self.logger.info(
                f"RL shadow mode: enabled (min_multiplier={self.rl_shadow_min_multiplier:.2f})"
            )
        except Exception as e:
            self.rl_agent = None
            self.rl_shadow_mode = False
            self.logger.warning(f"RL shadow mode disabled: RL agent init failed: {e}")

        # Restore shadow books if available
        try:
            report_path = "data/state/rl_shadow_report.json"
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                books = payload.get("books", {})
                for key in ("baseline", "rl"):
                    if key not in books:
                        continue
                    book = books[key]
                    dst = self.shadow_books[key]
                    dst["balance"] = float(book.get("balance", dst["balance"]))
                    dst["peak_equity"] = float(book.get("peak_equity", dst["peak_equity"]))
                    dst["max_drawdown_pct"] = float(book.get("max_drawdown_pct", dst["max_drawdown_pct"]))
                    dst["trades"] = int(book.get("trades", dst["trades"]))
                    dst["wins"] = int(book.get("wins", dst["wins"]))
                    dst["losses"] = int(book.get("losses", dst["losses"]))
                    dst["realized_pnl"] = float(book.get("realized_pnl", dst["realized_pnl"]))

                    restored_positions = {}
                    for sym, raw in (book.get("positions", {}) or {}).items():
                        try:
                            restored_positions[sym] = ShadowPosition(
                                symbol=raw["symbol"],
                                direction=raw["direction"],
                                entry_price=float(raw["entry_price"]),
                                size=float(raw["size"]),
                                entry_cycle=int(raw.get("entry_cycle", self.cycle)),
                                stop_loss_pct=float(raw.get("stop_loss_pct", cfg.STOP_LOSS_PCT)),
                                max_price=float(raw.get("max_price", raw["entry_price"])),
                                peak_pnl_pct=float(raw.get("peak_pnl_pct", 0.0)),
                                stale_since_cycle=(int(raw["stale_since_cycle"]) if raw.get("stale_since_cycle") is not None else None),
                                is_futures=bool(raw.get("is_futures", False)),
                                source=raw.get("source", key),
                                rl_state=str(raw.get("rl_state", "")),
                                rl_action=float(raw.get("rl_action", 1.0)),
                                rl_confidence=float(raw.get("rl_confidence", 0.5)),
                            )
                        except Exception:
                            continue
                    dst["positions"] = restored_positions
                self.logger.info(
                    "RL shadow mode: restored previous report state"
                )
        except Exception as e:
            self.logger.warning(f"RL shadow restore skipped: {e}")

    def _shadow_pnl_pct(self, position: ShadowPosition, fill_price: float) -> float:
        if position.direction == "LONG":
            return (fill_price - position.entry_price) / max(position.entry_price, 1e-12) * 100
        return (position.entry_price - fill_price) / max(position.entry_price, 1e-12) * 100

    def _shadow_pnl_value(self, position: ShadowPosition, fill_price: float) -> float:
        return position.size * (self._shadow_pnl_pct(position, fill_price) / 100.0)

    def _shadow_mark_equity(self, strategy: str) -> float:
        book = self.shadow_books[strategy]
        equity = float(book["balance"])
        for position in book["positions"].values():
            hist = self.market_data.price_history.get(position.symbol, [])
            if not hist:
                continue
            mark_raw = hist[-1]
            fill_mark = self._apply_slippage(
                mark_raw,
                position.direction,
                is_entry=False,
                is_futures=position.is_futures,
            )
            equity += self._shadow_pnl_value(position, fill_mark)
        return equity

    def _shadow_update_drawdown(self, strategy: str):
        book = self.shadow_books[strategy]
        equity = self._shadow_mark_equity(strategy)
        if equity > book["peak_equity"]:
            book["peak_equity"] = equity
        peak = max(book["peak_equity"], 1e-12)
        drawdown_pct = (equity - peak) / peak * 100.0
        if drawdown_pct < book["max_drawdown_pct"]:
            book["max_drawdown_pct"] = drawdown_pct

    def _shadow_record_event(self, event: Dict):
        if not self.rl_shadow_mode:
            return
        payload = {
            "ts": datetime.now().isoformat(),
            "cycle": self.cycle,
            **event,
        }
        self.rl_shadow_recent_events.append(payload)
        if len(self.rl_shadow_recent_events) > self.rl_shadow_max_events:
            self.rl_shadow_recent_events = self.rl_shadow_recent_events[-self.rl_shadow_max_events:]

        try:
            os.makedirs("data/state", exist_ok=True)
            with open("data/state/rl_shadow_events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            self.logger.debug(f"RL shadow event write failed: {e}")

    def _shadow_open_position(self, strategy: str, signal: Signal, size: float, rl_decision: Optional[Dict[str, float]] = None):
        if size <= 0:
            return

        book = self.shadow_books[strategy]
        is_futures = signal.symbol.startswith("PI_")
        # Futures: count per-direction (allow 1 long + 1 short per symbol)
        if is_futures:
            per_symbol_count = sum(1 for p in book["positions"].values()
                                   if p.symbol == signal.symbol and p.direction == signal.direction)
        else:
            per_symbol_count = sum(1 for p in book["positions"].values() if p.symbol == signal.symbol)
        per_symbol_limit = cfg.MAX_POSITIONS_PER_SYMBOL_FUTURES if is_futures else cfg.MAX_POSITIONS_PER_SYMBOL_SPOT
        if per_symbol_count >= per_symbol_limit:
            return
        raw_price = self.market_data.price_history.get(signal.symbol, [])[-1]
        entry_price = self._apply_slippage(raw_price, signal.direction, is_entry=True, is_futures=is_futures)
        stop_loss_pct = cfg.FUTURES_STOP_LOSS if is_futures else cfg.STOP_LOSS_PCT

        position = ShadowPosition(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry_price,
            size=size,
            entry_cycle=self.cycle,
            stop_loss_pct=stop_loss_pct,
            max_price=entry_price,
            peak_pnl_pct=0.0,
            stale_since_cycle=None,
            is_futures=is_futures,
            source=strategy,
            rl_state=str((rl_decision or {}).get("state", "")),
            rl_action=float((rl_decision or {}).get("multiplier", 1.0)),
            rl_confidence=float((rl_decision or {}).get("confidence", 0.5)),
        )

        entry_fee = size * self.rl_shadow_fee_rate
        book["balance"] -= entry_fee
        position_key = self._next_shadow_position_key(strategy, signal.symbol)
        book["positions"][position_key] = position
        self._shadow_update_drawdown(strategy)
        self._shadow_record_event({
            "type": "open",
            "strategy": strategy,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "size": round(size, 4),
            "entry_price": round(entry_price, 6),
            "entry_fee": round(entry_fee, 6),
            "confidence": round(signal.confidence, 4),
            "rl_action": round(position.rl_action, 4) if strategy == "rl" else 1.0,
            "rl_confidence": round(position.rl_confidence, 4) if strategy == "rl" else 0.5,
        })

    def _build_rl_state_for_symbol(self, symbol: str) -> str:
        """Build RL state from latest cached market context for online learning."""
        hist = self.market_data.price_history.get(symbol, [])
        if len(hist) < 20:
            return "0.00_0.0000_0.0_50.0_0.50"

        vol = self.market_data.calculate_volatility(symbol)
        rsi = self.market_data.calculate_rsi(symbol)
        _, slope = self.market_data.calculate_trend(symbol)

        ml_conf = 0.5
        if len(hist) >= 50 and self.ml_model.model is not None:
            try:
                ml_conf = float(self.ml_model.predict(hist).get("confidence", 0.5))
            except Exception:
                ml_conf = 0.5

        sentiment_cache = getattr(self.sentiment, "cache", {}) or {}
        global_sent = float(sentiment_cache.get("global", 0.0) or 0.0)
        coin_sent = sentiment_cache.get("coins", {}) or {}
        try:
            sent = float(self.sentiment.get_signal(symbol, global_sent, coin_sent))
        except Exception:
            sent = 0.0

        if self.rl_agent is None:
            return f"{sent:.2f}_{vol:.4f}_{(slope * 10000.0):.1f}_{rsi:.1f}_{ml_conf:.2f}"

        return self.rl_agent.get_state(
            sentiment=sent,
            volatility=vol,
            ml_confidence=ml_conf,
            rsi=rsi,
            trend=slope * 10000.0,
        )

    def _shadow_close_position(self, strategy: str, position_key: str, reason: str):
        book = self.shadow_books[strategy]
        position = book["positions"].get(position_key)
        if not position:
            return

        hist = self.market_data.price_history.get(position.symbol, [])
        if not hist:
            return

        raw_price = hist[-1]
        exit_price = self._apply_slippage(raw_price, position.direction, is_entry=False, is_futures=position.is_futures)
        pnl_value = self._shadow_pnl_value(position, exit_price)
        pnl_pct = self._shadow_pnl_pct(position, exit_price)
        exit_fee = position.size * self.rl_shadow_fee_rate

        book["balance"] += pnl_value - exit_fee
        book["trades"] += 1
        book["realized_pnl"] += pnl_value - exit_fee
        if pnl_value - exit_fee >= 0:
            book["wins"] += 1
        else:
            book["losses"] += 1

        # Online RL learning from shadow outcomes only (never from live execution path)
        if strategy == "rl" and self.rl_agent is not None and position.rl_state:
            try:
                reward = float(np.clip((pnl_value - exit_fee) / max(position.size, 1e-12), -2.0, 2.0))
                next_state = self._build_rl_state_for_symbol(position.symbol)
                self.rl_agent.update_q_value(
                    position.rl_state,
                    position.rl_action,
                    reward,
                    next_state,
                    done=True,
                    symbol=position.symbol,
                )
                self.rl_agent.log_trade(position.symbol, position.direction, reward, position.rl_action)
            except Exception as e:
                self.logger.debug(f"RL shadow learn failed for {position.symbol}: {e}")

        del book["positions"][position_key]
        self._shadow_update_drawdown(strategy)
        self._shadow_record_event({
            "type": "close",
            "strategy": strategy,
            "symbol": position.symbol,
            "direction": position.direction,
            "size": round(position.size, 4),
            "exit_price": round(exit_price, 6),
            "reason": reason,
            "pnl_value": round(pnl_value - exit_fee, 6),
            "pnl_pct": round(pnl_pct, 4),
            "rl_action": round(position.rl_action, 4) if strategy == "rl" else 1.0,
        })

    def _shadow_check_risk(self):
        """Apply baseline risk exit logic to both shadow books under same fill model."""
        if not self.rl_shadow_mode:
            return

        for strategy in ("baseline", "rl"):
            book = self.shadow_books[strategy]
            for position_key, position in list(book["positions"].items()):
                hist = self.market_data.price_history.get(position.symbol, [])
                if not hist:
                    continue

                current_price = hist[-1]
                if position.direction == "LONG" and current_price > position.max_price:
                    position.max_price = current_price
                elif position.direction == "SHORT" and current_price < position.max_price:
                    position.max_price = current_price

                mark_fill = self._apply_slippage(
                    current_price,
                    position.direction,
                    is_entry=False,
                    is_futures=position.is_futures,
                )
                pnl_pct = self._shadow_pnl_pct(position, mark_fill)
                if pnl_pct > position.peak_pnl_pct:
                    position.peak_pnl_pct = pnl_pct

                should_exit = False
                exit_reason = ""
                if pnl_pct <= position.stop_loss_pct:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                elif pnl_pct >= cfg.TAKE_PROFIT_PCT:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
                elif position.direction == "LONG":
                    drawdown_from_max = (
                        (position.max_price - current_price)
                        / max(position.max_price, 1e-12)
                        * 100
                    )
                    if drawdown_from_max >= cfg.TRAILING_STOP_PCT and pnl_pct > 1.0:
                        should_exit = True
                        exit_reason = "TRAILING_STOP"
                elif position.direction == "SHORT":
                    rebound_from_min = (
                        (current_price - position.max_price)
                        / max(position.max_price, 1e-12)
                        * 100
                    )
                    if rebound_from_min >= cfg.TRAILING_STOP_PCT and pnl_pct > 1.0:
                        should_exit = True
                        exit_reason = "TRAILING_STOP"

                if not should_exit:
                    stale_minutes = cfg.STALE_EXIT_STALE_MIN_FUTURES if position.is_futures else cfg.STALE_EXIT_STALE_MIN_SPOT
                    stale_cycles = max(1, int((stale_minutes * 60) / max(cfg.RISK_CHECK_INTERVAL, 1)))
                    stale_condition = (
                        position.peak_pnl_pct >= cfg.STALE_EXIT_PROFIT_ARM_PCT
                        and pnl_pct > 0
                        and pnl_pct <= (position.peak_pnl_pct - cfg.STALE_EXIT_DECAY_PCT)
                    )
                    if stale_condition:
                        if position.stale_since_cycle is None:
                            position.stale_since_cycle = self.cycle
                        elif (self.cycle - position.stale_since_cycle) >= stale_cycles:
                            should_exit = True
                            exit_reason = "STALE_PROFIT_DECAY"
                    else:
                        position.stale_since_cycle = None

                if should_exit:
                    self._shadow_close_position(strategy, position_key, exit_reason)

            self._shadow_update_drawdown(strategy)

    def _rl_shadow_decision(self, signal: Signal) -> Dict[str, float]:
        """Get RL shadow action (size multiplier) for this signal context."""
        if not self.rl_shadow_mode or self.rl_agent is None:
            return {
                "enabled": False,
                "multiplier": 1.0,
                "confidence": 0.5,
                "trade": False,
            }

        trend_value = signal.trend_slope * 10000.0
        state = self.rl_agent.get_state(
            sentiment=signal.sentiment,
            volatility=signal.volatility,
            ml_confidence=signal.confidence,
            rsi=signal.rsi,
            trend=trend_value,
        )
        multiplier = float(self.rl_agent.get_action(state))
        confidence = float(self.rl_agent.get_confidence(state, action_type=signal.direction))
        trade = multiplier >= self.rl_shadow_min_multiplier
        return {
            "enabled": True,
            "multiplier": multiplier,
            "confidence": confidence,
            "trade": trade,
            "state": state,
        }

    def _persist_rl_shadow_report(self):
        if not self.rl_shadow_mode:
            return

        try:
            os.makedirs("data/state", exist_ok=True)
            books = {}
            for strategy in ("baseline", "rl"):
                book = self.shadow_books[strategy]
                positions = {
                    sym: asdict(pos)
                    for sym, pos in book["positions"].items()
                }
                books[strategy] = {
                    "balance": round(book["balance"], 6),
                    "equity": round(self._shadow_mark_equity(strategy), 6),
                    "peak_equity": round(book["peak_equity"], 6),
                    "max_drawdown_pct": round(book["max_drawdown_pct"], 4),
                    "trades": int(book["trades"]),
                    "wins": int(book["wins"]),
                    "losses": int(book["losses"]),
                    "win_rate": round((book["wins"] / max(book["trades"], 1)) * 100.0, 2),
                    "realized_pnl": round(book["realized_pnl"], 6),
                    "positions": positions,
                }

            report = {
                "updated_at": datetime.now().isoformat(),
                "cycle": self.cycle,
                "mode": "shadow",
                "rl_shadow_min_multiplier": self.rl_shadow_min_multiplier,
                "books": books,
                "delta": {
                    "equity": round(books["rl"]["equity"] - books["baseline"]["equity"], 6),
                    "realized_pnl": round(books["rl"]["realized_pnl"] - books["baseline"]["realized_pnl"], 6),
                    "max_drawdown_pct": round(books["rl"]["max_drawdown_pct"] - books["baseline"]["max_drawdown_pct"], 4),
                },
                "recent_events": self.rl_shadow_recent_events[-80:],
            }
            with open("data/state/rl_shadow_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            if self.rl_agent is not None:
                self.rl_agent.save_agent("data/state/rl_agent.json")
        except Exception as e:
            self.logger.warning(f"RL shadow report save failed: {e}")

    def _init_api(self):
        """Initialize exchange API."""
        if not COINBASE_AVAILABLE:
            self.logger.warning("Coinbase SDK not available; using public endpoints only")
            return

        api_key = os.getenv("COINBASE_API_KEY")
        api_secret = os.getenv("COINBASE_API_SECRET")

        if not api_key or not api_secret:
            self.logger.warning("Coinbase credentials missing; using public endpoints only")
            return

        if not cfg.PAPER_TRADING and (len(api_key) < 20 or len(api_secret) < 20):
            raise ValueError("Invalid Coinbase credentials: too short for live mode")

        try:
            self.client = Client(api_key, api_secret)
            self.logger.info("Coinbase API connected")
        except Exception as e:
            self.client = None
            self.logger.error(f"API connection failed: {e}")
            if not cfg.PAPER_TRADING:
                raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown gracefully."""
        self.logger.info("Shutdown signal received...")
        self.running = False

    def _parse_datetime(self, raw_value) -> Optional[datetime]:
        if isinstance(raw_value, datetime):
            return raw_value
        if isinstance(raw_value, str) and raw_value.strip():
            try:
                return datetime.fromisoformat(raw_value)
            except ValueError:
                return None
        return None

    def _next_position_key(self, symbol: str) -> str:
        if symbol not in self.positions:
            return symbol
        idx = 2
        while True:
            candidate = f"{symbol}__{idx}"
            if candidate not in self.positions:
                return candidate
            idx += 1

    def _next_shadow_position_key(self, strategy: str, symbol: str) -> str:
        positions = self.shadow_books[strategy]["positions"]
        if symbol not in positions:
            return symbol
        idx = 2
        while True:
            candidate = f"{symbol}__{idx}"
            if candidate not in positions:
                return candidate
            idx += 1

    def _load_state(self):
        """Load previous state."""
        try:
            if os.path.exists("data/state/positions.json"):
                with open("data/state/positions.json", 'r') as f:
                    data = json.load(f)
                    for key, pos in data.items():
                        payload = dict(pos)
                        payload["entry_time"] = self._parse_datetime(payload.get("entry_time")) or datetime.now()
                        payload["stale_since"] = self._parse_datetime(payload.get("stale_since"))
                        if "peak_pnl_pct" not in payload:
                            payload["peak_pnl_pct"] = 0.0
                        position = Position(**payload)
                        safe_key = key if key not in self.positions else self._next_position_key(position.symbol)
                        self.positions[safe_key] = position
                self.logger.info(f"Loaded {len(self.positions)} positions")
        except Exception as e:
            self.logger.warning(f"Could not load state: {e}")

        # Restore paper balances from last saved state
        try:
            bal_path = "data/state/paper_balances.json"
            if os.path.exists(bal_path):
                with open(bal_path, 'r') as f:
                    bal = json.load(f)
                self.balance_spot = float(bal.get("spot", self.balance_spot))
                self.balance_futures = float(bal.get("futures", self.balance_futures))
                self.risk_manager.update_balance(self.balance_spot + self.balance_futures)

                # Restore circuit-breaker state so it survives restarts
                saved_peak = float(bal.get("peak_balance", 0.0))
                if saved_peak > 0:
                    self.risk_manager.peak_balance = max(
                        saved_peak, self.risk_manager.current_balance
                    )
                saved_daily_pnl = float(bal.get("daily_pnl", 0.0))
                saved_daily_date = bal.get("daily_pnl_date", "")
                if saved_daily_date == datetime.now().date().isoformat():
                    self.risk_manager.daily_pnl = saved_daily_pnl
                self.risk_manager.consecutive_losses = int(bal.get("consecutive_losses", 0))

                self.logger.info(
                    f"Loaded balances: Spot=${self.balance_spot:,.2f} Futures=${self.balance_futures:,.2f} "
                    f"PeakBal=${self.risk_manager.peak_balance:,.2f} DailyPnL=${self.risk_manager.daily_pnl:+.2f} "
                    f"ConsecLosses={self.risk_manager.consecutive_losses}"
                )
        except Exception as e:
            self.logger.warning(f"Could not load balances: {e}")

        try:
            spot_hist_path = "data/state/spot_price_history.json"
            if os.path.exists(spot_hist_path):
                with open(spot_hist_path, 'r') as f:
                    hist = json.load(f)
                loaded_symbols = 0
                loaded_points = 0
                for symbol, prices in hist.items():
                    if isinstance(prices, list) and prices:
                        self.market_data.price_history[symbol] = [float(p) for p in prices[-500:]]
                        loaded_symbols += 1
                        loaded_points += len(self.market_data.price_history[symbol])
                if loaded_symbols > 0:
                    self.logger.info(
                        f"Loaded spot history: {loaded_symbols} symbols, {loaded_points} points"
                    )
        except Exception as e:
            self.logger.warning(f"Could not load spot history: {e}")

        try:
            futures_hist_path = "data/state/futures_price_history.json"
            if os.path.exists(futures_hist_path):
                with open(futures_hist_path, 'r') as f:
                    hist = json.load(f)
                loaded_symbols = 0
                loaded_points = 0
                for symbol, prices in hist.items():
                    if isinstance(prices, list) and prices:
                        existing = self.market_data.price_history.get(symbol, [])
                        merged = (existing + [float(p) for p in prices])[-500:]
                        self.market_data.price_history[symbol] = merged
                        loaded_symbols += 1
                        loaded_points += len(merged)
                if loaded_symbols > 0:
                    self.logger.info(
                        f"Loaded futures history: {loaded_symbols} symbols, {loaded_points} points"
                    )
        except Exception as e:
            self.logger.warning(f"Could not load futures history: {e}")

    def _trim_stacked_positions(self):
        """On startup, close duplicate positions that exceed the new per-symbol limits.

        Keeps the BEST position per symbol (highest peak_pnl_pct) and closes the rest
        at current simulated prices.  For futures, limits are enforced per-direction
        (1 long + 1 short allowed per symbol).
        """
        from collections import defaultdict
        # For futures: group by symbol+direction; for spot: group by symbol only
        symbol_groups: Dict[str, List[Tuple[str, Position]]] = defaultdict(list)
        for key, pos in self.positions.items():
            is_futures = pos.symbol.startswith("PI_")
            if is_futures:
                group_key = f"{pos.symbol}:{pos.direction}"
            else:
                group_key = pos.symbol
            symbol_groups[group_key].append((key, pos))

        trimmed = 0
        for group_key, entries in symbol_groups.items():
            symbol = entries[0][1].symbol
            is_futures = symbol.startswith("PI_")
            # For futures per-direction groups, limit is 1 per direction
            # (total per symbol = 2 via MAX_POSITIONS_PER_SYMBOL_FUTURES)
            if is_futures:
                per_group_limit = 1  # 1 per direction
            else:
                per_group_limit = cfg.MAX_POSITIONS_PER_SYMBOL_SPOT
            if len(entries) <= per_group_limit:
                continue

            # Sort: keep highest peak_pnl_pct first
            entries.sort(key=lambda kp: kp[1].peak_pnl_pct, reverse=True)
            keepers = entries[:per_group_limit]
            to_close = entries[per_group_limit:]

            for key, pos in to_close:
                hist = self.market_data.price_history.get(symbol, [])
                if hist:
                    price = hist[-1]
                else:
                    price = pos.entry_price  # Fallback

                if pos.direction == "LONG":
                    pnl_pct = (price - pos.entry_price) / max(pos.entry_price, 1e-12) * 100
                else:
                    pnl_pct = (pos.entry_price - price) / max(pos.entry_price, 1e-12) * 100

                self._close_position(key, price, "STARTUP_TRIM_STACKED", pnl_pct)
                trimmed += 1

        if trimmed:
            self.logger.warning(
                f"STARTUP TRIM: Closed {trimmed} stacked positions to enforce per-symbol limits"
            )
            self._save_state()

    def _ensure_model_ready(self):
        """Train model once enough data exists if startup training had no data."""
        if self.ml_model.model is not None:
            return

        if self.cycle - self.last_train_attempt_cycle < cfg.TRADE_CYCLE_INTERVAL:
            return

        self.last_train_attempt_cycle = self.cycle
        ready_series = sum(
            1 for prices in self.market_data.price_history.values()
            if len(prices) >= 60
        )
        total_points = sum(len(prices) for prices in self.market_data.price_history.values())

        if ready_series < 2 or total_points < 240:
            self.logger.info(
                f"Model not ready: need more data (ready_series={ready_series}, points={total_points})"
            )
            return

        self.logger.info("Model training trigger: sufficient live history collected")
        self.ml_model.load_or_train(self.market_data.price_history)
        if self.ml_model.model is not None:
            self.logger.info("Model is now ready for live signal generation")
        else:
            self.logger.info("Model still not ready after training attempt")

    def _save_state(self):
        """Save current state."""
        try:
            os.makedirs("data/state", exist_ok=True)
            with open("data/state/positions.json", 'w') as f:
                json.dump({k: asdict(v) for k, v in self.positions.items()}, f, default=str)

            # Persist paper balances so dashboards/monitors can read them
            with open("data/state/paper_balances.json", 'w') as f:
                json.dump({
                    "spot": round(self.balance_spot, 4),
                    "futures": round(self.balance_futures, 4),
                    "peak_balance": round(self.risk_manager.peak_balance, 4),
                    "daily_pnl": round(self.risk_manager.daily_pnl, 4),
                    "consecutive_losses": self.risk_manager.consecutive_losses,
                    "daily_pnl_date": self.risk_manager.last_reset.date().isoformat(),
                }, f)

            serializable_history = {
                symbol: [float(price) for price in prices[-500:]]
                for symbol, prices in self.market_data.price_history.items()
            }
            with open("data/state/spot_price_history.json", 'w') as f:
                json.dump(serializable_history, f)
            with open("data/state/price_history.json", 'w') as f:
                json.dump(serializable_history, f)

            futures_history = {
                symbol: [float(price) for price in self.market_data.price_history.get(symbol, [])[-500:]]
                for symbol in cfg.FUTURES_SYMBOLS
                if self.market_data.price_history.get(symbol)
            }
            with open("data/state/futures_price_history.json", 'w') as f:
                json.dump(futures_history, f)
        except Exception as e:
            self.logger.error(f"Save state failed: {e}")

    def _fetch_public_spot_price(self, symbol: str) -> Optional[float]:
        """Fetch spot price using Coinbase public endpoint (no auth required)."""
        try:
            url = f"https://api.coinbase.com/v2/prices/{symbol}/spot"
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            payload = resp.json()
            return float(payload["data"]["amount"])
        except Exception as e:
            self.logger.debug(f"Public price fetch failed for {symbol}: {e}")
            return None

    def _apply_slippage(self, raw_price: float, direction: str, is_entry: bool, is_futures: bool) -> float:
        """Apply conservative slippage to simulated fills."""
        bps = cfg.SLIPPAGE_BPS_FUTURES if is_futures else cfg.SLIPPAGE_BPS_SPOT
        slip = max(0.0, bps) / 10000.0

        if direction == "LONG":
            multiplier = (1 + slip) if is_entry else (1 - slip)
        else:
            multiplier = (1 - slip) if is_entry else (1 + slip)

        return raw_price * multiplier

    def _sum_depth_levels(self, levels: List, max_levels: int) -> float:
        """Convert bids/asks levels into aggregate USD depth."""
        depth = 0.0
        for level in levels[:max_levels]:
            price = None
            size = None

            if isinstance(level, dict):
                price = level.get("price") or level.get("limitPrice") or level.get("rate")
                size = level.get("size") or level.get("qty") or level.get("quantity") or level.get("volume")
            elif isinstance(level, list) and len(level) >= 2:
                price = level[0]
                size = level[1]

            if price is None or size is None:
                continue

            try:
                depth += float(price) * float(size)
            except (TypeError, ValueError):
                continue

        return depth

    def _fetch_spot_depth_usd(self, symbol: str) -> Optional[float]:
        """Fetch Coinbase orderbook depth in USD."""
        try:
            url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            payload = resp.json()

            bids = payload.get("bids", [])
            asks = payload.get("asks", [])
            bid_depth = self._sum_depth_levels(bids, cfg.ORDERBOOK_LEVELS)
            ask_depth = self._sum_depth_levels(asks, cfg.ORDERBOOK_LEVELS)
            return min(bid_depth, ask_depth)
        except Exception as e:
            self.logger.debug(f"Spot depth fetch failed for {symbol}: {e}")
            return None

    def _fetch_futures_depth_usd(self, symbol: str) -> Optional[float]:
        """Fetch Kraken futures orderbook depth in USD."""
        try:
            url = f"https://futures.kraken.com/derivatives/api/v3/orderbook?symbol={symbol}"
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            payload = resp.json()
            book = payload.get("orderBook", payload)

            bids = book.get("bids", [])
            asks = book.get("asks", [])
            bid_depth = self._sum_depth_levels(bids, cfg.ORDERBOOK_LEVELS)
            ask_depth = self._sum_depth_levels(asks, cfg.ORDERBOOK_LEVELS)
            return min(bid_depth, ask_depth)
        except Exception as e:
            self.logger.debug(f"Futures depth fetch failed for {symbol}: {e}")
            return None

    def _passes_liquidity_gate(self, symbol: str, is_futures: bool, size_usd: float) -> Tuple[bool, float]:
        """Return if market depth can support the intended order size."""
        if not cfg.USE_LIQUIDITY_GATE:
            return True, 0.0

        min_depth = cfg.MIN_DEPTH_USD_FUTURES if is_futures else cfg.MIN_DEPTH_USD_SPOT
        required_depth = max(min_depth, size_usd * cfg.DEPTH_TO_TRADE_RATIO)
        depth = self._fetch_futures_depth_usd(symbol) if is_futures else self._fetch_spot_depth_usd(symbol)

        if depth is None:
            return cfg.LIQUIDITY_GATE_FAIL_OPEN, 0.0

        return depth >= required_depth, depth

    def _alert_cooldown_ok(self, key: str) -> bool:
        last = self.last_alert_at.get(key)
        if not last:
            return True
        elapsed = datetime.now() - last
        return elapsed >= timedelta(minutes=cfg.ALERT_COOLDOWN_MIN)

    def _send_alert(self, key: str, message: str):
        """Send alert through log + optional webhook/email."""
        if not self._alert_cooldown_ok(key):
            return

        self.last_alert_at[key] = datetime.now()
        self.logger.warning(f"ALERT[{key}] {message}")

        if self.alert_webhook_url:
            try:
                requests.post(
                    self.alert_webhook_url,
                    json={"type": key, "message": message, "timestamp": datetime.now().isoformat()},
                    timeout=8,
                )
            except Exception as e:
                self.logger.warning(f"Webhook alert failed: {e}")

        if all([
            self.alert_email_to,
            self.alert_email_from,
            self.alert_smtp_host,
            self.alert_smtp_user,
            self.alert_smtp_password,
        ]):
            try:
                msg = EmailMessage()
                msg["Subject"] = f"Trading Bot Alert: {key}"
                msg["From"] = self.alert_email_from
                msg["To"] = self.alert_email_to
                msg.set_content(message)

                with smtplib.SMTP(self.alert_smtp_host, self.alert_smtp_port, timeout=10) as smtp:
                    smtp.starttls()
                    smtp.login(self.alert_smtp_user, self.alert_smtp_password)
                    smtp.send_message(msg)
            except Exception as e:
                self.logger.warning(f"Email alert failed: {e}")

    def _check_alerts(self):
        """Check drawdown/daily loss alert conditions."""
        if self.risk_manager.peak_balance > 0:
            drawdown_pct = (
                (self.risk_manager.current_balance - self.risk_manager.peak_balance)
                / self.risk_manager.peak_balance
                * 100
            )
            if drawdown_pct <= cfg.ALERT_DRAWDOWN_PCT:
                self._send_alert(
                    "drawdown",
                    f"Drawdown alert: {drawdown_pct:.2f}% (threshold {cfg.ALERT_DRAWDOWN_PCT:.2f}%). "
                    f"Balance ${self.risk_manager.current_balance:,.2f}.",
                )

        base_balance = cfg.INITIAL_SPOT_BALANCE + cfg.INITIAL_FUTURES_BALANCE
        daily_loss_pct = (self.risk_manager.daily_pnl / base_balance) * 100 if base_balance > 0 else 0.0
        if daily_loss_pct <= cfg.ALERT_DAILY_LOSS_PCT:
            self._send_alert(
                "daily_loss",
                f"Daily loss alert: {daily_loss_pct:.2f}% (threshold {cfg.ALERT_DAILY_LOSS_PCT:.2f}%). "
                f"Daily P&L ${self.risk_manager.daily_pnl:+.2f}.",
            )

    def _log_trade_csv(self, position: Position, exit_price: float, pnl_value: float, reason: str):
        """Append closed trade details to CSV for accounting/tax tracking."""
        try:
            os.makedirs("data", exist_ok=True)
            path = "data/trades.csv"
            file_exists = os.path.exists(path)
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "symbol", "direction", "entry_price", "exit_price",
                        "size_usd", "pnl_usd", "pnl_pct", "exit_reason", "entry_reason"
                    ])
                writer.writerow([
                    datetime.now().isoformat(),
                    position.symbol,
                    position.direction,
                    f"{position.entry_price:.8f}",
                    f"{exit_price:.8f}",
                    f"{position.size:.2f}",
                    f"{pnl_value:.2f}",
                    f"{((exit_price - position.entry_price) / position.entry_price * 100 if position.direction == 'LONG' else (position.entry_price - exit_price) / position.entry_price * 100):.4f}",
                    reason,
                    position.entry_reason,
                ])
        except Exception as e:
            self.logger.error(f"Trade CSV logging failed: {e}")

    def _fetch_kraken_futures_tickers(self) -> Dict[str, float]:
        """Fetch futures tickers from Kraken Futures public endpoint."""
        prices: Dict[str, float] = {}
        try:
            resp = requests.get("https://futures.kraken.com/derivatives/api/v3/tickers", timeout=8)
            resp.raise_for_status()
            payload = resp.json()
            tickers = payload.get("tickers", [])
            target = set(cfg.FUTURES_SYMBOLS)

            for item in tickers:
                symbol = item.get("symbol")
                if symbol not in target:
                    continue

                raw_price = item.get("markPrice", item.get("last"))
                if raw_price is None:
                    continue

                try:
                    prices[symbol] = float(raw_price)
                except (TypeError, ValueError):
                    continue
        except Exception as e:
            self.logger.warning(f"Kraken futures fetch failed: {e}")

        return prices

    def _extract_market_price(self, payload: Dict) -> Optional[float]:
        """Extract a usable price from common market payload formats."""
        if not isinstance(payload, dict):
            return None

        direct_fields = (
            "price",
            "markPrice",
            "mark_price",
            "last",
            "last_price",
            "close",
            "mid_market",
        )
        for field in direct_fields:
            value = payload.get(field)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        data_payload = payload.get("data")
        if isinstance(data_payload, dict):
            nested = self._extract_market_price(data_payload)
            if nested is not None:
                return nested

        trades = payload.get("trades")
        if isinstance(trades, list) and trades:
            first_trade = trades[0]
            if isinstance(first_trade, dict):
                trade_price = first_trade.get("price")
                if trade_price is not None:
                    try:
                        return float(trade_price)
                    except (TypeError, ValueError):
                        return None

        return None

    def _fetch_coinbase_product_price(self, product_id: str) -> Optional[float]:
        """Fetch a single Coinbase product price across public endpoints."""
        endpoints = (
            f"https://api.exchange.coinbase.com/products/{product_id}/ticker",
            f"https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/ticker",
        )

        for url in endpoints:
            try:
                resp = requests.get(url, timeout=8)
                if resp.status_code >= 400:
                    continue

                payload = resp.json()
                parsed = self._extract_market_price(payload)
                if parsed is not None:
                    return parsed
            except Exception as e:
                self.logger.debug(f"Coinbase product fetch failed ({product_id}) via {url}: {e}")

        return None

    def _fetch_coinbase_futures_tickers(self) -> Dict[str, float]:
        """Fetch futures prices from Coinbase product endpoints."""
        prices: Dict[str, float] = {}

        for symbol in cfg.FUTURES_SYMBOLS:
            product_ids = self.coinbase_futures_product_map.get(symbol, ())
            for product_id in product_ids:
                price = self._fetch_coinbase_product_price(product_id)
                if price is not None:
                    prices[symbol] = price
                    break

        return prices

    def fetch_futures_prices(self):
        """Fetch configured futures prices from Coinbase with Kraken fallback."""
        coinbase_prices: Dict[str, float] = {}
        kraken_prices: Dict[str, float] = {}
        source_counts = {"coinbase": 0, "kraken": 0, "spot_proxy": 0, "missing": 0}

        if self.enable_coinbase_futures_data:
            coinbase_prices = self._fetch_coinbase_futures_tickers()

        if self.enable_kraken_futures_fallback:
            kraken_prices = self._fetch_kraken_futures_tickers()

        for symbol in cfg.FUTURES_SYMBOLS:
            price = coinbase_prices.get(symbol)
            source = "coinbase"

            if price is None and self.enable_kraken_futures_fallback:
                price = kraken_prices.get(symbol)
                source = "kraken"

            if price is None:
                spot_symbol = symbol.replace("PI_", "").replace("USD", "-USD")
                spot_hist = self.market_data.price_history.get(spot_symbol, [])
                if spot_hist:
                    price = float(spot_hist[-1])
                    source = "spot_proxy"

            if price is not None:
                self.market_data.update_price(symbol, price)
                source_counts[source] += 1
                self.logger.debug(f"Futures price {symbol} via {source}: {price:.2f}")
            else:
                source_counts["missing"] += 1
                self.logger.warning(
                    f"No futures price for {symbol} (coinbase_enabled={self.enable_coinbase_futures_data}, "
                    f"kraken_fallback={self.enable_kraken_futures_fallback})"
                )

        self.logger.info(
            f"Futures data status: coinbase={source_counts['coinbase']} | "
            f"kraken_fallback={source_counts['kraken']} | "
            f"spot_proxy={source_counts['spot_proxy']} | "
            f"missing={source_counts['missing']}"
        )

    def _calculate_symbol_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Calculate return correlation for two symbols using recent history."""
        prices_a = self.market_data.price_history.get(symbol_a, [])
        prices_b = self.market_data.price_history.get(symbol_b, [])

        if len(prices_a) < 12 or len(prices_b) < 12:
            return 0.0

        lookback = min(self.correlation_lookback, len(prices_a), len(prices_b))
        slice_a = np.array(prices_a[-lookback:], dtype=float)
        slice_b = np.array(prices_b[-lookback:], dtype=float)

        ret_a = np.diff(slice_a) / np.maximum(slice_a[:-1], 1e-12)
        ret_b = np.diff(slice_b) / np.maximum(slice_b[:-1], 1e-12)

        if len(ret_a) < 8 or len(ret_b) < 8:
            return 0.0

        corr_matrix = np.corrcoef(ret_a, ret_b)
        corr = corr_matrix[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)

    def _passes_correlation_gate(self, new_symbol: str, existing_symbols: List[str]) -> Tuple[bool, float]:
        """Return if symbol is within correlation limits and the max absolute correlation observed."""
        if not existing_symbols:
            return True, 0.0

        max_abs_corr = 0.0
        for existing in existing_symbols:
            corr = abs(self._calculate_symbol_correlation(new_symbol, existing))
            if corr > max_abs_corr:
                max_abs_corr = corr
            if corr > cfg.MAX_CORRELATION:
                return False, max_abs_corr

        return True, max_abs_corr

    def _maybe_retrain_model(self):
        """Retrain model periodically on newly collected history."""
        if datetime.now() - self.last_model_retrain < timedelta(hours=cfg.MODEL_RETRAIN_HOURS):
            return

        ready_series = sum(1 for prices in self.market_data.price_history.values() if len(prices) >= 60)
        total_points = sum(len(prices) for prices in self.market_data.price_history.values())
        if ready_series < 2 or total_points < 240:
            self.logger.info(
                f"Skip scheduled retrain: insufficient data (ready_series={ready_series}, points={total_points})"
            )
            self.last_model_retrain = datetime.now()
            return

        self.logger.info("Scheduled model retrain started")
        try:
            self.ml_model._train(self.market_data.price_history)
            self.logger.info("Scheduled model retrain completed")
        except Exception as e:
            self.logger.warning(f"Scheduled retrain failed: {e}")
        finally:
            self.last_model_retrain = datetime.now()

    def fetch_prices(self):
        """Fetch current prices."""
        symbols = list(cfg.SPOT_SYMBOLS)

        for i, symbol in enumerate(symbols):
            price: Optional[float] = None
            retries = cfg.PRICE_FETCH_RETRIES

            while retries > 0 and price is None:
                try:
                    if self.client:
                        price = float(self.client.get_spot_price(currency_pair=symbol)['amount'])
                    else:
                        price = self._fetch_public_spot_price(symbol)
                except Exception as e:
                    self.logger.debug(f"Authenticated fetch failed for {symbol}: {e}")
                    price = self._fetch_public_spot_price(symbol)

                if price is None:
                    retries -= 1
                    if retries > 0:
                        time.sleep(cfg.PRICE_FETCH_RETRY_DELAY_SEC)

            if price is not None:
                self.market_data.update_price(symbol, price)
            else:
                self.logger.error(f"Failed to fetch {symbol} after {cfg.PRICE_FETCH_RETRIES} retries")

            if i < len(symbols) - 1:
                time.sleep(cfg.PRICE_FETCH_RATE_LIMIT_SEC)

        if self.enable_futures_data:
            self.fetch_futures_prices()

    def check_risk(self):
        """Check stop losses and take profits."""
        current_prices = {}

        for position_key, position in list(self.positions.items()):
            symbol = position.symbol
            # Get current price
            hist = self.market_data.price_history.get(symbol, [])
            if not hist:
                continue

            current_price = hist[-1]
            current_prices[position_key] = current_price

            # Calculate P&L
            if position.direction == "LONG":
                pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
            else:
                pnl_pct = (position.entry_price - current_price) / position.entry_price * 100
            if pnl_pct > position.peak_pnl_pct:
                position.peak_pnl_pct = pnl_pct

            # Update max price for trailing stop
            if position.direction == "LONG" and current_price > position.max_price:
                position.max_price = current_price
            elif position.direction == "SHORT" and current_price < position.max_price:
                position.max_price = current_price

            # Check exits
            should_exit = False
            exit_reason = ""
            stop_loss_pct = cfg.FUTURES_STOP_LOSS if symbol.startswith("PI_") else cfg.STOP_LOSS_PCT

            # Stop Loss
            if pnl_pct <= stop_loss_pct:
                should_exit = True
                exit_reason = "STOP_LOSS"

            # Take Profit
            elif pnl_pct >= cfg.TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = "TAKE_PROFIT"

            # Trailing Stop
            elif position.direction == "LONG":
                drawdown_from_max = (position.max_price - current_price) / max(position.max_price, 1e-12) * 100
                if drawdown_from_max >= cfg.TRAILING_STOP_PCT and pnl_pct > 1.0:
                    should_exit = True
                    exit_reason = "TRAILING_STOP"
            elif position.direction == "SHORT":
                rebound_from_min = (current_price - position.max_price) / max(position.max_price, 1e-12) * 100
                if rebound_from_min >= cfg.TRAILING_STOP_PCT and pnl_pct > 1.0:
                    should_exit = True
                    exit_reason = "TRAILING_STOP"

            if not should_exit:
                stale_minutes = cfg.STALE_EXIT_STALE_MIN_FUTURES if symbol.startswith("PI_") else cfg.STALE_EXIT_STALE_MIN_SPOT
                stale_condition = (
                    position.peak_pnl_pct >= cfg.STALE_EXIT_PROFIT_ARM_PCT
                    and pnl_pct > 0
                    and pnl_pct <= (position.peak_pnl_pct - cfg.STALE_EXIT_DECAY_PCT)
                )
                if stale_condition:
                    if position.stale_since is None:
                        position.stale_since = datetime.now()
                    elif (datetime.now() - position.stale_since) >= timedelta(minutes=stale_minutes):
                        should_exit = True
                        exit_reason = "STALE_PROFIT_DECAY"
                else:
                    position.stale_since = None

            # Max hold time exit — close flat positions that go nowhere
            if not should_exit:
                max_hold_h = cfg.MAX_HOLD_HOURS_FUTURES if symbol.startswith("PI_") else cfg.MAX_HOLD_HOURS_SPOT
                hold_duration = datetime.now() - position.entry_time
                if hold_duration >= timedelta(hours=max_hold_h):
                    if abs(pnl_pct) < cfg.MAX_HOLD_FLAT_BAND_PCT:
                        should_exit = True
                        exit_reason = "MAX_HOLD_FLAT"
                    elif hold_duration >= timedelta(hours=max_hold_h * 2):
                        # Double the max hold time: close regardless of P/L
                        should_exit = True
                        exit_reason = "MAX_HOLD_FORCED"

            if should_exit:
                self._close_position(position_key, current_price, exit_reason, pnl_pct)

        return current_prices

    def _close_position(self, position_key: str, price: float, reason: str, pnl_pct: float):
        """Close a position."""
        position = self.positions.get(position_key)
        if not position:
            return

        symbol = position.symbol

        is_futures = symbol.startswith("PI_")
        fill_price = self._apply_slippage(price, position.direction, is_entry=False, is_futures=is_futures)

        # Calculate P&L value
        if position.direction == "LONG":
            pnl_value = position.size * (fill_price - position.entry_price) / position.entry_price
        else:
            pnl_value = position.size * (position.entry_price - fill_price) / position.entry_price

        # Update balance (paper trading)
        if cfg.PAPER_TRADING:
            if symbol.startswith("PI_"):
                self.balance_futures += pnl_value
            else:
                self.balance_spot += pnl_value
            self.risk_manager.update_balance(self.balance_spot + self.balance_futures)

        # Record
        self.risk_manager.record_trade(pnl_value)
        self._log_trade_csv(position, fill_price, pnl_value, reason)

        # Track per-symbol results for auto-pause
        self._record_symbol_result(symbol, is_win=(pnl_value > 0))

        # Track per-direction results for direction pause
        self._record_direction_result(position.direction, is_win=(pnl_value > 0))

        realized_pct = (pnl_value / max(position.size, 1e-12)) * 100
        self.logger.info(
            f"CLOSE {position.direction} {symbol} @ ${fill_price:.2f} | {reason} | "
            f"P&L: ${pnl_value:+.2f} ({realized_pct:+.2f}%)"
        )

        del self.positions[position_key]
        self._save_state()

    def _is_symbol_paused(self, symbol: str) -> bool:
        """Check if symbol is auto-paused due to consecutive losses."""
        pause_until = self.symbol_paused_until.get(symbol)
        if pause_until and datetime.now() < pause_until:
            return True
        elif pause_until:
            # Pause expired — reset
            del self.symbol_paused_until[symbol]
            self.symbol_consecutive_losses[symbol] = 0
        return False

    def _record_symbol_result(self, symbol: str, is_win: bool):
        """Track per-symbol consecutive losses for auto-pause."""
        base_symbol = symbol.split('__')[0]  # Handle multi-position keys
        if is_win:
            self.symbol_consecutive_losses[base_symbol] = 0
        else:
            self.symbol_consecutive_losses[base_symbol] = (
                self.symbol_consecutive_losses.get(base_symbol, 0) + 1
            )
            if self.symbol_consecutive_losses[base_symbol] >= cfg.SYMBOL_PAUSE_CONSECUTIVE_LOSSES:
                pause_hours = 2  # Pause for 2 hours
                self.symbol_paused_until[base_symbol] = datetime.now() + timedelta(hours=pause_hours)
                self.logger.warning(
                    f"AUTO-PAUSE: {base_symbol} paused for {pause_hours}h after "
                    f"{self.symbol_consecutive_losses[base_symbol]} consecutive losses"
                )

    def _record_direction_result(self, direction: str, is_win: bool):
        """Track per-direction rolling win rate for adaptive pause."""
        results = self.direction_recent_results.get(direction, [])
        results.append(is_win)
        # Keep only the last LOOKBACK trades
        if len(results) > cfg.DIRECTION_PAUSE_LOOKBACK:
            results = results[-cfg.DIRECTION_PAUSE_LOOKBACK:]
        self.direction_recent_results[direction] = results

        # Check if we have enough trades to judge
        if len(results) >= cfg.DIRECTION_PAUSE_MIN_TRADES:
            wins = sum(1 for r in results if r)
            wr = wins / len(results)
            if wr < cfg.DIRECTION_PAUSE_MIN_WR:
                # Pause this direction
                pause_until = datetime.now() + timedelta(hours=cfg.DIRECTION_PAUSE_HOURS)
                self.direction_paused_until[direction] = pause_until
                self.logger.warning(
                    f"DIRECTION-PAUSE: {direction} paused for {cfg.DIRECTION_PAUSE_HOURS}h | "
                    f"WR={wr:.1%} ({wins}W/{len(results)-wins}L last {len(results)}) < {cfg.DIRECTION_PAUSE_MIN_WR:.0%} threshold"
                )

    def _is_direction_paused(self, direction: str) -> bool:
        """Check if a direction (LONG/SHORT) is auto-paused due to poor performance."""
        pause_until = self.direction_paused_until.get(direction)
        if pause_until and datetime.now() < pause_until:
            return True
        elif pause_until:
            # Pause expired — reset
            self.direction_paused_until[direction] = None
            self.logger.info(f"DIRECTION-UNPAUSE: {direction} pause expired, direction re-enabled")
        return False

    def generate_signals(self) -> List[Signal]:
        """Generate trading signals with quality filters."""
        signals = []
        # NOTE: AGGRESSIVE_FUTURES_BURST no longer bypasses quality gates.
        # It only controls trade cycle speed (already handled in __post_init__).

        # Get sentiment
        global_sent, coin_sent = self.sentiment.fetch_sentiment()
        sentiment_status = self.sentiment.status()
        self.logger.info(
            f"Sentiment status: source={sentiment_status['source']} value={sentiment_status['value']} "
            f"last_success={sentiment_status['last_success']}"
        )

        symbols = list(cfg.SPOT_SYMBOLS)
        if self.enable_futures_data:
            symbols.extend(cfg.FUTURES_SYMBOLS)
            futures_points = [
                f"{symbol}:{len(self.market_data.price_history.get(symbol, []))}"
                for symbol in cfg.FUTURES_SYMBOLS
            ]
            self.logger.info("Futures history points: " + ", ".join(futures_points))

        skipped_side = 0
        skipped_paused = 0
        skipped_confidence = 0
        skipped_trend_mismatch = 0
        counter_trend_taken = 0

        skipped_blacklist = 0
        skipped_direction_pause = 0

        for symbol in symbols:
            # Symbol blacklist check
            if symbol in cfg.SYMBOL_BLACKLIST:
                skipped_blacklist += 1
                continue

            hist = self.market_data.price_history.get(symbol, [])
            if len(hist) < 50:
                continue

            # Per-symbol auto-pause check
            if self._is_symbol_paused(symbol):
                skipped_paused += 1
                continue

            # ML Prediction
            ml_pred = self.ml_model.predict(hist)
            ml_conf = ml_pred['confidence']
            ml_direction = ml_pred['direction']  # >0.5 = up, <0.5 = down

            # Strict confidence gate — no bypasses
            if ml_conf < cfg.MIN_ML_CONFIDENCE:
                skipped_confidence += 1
                continue

            # Technicals
            rsi = self.market_data.calculate_rsi(symbol)
            trend, slope = self.market_data.calculate_trend(symbol)
            vol = self.market_data.calculate_volatility(symbol)

            # SIDE-market filter: skip sideways markets unless ML is very confident
            if cfg.SIDE_MARKET_FILTER and trend == "SIDE":
                if ml_conf < cfg.SIDE_MARKET_ML_OVERRIDE:
                    skipped_side += 1
                    continue

            # Sentiment
            sent = self.sentiment.get_signal(symbol, global_sent, coin_sent)

            # Correlation check
            existing_symbols = [s for s in {pos.symbol for pos in self.positions.values()} if s != symbol]
            _, corr = self._passes_correlation_gate(symbol, existing_symbols)

            # Determine direction — TREND FOLLOWING only in trending markets
            direction = None
            confidence = ml_conf
            long_trigger = cfg.MIN_ENSEMBLE_SCORE
            short_trigger = 1 - cfg.MIN_ENSEMBLE_SCORE

            if self.direction_bias == "short_lean":
                long_trigger = min(0.98, long_trigger + self.direction_bias_strength)
                short_trigger = max(0.01, short_trigger - self.direction_bias_strength)
            elif self.direction_bias == "long_lean":
                long_trigger = max(0.51, long_trigger - self.direction_bias_strength)
                short_trigger = min(0.49, short_trigger + self.direction_bias_strength)

            if ml_direction > long_trigger:
                # Potential Long — require UP trend (or very strong ML override in SIDE/counter)
                if trend == "UP":
                    if rsi < cfg.MAX_RSI_LONG:
                        direction = "LONG"
                        confidence = ml_conf  # Use raw ML confidence
                elif trend == "SIDE" and ml_conf >= cfg.SIDE_MARKET_ML_OVERRIDE:
                    if rsi < cfg.MAX_RSI_LONG and sent >= -cfg.SENTIMENT_THRESHOLD:
                        direction = "LONG"
                        confidence = ml_conf * 0.85  # Penalize SIDE
                elif trend == "DOWN" and ml_conf >= cfg.COUNTER_TREND_ML_OVERRIDE:
                    # Counter-trend: model strongly predicts UP against DOWN trend
                    if rsi < cfg.MAX_RSI_LONG:
                        direction = "LONG"
                        confidence = ml_conf * 0.70  # Heavy penalty
                        counter_trend_taken += 1

            elif ml_direction < short_trigger:
                # Potential Short — require DOWN trend (or very strong ML override in SIDE/counter)
                if trend == "DOWN":
                    if rsi > cfg.MIN_RSI_SHORT:
                        direction = "SHORT"
                        confidence = ml_conf  # Use raw ML confidence
                elif trend == "SIDE" and ml_conf >= cfg.SIDE_MARKET_ML_OVERRIDE:
                    if rsi > cfg.MIN_RSI_SHORT and sent <= cfg.SENTIMENT_THRESHOLD:
                        direction = "SHORT"
                        confidence = ml_conf * 0.85  # Penalize SIDE
                elif trend == "UP" and ml_conf >= cfg.COUNTER_TREND_ML_OVERRIDE:
                    # Counter-trend: model strongly predicts DOWN against UP trend
                    if rsi > cfg.MIN_RSI_SHORT:
                        direction = "SHORT"
                        confidence = ml_conf * 0.70  # Heavy penalty
                        counter_trend_taken += 1

            # SHORT-only / LONG-only mode filter
            if direction == "LONG" and cfg.DIRECTION_MODE == "short_only":
                direction = None  # Block LONG entries
            elif direction == "SHORT" and cfg.DIRECTION_MODE == "long_only":
                direction = None  # Block SHORT entries

            # Direction Performance Tracker — skip if direction is paused
            if direction and self._is_direction_paused(direction):
                skipped_direction_pause += 1
                direction = None  # Block this direction

            if not direction and ml_conf >= cfg.MIN_ML_CONFIDENCE:
                skipped_trend_mismatch += 1

            # NOTE: Direction bias confidence adjustments REMOVED.
            # They caused compounding SHORT bias when combined with an already-biased ML model.
            # The direction_bias trigger adjustments above are sufficient for any lean preference.

            if direction:
                signals.append(Signal(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    ml_score=ml_direction,
                    rsi=rsi,
                    trend=trend,
                    sentiment=sent,
                    correlation=corr,
                    volatility=vol,
                    trend_slope=slope,
                    reason=f"ML:{ml_direction:.2f}|Trend:{trend}|RSI:{rsi:.0f}|Sent:{sent:+.2f}",
                    timestamp=datetime.now()
                ))

        if skipped_side or skipped_paused or skipped_confidence or skipped_trend_mismatch or skipped_blacklist or skipped_direction_pause:
            self.logger.info(
                f"Signal filters: skipped_SIDE={skipped_side} skipped_paused={skipped_paused} "
                f"skipped_low_conf={skipped_confidence} skipped_trend_mismatch={skipped_trend_mismatch} "
                f"skipped_blacklist={skipped_blacklist} skipped_dir_pause={skipped_direction_pause} "
                f"direction_mode={cfg.DIRECTION_MODE}"
            )
        if counter_trend_taken:
            self.logger.info(f"Counter-trend signals generated: {counter_trend_taken}")

        return signals

    def execute_signals(self, signals: List[Signal]):
        """Execute trading signals."""
        # Check risk manager first
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            self.logger.warning(f"Trading blocked: {reason}")
            return

        # Current symbol occupancy — futures counted per-direction (allow long+short)
        symbol_position_counts: Dict[str, int] = {}
        symbol_direction_counts: Dict[str, int] = {}  # "SYMBOL:DIRECTION" -> count
        for position in self.positions.values():
            symbol_position_counts[position.symbol] = symbol_position_counts.get(position.symbol, 0) + 1
            sd_key = f"{position.symbol}:{position.direction}"
            symbol_direction_counts[sd_key] = symbol_direction_counts.get(sd_key, 0) + 1

        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)

        spot_positions = sum(1 for symbol in self.positions if not symbol.startswith("PI_"))
        futures_positions = sum(1 for symbol in self.positions if symbol.startswith("PI_"))

        for signal in signals:
            rl_decision = self._rl_shadow_decision(signal)

            is_futures = signal.symbol.startswith("PI_")
            position_count = futures_positions if is_futures else spot_positions
            per_symbol_limit = cfg.MAX_POSITIONS_PER_SYMBOL_FUTURES if is_futures else cfg.MAX_POSITIONS_PER_SYMBOL_SPOT
            # Futures: count per-direction (1 long + 1 short allowed)
            # Spot: count all positions for symbol
            if is_futures:
                sd_key = f"{signal.symbol}:{signal.direction}"
                per_symbol_count = symbol_direction_counts.get(sd_key, 0)
            else:
                per_symbol_count = symbol_position_counts.get(signal.symbol, 0)

            # Check per-symbol auto-pause (consecutive losses)
            if self._is_symbol_paused(signal.symbol):
                self._shadow_record_event({
                    "type": "decision",
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "baseline": "SKIP_SYMBOL_PAUSED",
                    "rl_trade": rl_decision.get("trade", False),
                    "rl_multiplier": round(float(rl_decision.get("multiplier", 1.0)), 4),
                })
                continue

            if per_symbol_count >= per_symbol_limit:
                self._shadow_record_event({
                    "type": "decision",
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "baseline": "SKIP_MAX_PER_SYMBOL",
                    "per_symbol_count": int(per_symbol_count),
                    "per_symbol_limit": int(per_symbol_limit),
                    "rl_trade": rl_decision.get("trade", False),
                    "rl_multiplier": round(float(rl_decision.get("multiplier", 1.0)), 4),
                })
                continue

            # Check max positions by asset class
            if is_futures:
                if futures_positions >= cfg.MAX_POSITIONS_FUTURES:
                    self._shadow_record_event({
                        "type": "decision",
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "baseline": "SKIP_MAX_FUTURES",
                        "rl_trade": rl_decision.get("trade", False),
                        "rl_multiplier": round(float(rl_decision.get("multiplier", 1.0)), 4),
                    })
                    continue
            else:
                if spot_positions >= cfg.MAX_POSITIONS_SPOT:
                    self._shadow_record_event({
                        "type": "decision",
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "baseline": "SKIP_MAX_SPOT",
                        "rl_trade": rl_decision.get("trade", False),
                        "rl_multiplier": round(float(rl_decision.get("multiplier", 1.0)), 4),
                    })
                    continue

            # Check correlation against existing open positions
            existing_symbols_list = [s for s in {pos.symbol for pos in self.positions.values()} if s != signal.symbol]
            passes_corr, max_corr = self._passes_correlation_gate(signal.symbol, existing_symbols_list)
            if not passes_corr:
                self.logger.info(
                    f"Skip {signal.symbol}: correlation {max_corr:.2f} exceeds limit {cfg.MAX_CORRELATION:.2f}"
                )
                self._shadow_record_event({
                    "type": "decision",
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "baseline": "SKIP_CORRELATION",
                    "max_corr": round(float(max_corr), 4),
                    "rl_trade": rl_decision.get("trade", False),
                    "rl_multiplier": round(float(rl_decision.get("multiplier", 1.0)), 4),
                })
                continue

            # Calculate size
            total_balance = self.balance_spot + self.balance_futures
            size = self.risk_manager.calculate_position_size(
                total_balance,
                signal.confidence,
                self.market_data.calculate_volatility(signal.symbol),
                position_count
            )

            min_trade_size = max(10.0, total_balance * cfg.MIN_POSITION_PCT)
            # For futures, leverage reduces capital needed — adjust min accordingly
            effective_min = min_trade_size / cfg.FUTURES_LEVERAGE if is_futures and cfg.FUTURES_LEVERAGE > 1 else min_trade_size

            if is_futures and cfg.FUTURES_LEVERAGE > 1:
                size = size / cfg.FUTURES_LEVERAGE

            base_size = size
            live_rl_multiplier = 1.0
            if self.rl_live_size_control and bool(rl_decision.get("enabled", False)):
                try:
                    live_rl_multiplier = float(rl_decision.get("multiplier", 1.0))
                except Exception:
                    live_rl_multiplier = 1.0
                live_rl_multiplier = max(self.rl_live_size_min_mult, min(live_rl_multiplier, self.rl_live_size_max_mult))
                size = base_size * live_rl_multiplier

            if size < effective_min:
                self._shadow_record_event({
                    "type": "decision",
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "baseline": "SKIP_MIN_SIZE",
                    "base_size": round(float(base_size), 4),
                    "size": round(float(size), 4),
                    "min_trade_size": round(float(effective_min), 4),
                    "rl_live_size_multiplier": round(float(live_rl_multiplier), 4),
                    "rl_trade": rl_decision.get("trade", False),
                    "rl_multiplier": round(float(rl_decision.get("multiplier", 1.0)), 4),
                })
                continue

            # Get current price
            raw_price = self.market_data.price_history[signal.symbol][-1]
            price = self._apply_slippage(raw_price, signal.direction, is_entry=True, is_futures=is_futures)

            # Liquidity gate
            passes_liquidity, depth = self._passes_liquidity_gate(signal.symbol, is_futures, size)
            if not passes_liquidity:
                self.logger.info(
                    f"Skip {signal.symbol}: insufficient depth ${depth:,.0f} for size ${size:,.0f}"
                )
                self._shadow_record_event({
                    "type": "decision",
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "baseline": "SKIP_LIQUIDITY",
                    "depth": round(float(depth), 2),
                    "size": round(float(size), 2),
                    "rl_trade": rl_decision.get("trade", False),
                    "rl_multiplier": round(float(rl_decision.get("multiplier", 1.0)), 4),
                })
                continue

            # Shadow-mode tracking (same fill model, no live behavior change)
            self._shadow_open_position("baseline", signal, size)
            rl_multiplier = float(rl_decision.get("multiplier", 1.0))
            rl_size = size * rl_multiplier
            rl_trade = bool(rl_decision.get("trade", False))
            if rl_trade and rl_size >= min_trade_size:
                self._shadow_open_position("rl", signal, rl_size, rl_decision=rl_decision)

            self._shadow_record_event({
                "type": "decision",
                "symbol": signal.symbol,
                "direction": signal.direction,
                "baseline": "OPEN",
                "baseline_size": round(float(base_size), 4),
                "live_size": round(float(size), 4),
                "rl_live_size_multiplier": round(float(live_rl_multiplier), 4),
                "rl_trade": rl_trade,
                "rl_multiplier": round(rl_multiplier, 4),
                "rl_size": round(float(rl_size), 4),
                "rl_confidence": round(float(rl_decision.get("confidence", 0.5)), 4),
            })

            # Set stops
            stop_loss_pct = cfg.FUTURES_STOP_LOSS if is_futures else cfg.STOP_LOSS_PCT
            if signal.direction == "LONG":
                stop = price * (1 + stop_loss_pct / 100)
                target = price * (1 + cfg.TAKE_PROFIT_PCT / 100)
            else:
                stop = price * (1 - stop_loss_pct / 100)
                target = price * (1 - cfg.TAKE_PROFIT_PCT / 100)

            # Create position
            position = Position(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=price,
                size=size,
                entry_time=datetime.now(),
                stop_loss=stop,
                take_profit=target,
                max_price=price,
                ml_confidence=signal.confidence,
                entry_reason=signal.reason
            )

            # Execute (paper trading)
            if cfg.DRY_RUN:
                self.logger.info(
                    f"DRY RUN: Would OPEN {signal.direction} {signal.symbol} @ ${price:.2f} | "
                    f"Size: ${size:.2f} | Conf: {signal.confidence:.2%}"
                )
                continue

            if cfg.PAPER_TRADING:
                fee = size * 0.001  # 0.1% fee
                if is_futures:
                    self.balance_futures -= fee
                else:
                    self.balance_spot -= fee

            position_key = self._next_position_key(signal.symbol)
            self.positions[position_key] = position
            if is_futures:
                futures_positions += 1
            else:
                spot_positions += 1
            symbol_position_counts[signal.symbol] = symbol_position_counts.get(signal.symbol, 0) + 1
            sd_key = f"{signal.symbol}:{signal.direction}"
            symbol_direction_counts[sd_key] = symbol_direction_counts.get(sd_key, 0) + 1

            self.logger.info(
                f"OPEN {signal.direction} {signal.symbol} @ ${price:.2f} | "
                f"Size: ${size:.2f} | Conf: {signal.confidence:.2%} | "
                f"RLx:{live_rl_multiplier:.2f} | "
                f"Reason: {signal.reason}"
            )

            self._save_state()

    def log_status(self):
        """Log current status."""
        total_balance = self.balance_spot + self.balance_futures
        pnl = total_balance - (cfg.INITIAL_SPOT_BALANCE + cfg.INITIAL_FUTURES_BALANCE)
        spot_pnl = self.balance_spot - cfg.INITIAL_SPOT_BALANCE
        futures_pnl = self.balance_futures - cfg.INITIAL_FUTURES_BALANCE

        self.logger.info(
            f"Total: ${total_balance:,.2f} | "
            f"Spot: ${self.balance_spot:,.2f} ({spot_pnl:+.2f}) | "
            f"Futures: ${self.balance_futures:,.2f} ({futures_pnl:+.2f}) | "
            f"P&L: ${pnl:+.2f} | "
            f"Positions: {len(self.positions)} | "
            f"Daily: {self.risk_manager.daily_pnl:+.2f}"
        )

    def run(self):
        """Main loop."""
        while self.running:
            try:
                self.cycle += 1
                is_trade_cycle = (self.cycle % cfg.TRADE_CYCLE_INTERVAL == 0)

                self.logger.info(f"--- Cycle {self.cycle} {'[TRADE]' if is_trade_cycle else '[RISK]'} ---")

                # 1. Fetch prices
                self.fetch_prices()

                # 2. Check risk (every cycle)
                self.check_risk()
                self._shadow_check_risk()
                self._check_alerts()

                # 3. Trade decisions (every N cycles)
                if is_trade_cycle:
                    self._ensure_model_ready()
                    self._maybe_retrain_model()
                    signals = self.generate_signals()
                    if signals:
                        self.execute_signals(signals)

                # 4. Log status
                if self.cycle % 5 == 0:
                    self.log_status()

                # 5. Save state
                if self.cycle % 10 == 0:
                    self._save_state()
                    self._persist_rl_shadow_report()

                # Sleep
                for _ in range(cfg.RISK_CHECK_INTERVAL):
                    if not self.running:
                        break
                    time.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)

        self.logger.info("Bot stopped gracefully")
        self._persist_rl_shadow_report()
        self._save_state()

# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """Create and run the trading bot."""
    bot = TradingBot()
    bot.run()

if __name__ == "__main__":
    main()
