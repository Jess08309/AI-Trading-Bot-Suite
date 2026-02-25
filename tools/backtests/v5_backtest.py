"""
V5 Backtester — mirrors the EXACT signal & exit logic of trading_engine.py v5.0 QUALITY-FIRST.

Downloads 1-MINUTE candle data from Coinbase for all traded symbols, then:
1. Trains the same GradientBoosting ML model (balanced, undersampled)
2. Generates signals using the same thresholds: MIN_ML_CONFIDENCE, SIDE filter,
   trend following, direction triggers, RSI gates, counter-trend override
3. Applies the same exit rules: STOP_LOSS, TAKE_PROFIT, TRAILING_STOP,
   MAX_HOLD_FLAT, STALE_PROFIT_DECAY, MAX_HOLD_FORCED
4. Reports performance by direction (LONG / SHORT), per-symbol, and overall

Run:
    cd D:\\042021\\CryptoBot
    .venv\\Scripts\\python.exe tools/backtests/v5_backtest.py

Flags:
    --days 90         How many days of history to download (default 90)
    --balance 2500    Starting balance per side (default 2500)
    --direction both  Direction mode: both / short_only / long_only
    --skip-download   Re-use previously downloaded data
    --verbose         Print every trade
"""

import os, sys, csv, json, time, math, argparse, logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

import requests
import numpy as np

# ── project root so we can import the real ML model code ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# Import the real 15-indicator engine
try:
    from cryptotrades.utils.technical_indicators import compute_all_indicators as _compute_indicators
except ImportError:
    try:
        from utils.technical_indicators import compute_all_indicators as _compute_indicators
    except ImportError:
        _compute_indicators = None

# Individual indicator functions for VECTORIZED computation (precompute once)
try:
    from cryptotrades.utils.technical_indicators import (
        rsi as _rsi_full, macd as _macd_full, stochastic as _stoch_full,
        cci as _cci_full, rate_of_change as _roc_full, momentum as _momentum_full,
        williams_r as _wr_full, ultimate_oscillator as _uo_full, trix as _trix_full,
        chande_momentum_oscillator as _cmo_full, atr_approx as _atr_full,
        trend_strength as _ts_full, bollinger_bands as _bb_full,
        mean_reversion_score as _mr_full, volatility_ratio as _vr_full,
    )
    _HAS_VECTORIZED = True
except ImportError:
    _HAS_VECTORIZED = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("v5_backtest")

# Canonical 15-feature order (must match trading_engine.py)
_ML_FEATURE_NAMES = [
    "rsi_14", "macd_histogram", "stoch_k", "cci_20", "roc_10",
    "momentum_10", "williams_r", "ultimate_osc", "trix_15", "cmo_14",
    "atr_14", "trend_strength", "bb_position", "mean_reversion", "vol_ratio",
]

# =====================================================================
# CONFIG — mirrors trading_engine.py defaults
# =====================================================================
SPOT_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
    "DOGE-USD", "LTC-USD", "LINK-USD", "XRP-USD", "DOT-USD",
    "UNI-USD", "ATOM-USD", "XLM-USD", "BCH-USD", "NEAR-USD",
    "AAVE-USD", "MATIC-USD",
]

# Symbols we can actually download OHLC from Coinbase
DOWNLOADABLE = SPOT_SYMBOLS  # Coinbase REST only serves spot product candles

DATA_DIR = os.path.join(PROJECT_ROOT, "tools", "backtests", "data")
CACHE_CSV = os.path.join(DATA_DIR, "minute_prices.csv")
BARSPERHOUR = 60  # 1-minute candles → 60 bars per hour

# ── V5 Engine defaults — updated to match 15-indicator momentum suite ──
MIN_ML_CONFIDENCE   = 0.66
MIN_ENSEMBLE_SCORE  = 0.62
SIDE_MARKET_FILTER  = True
SIDE_ML_OVERRIDE    = 0.75
COUNTER_TREND_OVERRIDE = 0.92
MIN_TREND_SLOPE     = 0.0005
MAX_RSI_LONG        = 65.0
MIN_RSI_SHORT       = 35.0
SENTIMENT_THRESHOLD = 0.15

STOP_LOSS_PCT       = -2.0
TAKE_PROFIT_PCT     = 2.5
TRAILING_STOP_PCT   = 1.5
MAX_HOLD_HOURS      = 4
MAX_HOLD_FLAT_BAND  = 0.5
STALE_PROFIT_HOURS  = 2
STALE_PROFIT_MIN    = 0.3

MAX_POSITION_PCT    = 0.08
MIN_POSITION_PCT    = 0.02


# =====================================================================
# DATA DOWNLOAD
# =====================================================================

def download_coinbase_candles(symbol: str, days: int) -> List[dict]:
    """Download 1-minute candles from Coinbase REST (granularity=60)."""
    base = "https://api.exchange.coinbase.com/products"
    gran = 60  # 1-minute candles
    end = datetime.now()
    start = end - timedelta(days=days)
    all_candles = []

    current = start
    while current < end:
        chunk_end = min(current + timedelta(minutes=299), end)  # max 300 per req
        url = (f"{base}/{symbol}/candles"
               f"?start={current.isoformat()}"
               f"&end={chunk_end.isoformat()}"
               f"&granularity={gran}")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                for c in data:
                    ts, low, high, opn, close, vol = c
                    all_candles.append({
                        "timestamp": datetime.fromtimestamp(ts).isoformat(),
                        "ts_epoch": ts,
                        "symbol": symbol,
                        "open": float(opn),
                        "high": float(high),
                        "low": float(low),
                        "close": float(close),
                        "volume": float(vol),
                    })
            elif r.status_code == 404:
                logger.warning(f"  {symbol}: 404 — not on Coinbase, skipping")
                return []
            else:
                logger.warning(f"  {symbol}: HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"  {symbol}: error {e}")
        current = chunk_end
        time.sleep(0.35)  # rate limit

    # Deduplicate + sort ascending
    seen = set()
    unique = []
    for c in all_candles:
        if c["ts_epoch"] not in seen:
            seen.add(c["ts_epoch"])
            unique.append(c)
    unique.sort(key=lambda x: x["ts_epoch"])
    return unique


def download_all(symbols: List[str], days: int) -> Dict[str, List[dict]]:
    """Download 1-minute data for all symbols, save to CSV cache."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_data: Dict[str, List[dict]] = {}
    for sym in symbols:
        print(f"  Downloading {sym}...", end=" ", flush=True)
        candles = download_coinbase_candles(sym, days)
        if candles:
            all_data[sym] = candles
            print(f"{len(candles)} candles")
        else:
            print("SKIPPED")
    # Save cache
    with open(CACHE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        for sym, rows in all_data.items():
            for c in rows:
                w.writerow([c["timestamp"], c["symbol"],
                            c["open"], c["high"], c["low"], c["close"], c["volume"]])
    logger.info(f"Saved {sum(len(v) for v in all_data.values())} total candles to {CACHE_CSV}")
    return all_data


def load_cache() -> Dict[str, List[dict]]:
    """Load previously downloaded data from CSV cache."""
    if not os.path.exists(CACHE_CSV):
        raise FileNotFoundError(f"No cached data at {CACHE_CSV}. Run without --skip-download.")
    data: Dict[str, List[dict]] = defaultdict(list)
    with open(CACHE_CSV) as f:
        for row in csv.DictReader(f):
            data[row["symbol"]].append({
                "timestamp": row["timestamp"],
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "open": float(row["open"]),
                "volume": float(row["volume"]),
            })
    logger.info(f"Loaded cache: {len(data)} symbols, {sum(len(v) for v in data.values())} candles")
    return dict(data)


# =====================================================================
# ML MODEL (mirrors trading_engine.py MLModel exactly)
# =====================================================================

def extract_features(prices: List[float]) -> Optional[List[float]]:
    """Extract 15 indicator features using compute_all_indicators (mirrors live engine)."""
    if len(prices) < 30 or _compute_indicators is None:
        return None
    try:
        feat_dict = _compute_indicators(prices)
        if not feat_dict:
            return None
        return [float(feat_dict.get(name, 0.0)) for name in _ML_FEATURE_NAMES]
    except Exception:
        return None


def precompute_indicator_arrays(prices: List[float]) -> Optional[np.ndarray]:
    """Compute all 15 indicators as full arrays over the price series.

    Returns a (15, N) numpy array. Computed ONCE per symbol, then indexed
    by bar number — this is 1000x faster than recomputing at every bar.
    """
    if not _HAS_VECTORIZED:
        return None
    n = len(prices)
    if n < 30:
        return None
    try:
        rsi_arr = _rsi_full(prices, 14)
        _ml, _sig, macd_hist_arr = _macd_full(prices)
        stoch_k_arr, _ = _stoch_full(prices, 14)
        cci_arr = _cci_full(prices, 20)
        roc_arr = _roc_full(prices, min(10, n - 1))
        mom_arr = _momentum_full(prices, min(10, n - 1))
        wr_arr = _wr_full(prices, 14)
        uo_arr = _uo_full(prices, 7, 14, min(28, n - 1))
        trix_arr = _trix_full(prices, min(15, max(3, n // 4)))
        cmo_arr = _cmo_full(prices, 14)
        atr_arr = _atr_full(prices, 14)
        p_arr = np.array(prices, dtype=float)
        atr_norm = np.where(p_arr > 0, atr_arr / p_arr, 0.0)
        ts_arr = _ts_full(prices, min(20, n))
        bb_up, _bb_mid, bb_lo = _bb_full(prices, min(20, n))
        bb_w = bb_up - bb_lo
        bb_pos = np.where((bb_w > 0) & ~np.isnan(bb_up) & ~np.isnan(bb_lo),
                          (p_arr - bb_lo) / bb_w, 0.5)
        mr_arr = _mr_full(prices, min(20, n))
        vr_arr = _vr_full(prices, 5, min(20, n))

        result = np.stack([rsi_arr, macd_hist_arr, stoch_k_arr, cci_arr, roc_arr,
                           mom_arr, wr_arr, uo_arr, trix_arr, cmo_arr,
                           atr_norm, ts_arr, bb_pos, mr_arr, vr_arr])
        defaults = [50., 0., 50., 0., 0., 0., -50., 50., 0., 0., 0., 0., 0.5, 0., 1.]
        for row, d in enumerate(defaults):
            result[row] = np.where(np.isnan(result[row]), d, result[row])
        return result  # shape (15, n)
    except Exception:
        return None


def train_model(price_series: Dict[str, List[float]]) -> Optional[GradientBoostingClassifier]:
    """Train a GBM exactly like the live engine (balanced, undersampled).
    VECTORIZED: precomputes indicator arrays per symbol for O(n) extraction."""
    X, y = [], []
    for symbol, prices in price_series.items():
        max_pts = 2000
        recent = prices[-max_pts:] if len(prices) > max_pts else prices
        if len(recent) < 60:
            continue

        # Vectorized path: compute indicators once, index by bar
        indicators = precompute_indicator_arrays(recent)
        if indicators is not None:
            for i in range(30, len(recent) - 10):
                future_ret = (recent[i + 10] - recent[i]) / recent[i]
                if abs(future_ret) < 0.002:
                    continue
                label = 1 if future_ret > 0.005 else 0
                X.append(indicators[:, i].tolist())
                y.append(label)
        else:
            # Fallback: per-bar computation (slow)
            for i in range(30, len(recent) - 10):
                feat = extract_features(recent[:i])
                if feat is not None:
                    future_ret = (recent[i + 10] - recent[i]) / recent[i]
                    if abs(future_ret) < 0.002:
                        continue
                    label = 1 if future_ret > 0.005 else 0
                    X.append(feat)
                    y.append(label)

    if len(X) < 200:
        logger.error(f"Only {len(X)} training samples — not enough")
        return None

    X, y = np.array(X), np.array(y)
    ratio = np.mean(y)
    logger.info(f"Training: {len(y)} samples, {int(np.sum(y))} UP ({ratio:.1%}), {len(y)-int(np.sum(y))} DOWN ({1-ratio:.1%})")

    # Undersample if imbalanced (mirrors engine)
    if ratio < 0.35 or ratio > 0.65:
        up_idx = np.where(y == 1)[0]
        dn_idx = np.where(y == 0)[0]
        minority = min(len(up_idx), len(dn_idx))
        if minority < 50:
            logger.error("Too few minority samples")
            return None
        if len(up_idx) > len(dn_idx):
            up_idx = np.random.choice(up_idx, size=minority, replace=False)
        else:
            dn_idx = np.random.choice(dn_idx, size=minority, replace=False)
        idx = np.concatenate([up_idx, dn_idx])
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]
        logger.info(f"After undersampling: {len(y)} samples, ratio={np.mean(y):.1%}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)
    sw = compute_sample_weight('balanced', y_train)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=sw)
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)
    logger.info(f"Model: train={train_acc:.2%}, test={test_acc:.2%}")

    # Log top feature importances
    try:
        importances = model.feature_importances_
        top = sorted(zip(_ML_FEATURE_NAMES, importances), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"{n}={v:.3f}" for n, v in top)
        logger.info(f"Top features: {top_str}")
    except Exception:
        pass

    if test_acc < 0.52:
        logger.warning(f"Model test accuracy {test_acc:.2%} below 52% — using anyway for backtest")
    return model


def ml_predict(model: GradientBoostingClassifier, prices: List[float],
               precomputed_features: Optional[List[float]] = None) -> Dict[str, float]:
    """Predict exactly like trading_engine MLModel.predict.
    If precomputed_features is given, skip indicator calculation (fast path)."""
    if model is None:
        return {"direction": 0.5, "confidence": 0.5}
    if precomputed_features is not None:
        feat = precomputed_features
    elif len(prices) < 30:
        return {"direction": 0.5, "confidence": 0.5}
    else:
        feat = extract_features(prices)
    if feat is None:
        return {"direction": 0.5, "confidence": 0.5}
    try:
        feat_arr = np.nan_to_num(np.array(feat).reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        proba = model.predict_proba(feat_arr)[0]
        up_prob = proba[1]
        confidence = max(proba[0], proba[1])
        return {"direction": up_prob, "confidence": confidence}
    except Exception:
        return {"direction": 0.5, "confidence": 0.5}


# =====================================================================
# TECHNICAL INDICATORS (mirror engine helpers)
# =====================================================================

def calc_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    p = np.array(prices)
    deltas = np.diff(p[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gains)
    avg_l = np.mean(losses)
    if avg_l == 0:
        return 100.0
    return 100 - (100 / (1 + avg_g / avg_l))


def calc_trend(prices: List[float], lookback: int = 20) -> Tuple[str, float]:
    if len(prices) < lookback:
        return "SIDE", 0.0
    recent = prices[-lookback:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0] / np.mean(recent)
    if slope > MIN_TREND_SLOPE:
        return "UP", slope
    elif slope < -MIN_TREND_SLOPE:
        return "DOWN", slope
    return "SIDE", slope


def calc_volatility(prices: List[float], lookback: int = 20) -> float:
    if len(prices) < lookback:
        return 0.01
    p = np.array(prices[-lookback:])
    returns = np.diff(p) / p[:-1]
    return float(np.std(returns))


def kelly_size(balance: float, confidence: float, volatility: float, n_pos: int) -> float:
    """Mirrors engine calculate_position_size."""
    kelly = max(0.01, (2 * confidence) - 1)
    vol_adj = 1.0 / (1 + volatility * 10)
    size = balance * MAX_POSITION_PCT * kelly * vol_adj
    pos_factor = max(0.3, 1 - (n_pos * 0.15))
    size *= pos_factor
    return max(balance * MIN_POSITION_PCT, min(size, balance * MAX_POSITION_PCT))


# =====================================================================
# SIGNAL GENERATION (mirrors generate_signals exactly)
# =====================================================================

@dataclass
class Signal:
    symbol: str
    direction: str  # LONG or SHORT
    confidence: float
    ml_direction: float
    rsi: float
    trend: str
    reason: str


def generate_signal(
    model: GradientBoostingClassifier,
    symbol: str,
    prices: List[float],
    direction_mode: str = "both",
    blacklist: set = None,
    sentiment: float = 0.0,
    precomputed_features: Optional[List[float]] = None,
) -> Optional[Signal]:
    """Generate a single signal for one symbol, exactly mirroring v5 logic."""
    if blacklist and symbol in blacklist:
        return None
    if len(prices) < 30:
        return None

    pred = ml_predict(model, prices, precomputed_features=precomputed_features)
    ml_conf = pred["confidence"]
    ml_dir  = pred["direction"]

    if ml_conf < MIN_ML_CONFIDENCE:
        return None

    rsi   = calc_rsi(prices)
    trend, slope = calc_trend(prices)
    vol   = calc_volatility(prices)

    # SIDE-market filter
    if SIDE_MARKET_FILTER and trend == "SIDE":
        if ml_conf < SIDE_ML_OVERRIDE:
            return None

    # Direction triggers
    long_trigger  = MIN_ENSEMBLE_SCORE
    short_trigger = 1 - MIN_ENSEMBLE_SCORE

    direction = None
    confidence = ml_conf

    if ml_dir > long_trigger:
        if trend == "UP":
            if rsi < MAX_RSI_LONG:
                direction = "LONG"
                confidence = ml_conf
        elif trend == "SIDE" and ml_conf >= SIDE_ML_OVERRIDE:
            if rsi < MAX_RSI_LONG and sentiment >= -SENTIMENT_THRESHOLD:
                direction = "LONG"
                confidence = ml_conf * 0.85
        elif trend == "DOWN" and ml_conf >= COUNTER_TREND_OVERRIDE:
            if rsi < MAX_RSI_LONG:
                direction = "LONG"
                confidence = ml_conf * 0.70

    elif ml_dir < short_trigger:
        if trend == "DOWN":
            if rsi > MIN_RSI_SHORT:
                direction = "SHORT"
                confidence = ml_conf
        elif trend == "SIDE" and ml_conf >= SIDE_ML_OVERRIDE:
            if rsi > MIN_RSI_SHORT and sentiment <= SENTIMENT_THRESHOLD:
                direction = "SHORT"
                confidence = ml_conf * 0.85
        elif trend == "UP" and ml_conf >= COUNTER_TREND_OVERRIDE:
            if rsi > MIN_RSI_SHORT:
                direction = "SHORT"
                confidence = ml_conf * 0.70

    # Direction mode filter
    if direction == "LONG" and direction_mode == "short_only":
        direction = None
    elif direction == "SHORT" and direction_mode == "long_only":
        direction = None

    if not direction:
        return None

    return Signal(
        symbol=symbol, direction=direction, confidence=confidence,
        ml_direction=ml_dir, rsi=rsi, trend=trend,
        reason=f"ML:{ml_dir:.2f}|Trend:{trend}|RSI:{rsi:.0f}"
    )


# =====================================================================
# POSITION + EXIT LOGIC
# =====================================================================

@dataclass
class Position:
    symbol: str
    direction: str  # LONG / SHORT
    entry_price: float
    size_usd: float
    confidence: float
    entry_idx: int
    max_price: float   # for trailing stop (LONG)
    min_price: float   # for trailing stop (SHORT)


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    hold_bars: int


def check_exit(pos: Position, current_price: float, bar_idx: int) -> Optional[str]:
    """Check if position should be exited — mirrors engine _check_exits."""
    hold_bars = bar_idx - pos.entry_idx

    if pos.direction == "LONG":
        pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
        if current_price > pos.max_price:
            pos.max_price = current_price
        drawdown = ((pos.max_price - current_price) / pos.max_price) * 100
    else:  # SHORT
        pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
        if current_price < pos.min_price:
            pos.min_price = current_price
        drawdown = ((current_price - pos.min_price) / pos.min_price) * 100

    # 1. Stop loss
    if pnl_pct <= STOP_LOSS_PCT:
        return "STOP_LOSS"

    # 2. Take profit
    if pnl_pct >= TAKE_PROFIT_PCT:
        return "TAKE_PROFIT"

    # 3. Trailing stop (only if in profit > 1%)
    if drawdown >= TRAILING_STOP_PCT and pnl_pct > 1.0:
        return "TRAILING_STOP"

    # 4. Stale profit decay (profitable but stalling)
    if hold_bars >= STALE_PROFIT_HOURS * BARSPERHOUR and 0 < pnl_pct < STALE_PROFIT_MIN:
        return "STALE_PROFIT_DECAY"

    # 5. Max hold flat
    if hold_bars >= MAX_HOLD_HOURS * BARSPERHOUR:
        if abs(pnl_pct) < MAX_HOLD_FLAT_BAND:
            return "MAX_HOLD_FLAT"
        # 6. Max hold forced (2x max hold)
        if hold_bars >= MAX_HOLD_HOURS * BARSPERHOUR * 2:
            return "MAX_HOLD_FORCED"

    return None


def close_position(pos: Position, exit_price: float, bar_idx: int, reason: str) -> Trade:
    if pos.direction == "LONG":
        pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
    else:
        pnl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100
    pnl_usd = pos.size_usd * (pnl_pct / 100)
    return Trade(
        symbol=pos.symbol, direction=pos.direction,
        entry_price=pos.entry_price, exit_price=exit_price,
        size_usd=pos.size_usd, pnl_usd=pnl_usd, pnl_pct=pnl_pct,
        exit_reason=reason, hold_bars=bar_idx - pos.entry_idx,
    )


# =====================================================================
# MAIN BACKTEST ENGINE
# =====================================================================

def run_backtest(
    all_prices: Dict[str, List[float]],
    direction_mode: str = "both",
    blacklist: set = None,
    starting_balance: float = 2500.0,
    verbose: bool = False,
) -> Tuple[List[Trade], float]:
    """
    Walk-forward backtest.
    - Trains model on first 40% of data
    - Trades on remaining 60% 
    - Retrains every 200 bars (simulates live retrain)
    """
    if blacklist is None:
        blacklist = set()

    # Determine bar count — use LONGEST series so each symbol uses all its data
    symbols = [s for s in all_prices if s not in blacklist and len(all_prices[s]) >= 200]
    if not symbols:
        logger.error("No symbols with enough data")
        return [], starting_balance

    sym_lengths = {s: len(all_prices[s]) for s in symbols}
    max_bars = max(sym_lengths.values())
    min_bars_info = min(sym_lengths.values())
    logger.info(f"Backtesting {len(symbols)} symbols over {max_bars} 1-min bars "
                f"(~{max_bars/1440:.0f} days max, ~{min_bars_info/1440:.0f} days min)")
    for s in symbols:
        logger.info(f"  {s}: {sym_lengths[s]} bars ({sym_lengths[s]/1440:.1f} days)")

    # ── Precompute indicator arrays for ALL symbols (vectorized, done once) ──
    logger.info("Precomputing indicator arrays for all symbols...")
    precomp_t0 = time.time()
    indicator_arrays: Dict[str, Optional[np.ndarray]] = {}
    for s in symbols:
        indicator_arrays[s] = precompute_indicator_arrays(all_prices[s])
    logger.info(f"Indicators precomputed in {time.time() - precomp_t0:.1f}s")

    # Split: first 40% of the LONGEST series for initial training window
    split_idx = int(max_bars * 0.4)
    retrain_interval = 12000  # retrain every 12000 bars (~200 hours = ~8 days at 1-min)

    # Initial training — each symbol contributes what it has up to split_idx
    train_series = {}
    for s in symbols:
        end = min(split_idx, sym_lengths[s])
        if end >= 60:  # need at least 60 bars for training
            train_series[s] = all_prices[s][:end]
    model = train_model(train_series)
    if model is None:
        logger.error("Initial model training failed")
        return [], starting_balance

    # Trading loop
    balance = starting_balance
    positions: Dict[str, Position] = {}  # symbol → Position
    all_trades: List[Trade] = []
    last_retrain = split_idx
    progress_interval = max(1, (max_bars - split_idx) // 20)  # 5% progress updates

    for bar in range(split_idx, max_bars):
        # Which symbols have data at this bar?
        active_symbols = [s for s in symbols if bar < sym_lengths[s]]

        # ── Progress logging ──
        if (bar - split_idx) % progress_interval == 0:
            pct = (bar - split_idx) / (max_bars - split_idx) * 100
            logger.info(f"  Progress: {pct:.0f}% (bar {bar}/{max_bars}, "
                        f"{len(all_trades)} trades, balance ${balance:+.2f})")

        # ── Retrain periodically ──
        if bar - last_retrain >= retrain_interval:
            retrain_series = {}
            for s in symbols:
                end = min(bar, sym_lengths[s])
                if end >= 60:
                    retrain_series[s] = all_prices[s][:end]
            new_model = train_model(retrain_series)
            if new_model is not None:
                model = new_model
            last_retrain = bar

        # ── Check exits on open positions ──
        closed_this_bar = []
        for sym, pos in list(positions.items()):
            # If this symbol ran out of data, force-close
            if bar >= sym_lengths[sym]:
                last_price = all_prices[sym][sym_lengths[sym] - 1]
                trade = close_position(pos, last_price, bar, "END_OF_DATA")
                all_trades.append(trade)
                balance += trade.pnl_usd
                closed_this_bar.append(sym)
                if verbose:
                    print(f"  [{bar}] CLOSE {trade.direction} {sym} @ {last_price:.4f} "
                          f"P&L: ${trade.pnl_usd:+.2f} ({trade.pnl_pct:+.2f}%) "
                          f"[END_OF_DATA]")
                continue

            price = all_prices[sym][bar]
            reason = check_exit(pos, price, bar)
            if reason:
                trade = close_position(pos, price, bar, reason)
                all_trades.append(trade)
                balance += trade.pnl_usd  # P&L settles back to balance
                closed_this_bar.append(sym)
                if verbose:
                    print(f"  [{bar}] CLOSE {trade.direction} {sym} @ {price:.4f} "
                          f"P&L: ${trade.pnl_usd:+.2f} ({trade.pnl_pct:+.2f}%) "
                          f"[{trade.exit_reason}]")
        for sym in closed_this_bar:
            del positions[sym]

        # ── Generate new signals (using precomputed features — fast!) ──
        for sym in active_symbols:
            if sym in positions:
                continue  # already have a position
            # Fast path: grab features from precomputed indicator arrays
            features = None
            if indicator_arrays.get(sym) is not None and bar < indicator_arrays[sym].shape[1]:
                features = indicator_arrays[sym][:, bar].tolist()
            prices_up_to = all_prices[sym][:bar + 1]
            sig = generate_signal(model, sym, prices_up_to, direction_mode, blacklist,
                                  precomputed_features=features)
            if sig is None:
                continue

            vol = calc_volatility(prices_up_to)
            size = kelly_size(balance, sig.confidence, vol, len(positions))
            if size < 10 or size > balance * 0.5:
                continue

            price = all_prices[sym][bar]
            positions[sym] = Position(
                symbol=sym, direction=sig.direction,
                entry_price=price, size_usd=size,
                confidence=sig.confidence, entry_idx=bar,
                max_price=price, min_price=price,
            )
            balance -= 0  # paper — size tracked in position, P&L settles on close
            if verbose:
                print(f"  [{bar}] OPEN {sig.direction} {sym} @ {price:.4f} "
                      f"${size:.2f} conf={sig.confidence:.2f} [{sig.reason}]")

    # Close any remaining positions at end
    for sym, pos in positions.items():
        last_idx = sym_lengths[sym] - 1
        price = all_prices[sym][last_idx]
        trade = close_position(pos, price, max_bars - 1, "END_OF_DATA")
        all_trades.append(trade)
        balance += trade.pnl_usd

    ending_balance = starting_balance + sum(t.pnl_usd for t in all_trades)
    return all_trades, ending_balance


# =====================================================================
# REPORTING
# =====================================================================

def print_report(trades: List[Trade], starting_bal: float, label: str = ""):
    total_pnl = sum(t.pnl_usd for t in trades)
    wins  = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    avg_w = np.mean([t.pnl_usd for t in wins]) if wins else 0
    avg_l = np.mean([t.pnl_usd for t in losses]) if losses else 0

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS{' — ' + label if label else ''}")
    print(f"{'='*60}")
    print(f"  Total Trades:    {len(trades)}")
    print(f"  Win Rate:        {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Total P&L:       ${total_pnl:+.2f}")
    print(f"  Avg Win:         ${avg_w:+.2f}")
    print(f"  Avg Loss:        ${avg_l:+.2f}")
    ending = starting_bal + total_pnl
    print(f"  Return:          {(total_pnl/starting_bal)*100:+.2f}%")
    print(f"  Balance:         ${starting_bal:.2f} -> ${ending:.2f}")

    # Direction split
    longs  = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]
    if longs:
        lw = len([t for t in longs if t.pnl_usd > 0])
        lp = sum(t.pnl_usd for t in longs)
        print(f"\n  LONG:  {len(longs)} trades, {lw/len(longs)*100:.1f}% WR, P&L ${lp:+.2f}")
    if shorts:
        sw = len([t for t in shorts if t.pnl_usd > 0])
        sp = sum(t.pnl_usd for t in shorts)
        print(f"  SHORT: {len(shorts)} trades, {sw/len(shorts)*100:.1f}% WR, P&L ${sp:+.2f}")

    # Exit reasons
    reasons = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        reasons[t.exit_reason]["count"] += 1
        reasons[t.exit_reason]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            reasons[t.exit_reason]["wins"] += 1

    print(f"\n  EXIT REASONS:")
    for r, d in sorted(reasons.items(), key=lambda x: x[1]["pnl"]):
        wr_r = d["wins"] / d["count"] * 100 if d["count"] else 0
        print(f"    {r:22s}: {d['count']:4d} trades  {wr_r:5.1f}% WR  P&L ${d['pnl']:+8.2f}")

    # Per-symbol
    sym_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        sym_stats[t.symbol]["count"] += 1
        sym_stats[t.symbol]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            sym_stats[t.symbol]["wins"] += 1

    print(f"\n  SYMBOL BREAKDOWN (sorted by P&L):")
    for sym, d in sorted(sym_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr_s = d["wins"] / d["count"] * 100 if d["count"] else 0
        print(f"    {sym:15s}: {d['count']:4d} trades  {wr_s:5.1f}% WR  P&L ${d['pnl']:+8.2f}")

    # Drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t.pnl_usd
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
    print(f"\n  Max Drawdown:    ${max_dd:.2f}")
    print(f"{'='*60}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="V5 Engine Backtester")
    parser.add_argument("--days", type=int, default=30, help="Days of history (default 30)")
    parser.add_argument("--balance", type=float, default=2500.0, help="Starting balance (default 2500)")
    parser.add_argument("--direction", type=str, default="both", choices=["both", "short_only", "long_only"])
    parser.add_argument("--blacklist", type=str, default="", help="Comma-separated symbols to blacklist")
    parser.add_argument("--skip-download", action="store_true", help="Use cached data")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    blacklist = set(s.strip() for s in args.blacklist.split(",") if s.strip())

    print(f"\n{'='*60}")
    print(f"V5 BACKTEST — {args.days} days, ${args.balance:.0f} balance")
    print(f"Direction: {args.direction}  Blacklist: {blacklist or 'none'}")
    print(f"{'='*60}\n")

    # ── Get data ──
    if args.skip_download:
        raw_data = load_cache()
    else:
        print("Downloading 1-minute candle data from Coinbase...")
        raw_data = download_all(DOWNLOADABLE, args.days)

    # Convert to {symbol: [close_prices]}
    all_prices: Dict[str, List[float]] = {}
    for sym, candles in raw_data.items():
        closes = [c["close"] for c in candles]
        if len(closes) >= 200:
            all_prices[sym] = closes
            logger.info(f"  {sym}: {len(closes)} bars ({len(closes)/1440:.1f} days)")
        else:
            logger.warning(f"  {sym}: only {len(closes)} bars — skipping")

    if not all_prices:
        logger.error("No symbols with sufficient data")
        return

    # ── Run: BOTH directions ──
    print("\n" + "="*60)
    print("SCENARIO 1: BOTH directions (no restrictions)")
    print("="*60)
    trades_both, bal_both = run_backtest(all_prices, "both", blacklist, args.balance, args.verbose)
    if trades_both:
        print_report(trades_both, args.balance, "BOTH DIRECTIONS")

    # ── Run: SHORT only ──
    print("\n" + "="*60)
    print("SCENARIO 2: SHORT only")
    print("="*60)
    trades_short, bal_short = run_backtest(all_prices, "short_only", blacklist, args.balance, args.verbose)
    if trades_short:
        print_report(trades_short, args.balance, "SHORT ONLY")

    # ── Run: LONG only ──
    print("\n" + "="*60)
    print("SCENARIO 3: LONG only")
    print("="*60)
    trades_long, bal_long = run_backtest(all_prices, "long_only", blacklist, args.balance, args.verbose)
    if trades_long:
        print_report(trades_long, args.balance, "LONG ONLY")

    # ── Run: With blacklist vs without ──
    if blacklist:
        print("\n" + "="*60)
        print("SCENARIO 4: BOTH directions, NO blacklist")
        print("="*60)
        trades_noblack, bal_noblack = run_backtest(all_prices, "both", set(), args.balance, args.verbose)
        if trades_noblack:
            print_report(trades_noblack, args.balance, "NO BLACKLIST")

    # ── Summary comparison ──
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    scenarios = [
        ("BOTH", trades_both, bal_both),
        ("SHORT_ONLY", trades_short, bal_short),
        ("LONG_ONLY", trades_long, bal_long),
    ]
    if blacklist:
        scenarios.append(("NO_BLACKLIST", trades_noblack, bal_noblack))

    print(f"  {'Scenario':15s} {'Trades':>7s} {'WR':>7s} {'P&L':>10s} {'Return':>8s}")
    print(f"  {'-'*50}")
    for name, trades, bal in scenarios:
        if trades:
            wr = len([t for t in trades if t.pnl_usd > 0]) / len(trades) * 100
            pnl = sum(t.pnl_usd for t in trades)
            ret = (pnl / args.balance) * 100
            print(f"  {name:15s} {len(trades):7d} {wr:6.1f}% ${pnl:+9.2f} {ret:+7.2f}%")
        else:
            print(f"  {name:15s}       0     N/A       N/A     N/A")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
