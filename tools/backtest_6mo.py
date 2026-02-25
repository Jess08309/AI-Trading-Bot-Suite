"""
6-Month Walk-Forward Backtest
=============================
Uses the 6-month 1-minute candle data already downloaded in data/historical/1min/
Mirrors the EXACT v5 signal & exit logic from trading_engine.py.

Walk-forward: trains on first 40%, tests on remaining 60%, retrains periodically.
Uses precomputed indicator arrays for speed (~100x faster than per-bar computation).

Run:
    cd c:\\Bot
    .venv\\Scripts\\python.exe tools\\backtest_6mo.py
    .venv\\Scripts\\python.exe tools\\backtest_6mo.py --use-production-model
    .venv\\Scripts\\python.exe tools\\backtest_6mo.py --verbose
"""

import os, sys, time, math, argparse, logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

try:
    from cryptotrades.utils.technical_indicators import compute_all_indicators as _compute_indicators
except ImportError:
    _compute_indicators = None

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
logger = logging.getLogger("backtest_6mo")

# Also log to file for monitoring
_log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest", "backtest_6mo_output.txt")
os.makedirs(os.path.dirname(_log_file), exist_ok=True)
_fh = logging.FileHandler(_log_file, mode="w", encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_fh)

# Canonical 15-feature order (must match trading_engine.py)
_ML_FEATURE_NAMES = [
    "rsi_14", "macd_histogram", "stoch_k", "cci_20", "roc_10",
    "momentum_10", "williams_r", "ultimate_osc", "trix_15", "cmo_14",
    "atr_14", "trend_strength", "bb_position", "mean_reversion", "vol_ratio",
]

# =====================================================================
# CONFIG — mirrors locked_profile.json exactly
# =====================================================================
SPOT_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "AAVE-USD",
    # v5: ONLY proven winners + high-liquidity majors
    # EXCLUDED: NEAR (-$10.83), DOT (-$7.37), ATOM (-$6.93), LTC (-$4.91),
    #           BCH (-$3.26), UNI (-$21.56), XLM (-$9.49)
    #           ADA, AVAX, DOGE, LINK, MATIC — marginal/neg in v4
]

# ── Futures symbols (mapped to spot data, simulated with leverage) ──
FUTURES_SYMBOLS = [
    "PI_XBTUSD", "PI_ETHUSD", "PI_SOLUSD", "PI_XRPUSD", "PI_AAVEUSD",
    "PI_ADAUSD", "PI_AVAXUSD", "PI_DOGEUSD", "PI_LINKUSD", "PI_DOTUSD",
    "PI_BCHUSD", "PI_LTCUSD",
]

# Map futures symbol → spot data file
FUTURES_TO_SPOT = {
    "PI_XBTUSD": "BTC-USD", "PI_ETHUSD": "ETH-USD", "PI_SOLUSD": "SOL-USD",
    "PI_XRPUSD": "XRP-USD", "PI_AAVEUSD": "AAVE-USD", "PI_ADAUSD": "ADA-USD",
    "PI_AVAXUSD": "AVAX-USD", "PI_DOGEUSD": "DOGE-USD", "PI_LINKUSD": "LINK-USD",
    "PI_DOTUSD": "DOT-USD", "PI_BCHUSD": "BCH-USD", "PI_LTCUSD": "LTC-USD",
}

# Futures parameters
FUTURES_LEVERAGE       = 2
FUTURES_STOP_LOSS_PCT  = -3.0     # wider SL for leveraged (effective -6% on capital)
FUTURES_SLIPPAGE_BPS   = 8        # higher slippage on futures
FUTURES_MAX_HOLD_BARS  = 120      # 2h max hold for futures
FUTURES_STALE_MIN_BARS = 20

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "historical", "1min")

MIN_ML_CONFIDENCE   = 0.80       # v10: slightly relaxed from v9 (0.82) for more opportunities, RSI gate compensates
MIN_ENSEMBLE_SCORE  = 0.62
SIDE_MARKET_FILTER  = True
SIDE_ML_OVERRIDE    = 0.88       # v11b: allow SIDE entries for both directions at high conf
COUNTER_TREND_OVERRIDE = 0.95   # v11b: allow counter-trend only at very high conf
MIN_TREND_SLOPE     = 0.0005
MAX_RSI_LONG        = 65.0       # relaxed RSI gate
MIN_RSI_SHORT       = 38.0
SENTIMENT_THRESHOLD = 0.15
MIN_VOLATILITY      = 0.002      # v9: higher vol floor — skip dead markets
MAX_VOLATILITY      = 0.006      # v9: vol ceiling — skip extreme volatility (high SL risk)

STOP_LOSS_PCT       = -1.2       # v10: tighter SL to reduce loss per hit (v9's selective entries should survive)
TAKE_PROFIT_PCT     = 2.5        # v6: achievable TP. R:R = 2.5/1.5 = 1.67:1
TRAILING_STOP_PCT   = 0.8        # v6: trail at 0.8% drawdown from peak
MAX_HOLD_BARS_SPOT  = 180        # 3h max hold
MAX_HOLD_FLAT_BAND  = 0.3        # reasonable flat band
MAX_HOLD_FORCED_BARS = 6 * 60    # 6 hours hard cap
# v6: FEE-AWARE STALE EXIT — arm-decay must exceed fees (0.3%)
# ARM(1.2%) - DECAY(0.5%) = 0.7% exit > 0.3% fees ✓ (net +0.4%)
# v3 had 68.6% WR because 1.0-0.6=0.4>0.3. v4 crashed: 0.5-0.3=0.2<0.3
STALE_ARM_PCT       = 1.2        # v6: arm after 1.2% peak
STALE_DECAY_PCT     = 0.5        # v6: decay 0.5% from peak → exits at ~0.7%
STALE_MIN_BARS      = 25         # v6: 25 bars min before stale exit

MAX_POSITION_PCT    = 0.40       # v11: aggressive sizing for real profit
MIN_POSITION_PCT    = 0.05
MAX_POSITIONS       = 20         # v11: allow many concurrent positions
MAX_CORRELATION     = 0.85       # v11: relaxed — allow correlated spot+futures
FEE_PCT             = 0.001      # 0.1% per side
SLIPPAGE_BPS        = 5          # 5 basis points

SIGNAL_INTERVAL     = 5          # check signals every 5 bars (5 min, like live TRADE cycle)

# Timeframe multiplier (set by --timeframe arg, used to scale bar-count params)
_TF = 1  # 1 = 1min, 5 = 5min, 10 = 10min

# ── Signal diagnostics (populated during backtest) ──
_diag = {"low_conf": 0, "side_filter": 0, "no_direction": 0, "rsi_gate": 0,
         "corr_gate": 0, "circuit": 0, "dir_pause": 0, "signals_gen": 0, "trades_opened": 0}

# =====================================================================
# DATA LOADING
# =====================================================================

def aggregate_to_timeframe(prices: np.ndarray, tf: int) -> np.ndarray:
    """Aggregate 1-min close prices into tf-minute bars by taking the close of each window."""
    if tf <= 1:
        return prices
    n = len(prices)
    n_bars = n // tf
    if n_bars < 100:
        return prices
    # Take the last price of each tf-bar window (= the close of that candle)
    return np.array([prices[(i + 1) * tf - 1] for i in range(n_bars)])


def load_6mo_data(tf: int = 1, include_futures: bool = False) -> Dict[str, np.ndarray]:
    """Load all per-symbol 1-min CSVs from data/historical/1min/, optionally aggregate.
    If include_futures, also load futures symbols using spot data as proxy."""
    all_prices = {}
    bars_per_day = 1440 // tf

    # Load spot symbols
    for sym in SPOT_SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}_1min.csv")
        if not os.path.exists(path):
            logger.warning(f"  {sym}: no data file, skipping")
            continue
        df = pd.read_csv(path)
        closes = df["close"].values.astype(np.float64)
        if tf > 1:
            closes = aggregate_to_timeframe(closes, tf)
        if len(closes) < 500:
            logger.warning(f"  {sym}: only {len(closes)} bars, skipping")
            continue
        all_prices[sym] = closes
        logger.info(f"  {sym}: {len(closes):,} bars ({len(closes)/bars_per_day:.0f} days)")

    # Load futures symbols (use spot data as proxy — perps track spot within ~0.1%)
    if include_futures:
        for fut_sym in FUTURES_SYMBOLS:
            spot_sym = FUTURES_TO_SPOT.get(fut_sym)
            if not spot_sym:
                continue
            path = os.path.join(DATA_DIR, f"{spot_sym}_1min.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            closes = df["close"].values.astype(np.float64)
            if tf > 1:
                closes = aggregate_to_timeframe(closes, tf)
            if len(closes) < 500:
                continue
            all_prices[fut_sym] = closes
            logger.info(f"  {fut_sym}: {len(closes):,} bars (proxy: {spot_sym})")

    return all_prices


# =====================================================================
# INDICATOR PRECOMPUTATION
# =====================================================================

def precompute_indicators(prices_list: List[float], tf: int = 1) -> Optional[np.ndarray]:
    """Compute all 15 indicators as (15, N) array. Done once per symbol.
    
    When tf > 1, all indicator periods are scaled down so they cover
    the same real-time window as they would on 1-min candles.
    E.g. RSI-14 on 1min (14 min window) → RSI-3 on 5min (15 min window).
    """
    if not _HAS_VECTORIZED:
        return None
    n = len(prices_list)
    if n < 30:
        return None
    prices = list(prices_list)

    # Scale indicator periods to preserve real-time lookback windows
    def sp(period_1m: int, minimum: int = 2) -> int:
        """Scale period for timeframe, with a sane minimum."""
        return max(minimum, round(period_1m / tf))

    try:
        rsi_arr = _rsi_full(prices, sp(14))
        _ml, _sig, macd_hist_arr = _macd_full(prices, sp(12, 3), sp(26, 5), sp(9, 2))
        stoch_k_arr, _ = _stoch_full(prices, sp(14))
        cci_arr = _cci_full(prices, sp(20))
        roc_p = sp(10)
        roc_arr = _roc_full(prices, min(roc_p, n - 1))
        mom_arr = _momentum_full(prices, min(roc_p, n - 1))
        wr_arr = _wr_full(prices, sp(14))
        uo_p1, uo_p2, uo_p3 = sp(7, 2), sp(14), sp(28)
        uo_arr = _uo_full(prices, uo_p1, uo_p2, min(uo_p3, n - 1))
        trix_p = sp(15, 3)
        trix_arr = _trix_full(prices, min(trix_p, max(3, n // 4)))
        cmo_arr = _cmo_full(prices, sp(14))
        atr_arr = _atr_full(prices, sp(14))
        p_arr = np.array(prices, dtype=float)
        atr_norm = np.where(p_arr > 0, atr_arr / p_arr, 0.0)
        ts_p = sp(20)
        ts_arr = _ts_full(prices, min(ts_p, n))
        bb_p = sp(20)
        bb_up, _bb_mid, bb_lo = _bb_full(prices, min(bb_p, n))
        bb_w = bb_up - bb_lo
        bb_pos = np.where((bb_w > 0) & ~np.isnan(bb_up) & ~np.isnan(bb_lo),
                          (p_arr - bb_lo) / bb_w, 0.5)
        mr_p = sp(20)
        mr_arr = _mr_full(prices, min(mr_p, n))
        vr_short, vr_long = sp(5, 2), sp(20)
        vr_arr = _vr_full(prices, vr_short, min(vr_long, n))

        result = np.stack([rsi_arr, macd_hist_arr, stoch_k_arr, cci_arr, roc_arr,
                           mom_arr, wr_arr, uo_arr, trix_arr, cmo_arr,
                           atr_norm, ts_arr, bb_pos, mr_arr, vr_arr])
        defaults = [50., 0., 50., 0., 0., 0., -50., 50., 0., 0., 0., 0., 0.5, 0., 1.]
        for row, d in enumerate(defaults):
            result[row] = np.where(np.isnan(result[row]), d, result[row])
        return result
    except Exception as e:
        logger.warning(f"precompute_indicators error: {e}")
        return None


# =====================================================================
# ML MODEL
# =====================================================================

def extract_features(prices: List[float]) -> Optional[List[float]]:
    if len(prices) < 30 or _compute_indicators is None:
        return None
    try:
        feat_dict = _compute_indicators(prices)
        if not feat_dict:
            return None
        return [float(feat_dict.get(name, 0.0)) for name in _ML_FEATURE_NAMES]
    except Exception:
        return None


def train_model_walkforward(price_series: Dict[str, np.ndarray],
                            indicator_arrays: Dict[str, np.ndarray],
                            end_bar: int) -> Optional[GradientBoostingClassifier]:
    """Train GBM on data up to end_bar using precomputed indicators."""
    X, y = [], []
    for sym, prices in price_series.items():
        max_idx = min(end_bar, len(prices))
        if max_idx < 60:
            continue
        ind = indicator_arrays.get(sym)
        if ind is None:
            continue

        # Sample every 5th bar for training efficiency
        for i in range(30, max_idx - 10, 5):
            future_ret = (prices[i + 10] - prices[i]) / prices[i]
            if abs(future_ret) < 0.002:
                continue
            label = 1 if future_ret > 0.005 else 0
            if i < ind.shape[1]:
                X.append(ind[:, i].tolist())
                y.append(label)

    if len(X) < 200:
        logger.warning(f"Only {len(X)} training samples — not enough")
        return None

    X, y = np.array(X), np.array(y)
    ratio = np.mean(y)

    # Undersample if imbalanced
    up_idx = np.where(y == 1)[0]
    dn_idx = np.where(y == 0)[0]
    if ratio < 0.35 or ratio > 0.65:
        minority = min(len(up_idx), len(dn_idx))
        if minority < 50:
            return None
        if len(up_idx) > len(dn_idx):
            up_idx = np.random.choice(up_idx, size=minority, replace=False)
        else:
            dn_idx = np.random.choice(dn_idx, size=minority, replace=False)
        idx = np.concatenate([up_idx, dn_idx])
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)
    sw = compute_sample_weight('balanced', y_train)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=sw)
    test_acc = model.score(X_test, y_test)
    logger.info(f"  Model trained: {len(y)} samples, test={test_acc:.2%}")
    return model


def ml_predict(model, feat: List[float]) -> Dict[str, float]:
    """Predict using precomputed features."""
    if model is None or feat is None:
        return {"direction": 0.5, "confidence": 0.5}
    try:
        feat_arr = np.nan_to_num(np.array(feat).reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        proba = model.predict_proba(feat_arr)[0]
        up_prob = proba[1]
        confidence = max(proba[0], proba[1])
        return {"direction": up_prob, "confidence": confidence}
    except Exception:
        return {"direction": 0.5, "confidence": 0.5}


def load_production_model():
    """Load the actual production market_model.joblib."""
    import joblib
    model_path = os.path.join(PROJECT_ROOT, "data", "models", "market_model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info(f"Loaded production model from {model_path}")
        return model
    logger.error(f"Production model not found at {model_path}")
    return None


# =====================================================================
# TECHNICAL HELPERS
# =====================================================================

def calc_rsi(prices: np.ndarray, idx: int, period: int = 14) -> float:
    if idx < period + 1:
        return 50.0
    p = prices[max(0, idx - period):idx + 1]
    deltas = np.diff(p)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gains)
    avg_l = np.mean(losses)
    if avg_l == 0:
        return 100.0
    return 100 - (100 / (1 + avg_g / avg_l))


def calc_trend(prices: np.ndarray, idx: int, lookback: int = 200) -> Tuple[str, float]:
    """Trend on ~10-min aggregated candles, 20-period lookback.
    Aggregates into 10-min bars from whatever timeframe is loaded."""
    lb = max(20, lookback // _TF)  # scale lookback for timeframe
    start = max(0, idx - lb)
    if idx - start < 20:
        return "SIDE", 0.0
    # Aggregate into ~10-min bars: bar_size = 10/TF (e.g., 1→10, 5→2, 10→1)
    window = prices[start:idx + 1]
    n = len(window)
    bar_size = max(1, 10 // _TF)
    n_bars = n // bar_size
    if n_bars < 5:
        return "SIDE", 0.0
    if bar_size > 1:
        agg_closes = np.array([window[(i + 1) * bar_size - 1] for i in range(n_bars)])
    else:
        agg_closes = window  # already 10-min bars, no aggregation needed
    # Use last 20 aggregated bars
    recent = agg_closes[-20:]
    if len(recent) < 5:
        return "SIDE", 0.0
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0] / np.mean(recent)
    if slope > MIN_TREND_SLOPE:
        return "UP", slope
    elif slope < -MIN_TREND_SLOPE:
        return "DOWN", slope
    return "SIDE", slope


def calc_volatility(prices: np.ndarray, idx: int, lookback: int = 20) -> float:
    start = max(0, idx - lookback)
    if idx - start < 5:
        return 0.01
    p = prices[start:idx + 1]
    returns = np.diff(p) / p[:-1]
    return float(np.std(returns))


def calc_correlation(p1: np.ndarray, p2: np.ndarray, idx: int, lookback: int = 60) -> float:
    """Rolling correlation between two price series."""
    start = max(0, idx - lookback)
    s1 = p1[start:idx + 1]
    s2 = p2[start:min(idx + 1, len(p2))]
    min_len = min(len(s1), len(s2))
    if min_len < 10:
        return 0.0
    s1, s2 = s1[-min_len:], s2[-min_len:]
    r1 = np.diff(s1) / s1[:-1]
    r2 = np.diff(s2) / s2[:-1]
    if np.std(r1) == 0 or np.std(r2) == 0:
        return 0.0
    return float(np.corrcoef(r1, r2)[0, 1])


def kelly_size(balance: float, confidence: float, volatility: float, n_pos: int, is_futures: bool = False) -> float:
    kelly = max(0.01, (2 * confidence) - 1)
    vol_adj = 1.0 / (1 + volatility * 10)
    size = balance * MAX_POSITION_PCT * kelly * vol_adj
    pos_factor = max(0.3, 1 - (n_pos * 0.05))  # gentler decay with more positions
    size *= pos_factor
    # Futures: leverage means less capital needed for same exposure
    if is_futures:
        size = size / FUTURES_LEVERAGE
    size = max(balance * MIN_POSITION_PCT, min(size, balance * MAX_POSITION_PCT))
    return size


# =====================================================================
# POSITION + TRADE
# =====================================================================

@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    size_usd: float
    confidence: float
    entry_bar: int
    peak_pnl_pct: float = 0.0
    stale_bars: int = 0

    def pnl_pct(self, price: float) -> float:
        if self.direction == "LONG":
            return ((price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - price) / self.entry_price) * 100


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


# =====================================================================
# EXIT LOGIC — mirrors engine exactly
# =====================================================================

def _is_futures(sym: str) -> bool:
    """Check if symbol is a futures contract."""
    return sym.startswith("PI_")


def check_exit(pos: Position, price: float, bar: int) -> Optional[str]:
    hold_bars = bar - pos.entry_bar
    pnl = pos.pnl_pct(price)
    is_fut = _is_futures(pos.symbol)

    # Futures: P&L is leveraged
    if is_fut:
        pnl_effective = pnl * FUTURES_LEVERAGE
    else:
        pnl_effective = pnl

    # Track peak P&L (on effective/leveraged basis)
    if pnl_effective > pos.peak_pnl_pct:
        pos.peak_pnl_pct = pnl_effective
        pos.stale_bars = 0
    else:
        pos.stale_bars += 1

    # 1. Stop loss (futures use wider SL)
    sl = FUTURES_STOP_LOSS_PCT if is_fut else STOP_LOSS_PCT
    if pnl_effective <= sl:
        return "STOP_LOSS"

    # 2. Take profit
    if pnl_effective >= TAKE_PROFIT_PCT:
        return "TAKE_PROFIT"

    # 3. Trailing stop — drawdown from peak while in profit
    if pnl_effective > 0.8 and (pos.peak_pnl_pct - pnl_effective) >= TRAILING_STOP_PCT:  # v6: trigger at 0.8%
        return "TRAILING_STOP"

    # 4. Stale profit decay — peaked above arm threshold, now decaying
    stale_min = FUTURES_STALE_MIN_BARS if is_fut else STALE_MIN_BARS
    if (pos.peak_pnl_pct >= STALE_ARM_PCT and
        pnl_effective > 0 and
        (pos.peak_pnl_pct - pnl_effective) >= STALE_DECAY_PCT and
        pos.stale_bars >= stale_min):
        return "STALE_PROFIT_DECAY"

    # 5. Max hold flat (futures have shorter hold limit)
    max_hold = FUTURES_MAX_HOLD_BARS if is_fut else MAX_HOLD_BARS_SPOT
    if hold_bars >= max_hold and abs(pnl_effective) < MAX_HOLD_FLAT_BAND:
        return "MAX_HOLD_FLAT"

    # 6. Max hold forced
    if hold_bars >= MAX_HOLD_FORCED_BARS:
        return "MAX_HOLD_FORCED"

    return None


def close_trade(pos: Position, price: float, bar: int, reason: str) -> Trade:
    pnl_pct = pos.pnl_pct(price)
    is_fut = _is_futures(pos.symbol)
    # Futures: P&L is multiplied by leverage
    if is_fut:
        pnl_pct *= FUTURES_LEVERAGE
    # Apply slippage + fees (futures have higher slippage)
    slippage = (FUTURES_SLIPPAGE_BPS if is_fut else SLIPPAGE_BPS) / 10000
    fee = FEE_PCT
    cost_pct = (slippage + fee) * 2  # entry + exit
    net_pnl_pct = pnl_pct - (cost_pct * 100)
    pnl_usd = pos.size_usd * (net_pnl_pct / 100)
    return Trade(
        symbol=pos.symbol, direction=pos.direction,
        entry_price=pos.entry_price, exit_price=price,
        size_usd=pos.size_usd, pnl_usd=pnl_usd, pnl_pct=net_pnl_pct,
        exit_reason=reason, hold_bars=bar - pos.entry_bar,
    )


# =====================================================================
# SIGNAL GENERATION — mirrors engine exactly
# =====================================================================

def generate_signal(model, sym: str, feat: List[float],
                    prices: np.ndarray, bar: int,
                    direction_mode: str = "both") -> Optional[Tuple[str, float]]:
    """Returns (direction, confidence) or None."""
    pred = ml_predict(model, feat)
    ml_conf = pred["confidence"]
    ml_dir = pred["direction"]

    if ml_conf < MIN_ML_CONFIDENCE:
        _diag["low_conf"] += 1
        return None

    rsi = calc_rsi(prices, bar)
    trend, _ = calc_trend(prices, bar)

    # Volatility filter — skip dead/flat markets AND extreme volatility
    vol_lb = max(10, 60 // _TF)  # ~1-hour realized vol
    vol = calc_volatility(prices, bar, lookback=vol_lb)
    if vol < MIN_VOLATILITY or vol > MAX_VOLATILITY:
        _diag["side_filter"] += 1  # reuse counter for vol+side filtering
        return None

    # v11b: Direction-aware RSI momentum gate
    # For LONG: RSI rising = good momentum. For SHORT: RSI falling = good momentum.
    rsi_lb = max(1, 5 // _TF)  # 5-min momentum lookback
    rsi_gate_min = max(5, 35 // _TF)
    rsi_rising = True  # default: no gate
    if bar >= rsi_gate_min:
        rsi_now = calc_rsi(prices, bar)
        rsi_prev = calc_rsi(prices, bar - rsi_lb)
        rsi_rising = rsi_now >= rsi_prev
    # Gate is applied per-direction below, not as an early return

    # SIDE market filter — only trade in clear trend direction
    if SIDE_MARKET_FILTER and trend == "SIDE" and ml_conf < SIDE_ML_OVERRIDE:
        _diag["side_filter"] += 1
        return None

    # Momentum filter: require price above short MA
    ma_len = max(2, 10 // _TF)  # ~10-min MA
    ma_gate = max(5, 30 // _TF)
    if bar >= ma_gate:
        ma_short = float(np.mean(prices[bar - ma_len:bar + 1]))
        current = float(prices[bar])
    else:
        ma_short = float(prices[bar])
        current = ma_short

    direction = None
    confidence = ml_conf

    if ml_dir > MIN_ENSEMBLE_SCORE:
        # LONG candidate — require rising RSI momentum
        if not rsi_rising:
            _diag["rsi_gate"] += 1
        elif trend == "UP" and rsi < MAX_RSI_LONG and current >= ma_short:
            direction = "LONG"
            confidence = ml_conf
        elif trend == "SIDE" and ml_conf >= SIDE_ML_OVERRIDE and rsi < MAX_RSI_LONG:
            direction = "LONG"
            confidence = ml_conf * 0.90
        elif trend == "DOWN" and ml_conf >= COUNTER_TREND_OVERRIDE:
            direction = "LONG"
            confidence = ml_conf * 0.80  # heavy penalty for counter-trend
        else:
            _diag["rsi_gate"] += 1
    elif ml_dir < (1 - MIN_ENSEMBLE_SCORE):
        # SHORT candidate — require FALLING RSI momentum (opposite of long)
        if rsi_rising:
            _diag["rsi_gate"] += 1
        elif trend == "DOWN" and rsi > MIN_RSI_SHORT and current <= ma_short:
            direction = "SHORT"
            confidence = ml_conf
        elif trend == "DOWN" and rsi > MIN_RSI_SHORT:
            # v11b: DOWN trend is enough for shorts, don't require price < MA
            direction = "SHORT"
            confidence = ml_conf * 0.95
        elif trend == "SIDE" and ml_conf >= SIDE_ML_OVERRIDE and rsi > MIN_RSI_SHORT:
            direction = "SHORT"
            confidence = ml_conf * 0.90
        elif trend == "UP" and ml_conf >= COUNTER_TREND_OVERRIDE:
            direction = "SHORT"
            confidence = ml_conf * 0.80  # heavy penalty for counter-trend
        else:
            _diag["rsi_gate"] += 1
    else:
        _diag["no_direction"] += 1

    if direction == "LONG" and direction_mode == "short_only":
        return None
    if direction == "SHORT" and direction_mode == "long_only":
        return None
    if direction is None:
        return None

    _diag["signals_gen"] += 1
    return direction, confidence


# =====================================================================
# MAIN BACKTEST ENGINE
# =====================================================================

def run_backtest(
    all_prices: Dict[str, np.ndarray],
    indicator_arrays: Dict[str, np.ndarray],
    starting_balance: float = 2500.0,
    direction_mode: str = "both",
    use_production_model: bool = False,
    verbose: bool = False,
) -> Tuple[List[Trade], float, List[float]]:
    """
    Walk-forward backtest on 6 months of 1-minute data.
    Returns (trades, final_balance, equity_curve).
    """
    symbols = list(all_prices.keys())
    sym_lengths = {s: len(all_prices[s]) for s in symbols}
    max_bars = max(sym_lengths.values())

    bars_per_day = 1440 // _TF
    logger.info(f"Backtesting {len(symbols)} symbols over {max_bars:,} bars (~{max_bars/bars_per_day:.0f} days)")

    # Model setup
    if use_production_model:
        model = load_production_model()
        if model is None:
            return [], starting_balance, []
        split_idx = 30  # start trading immediately with production model
    else:
        # Walk-forward: train on first 40%, trade on 60%
        split_idx = int(max_bars * 0.4)
        logger.info(f"Training on first {split_idx:,} bars (~{split_idx/bars_per_day:.0f} days)...")
        model = train_model_walkforward(all_prices, indicator_arrays, split_idx)
        if model is None:
            logger.error("Initial training failed")
            return [], starting_balance, []

    retrain_interval = max(500, 20000 // _TF)  # retrain every ~14 days (scaled)
    last_retrain = split_idx

    # State
    balance = starting_balance
    positions: Dict[str, Position] = {}
    all_trades: List[Trade] = []
    equity_curve: List[float] = []

    # Circuit breaker state
    consecutive_losses = 0
    daily_start_balance = balance
    daily_bar_start = split_idx
    peak_balance = balance

    # Direction pause state
    recent_results: Dict[str, List[bool]] = defaultdict(list)  # direction -> [win/loss]
    last_opened_bar = split_idx  # track last trade opening for deadlock prevention

    progress_pct = 0
    total_bars = max_bars - split_idx

    for bar in range(split_idx, max_bars):
        # Progress tracking
        done = bar - split_idx
        new_pct = int(done / total_bars * 100) if total_bars > 0 else 100
        if new_pct >= progress_pct + 5:
            progress_pct = new_pct
            unrealized = sum(
                p.pnl_pct(all_prices[p.symbol][min(bar, len(all_prices[p.symbol])-1)]) * p.size_usd / 100
                for p in positions.values()
                if bar < len(all_prices[p.symbol])
            )
            logger.info(f"  {progress_pct:3d}% | bar {bar:,}/{max_bars:,} | "
                        f"trades={len(all_trades)} | bal=${balance:.2f} | "
                        f"unrealized=${unrealized:+.2f} | pos={len(positions)}")
            logger.info(f"       diag: low_conf={_diag['low_conf']} side_filter={_diag['side_filter']} "
                        f"no_dir={_diag['no_direction']} rsi={_diag['rsi_gate']} "
                        f"corr={_diag['corr_gate']} dir_pause={_diag['dir_pause']} signals={_diag['signals_gen']} "
                        f"opened={_diag['trades_opened']} consec_loss={consecutive_losses}")

        # Daily reset (every 1440/tf bars)
        if (bar - daily_bar_start) >= (1440 // _TF):
            daily_start_balance = balance
            daily_bar_start = bar
            consecutive_losses = 0

        # Equity curve (sample every ~60min worth of bars)
        eq_interval = max(1, 60 // _TF)
        if done % eq_interval == 0:
            unrealized = sum(
                p.pnl_pct(all_prices[p.symbol][min(bar, len(all_prices[p.symbol])-1)]) * p.size_usd / 100
                for p in positions.values()
                if bar < len(all_prices[p.symbol])
            )
            equity_curve.append(balance + unrealized)

        # ── Check exits every bar (RISK cycle) ──
        closed = []
        for sym, pos in list(positions.items()):
            if bar >= sym_lengths.get(sym, 0):
                price = all_prices[sym][-1]
                trade = close_trade(pos, price, bar, "END_OF_DATA")
                all_trades.append(trade)
                balance += trade.pnl_usd
                closed.append(sym)
                continue

            price = float(all_prices[sym][bar])
            reason = check_exit(pos, price, bar)
            if reason:
                trade = close_trade(pos, price, bar, reason)
                all_trades.append(trade)
                balance += trade.pnl_usd
                closed.append(sym)

                # Track for circuit breaker
                if trade.pnl_usd > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                if trade.pnl_usd > 0:
                    recent_results[trade.direction].append(True)
                else:
                    recent_results[trade.direction].append(False)
                # Keep last 20
                for d in recent_results:
                    recent_results[d] = recent_results[d][-20:]

                if verbose:
                    logger.info(f"  CLOSE {trade.direction} {sym} @ {price:.4f} "
                                f"P&L ${trade.pnl_usd:+.2f} ({trade.pnl_pct:+.2f}%) [{reason}]")

        for sym in closed:
            del positions[sym]

        # Circuit breakers
        if consecutive_losses >= 5:
            _diag["circuit"] += 1
            continue
        daily_loss = (balance - daily_start_balance) / daily_start_balance * 100
        if daily_loss <= -4.0:
            continue
        peak_balance = max(peak_balance, balance)
        drawdown_pct = (peak_balance - balance) / peak_balance * 100
        if drawdown_pct >= 8.0:
            continue

        # ── Retrain periodically (walk-forward) ──
        if not use_production_model and (bar - last_retrain) >= retrain_interval:
            logger.info(f"  Retraining at bar {bar:,}...")
            new_model = train_model_walkforward(all_prices, indicator_arrays, bar)
            if new_model is not None:
                model = new_model
            last_retrain = bar

        # ── Generate signals every SIGNAL_INTERVAL bars (TRADE cycle) ──
        if done % SIGNAL_INTERVAL != 0:
            continue

        active_symbols = [s for s in symbols if bar < sym_lengths.get(s, 0) and s not in positions]

        # Enforce max positions cap
        if len(positions) >= MAX_POSITIONS:
            continue

        for sym in active_symbols:
            ind = indicator_arrays.get(sym)
            if ind is None or bar >= ind.shape[1]:
                continue

            feat = ind[:, bar].tolist()
            result = generate_signal(model, sym, feat, all_prices[sym], bar, direction_mode)
            if result is None:
                continue

            direction, confidence = result

            # Direction pause — DISABLED (was causing deadlocks, filtering done by ML + trend)
            # dir_results = recent_results.get(direction, [])
            # if len(dir_results) >= 10 and sum(dir_results)/len(dir_results) < 0.35:
            #     _diag["dir_pause"] += 1
            #     continue

            # Correlation gate — skip if too correlated with existing positions
            corr_ok = True
            for held_sym, held_pos in positions.items():
                if held_sym not in all_prices or bar >= len(all_prices[held_sym]):
                    continue
                corr = abs(calc_correlation(all_prices[sym], all_prices[held_sym], bar))
                if corr > MAX_CORRELATION:
                    corr_ok = False
                    break
            if not corr_ok:
                _diag["corr_gate"] += 1
                continue

            # Position sizing
            vol = calc_volatility(all_prices[sym], bar)
            is_fut = _is_futures(sym)
            size = kelly_size(balance, confidence, vol, len(positions), is_futures=is_fut)
            if size < 10 or size > balance * 0.5:
                continue

            price = float(all_prices[sym][bar])
            positions[sym] = Position(
                symbol=sym, direction=direction,
                entry_price=price, size_usd=size,
                confidence=confidence, entry_bar=bar,
            )
            _diag["trades_opened"] += 1
            last_opened_bar = bar
            if verbose:
                rsi = calc_rsi(all_prices[sym], bar)
                trend, _ = calc_trend(all_prices[sym], bar)
                logger.info(f"  OPEN {direction} {sym} @ {price:.4f} ${size:.2f} "
                            f"conf={confidence:.2f} RSI={rsi:.0f} trend={trend}")

    # Close remaining positions
    for sym, pos in positions.items():
        price = float(all_prices[sym][-1])
        trade = close_trade(pos, price, max_bars - 1, "END_OF_BACKTEST")
        all_trades.append(trade)
        balance += trade.pnl_usd

    final_balance = starting_balance + sum(t.pnl_usd for t in all_trades)

    # Print signal diagnostics
    logger.info(f"\n  SIGNAL DIAGNOSTICS:")
    logger.info(f"    Low confidence (<{MIN_ML_CONFIDENCE}):  {_diag['low_conf']:,}")
    logger.info(f"    SIDE market filter:             {_diag['side_filter']:,}")
    logger.info(f"    No direction (0.38-0.62):       {_diag['no_direction']:,}")
    logger.info(f"    RSI gate:                       {_diag['rsi_gate']:,}")
    logger.info(f"    Correlation gate:               {_diag['corr_gate']:,}")
    logger.info(f"    Direction pause:                {_diag['dir_pause']:,}")
    logger.info(f"    Circuit breaker bars:           {_diag['circuit']:,}")
    logger.info(f"    Signals generated:              {_diag['signals_gen']:,}")
    logger.info(f"    Trades opened:                  {_diag['trades_opened']:,}")

    return all_trades, final_balance, equity_curve


# =====================================================================
# REPORTING
# =====================================================================

def print_report(trades: List[Trade], starting_bal: float, equity_curve: List[float],
                 label: str = "", test_days: float = 0):
    total_pnl = sum(t.pnl_usd for t in trades)
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    avg_w = np.mean([t.pnl_usd for t in wins]) if wins else 0
    avg_l = np.mean([t.pnl_usd for t in losses]) if losses else 0
    avg_hold = np.mean([t.hold_bars for t in trades]) / 60 if trades else 0  # hours

    # Profit factor
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    max_dd = 0
    max_dd_pct = 0
    if equity_curve:
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        max_dd = float(np.max(dd))
        max_dd_pct = float(np.max(dd / peak * 100))

    # Trade-based drawdown
    cumulative = 0
    peak_cum = 0
    max_trade_dd = 0
    for t in trades:
        cumulative += t.pnl_usd
        peak_cum = max(peak_cum, cumulative)
        dd = peak_cum - cumulative
        max_trade_dd = max(max_trade_dd, dd)

    ending = starting_bal + total_pnl
    ret_pct = (total_pnl / starting_bal) * 100

    print(f"\n{'='*65}")
    print(f"  BACKTEST RESULTS — {label}")
    print(f"{'='*65}")
    print(f"  Period:          ~{test_days:.0f} days of out-of-sample trading")
    print(f"  Starting:        ${starting_bal:,.2f}")
    print(f"  Ending:          ${ending:,.2f}")
    print(f"  Total P&L:       ${total_pnl:+,.2f} ({ret_pct:+.2f}%)")
    print(f"  Total Trades:    {len(trades)}")
    print(f"  Win Rate:        {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg Win:         ${avg_w:+.2f}")
    print(f"  Avg Loss:        ${avg_l:+.2f}")
    print(f"  Avg Hold:        {avg_hold:.1f} hours")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"  Max Drawdown:    ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
    if test_days > 0:
        daily_ret = ret_pct / test_days
        annualized = daily_ret * 365
        print(f"  Daily Avg:       {daily_ret:+.3f}%")
        print(f"  Annualized:      {annualized:+.1f}%")

    # Direction breakdown
    longs = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]
    print(f"\n  DIRECTION BREAKDOWN:")
    if longs:
        lw = len([t for t in longs if t.pnl_usd > 0])
        lp = sum(t.pnl_usd for t in longs)
        print(f"    LONG:  {len(longs):4d} trades  {lw/len(longs)*100:5.1f}% WR  P&L ${lp:+,.2f}")
    if shorts:
        sw = len([t for t in shorts if t.pnl_usd > 0])
        sp_pnl = sum(t.pnl_usd for t in shorts)
        print(f"    SHORT: {len(shorts):4d} trades  {sw/len(shorts)*100:5.1f}% WR  P&L ${sp_pnl:+,.2f}")

    # Spot vs Futures breakdown
    spot_trades = [t for t in trades if not _is_futures(t.symbol)]
    fut_trades = [t for t in trades if _is_futures(t.symbol)]
    if spot_trades and fut_trades:
        print(f"\n  MARKET BREAKDOWN:")
        sw = len([t for t in spot_trades if t.pnl_usd > 0])
        sp_pnl = sum(t.pnl_usd for t in spot_trades)
        sel = f"{sw/len(spot_trades)*100:5.1f}" if spot_trades else "  N/A"
        print(f"    SPOT:    {len(spot_trades):4d} trades  {sel}% WR  P&L ${sp_pnl:+,.2f}")
        fw = len([t for t in fut_trades if t.pnl_usd > 0])
        fp = sum(t.pnl_usd for t in fut_trades)
        fel = f"{fw/len(fut_trades)*100:5.1f}" if fut_trades else "  N/A"
        print(f"    FUTURES: {len(fut_trades):4d} trades  {fel}% WR  P&L ${fp:+,.2f}  ({FUTURES_LEVERAGE}x leverage)")

    # Exit reasons
    reasons = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        reasons[t.exit_reason]["count"] += 1
        reasons[t.exit_reason]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            reasons[t.exit_reason]["wins"] += 1

    print(f"\n  EXIT REASONS:")
    for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        wr_r = d["wins"] / d["count"] * 100 if d["count"] else 0
        print(f"    {r:22s}  {d['count']:5d} trades  {wr_r:5.1f}% WR  P&L ${d['pnl']:+,.2f}")

    # Per-symbol
    sym_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        sym_stats[t.symbol]["count"] += 1
        sym_stats[t.symbol]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            sym_stats[t.symbol]["wins"] += 1

    print(f"\n  SYMBOL BREAKDOWN (by P&L):")
    for sym, d in sorted(sym_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr_s = d["wins"] / d["count"] * 100 if d["count"] else 0
        print(f"    {sym:15s}  {d['count']:5d} trades  {wr_s:5.1f}% WR  P&L ${d['pnl']:+,.2f}")

    # Monthly breakdown
    print(f"\n  MONTHLY BREAKDOWN:")
    monthly = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    bar_to_month = {}
    for t in trades:
        # Approximate month from bar index
        approx_days = t.hold_bars / 1440
        monthly_key = f"Month {(sum(tt.hold_bars for tt in trades[:trades.index(t)])) // (30*1440) + 1}"
        # Simpler: just group by trade sequence
    # Group trades into ~30-day buckets
    if trades:
        trades_per_month = max(1, len(trades) // 6)
        for i, t in enumerate(trades):
            month_num = i // trades_per_month + 1
            key = f"Period {month_num}"
            monthly[key]["count"] += 1
            monthly[key]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                monthly[key]["wins"] += 1
        for m, d in sorted(monthly.items()):
            wr_m = d["wins"] / d["count"] * 100 if d["count"] else 0
            print(f"    {m:15s}  {d['count']:5d} trades  {wr_m:5.1f}% WR  P&L ${d['pnl']:+,.2f}")

    print(f"{'='*65}\n")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="6-Month Walk-Forward Backtest")
    parser.add_argument("--balance", type=float, default=5000.0, help="Starting balance (default 5000)")
    parser.add_argument("--direction", type=str, default="both", choices=["both", "short_only", "long_only"])
    parser.add_argument("--use-production-model", action="store_true",
                        help="Use the production market_model.joblib instead of walk-forward training")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--timeframe", type=int, default=1, choices=[1, 5, 10, 15],
                        help="Candle timeframe in minutes (1, 5, 10, or 15). Aggregates 1-min data.")
    parser.add_argument("--position-pct", type=float, default=None,
                        help="Max position size as %% of balance (e.g. 25 for 25%%). Overrides Kelly sizing.")
    parser.add_argument("--futures", action="store_true", default=True,
                        help="Include futures symbols (PI_* mapped to spot data with leverage). Default: ON.")
    parser.add_argument("--no-futures", dest="futures", action="store_false",
                        help="Disable futures symbols.")
    parser.add_argument("--all-symbols", action="store_true", default=False,
                        help="Use ALL 17 spot symbols instead of just the v10 winners.")
    args = parser.parse_args()

    # ── Override position sizing if requested ──
    global MAX_POSITION_PCT, MIN_POSITION_PCT
    if args.position_pct is not None:
        MAX_POSITION_PCT = args.position_pct / 100.0
        MIN_POSITION_PCT = MAX_POSITION_PCT * 0.5  # min = half of max

    # ── Expand symbols if requested ──
    global SPOT_SYMBOLS
    if args.all_symbols:
        SPOT_SYMBOLS = [
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
            "DOGE-USD", "LINK-USD", "XRP-USD", "LTC-USD", "UNI-USD",
            "XLM-USD", "BCH-USD", "DOT-USD", "MATIC-USD", "ATOM-USD",
            "NEAR-USD", "AAVE-USD",
        ]

    # ── Scale bar-count parameters for timeframe ──
    global _TF, MAX_HOLD_BARS_SPOT, MAX_HOLD_FORCED_BARS, STALE_MIN_BARS
    global SIGNAL_INTERVAL, MIN_VOLATILITY, MAX_VOLATILITY
    tf = args.timeframe
    _TF = tf
    if tf > 1:
        MAX_HOLD_BARS_SPOT = max(6, 180 // tf)       # 3h hold in bars
        MAX_HOLD_FORCED_BARS = max(12, 360 // tf)     # 6h hard cap in bars
        STALE_MIN_BARS = max(3, 25 // tf)             # stale detection window
        SIGNAL_INTERVAL = max(1, 5 // tf)             # signal check frequency
        # Volatility scales with sqrt(tf) — longer bars have higher per-bar vol
        vol_scale = math.sqrt(tf)
        MIN_VOLATILITY = 0.002 * vol_scale
        MAX_VOLATILITY = 0.006 * vol_scale

    tf_label = f"{tf}min" if tf > 1 else "1min"
    fut_label = "SPOT+FUTURES" if args.futures else "SPOT ONLY"
    print(f"\n{'='*65}")
    print(f"  6-MONTH WALK-FORWARD BACKTEST  [{tf_label} candles]")
    print(f"  Balance: ${args.balance:,.2f}  |  Direction: {args.direction}  |  {fut_label}")
    print(f"  Position sizing: {MAX_POSITION_PCT*100:.0f}% max  |  Max positions: {MAX_POSITIONS}")
    print(f"  Model: {'PRODUCTION' if args.use_production_model else 'WALK-FORWARD (train 40% / test 60%)'}")
    print(f"  Fee: {FEE_PCT*100:.1f}% + {SLIPPAGE_BPS}bps slippage per side")
    if args.futures:
        print(f"  Futures: {FUTURES_LEVERAGE}x leverage  |  SL: {FUTURES_STOP_LOSS_PCT}%  |  Slippage: {FUTURES_SLIPPAGE_BPS}bps")
    if tf > 1:
        print(f"  Timeframe: {tf}min  |  Hold: {MAX_HOLD_BARS_SPOT}b/{MAX_HOLD_FORCED_BARS}b  |  Stale: {STALE_MIN_BARS}b  |  Vol: {MIN_VOLATILITY:.4f}-{MAX_VOLATILITY:.4f}")
    print(f"{'='*65}\n")

    # Load data
    logger.info(f"Loading 6-month {tf_label} candle data...")
    all_prices = load_6mo_data(tf=tf, include_futures=args.futures)
    if not all_prices:
        logger.error("No data loaded")
        return

    total_bars = sum(len(v) for v in all_prices.values())
    logger.info(f"Total: {total_bars:,} bars across {len(all_prices)} symbols\n")

    # Precompute all indicators
    logger.info("Precomputing indicator arrays (this may take a minute)...")
    t0 = time.time()
    indicator_arrays = {}
    for sym in all_prices:
        indicator_arrays[sym] = precompute_indicators(list(all_prices[sym]), tf=tf)
        status = "OK" if indicator_arrays[sym] is not None else "FAIL"
        bars = indicator_arrays[sym].shape[1] if indicator_arrays[sym] is not None else 0
        logger.info(f"  {sym}: {status} ({bars:,} bars)")
    if tf > 1:
        logger.info(f"  (indicator periods scaled by 1/{tf} to preserve real-time windows)")
    logger.info(f"Indicators precomputed in {time.time() - t0:.1f}s\n")

    # Run backtest
    logger.info("Starting walk-forward backtest...")
    t0 = time.time()
    trades, final_bal, equity = run_backtest(
        all_prices, indicator_arrays,
        starting_balance=args.balance,
        direction_mode=args.direction,
        use_production_model=args.use_production_model,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0
    logger.info(f"Backtest completed in {elapsed:.1f}s\n")

    if trades:
        max_bars = max(len(v) for v in all_prices.values())
        split_idx = 30 if args.use_production_model else int(max_bars * 0.4)
        test_bars = max_bars - split_idx
        bars_per_day = 1440 // tf
        test_days = test_bars / bars_per_day

        model_label = "PRODUCTION MODEL" if args.use_production_model else "WALK-FORWARD"
        print_report(trades, args.balance, equity,
                     f"{model_label} | {args.direction.upper()}", test_days)
    else:
        print("\nNo trades generated!")

    # Save trade log
    tf_suffix = f"_{tf}m" if tf > 1 else ""
    log_path = os.path.join(PROJECT_ROOT, "data", "backtest", f"backtest_6mo{tf_suffix}_trades.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["symbol", "direction", "entry_price", "exit_price", "size_usd",
                     "pnl_usd", "pnl_pct", "exit_reason", "hold_bars"])
        for t in trades:
            w.writerow([t.symbol, t.direction, f"{t.entry_price:.6f}", f"{t.exit_price:.6f}",
                        f"{t.size_usd:.2f}", f"{t.pnl_usd:.2f}", f"{t.pnl_pct:.2f}",
                        t.exit_reason, t.hold_bars])
    logger.info(f"Trade log saved to {log_path}")


if __name__ == "__main__":
    main()
