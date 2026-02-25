"""Train the live bot's ML model using 30-day 1-MINUTE historical data.

VECTORIZED approach: computes all 15 indicator arrays ONCE per symbol,
then indexes into them — O(n) instead of O(n²). Finishes in seconds.

The resulting model is saved to the bot's model path so the live engine
picks it up automatically.

Usage:
    python tools/train_from_historical.py
"""

import os
import sys
import csv
import time
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from joblib import dump

# Import INDIVIDUAL indicator functions for vectorized computation
try:
    from cryptotrades.utils.technical_indicators import (
        rsi, macd, stochastic, cci, rate_of_change, momentum, williams_r,
        ultimate_oscillator, trix, chande_momentum_oscillator,
        atr_approx, trend_strength, bollinger_bands, mean_reversion_score,
        volatility_ratio,
    )
except ImportError as e:
    print(f"ERROR: Could not import indicators: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_historical")

# ── Feature names (must match trading_engine.py exactly) ──
_ML_FEATURE_NAMES = [
    "rsi_14", "macd_histogram", "stoch_k", "cci_20", "roc_10",
    "momentum_10", "williams_r", "ultimate_osc", "trix_15", "cmo_14",
    "atr_14", "trend_strength", "bb_position", "mean_reversion", "vol_ratio",
]

CACHE_CSV = os.path.join(os.path.dirname(__file__), "backtests", "data", "minute_prices.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "trading_model.joblib")

# How many bars in the future to look for labelling
LOOKAHEAD = 10
# Minimum absolute return to count as non-flat
MIN_MOVE = 0.002
# Threshold for UP label
UP_THRESHOLD = 0.005


def load_hourly_cache() -> Dict[str, List[float]]:
    """Load close prices from backtest cache CSV (1-minute candles)."""
    if not os.path.exists(CACHE_CSV):
        logger.error(f"No cached data at {CACHE_CSV}")
        sys.exit(1)

    data: Dict[str, List[float]] = defaultdict(list)
    with open(CACHE_CSV) as f:
        for row in csv.DictReader(f):
            data[row["symbol"]].append(float(row["close"]))

    logger.info(f"Loaded {len(data)} symbols, {sum(len(v) for v in data.values())} 1-min candles")
    for sym in sorted(data.keys()):
        logger.info(f"  {sym}: {len(data[sym])} bars ({len(data[sym])/1440:.1f} days)")
    return dict(data)


def compute_indicator_arrays(prices: List[float]) -> Optional[np.ndarray]:
    """Compute all 15 indicators as full arrays over the price series.

    Returns a (15, N) numpy array where each row is one indicator's full series,
    or None if prices too short.
    """
    n = len(prices)
    if n < 30:
        return None

    try:
        # 1. RSI (14)
        rsi_arr = rsi(prices, 14)

        # 2. MACD histogram
        _macd_line, _signal, macd_hist_arr = macd(prices)

        # 3. Stochastic %K
        stoch_k_arr, _stoch_d = stochastic(prices, 14)

        # 4. CCI (20)
        cci_arr = cci(prices, 20)

        # 5. ROC (10)
        roc_arr = rate_of_change(prices, min(10, n - 1))

        # 6. Momentum (10)
        mom_arr = momentum(prices, min(10, n - 1))

        # 7. Williams %R (14)
        wr_arr = williams_r(prices, 14)

        # 8. Ultimate Oscillator
        uo_arr = ultimate_oscillator(prices, 7, 14, min(28, n - 1))

        # 9. TRIX (15)
        trix_arr = trix(prices, min(15, max(3, n // 4)))

        # 10. CMO (14)
        cmo_arr = chande_momentum_oscillator(prices, 14)

        # 11. ATR (14) normalized by price
        atr_arr = atr_approx(prices, 14)
        price_arr = np.array(prices, dtype=float)
        atr_norm_arr = np.where(price_arr > 0, atr_arr / price_arr, 0.0)

        # 12. Trend Strength (20)
        ts_arr = trend_strength(prices, min(20, n))

        # 13. Bollinger Band position (20)
        bb_upper, _bb_mid, bb_lower = bollinger_bands(prices, min(20, n))
        bb_width = bb_upper - bb_lower
        bb_pos_arr = np.where(
            (bb_width > 0) & ~np.isnan(bb_upper) & ~np.isnan(bb_lower),
            (price_arr - bb_lower) / bb_width,
            0.5,
        )

        # 14. Mean Reversion Z-score (20)
        mr_arr = mean_reversion_score(prices, min(20, n))

        # 15. Volatility Ratio (5 vs 20)
        vr_arr = volatility_ratio(prices, 5, min(20, n))

        # Stack into (15, N) array
        result = np.stack([
            rsi_arr, macd_hist_arr, stoch_k_arr, cci_arr, roc_arr,
            mom_arr, wr_arr, uo_arr, trix_arr, cmo_arr,
            atr_norm_arr, ts_arr, bb_pos_arr, mr_arr, vr_arr,
        ])

        # Replace NaN with defaults
        defaults = [50.0, 0.0, 50.0, 0.0, 0.0, 0.0, -50.0, 50.0, 0.0, 0.0,
                     0.0, 0.0, 0.5, 0.0, 1.0]
        for row_idx, default in enumerate(defaults):
            result[row_idx] = np.where(np.isnan(result[row_idx]), default, result[row_idx])

        return result  # shape (15, n)
    except Exception as e:
        logger.warning(f"Indicator computation failed: {e}")
        return None


def train_model(price_series: Dict[str, List[float]]):
    """
    Train a GBM using the same logic as trading_engine.py MLModel._train(),
    but leveraging 180 days of hourly data instead of ~8 hours of 1-min data.

    VECTORIZED: computes all indicators ONCE per symbol, then builds feature
    matrix by indexing into the arrays. O(n) per symbol, finishes in seconds.
    """
    logger.info("=" * 60)
    logger.info("TRAINING MODEL FROM 30-DAY 1-MINUTE HISTORICAL DATA")
    logger.info("=" * 60)

    X, y = [], []

    for symbol, prices in sorted(price_series.items()):
        n = len(prices)
        if n < 60:
            continue

        t0 = time.time()
        indicators = compute_indicator_arrays(prices)  # (15, n)
        if indicators is None:
            logger.warning(f"  {symbol}: indicator computation failed, skipping")
            continue

        price_arr = np.array(prices, dtype=float)
        count = 0

        # Minimum bar to use (need 30 bars warmup for indicators)
        start_bar = 30
        # Maximum bar (need LOOKAHEAD bars ahead for labels)
        end_bar = n - LOOKAHEAD

        for i in range(start_bar, end_bar):
            future_ret = (price_arr[i + LOOKAHEAD] - price_arr[i]) / price_arr[i]
            if abs(future_ret) < MIN_MOVE:
                continue  # skip flat moves

            features = indicators[:, i].tolist()  # 15 features at bar i
            label = 1 if future_ret > UP_THRESHOLD else 0
            X.append(features)
            y.append(label)
            count += 1

        elapsed = time.time() - t0
        logger.info(f"  {symbol}: {count} samples from {n} bars ({elapsed:.1f}s)")

    if len(X) < 200:
        logger.error(f"Only {len(X)} samples — not enough to train")
        return None

    X, y = np.array(X), np.array(y)
    ratio = np.mean(y)
    n_up = int(np.sum(y))
    n_down = len(y) - n_up
    logger.info(f"Total: {len(y)} samples, {n_up} UP ({ratio:.1%}), {n_down} DOWN ({1-ratio:.1%})")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, stratify=y
    )
    sw = compute_sample_weight('balanced', y_train)

    logger.info("Training GradientBoostingClassifier (200 trees, depth 4)...")
    t0 = time.time()

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sw)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    elapsed = time.time() - t0

    logger.info(f"Model trained in {elapsed:.1f}s — Train: {train_acc:.2%}, Test: {test_acc:.2%}")

    # Feature importances
    importances = model.feature_importances_
    top = sorted(zip(_ML_FEATURE_NAMES, importances), key=lambda x: -x[1])
    logger.info("Feature importances:")
    for name, imp in top:
        bar = "█" * int(imp * 100)
        logger.info(f"  {name:20s} {imp:.3f} {bar}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    logger.info(f"Saved to {MODEL_PATH}")

    # Versioned copy
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = os.path.join(MODEL_DIR, f"trading_model_{ts}_acc{int(test_acc*100)}%.joblib")
    dump(model, versioned)
    logger.info(f"Versioned copy: {versioned}")

    logger.info("=" * 60)
    logger.info(f"DONE — Test accuracy: {test_acc:.2%}")
    logger.info(f"The live bot will use this model on next restart or retrain cycle.")
    logger.info("=" * 60)

    return model


if __name__ == "__main__":
    prices = load_hourly_cache()
    train_model(prices)
