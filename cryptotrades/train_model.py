"""
Train a real trading model with proper feature engineering.
Matches the engine's 15 momentum indicators exactly.
Uses walk-forward validation — no future leakage.
"""

import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from datetime import datetime, timedelta

print("=" * 60)
print("CRYPTO TRADING MODEL TRAINER (v2 — aligned with engine)")
print("=" * 60)

# Fetch 2 years of hourly data for multiple coins
print("\n[1/5] Fetching historical data...")

coins = ["bitcoin", "ethereum", "solana", "cardano"]
all_prices = {}

for coin in coins:
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "365",
        "interval": "daily"
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        prices = [float(p[1]) for p in data['prices']]
        all_prices[coin] = prices
        print(f"  {coin}: {len(prices)} days")
    except Exception as e:
        print(f"  {coin}: FAILED ({e})")

if not all_prices:
    print("ERROR: No data fetched")
    exit(1)

# Build 15-feature dataset matching engine's _ML_FEATURE_NAMES exactly
print("\n[2/5] Engineering 15 features (matching engine)...")

def compute_indicators(prices):
    """Compute the same 15 indicators as the engine's technical_indicators.py"""
    s = pd.Series(prices)
    n = len(s)
    features = {}

    # 1. RSI-14
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    features['rsi_14'] = ((100 - (100 / (1 + rs))) / 100).iloc[-1]  # Normalized 0-1

    # 2. MACD histogram
    ema12 = s.ewm(span=12).mean()
    ema26 = s.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line
    features['macd_histogram'] = (macd_hist / s).iloc[-1]  # Normalized by price

    # 3. Stochastic K
    low14 = s.rolling(14).min()
    high14 = s.rolling(14).max()
    stoch_k = (s - low14) / (high14 - low14)
    features['stoch_k'] = stoch_k.iloc[-1] if not np.isnan(stoch_k.iloc[-1]) else 0.5

    # 4. CCI-20
    tp = s  # Using close only (no high/low in daily data)
    sma20 = tp.rolling(20).mean()
    mad20 = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma20) / (0.015 * mad20)
    features['cci_20'] = (cci / 200).iloc[-1]  # Normalized

    # 5. ROC-10
    features['roc_10'] = ((s.iloc[-1] / s.iloc[-11]) - 1) if n > 11 else 0.0

    # 6. Momentum-10
    features['momentum_10'] = ((s.iloc[-1] - s.iloc[-11]) / s.iloc[-11]) if n > 11 else 0.0

    # 7. Williams %R
    highest_14 = s.rolling(14).max()
    lowest_14 = s.rolling(14).min()
    williams = (highest_14 - s) / (highest_14 - lowest_14)
    features['williams_r'] = williams.iloc[-1] if not np.isnan(williams.iloc[-1]) else 0.5

    # 8. Ultimate Oscillator (simplified with close-only)
    bp = s - s.shift(1).clip(upper=s)
    tr = (s.shift(1).clip(lower=s) - s.shift(1).clip(upper=s)).abs().clip(lower=0.01)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    uo = (4 * avg7 + 2 * avg14 + avg28) / 7
    features['ultimate_osc'] = uo.iloc[-1] if not np.isnan(uo.iloc[-1]) else 0.5

    # 9. TRIX-15
    ema1 = s.ewm(span=15).mean()
    ema2 = ema1.ewm(span=15).mean()
    ema3 = ema2.ewm(span=15).mean()
    trix = ema3.pct_change()
    features['trix_15'] = trix.iloc[-1] if not np.isnan(trix.iloc[-1]) else 0.0

    # 10. CMO-14
    up_sum = gain.rolling(14).sum()
    down_sum = loss.rolling(14).sum()
    cmo = (up_sum - down_sum) / (up_sum + down_sum)
    features['cmo_14'] = cmo.iloc[-1] if not np.isnan(cmo.iloc[-1]) else 0.0

    # 11. ATR-14 (normalized)
    # With only close prices, approximate with |close - prev_close|
    atr_proxy = s.diff().abs().rolling(14).mean()
    features['atr_14'] = (atr_proxy / s).iloc[-1] if s.iloc[-1] > 0 else 0.0

    # 12. Trend strength (slope of linear regression)
    if n >= 20:
        y_vals = s.iloc[-20:].values
        x_vals = np.arange(20)
        slope = np.polyfit(x_vals, y_vals, 1)[0]
        features['trend_strength'] = slope / s.iloc[-1]
    else:
        features['trend_strength'] = 0.0

    # 13. Bollinger Band position
    sma20_val = s.rolling(20).mean().iloc[-1]
    std20 = s.rolling(20).std().iloc[-1]
    if std20 > 0:
        features['bb_position'] = (s.iloc[-1] - sma20_val) / (2 * std20)
    else:
        features['bb_position'] = 0.0

    # 14. Mean reversion (z-score)
    sma50 = s.rolling(50).mean().iloc[-1] if n >= 50 else s.mean()
    std50 = s.rolling(50).std().iloc[-1] if n >= 50 else s.std()
    features['mean_reversion'] = (s.iloc[-1] - sma50) / std50 if std50 > 0 else 0.0

    # 15. Volume ratio (approximate with price volatility ratio)
    short_vol = s.pct_change().rolling(5).std().iloc[-1]
    long_vol = s.pct_change().rolling(20).std().iloc[-1]
    features['vol_ratio'] = short_vol / long_vol if long_vol > 0 else 1.0

    return features


# Feature order must match engine exactly
FEATURE_NAMES = [
    "rsi_14", "macd_histogram", "stoch_k", "cci_20", "roc_10",
    "momentum_10", "williams_r", "ultimate_osc", "trix_15", "cmo_14",
    "atr_14", "trend_strength", "bb_position", "mean_reversion", "vol_ratio",
]

X, y = [], []

for coin, prices in all_prices.items():
    if len(prices) < 60:
        continue
    for i in range(50, len(prices) - 10):
        try:
            feats = compute_indicators(prices[:i+1])
            row = [feats.get(f, 0.0) for f in FEATURE_NAMES]
            # Replace NaN/inf
            row = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in row]

            future_return = (prices[i + 5] - prices[i]) / prices[i]
            if abs(future_return) < 0.005:
                continue  # Skip flat moves
            label = 1 if future_return > 0.01 else 0  # 1% threshold
            X.append(row)
            y.append(label)
        except Exception:
            continue

X, y = np.array(X), np.array(y)
print(f"  Total samples: {len(X)}")
print(f"  UP: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"  DOWN: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")

# Walk-forward validation (no shuffle — temporal integrity)
print("\n[3/5] Walk-forward cross-validation (3 folds)...")

tscv = TimeSeriesSplit(n_splits=3, gap=int(len(X) * 0.02))
fold_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    weights = compute_sample_weight('balanced', y_tr)
    fold_model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        min_samples_split=20, min_samples_leaf=10,
        subsample=0.8, max_features=0.7, random_state=42
    )
    fold_model.fit(X_tr, y_tr, sample_weight=weights)
    score = fold_model.score(X_te, y_te)
    fold_scores.append(score)
    print(f"  Fold {fold_idx+1}: OOS accuracy = {score:.2%}")

avg_oos = np.mean(fold_scores)
print(f"  Average OOS: {avg_oos:.2%}")

# Final model on full data
print("\n[4/5] Training final model...")
weights = compute_sample_weight('balanced', y)
model = GradientBoostingClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.05,
    min_samples_split=20, min_samples_leaf=10,
    subsample=0.8, max_features=0.7, random_state=42
)
model.fit(X, y, sample_weight=weights)

# Save with accuracy in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
acc_str = f"{int(avg_oos * 100)}%"
filename = f"models/trading_model_{timestamp}_acc{acc_str}.joblib"
joblib.dump(model, filename)
print(f"\n  Saved: {filename}")

# Feature importance
print("\n[5/5] Feature importance:")
importances = model.feature_importances_
for feat, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
    print(f"   {feat:20s}: {imp:.2%}")

print(f"\n{'='*60}")
print(f"Walk-forward OOS accuracy: {avg_oos:.2%}")
print(f"Features: {len(FEATURE_NAMES)} (aligned with engine)")
print(f"Algorithm: GradientBoostingClassifier (aligned with engine)")
print(f"{'='*60}")

if avg_oos < 0.55:
    print("WARNING: OOS accuracy < 55% — model may not be profitable")
elif avg_oos < 0.60:
    print("CAUTION: OOS accuracy < 60% — marginal model")
else:
    print("Model looks reasonable for paper trading")
