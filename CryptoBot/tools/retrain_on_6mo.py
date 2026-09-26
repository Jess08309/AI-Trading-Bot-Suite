"""
Retrain the ML model on 1-minute candle data (spot only -- the feature set is
price-action-only and asset-class agnostic, so a single spot-trained model is
used for both the spot and futures backtest legs).
Reads per-symbol CSVs directly (avoids loading a giant unified CSV into memory).

This produces a much stronger model than the current one because:
  - 10-50x more training samples (millions vs hundreds of thousands)
  - Covers multiple market regimes (rallies, crashes, chop, ranging)
  - All 15 technical indicator features
  - GradientBoosting with TimeSeriesSplit (no look-ahead bias)

GATE-MODEL FREEZE: by default this trains on data/historical/1min/ (the same
6-month window the baseline backtest replays) -- fine for producing a strong
*production* model, but WRONG for the baseline gate: a model trained and
tested on the identical window has seen the test window's outcomes during
training (look-ahead leakage), and cannot be used to validate whether the
lab reproduces genuinely out-of-sample live behavior.

To produce a frozen, leakage-free GATE model instead, point this at a
strictly-earlier data directory and/or pass --end-date:
    python tools/download_pretrain_window.py --end <window_start_iso> --days 60
    python tools/retrain_on_6mo.py \\
        --data-dir data/historical/1min_pretrain \\
        --end-date <window_start_iso> \\
        --output data/models/market_model_frozen.joblib

--end-date is enforced even when pointed at the normal 6-month dir, as a
defense-in-depth check -- any row at/after the cutoff is dropped before
feature extraction.

Usage:
    python tools/retrain_on_6mo.py
    python tools/retrain_on_6mo.py --data-dir data/historical/1min_pretrain \\
        --end-date 2026-03-29T00:00:00+00:00 --output data/models/market_model_frozen.joblib
"""

import argparse
import os
import sys
import csv
import json
import time
import glob
import numpy as np
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cryptotrades"))

from cryptotrades.utils.technical_indicators import compute_all_indicators
from cryptotrades.utils.feature_engine import FEATURE_NAMES

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "historical", "1min")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "market_model.joblib")

LOOKBACK = 30          # Candles needed to compute indicators
PREDICTION_HORIZON = 5 # How many candles ahead to predict
SAMPLE_EVERY = 5       # Sample every Nth candle (reduces data size while preserving diversity)


def load_symbol_prices(filepath: str, end_cutoff_ts: int = None) -> list:
    """Load close prices from a per-symbol 1-min CSV, oldest first.

    If end_cutoff_ts is given, rows with timestamp >= cutoff are dropped
    (defense-in-depth against the gate model ever seeing in-window data).
    """
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row["timestamp"])
                if end_cutoff_ts is not None and ts >= end_cutoff_ts:
                    continue
                rows.append((ts, float(row["close"])))
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def build_features_for_symbol(prices: list, symbol: str,
                              sample_every: int = SAMPLE_EVERY) -> tuple:
    """Build feature/label arrays from a symbol's price history.

    Args:
        prices: List of close prices (oldest first)
        symbol: For logging
        sample_every: Sample every Nth candle to reduce dataset size

    Returns:
        (features_list, labels_list) of numpy arrays
    """
    n = len(prices)
    if n < LOOKBACK + PREDICTION_HORIZON + 10:
        return [], []

    features = []
    labels = []
    count = 0

    for i in range(LOOKBACK, n - PREDICTION_HORIZON, sample_every):
        window = prices[i - LOOKBACK:i + 1]
        indicators = compute_all_indicators(window)
        if not indicators:
            continue

        # Build feature array in canonical order
        feature_vec = np.array([indicators.get(name, 0.0) for name in FEATURE_NAMES])

        # Label: 1 if price goes up over next PREDICTION_HORIZON, 0 if down
        current = prices[i]
        future = prices[i + PREDICTION_HORIZON]
        if current <= 0:
            continue
        label = 1 if future > current else 0

        features.append(feature_vec)
        labels.append(label)
        count += 1

    return features, labels


def retrain(data_dir: str, model_path: str, end_date: str = None, meta_output: str = None):
    print("=" * 60)
    print("RETRAIN ML MODEL")
    print("=" * 60)
    print()

    end_cutoff_ts = None
    if end_date:
        cutoff_dt = datetime.fromisoformat(end_date)
        if cutoff_dt.tzinfo is None:
            cutoff_dt = cutoff_dt.replace(tzinfo=timezone.utc)
        end_cutoff_ts = int(cutoff_dt.timestamp())
        print(f"End-date cutoff: {cutoff_dt.isoformat()} (rows at/after this are dropped)")

    files = sorted(glob.glob(os.path.join(data_dir, "*_1min.csv")))
    if not files:
        print(f"No data files in {data_dir}")
        return

    print(f"Data dir:     {data_dir}")
    print(f"Symbol files: {len(files)}")
    print(f"Model output: {model_path}")
    print(f"Features:     {len(FEATURE_NAMES)}")
    print(f"Sample every: {SAMPLE_EVERY} candles")
    print()

    # Phase 1: Build features from all symbols
    all_features = []
    all_labels = []
    symbols_used = []
    min_ts = None
    max_ts = None
    start = time.time()

    for filepath in files:
        symbol = os.path.basename(filepath).replace("_1min.csv", "")
        print(f"  Processing {symbol}...", end=" ", flush=True)

        rows = load_symbol_prices(filepath, end_cutoff_ts)
        if rows:
            min_ts = rows[0][0] if min_ts is None else min(min_ts, rows[0][0])
            max_ts = rows[-1][0] if max_ts is None else max(max_ts, rows[-1][0])
        prices = [r[1] for r in rows]
        features, labels = build_features_for_symbol(prices, symbol)

        all_features.extend(features)
        all_labels.extend(labels)
        if features:
            symbols_used.append(symbol)
        print(f"{len(features):,} samples")

    elapsed_features = time.time() - start

    if not all_features:
        print("ERROR: No training samples generated!")
        return

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pos_ratio = np.mean(y)
    print()
    print(f"[1/3] Feature extraction done in {elapsed_features:.0f}s")
    print(f"  Total samples: {len(X):,}")
    print(f"  Class balance: {pos_ratio:.1%} UP / {1-pos_ratio:.1%} DOWN")
    print(f"  Features per sample: {X.shape[1]}")
    if min_ts and max_ts:
        print(f"  Actual data range used: "
              f"{datetime.fromtimestamp(min_ts, tz=timezone.utc).isoformat()} -> "
              f"{datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat()}")
    print()

    # Phase 2: Train GradientBoosting with TimeSeriesSplit
    print("[2/3] Training GradientBoosting (TimeSeriesSplit, 5 folds)...")
    start = time.time()

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_score = 0
    best_model = None
    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        cv_scores.append(score)
        print(f"  Fold {fold+1}/{n_splits}: accuracy={score:.4f} "
              f"(train={len(X_train):,}, test={len(X_test):,})")

        if score > best_score:
            best_score = score
            best_model = model

    elapsed_train = time.time() - start
    avg_score = np.mean(cv_scores)

    print()
    print(f"  Training done in {elapsed_train:.0f}s")
    print(f"  CV Accuracy: {avg_score:.4f} (avg {n_splits} folds)")
    print(f"  Best Fold:   {best_score:.4f}")
    print()

    # Phase 3: Save model
    print("[3/3] Saving model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Version backup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    version_path = model_path.replace(".joblib", f"_{timestamp}_acc{best_score:.0%}.joblib")
    dump(best_model, version_path)
    dump(best_model, model_path)

    # Feature importances
    importances = best_model.feature_importances_
    sorted_feats = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    print()
    print("  Feature importances:")
    for name, imp in sorted_feats:
        bar = "█" * int(imp * 100)
        print(f"    {name:20s} {imp:.4f} {bar}")

    print()
    print(f"  Model saved: {model_path}")
    print(f"  Backup:      {version_path}")

    meta = {
        "model_path": model_path,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": data_dir,
        "end_date_cutoff": end_date,
        "actual_data_start": datetime.fromtimestamp(min_ts, tz=timezone.utc).isoformat() if min_ts else None,
        "actual_data_end": datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat() if max_ts else None,
        "num_samples": int(len(X)),
        "symbols": symbols_used,
        "cv_accuracy": float(avg_score),
        "best_fold_accuracy": float(best_score),
    }
    meta_path = meta_output or model_path.replace(".joblib", "_training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Training metadata: {meta_path}")
    print()

    total_elapsed = elapsed_features + elapsed_train
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR,
                         help="Directory of per-symbol *_1min.csv files to train on")
    parser.add_argument("--output", default=MODEL_PATH, help="Where to save the trained model")
    parser.add_argument("--end-date", default=None,
                         help="ISO timestamp: rows at/after this are excluded (gate-model freeze)")
    parser.add_argument("--meta-output", default=None,
                         help="Where to save training metadata JSON (default: alongside --output)")
    args = parser.parse_args()
    retrain(args.data_dir, args.output, args.end_date, args.meta_output)
