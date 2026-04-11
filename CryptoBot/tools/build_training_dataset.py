"""
Build a massive training dataset from 6 months of 1-minute candles.
Converts per-symbol CSVs into the unified price_history.csv format
that the ML model trainer expects.

Also builds training data directly for maximum sample count.

Usage:
    python tools/build_training_dataset.py
"""

import os
import sys
import csv
import glob
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical", "1min")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "history", "price_history.csv")


def symbol_to_pair(filename: str) -> str:
    """Convert filename like BTC-USD_1min.csv to pair name BTC-USD."""
    return filename.replace("_1min.csv", "")


def build_unified_csv():
    """Merge all per-symbol 1-min CSVs into a single price_history.csv
    that the ML trainer expects: timestamp, pair, price
    """
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_1min.csv")))
    if not files:
        print(f"No data files found in {INPUT_DIR}")
        print("Run tools/download_6mo_candles.py first!")
        return 0

    print(f"Found {len(files)} symbol files in {INPUT_DIR}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    total_rows = 0

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, 'w', newline='') as out:
        writer = csv.writer(out)
        # No header — the feature engine reads headerless: timestamp, pair, price

        for filepath in files:
            basename = os.path.basename(filepath)
            pair = symbol_to_pair(basename)
            count = 0

            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = int(row["timestamp"])
                        close = float(row["close"])
                        # Convert epoch to ISO timestamp
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                        writer.writerow([dt.isoformat(), pair, close])
                        count += 1
                    except (ValueError, KeyError):
                        continue

            print(f"  {pair}: {count:,} rows")
            total_rows += count

    print()
    print(f"Total: {total_rows:,} rows written to {OUTPUT_CSV}")
    file_size = os.path.getsize(OUTPUT_CSV) / 1024 / 1024
    print(f"File size: {file_size:.1f} MB")

    return total_rows


def count_training_samples():
    """Preview how many training samples the feature engine would produce."""
    try:
        from cryptotrades.utils.feature_engine import FeatureEngine
        fe = FeatureEngine(lookback=30, prediction_horizon=5)

        print("\nCounting potential training samples...")
        X, y = fe.build_training_data_from_csv(OUTPUT_CSV, min_samples=100)

        if X is not None:
            import numpy as np
            pos_ratio = np.mean(y)
            print(f"  Training samples: {len(X):,}")
            print(f"  Class balance: {pos_ratio:.1%} UP / {1-pos_ratio:.1%} DOWN")
            print(f"  Features per sample: {X.shape[1]}")
        else:
            print("  Could not build training data — check format")
    except Exception as e:
        print(f"  Could not count samples: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("BUILD UNIFIED TRAINING DATASET")
    print("=" * 60)
    print()

    total = build_unified_csv()

    if total > 0:
        count_training_samples()

    print()
    print("Done. You can now retrain the model with:")
    print("  python -c \"from cryptotrades.utils.market_predictor import MarketPredictor; m = MarketPredictor(); m.train('data/history/price_history.csv')\"")
