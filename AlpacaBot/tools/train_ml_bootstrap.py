"""
Bootstrap ML Model Training
============================
Fetches 30 days of 10-min historical bars from Alpaca for the full scanner
universe, then trains the GradientBoosting direction model.

This lets us train the model BEFORE the bot has collected in-memory data,
solving the chicken-and-egg problem (bot trades badly without ML, but ML
needs data from the bot running).

Usage:
    cd C:\AlpacaBot
    .venv\Scripts\python train_ml_bootstrap.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bootstrap")

from core.config import Config
from core.api_client import AlpacaAPI
from core.scanner import SCANNER_UNIVERSE
from utils.ml_model import OptionsMLModel
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def main():
    config = Config()
    api = AlpacaAPI(config)

    log.info("Connecting to Alpaca...")
    if not api.connect():
        log.error("Failed to connect to Alpaca. Check API keys.")
        return False

    # Use all scanner universe symbols for training
    symbols = list(SCANNER_UNIVERSE)
    log.info(f"Fetching 30 days of 10-min bars for {len(symbols)} symbols...")

    tf = TimeFrame(10, TimeFrameUnit.Minute)
    price_dict = {}
    errors = 0

    for i, sym in enumerate(symbols):
        try:
            bars = api.get_bars(sym, tf, days=30)
            if bars:
                prices = [float(bar.close) for bar in bars]
                price_dict[sym] = np.array(prices)
                log.info(f"  [{i+1}/{len(symbols)}] {sym}: {len(prices)} bars")
            else:
                log.warning(f"  [{i+1}/{len(symbols)}] {sym}: no bars returned")
                errors += 1
            time.sleep(0.4)  # rate limit
        except Exception as e:
            log.warning(f"  [{i+1}/{len(symbols)}] {sym}: error - {e}")
            errors += 1
            time.sleep(0.5)

    log.info(f"\nData fetched: {len(price_dict)} symbols, {errors} errors")
    for sym, prices in sorted(price_dict.items(), key=lambda x: -len(x[1])):
        log.info(f"  {sym}: {len(prices)} bars ({len(prices)/39:.0f} trading days)")

    total_bars = sum(len(p) for p in price_dict.values())
    usable = sum(1 for p in price_dict.values() if len(p) >= 60)
    log.info(f"\nTotal: {total_bars} bars across {len(price_dict)} symbols ({usable} usable for training)")

    if usable < 2:
        log.error("Not enough usable symbols (need >= 2 with 60+ bars). Aborting.")
        return False

    # Train the model
    log.info("\n" + "=" * 60)
    log.info("TRAINING ML MODEL...")
    log.info("=" * 60)

    ml = OptionsMLModel(model_dir="data/models", min_accuracy=0.54)

    # Try with stride=1 first (more samples), fall back to stride=2 if too slow
    success = ml.train(price_dict, min_samples=200, stride=1)

    if success:
        log.info(f"\n{'=' * 60}")
        log.info(f"MODEL TRAINED SUCCESSFULLY!")
        log.info(f"  Test accuracy: {ml.test_accuracy:.1%}")
        log.info(f"  Train accuracy: {ml.train_accuracy:.1%}")
        log.info(f"  Top features:")
        for name, imp in list(ml.feature_importances.items())[:5]:
            log.info(f"    {name}: {imp:.4f}")
        log.info(f"  Saved to: {ml.model_path}")
        log.info(f"{'=' * 60}")
        return True
    else:
        log.error(f"\nModel training failed or rejected (accuracy < {ml.min_accuracy:.0%})")
        log.error(f"  Best accuracy achieved: {ml.test_accuracy:.1%}")
        return False


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
