"""Bootstrap ML model training from Alpaca historical data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.config import Config
from core.api_client import AlpacaAPI
from core.scanner import SCANNER_UNIVERSE
from utils.ml_model import OptionsMLModel

c = Config()
api = AlpacaAPI(c)
api.connect()

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
tf = TimeFrame(10, TimeFrameUnit.Minute)

symbols = list(SCANNER_UNIVERSE)
print(f"Fetching 30d of 10-min bars for {len(symbols)} symbols...")

price_dict = {}
for i, sym in enumerate(symbols):
    try:
        bars = api.get_bars(sym, tf, days=30)
        if bars and len(bars) >= 60:
            prices = [float(bar.close) for bar in bars]
            price_dict[sym] = np.array(prices)
            print(f"  [{i+1}/{len(symbols)}] {sym}: {len(prices)} bars")
        else:
            print(f"  [{i+1}/{len(symbols)}] {sym}: insufficient ({len(bars) if bars else 0} bars)")
    except Exception as e:
        print(f"  [{i+1}/{len(symbols)}] {sym}: ERROR {e}")

print(f"\nGot data for {len(price_dict)} symbols")
total = sum(len(v) for v in price_dict.values())
print(f"Total data points: {total}")

model = OptionsMLModel(model_dir="data/models", min_accuracy=0.51)
print(f"\nTraining with min_accuracy=51%...")
success = model.train(price_dict, min_samples=200, stride=3)

if success:
    print(f"\nSUCCESS: Model accepted with {model.test_accuracy:.1%} accuracy")
    print(f"Saved to: data/models/")
    print(f"Top features: {dict(list(model.feature_importances.items())[:5])}")
else:
    print(f"\nFAILED: Model rejected (below {model.min_accuracy:.0%} threshold)")
    if hasattr(model, 'test_accuracy') and model.test_accuracy:
        print(f"  Achieved: {model.test_accuracy:.1%}")
