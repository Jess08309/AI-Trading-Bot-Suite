"""Quick test: how far back can we get 10-min bars from Alpaca?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import Config
from core.api_client import AlpacaAPI
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
from datetime import datetime, timedelta

cfg = Config()
api = AlpacaAPI(cfg)
api.connect()

tf = TimeFrame(10, TimeFrameUnit.Minute)

# Test the wrapper vs direct
print("=== Testing wrapper (api.get_bars) ===")
bars_w = api.get_bars("MSFT", tf, days=365)
print(f"  Wrapper result: type={type(bars_w)}, len={len(bars_w)}")

print("\n=== Testing direct (api.data.get_stock_bars) ===")
req = StockBarsRequest(
    symbol_or_symbols="MSFT",
    timeframe=tf,
    start=datetime.now() - timedelta(days=365),
)
result = api.data.get_stock_bars(req)
print(f"  Direct result: type={type(result)}")

# Check if 'MSFT' is in the result
try:
    print(f"  'MSFT' in result: {'MSFT' in result}")
    bars = result["MSFT"]
    print(f"  result['MSFT']: type={type(bars)}, len={len(bars)}")
except Exception as e:
    print(f"  Error: {e}")

# Check .data attribute  
try:
    print(f"  result.data: type={type(result.data)}")
    print(f"  result.data keys: {list(result.data.keys())}")
except Exception as e:
    print(f"  .data error: {e}")

# Check dict()
try:
    d = dict(result)
    print(f"  dict(result) keys: {list(d.keys())}")
except Exception as e:
    print(f"  dict error: {e}")
