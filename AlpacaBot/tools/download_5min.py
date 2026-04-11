"""
Download 5-minute bars for scalp backtesting.
Gets 90 days of 5-min data from Alpaca for each symbol.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
os.makedirs("data/historical", exist_ok=True)

client = StockHistoricalDataClient(API_KEY, API_SECRET)

end = datetime.now()

# Download in 30-day chunks to avoid timeouts (90 days total)
for symbol in SYMBOLS:
    all_bars = []
    
    for chunk in range(3):  # 3 x 30 days = 90 days
        chunk_end = end - timedelta(days=chunk * 30)
        chunk_start = chunk_end - timedelta(days=30)
        
        print(f"  {symbol} chunk {chunk+1}/3: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
        
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=chunk_start,
                end=chunk_end,
            )
            bars = client.get_stock_bars(req)
            symbol_bars = bars.data.get(symbol, [])
            
            for b in symbol_bars:
                all_bars.append({
                    "timestamp": b.timestamp,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                })
        except Exception as e:
            print(f"    Error: {e}")
    
    if all_bars:
        df = pd.DataFrame(all_bars)
        df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
        path = f"data/historical/{symbol}_5min.csv"
        df.to_csv(path, index=False)
        
        days = df["timestamp"].apply(lambda x: str(x)[:10]).nunique()
        print(f"  {symbol}: {len(df)} bars across {days} trading days saved")
    else:
        print(f"  {symbol}: NO DATA")

print("\nDone!")
