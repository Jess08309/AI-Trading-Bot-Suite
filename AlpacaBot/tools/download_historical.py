"""
Download historical candles from Alpaca for backtesting.
Daily bars (1 year) + recent minute bars (10 days).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from core.config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


def download():
    config = Config()
    client = StockHistoricalDataClient(
        api_key=config.API_KEY,
        secret_key=config.API_SECRET,
    )
    os.makedirs("data/historical", exist_ok=True)

    end = datetime.now()

    # ── Daily bars: 1 year ──
    print("=== Downloading DAILY bars (1 year) ===")
    start_daily = end - timedelta(days=365)

    for symbol in SYMBOLS:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_daily,
                end=end,
            )
            bars = client.get_stock_bars(req)
            bar_list = bars.data.get(symbol, [])

            if not bar_list:
                print("no data")
                continue

            rows = []
            for bar in bar_list:
                rows.append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                })

            df = pd.DataFrame(rows)
            path = f"data/historical/{symbol}_daily.csv"
            df.to_csv(path, index=False)
            print(f"{len(df)} bars")

        except Exception as e:
            print(f"ERROR: {e}")

    # ── Minute bars: 10 days ──
    print("\n=== Downloading MINUTE bars (10 days) ===")
    start_min = end - timedelta(days=10)

    for symbol in SYMBOLS:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start_min,
                end=end,
            )
            bars = client.get_stock_bars(req)
            bar_list = bars.data.get(symbol, [])

            if not bar_list:
                print("no data")
                continue

            rows = []
            for bar in bar_list:
                rows.append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                })

            df = pd.DataFrame(rows)
            path = f"data/historical/{symbol}_1min.csv"
            df.to_csv(path, index=False)
            print(f"{len(df):,} bars")

        except Exception as e:
            print(f"ERROR: {e}")

    print("\nDone! Files in data/historical/")


if __name__ == "__main__":
    download()
