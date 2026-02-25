"""
Download 2 years of historical price data for backtesting.
Saves to CSV so backtest can run from local data (fast).

Run: python download_historical.py
"""

import os
import sys
import csv
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Output paths (override with HISTORICAL_OUTPUT_DIR if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("HISTORICAL_OUTPUT_DIR", os.path.join(BASE_DIR, "data", "backtest"))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "historical_prices_2yr.csv")

# Symbols to download (matches trading_engine.py)
SPOT_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
    "DOGE-USD", "MATIC-USD", "LTC-USD", "LINK-USD", "SHIB-USD",
    "XRP-USD", "DOT-USD", "UNI-USD", "ATOM-USD", "XLM-USD",
    "OP-USD", "ARB-USD", "SUI-USD"  # Added for AI learning
]

FUTURES_SYMBOLS = [
    "PI_XBTUSD", "PI_ETHUSD", "PI_SOLUSD", "PI_ADAUSD", "PI_AVAXUSD",
    "PI_DOGEUSD", "PI_MATICUSD", "PI_LTCUSD", "PI_LINKUSD", "PI_SHIBUSD",
    "PI_XRPUSD", "PI_DOTUSD", "PI_UNIUSD", "PI_ATOMUSD", "PI_XLMUSD"
]


def fetch_coinbase_historical(symbol: str, days: int = 730) -> List[Dict]:
    """
    Fetch historical daily candles from Coinbase API.
    Free tier: 300 candles per request, daily granularity = 86400
    """
    prices = []
    base_url = "https://api.exchange.coinbase.com/products"
    granularity = 86400  # Daily candles
    
    # Coinbase uses product IDs like BTC-USD
    product_id = symbol
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    print(f"  Fetching {symbol}...", end=" ", flush=True)
    
    try:
        # Coinbase limits to 300 candles per request, so chunk it
        current_start = start_time
        while current_start < end_time:
            current_end = min(current_start + timedelta(days=299), end_time)
            
            url = (f"{base_url}/{product_id}/candles"
                   f"?start={current_start.isoformat()}"
                   f"&end={current_end.isoformat()}"
                   f"&granularity={granularity}")
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                candles = response.json()
                # Coinbase format: [timestamp, low, high, open, close, volume]
                for candle in candles:
                    ts, low, high, open_p, close_p, volume = candle
                    prices.append({
                        "timestamp": datetime.utcfromtimestamp(ts).isoformat(),
                        "symbol": symbol,
                        "type": "spot",
                        "open": open_p,
                        "high": high,
                        "low": low,
                        "close": close_p,
                        "volume": volume,
                        "price": close_p  # Use close as the price
                    })
            elif response.status_code == 404:
                print(f"not found")
                return []
            else:
                print(f"error {response.status_code}")
                break
            
            current_start = current_end
            time.sleep(0.2)  # Rate limiting
        
        print(f"{len(prices)} candles")
        
    except Exception as e:
        print(f"error: {e}")
    
    return prices


def fetch_kraken_futures_historical(symbol: str, days: int = 730) -> List[Dict]:
    """
    Fetch historical candles from Kraken Futures API.
    """
    prices = []
    base_url = "https://futures.kraken.com/api/charts/v1"
    
    # Kraken uses different resolution format
    # 1D = daily candles
    resolution = "1D"
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    print(f"  Fetching {symbol}...", end=" ", flush=True)
    
    try:
        # Kraken Futures ticker format: PI_XBTUSD
        url = (f"{base_url}/{resolution}/{symbol}"
               f"?from={int(start_time.timestamp())}"
               f"&to={int(end_time.timestamp())}")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            candles = data.get("candles", [])
            
            for candle in candles:
                prices.append({
                    "timestamp": datetime.utcfromtimestamp(candle["time"]/1000).isoformat(),
                    "symbol": symbol,
                    "type": "futures",
                    "open": candle.get("open", 0),
                    "high": candle.get("high", 0),
                    "low": candle.get("low", 0),
                    "close": candle.get("close", 0),
                    "volume": candle.get("volume", 0),
                    "price": candle.get("close", 0)
                })
            
            print(f"{len(prices)} candles")
        else:
            print(f"error {response.status_code}")
            
    except Exception as e:
        print(f"error: {e}")
    
    return prices


def main():
    print("=" * 60)
    print("DOWNLOADING 2 YEARS OF HISTORICAL PRICE DATA")
    print("=" * 60)
    print(f"Output: {OUTPUT_CSV}")
    print(f"Spot symbols: {len(SPOT_SYMBOLS)}")
    print(f"Futures symbols: {len(FUTURES_SYMBOLS)}")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_prices = []
    
    # Download spot data
    print("\n--- SPOT DATA (Coinbase) ---")
    for symbol in SPOT_SYMBOLS:
        prices = fetch_coinbase_historical(symbol, days=730)
        all_prices.extend(prices)
        time.sleep(0.5)  # Rate limiting between symbols
    
    # Download futures data
    print("\n--- FUTURES DATA (Kraken) ---")
    for symbol in FUTURES_SYMBOLS:
        prices = fetch_kraken_futures_historical(symbol, days=730)
        all_prices.extend(prices)
        time.sleep(0.5)
    
    # Sort by timestamp
    all_prices.sort(key=lambda x: x["timestamp"])
    
    # Write to CSV
    print(f"\n--- SAVING {len(all_prices)} TOTAL RECORDS ---")
    
    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["timestamp", "symbol", "type", "open", "high", "low", "close", "volume", "price"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_prices)
    
    print(f"OK Saved to {OUTPUT_CSV}")
    print(f"File size: {os.path.getsize(OUTPUT_CSV) / 1024 / 1024:.2f} MB")
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    spot_count = sum(1 for p in all_prices if p["type"] == "spot")
    futures_count = sum(1 for p in all_prices if p["type"] == "futures")
    print(f"Spot candles:    {spot_count:,}")
    print(f"Futures candles: {futures_count:,}")
    print(f"Total:           {len(all_prices):,}")
    
    if all_prices:
        print(f"Date range: {all_prices[0]['timestamp'][:10]} to {all_prices[-1]['timestamp'][:10]}")
    
    print("\nNext step: Run backtest with --csv flag:")
    print(f"  python run_backtest.py --csv \"{OUTPUT_CSV}\"")


if __name__ == "__main__":
    main()
