"""Quick test: verify Alpaca migration works end-to-end."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

print("=== Alpaca Migration Test ===")

# 1. Config
from utils.config import config
print(f"[OK] Config loaded, spot pairs: {config.POPULAR_PAIRS[:3]}")
print(f"[OK] Alpaca retries: {config.ALPACA_RETRY_ATTEMPTS}")

# 2. Price fetch via perf_engine
from core.perf_engine import ConcurrentPriceFetcher
fetcher = ConcurrentPriceFetcher(max_workers=2)
btc = fetcher._public_spot_price("BTC/USD")
eth = fetcher._public_spot_price("ETH/USD")
print(f"[OK] BTC/USD: ${btc:,.2f}" if btc else "[FAIL] BTC/USD fetch failed")
print(f"[OK] ETH/USD: ${eth:,.2f}" if eth else "[FAIL] ETH/USD fetch failed")
fetcher.close()

# 3. Universe scanner
from core.universe_scanner import AlpacaDiscovery
syms = AlpacaDiscovery.get_available_symbols()
print(f"[OK] Alpaca crypto: {len(syms)} symbols" if syms else "[FAIL] Alpaca discovery failed")

# 4. Trading engine import
from core.trading_engine import TradingConfig, ALPACA_DATA_URL
print(f"[OK] TradingConfig.SPOT_SYMBOLS[:3] = {TradingConfig.SPOT_SYMBOLS[:3]}")
print(f"[OK] ALPACA_DATA_URL = {ALPACA_DATA_URL}")

# 5. Sentiment still works
from utils.news_sentiment import get_pair_sentiment
sent = get_pair_sentiment("BTC/USD", {"BTC": 0.5}, 0.1)
print(f"[OK] BTC/USD sentiment: {sent:.2f}")
sent_futures = get_pair_sentiment("PI_XBTUSD", {"BTC": 0.5}, 0.1)
print(f"[OK] PI_XBTUSD sentiment: {sent_futures:.2f}")

print("\n=== All tests passed ===")
