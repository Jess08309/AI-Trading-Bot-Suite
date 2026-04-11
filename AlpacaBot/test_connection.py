"""Quick connection test for Alpaca API."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.api_client import AlpacaAPI

config = Config()
print(config.summary())
print()

api = AlpacaAPI(config)
if api.connect():
    acct = api.get_account()
    print("Account connected!")
    print(f"  Equity:       ${acct['equity']:,.2f}")
    print(f"  Cash:         ${acct['cash']:,.2f}")
    print(f"  Buying Power: ${acct['buying_power']:,.2f}")
    print(f"  Market Open:  {api.is_market_open()}")

    price = api.get_latest_price("SPY")
    print(f"  SPY Price:    ${price}")

    price_qqq = api.get_latest_price("QQQ")
    print(f"  QQQ Price:    ${price_qqq}")
else:
    print("Connection FAILED - check API keys in .env")
