"""Quick script to check Alpaca options positions."""
import sys, json, os
sys.path.insert(0, "C:/PutSeller")
os.chdir("C:/PutSeller")

from dotenv import load_dotenv
load_dotenv("C:/PutSeller/.env")

from alpaca.trading.client import TradingClient
tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"], paper=True)

positions = tc.get_all_positions()
print(f"Total Alpaca positions: {len(positions)}")

options = [p for p in positions if hasattr(p, 'symbol') and len(str(p.symbol)) > 10]
print(f"Options positions: {len(options)}")

for p in options:
    print(f"  {p.symbol} | qty={p.qty} | side={p.side} | avg={p.avg_entry_price} | mkt={p.market_value} | unrealized={p.unrealized_pl}")
