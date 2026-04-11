"""Cancel all open orders on Alpaca to clear stuck positions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest

cfg = Config()
tc = TradingClient(api_key=cfg.API_KEY, secret_key=cfg.API_SECRET, paper=cfg.PAPER)

# Get all open orders
orders = tc.get_orders(GetOrdersRequest(status="open"))
print(f"Open orders: {len(orders)}")
for o in orders:
    print(f"  {o.id} | {o.side} {o.qty}x {o.symbol} | status={o.status} | submitted={o.submitted_at}")

# Cancel all open orders
if orders:
    cancelled = tc.cancel_orders()
    print(f"Cancelled all open orders: {len(cancelled)} responses")
else:
    print("No open orders to cancel")

# Check positions
positions = tc.get_all_positions()
print(f"\nOpen positions: {len(positions)}")
for p in positions:
    print(f"  {p.symbol} | qty={p.qty} | avg_entry=${float(p.avg_entry_price):.2f} | "
          f"current=${float(p.current_price):.2f} | unrealized=${float(p.unrealized_pl):.2f}")

# Check account
acct = tc.get_account()
print(f"\nAccount: equity=${float(acct.equity):,.2f} | cash=${float(acct.cash):,.2f}")
