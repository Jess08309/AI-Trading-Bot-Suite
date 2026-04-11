"""
Close reversed debit spreads created by the double-execution bug.

When a MLEG close order timed out (30s) but actually filled, AND the fallback
individual leg orders also filled, the net effect was reversing the position
(credit spread → debit spread in opposite direction).

These 9 reversed positions must be closed at market open.

Usage: Run manually at market open:
    C:\PutSeller\.venv\Scripts\python.exe C:\PutSeller\tools\_close_reversed.py
"""
import sys, os, time, re
sys.path.insert(0, "C:/PutSeller")
os.chdir("C:/PutSeller")

from dotenv import load_dotenv
load_dotenv("C:/PutSeller/.env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"], paper=True)

# ── Known reversed positions (from double-execution bug 2026-03-27) ──
# Format: (short_symbol, long_symbol, qty)
# "short" = currently short in Alpaca, "long" = currently long in Alpaca
# To close: buy to close shorts, sell to close longs
REVERSED = [
    ("AVGO260515P00270000",  "AVGO260515P00280000",  6),   # bear put debit
    ("GOOGL260501C00315000", "GOOGL260501C00305000",  10),  # bull call debit
    ("IBIT260515P00028000",  "IBIT260515P00033000",   20),  # bear put debit
    ("IBIT260515P00029000",  "IBIT260515P00034000",   4),   # bear put debit
    ("IWM260515P00215000",   "IWM260515P00225000",    11),  # bear put debit
    ("IWM260515P00220000",   "IWM260515P00230000",    5),   # bear put debit
    ("SLV260501P00050000",   "SLV260501P00055000",    10),  # bear put debit
    ("TQQQ260515P00030000",  "TQQQ260515P00035000",   2),   # bear put debit
    ("XLE260515P00053000",   "XLE260515P00058000",    4),   # bear put debit
]

def close_leg(symbol: str, qty: int, side: OrderSide, label: str):
    """Submit a market order to close one leg."""
    try:
        req = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=side, time_in_force=TimeInForce.DAY,
        )
        order = tc.submit_order(req)
        print(f"  {label}: {side.value} {qty}x {symbol} -> order {order.id}")
        return True
    except Exception as e:
        print(f"  FAILED {label}: {side.value} {qty}x {symbol}: {e}")
        return False


def main():
    # Check market status
    clock = tc.get_clock()
    if not clock.is_open:
        print(f"Market is CLOSED. Next open: {clock.next_open}")
        resp = input("Close positions anyway (pre-market)? [y/N]: ").strip().lower()
        if resp != 'y':
            print("Aborting. Run again when market is open.")
            return

    # Verify positions exist in Alpaca before closing
    positions = {str(p.symbol): p for p in tc.get_all_positions()}

    print(f"\n{'='*60}")
    print(f"CLOSING {len(REVERSED)} REVERSED DEBIT SPREADS")
    print(f"{'='*60}\n")

    closed = 0
    failed = 0

    for short_sym, long_sym, qty in REVERSED:
        # Verify both legs exist
        if short_sym not in positions:
            print(f"SKIP {short_sym}/{long_sym}: short leg not in Alpaca (already closed?)")
            continue
        if long_sym not in positions:
            print(f"SKIP {short_sym}/{long_sym}: long leg not in Alpaca (already closed?)")
            continue

        actual_short_qty = abs(int(positions[short_sym].qty))
        actual_long_qty = abs(int(positions[long_sym].qty))

        if actual_short_qty != qty or actual_long_qty != qty:
            print(f"WARNING {short_sym}: expected qty={qty}, Alpaca has short={actual_short_qty} long={actual_long_qty}")
            qty = min(actual_short_qty, actual_long_qty)  # close what we can

        print(f"Closing reversed spread: {short_sym} / {long_sym} x{qty}")

        # Buy to close the short leg
        ok1 = close_leg(short_sym, qty, OrderSide.BUY, "buy-to-close short")
        time.sleep(0.5)

        # Sell to close the long leg
        ok2 = close_leg(long_sym, qty, OrderSide.SELL, "sell-to-close long")
        time.sleep(0.5)

        if ok1 and ok2:
            closed += 1
        else:
            failed += 1

        print()

    print(f"\nDone: {closed} closed, {failed} failed")
    if failed:
        print("Re-run this script or check Alpaca dashboard for failed legs.")


if __name__ == "__main__":
    main()
