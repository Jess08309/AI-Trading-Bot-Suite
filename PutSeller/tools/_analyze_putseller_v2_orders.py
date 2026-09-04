import json
from collections import defaultdict

orders = json.load(open("/tmp/putseller_v2_orders.json"))
print("num orders", len(orders))

# direction: 0 = buy, 1 = sell (from QC OrderDirection enum: Buy=0, Sell=1)
fills = []
for o in orders:
    sym = o["symbol"]["value"]
    underlying = o["symbol"]["underlying"]["value"]
    right = "call" if "C" in sym[len(underlying):].strip()[6:7] else "put"
    # crude right detection: option symbol format SYMBOL YYMMDD[C|P]STRIKE
    qty = o["quantity"]
    price = o["price"]
    t = o["time"]
    fills.append({"symbol": sym, "underlying": underlying, "qty": qty, "price": price, "time": t})

fills.sort(key=lambda f: f["time"])
print("num fills", len(fills))

positions = defaultdict(list)
trades = []
for f in fills:
    sym, qty, price = f["symbol"], f["qty"], f["price"]
    lots = positions[sym]
    if qty > 0:  # buy
        # covering a short?
        remaining = qty
        while remaining > 1e-9 and lots and lots[0][0] < 0:
            lot_qty, lot_price, lot_time = lots[0]  # negative qty
            matched = min(-lot_qty, remaining)
            pnl = (lot_price - price) * matched * 100  # short leg: sold high, buy back low = profit
            trades.append({"symbol": sym, "underlying": f["underlying"], "leg": "short",
                            "open_t": lot_time, "close_t": f["time"], "open_price": lot_price,
                            "close_price": price, "qty": matched, "pnl": pnl})
            lots[0][0] += matched
            remaining -= matched
            if abs(lots[0][0]) <= 1e-9:
                lots.pop(0)
        if remaining > 1e-9:
            lots.append([remaining, price, f["time"]])
    else:  # sell
        remaining = -qty
        while remaining > 1e-9 and lots and lots[0][0] > 0:
            lot_qty, lot_price, lot_time = lots[0]
            matched = min(lot_qty, remaining)
            pnl = (price - lot_price) * matched * 100  # long leg: bought low sold high = profit
            trades.append({"symbol": sym, "underlying": f["underlying"], "leg": "long",
                            "open_t": lot_time, "close_t": f["time"], "open_price": lot_price,
                            "close_price": price, "qty": matched, "pnl": pnl})
            lots[0][0] -= matched
            remaining -= matched
            if lots[0][0] <= 1e-9:
                lots.pop(0)
        if remaining > 1e-9:
            lots.append([-remaining, price, f["time"]])

print("num closed leg round-trips", len(trades))
total_pnl = sum(t["pnl"] for t in trades)
print("total realized pnl (approx)", total_pnl)

short_pnl = sum(t["pnl"] for t in trades if t["leg"] == "short")
long_pnl = sum(t["pnl"] for t in trades if t["leg"] == "long")
print("short-leg total pnl", short_pnl, "| long-leg total pnl", long_pnl)

by_underlying = defaultdict(float)
for t in trades:
    by_underlying[t["underlying"]] += t["pnl"]
print("\n--- pnl by underlying (worst 15) ---")
for u, p in sorted(by_underlying.items(), key=lambda x: x[1])[:15]:
    print(f"  {u:8s} {p:12.2f}")
print("--- best 5 ---")
for u, p in sorted(by_underlying.items(), key=lambda x: x[1])[-5:]:
    print(f"  {u:8s} {p:12.2f}")

# open positions still open at end (unmatched)?
still_open = {sym: lots for sym, lots in positions.items() if lots}
print(f"\nsymbols with still-open lots at backtest end: {len(still_open)}")
