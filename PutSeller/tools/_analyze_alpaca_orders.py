import json
from collections import defaultdict

orders = json.load(open("/tmp/alpacabot_orders.json"))
print("num orders", len(orders))

fills = []
for o in orders:
    for e in o.get("events", []):
        if e.get("status") == "filled" and e.get("fillQuantity"):
            fills.append({
                "symbol": e["symbolValue"],
                "underlying": e.get("symbolPermtick", e["symbolValue"][:4]),
                "time": e["time"],
                "qty": e["fillQuantity"],
                "price": e["fillPrice"],
            })

fills.sort(key=lambda f: f["time"])
print("num fills", len(fills))

positions = defaultdict(list)
trades = []
for f in fills:
    sym = f["symbol"]
    qty = f["qty"]
    price = f["price"]
    lots = positions[sym]
    if qty > 0:
        lots.append([qty, price, f["time"]])
    else:
        remaining = -qty
        while remaining > 1e-9 and lots:
            lot_qty, lot_price, lot_time = lots[0]
            matched = min(lot_qty, remaining)
            pnl = (price - lot_price) * matched * 100
            trades.append({
                "symbol": sym, "open_t": lot_time, "close_t": f["time"],
                "open_price": lot_price, "close_price": price, "qty": matched, "pnl": pnl,
                "pct": (price - lot_price) / lot_price * 100 if lot_price else 0,
            })
            lots[0][0] -= matched
            remaining -= matched
            if lots[0][0] <= 1e-9:
                lots.pop(0)

trades.sort(key=lambda t: t["pnl"])
print("num closed round trips", len(trades))
print("\n--- worst 15 by pnl ---")
for t in trades[:15]:
    print(f"{t['symbol']:24s} open={t['open_price']:8.2f} close={t['close_price']:8.2f} qty={t['qty']:.0f} pnl={t['pnl']:10.2f} pct={t['pct']:8.1f}%")

print("\n--- best 5 ---")
for t in trades[-5:]:
    print(f"{t['symbol']:24s} open={t['open_price']:8.2f} close={t['close_price']:8.2f} qty={t['qty']:.0f} pnl={t['pnl']:10.2f} pct={t['pct']:8.1f}%")

total = sum(t["pnl"] for t in trades)
print("\ntotal realized pnl (approx, ignoring fees/still-open positions)", total)

# distribution of pct losses beyond -20% (stop loss threshold)
worse_than_stop = [t for t in trades if t["pct"] < -20]
print(f"\ntrades that closed WORSE than -20% stop-loss threshold: {len(worse_than_stop)} / {len(trades)}")
for t in worse_than_stop[:20]:
    print(f"  {t['symbol']:24s} pct={t['pct']:8.1f}% pnl={t['pnl']:10.2f} open_t={t['open_t']:.0f} close_t={t['close_t']:.0f} held_hrs={(t['close_t']-t['open_t'])/3600:.1f}")
