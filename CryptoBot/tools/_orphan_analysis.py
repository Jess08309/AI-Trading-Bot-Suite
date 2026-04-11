"""Analyze orphaned Alpaca positions not tracked by any bot."""
import requests, json, os

API_KEY = "PKFYHFB2A7EJEXQUKOEHCYLNR2"
API_SECRET = "2397JkVebdYJrVaVUobiJAqFWzYXA3o94tBXEJC83Y5S"
BASE = "https://paper-api.alpaca.markets"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

# === What bots are tracking ===
# PutSeller positions
ps_path = r"C:\PutSeller\data\state\positions.json"
ps_tracked = set()
if os.path.exists(ps_path):
    with open(ps_path) as f:
        ps = json.load(f)
    for k, v in ps.items():
        ps_tracked.add(v.get("short_symbol", ""))
        ps_tracked.add(v.get("long_symbol", ""))

# CallBuyer positions
cb_path = r"C:\CallBuyer\data\state\positions.json"
cb_tracked = set()
if os.path.exists(cb_path):
    with open(cb_path) as f:
        cb = json.load(f)
    for k, v in cb.items():
        cb_tracked.add(v.get("contract", ""))

# AlpacaBot positions
ab_path = r"C:\AlpacaBot\data\state\positions.json"
ab_tracked = set()
if os.path.exists(ab_path):
    with open(ab_path) as f:
        ab = json.load(f)
    for k, v in ab.items():
        ab_tracked.add(v.get("symbol", ""))
        if "short_leg_symbol" in v:
            ab_tracked.add(v["short_leg_symbol"])

all_tracked = ps_tracked | cb_tracked | ab_tracked
all_tracked.discard("")

print("=== BOT-TRACKED SYMBOLS ===")
print(f"PutSeller: {sorted(ps_tracked - {''})}")
print(f"CallBuyer: {sorted(cb_tracked - {''})}")
print(f"AlpacaBot: {sorted(ab_tracked - {''})}")
print(f"Total tracked: {len(all_tracked)} option symbols")

# === What Alpaca has ===
pos = requests.get(f"{BASE}/v2/positions", headers=HEADERS).json()
print(f"\n=== ALPACA HAS {len(pos)} POSITIONS ===")

# Group by underlying
from collections import defaultdict
by_underlying = defaultdict(list)
for p in pos:
    sym = p["symbol"]
    # Extract underlying from option symbol
    underlying = ""
    for i, c in enumerate(sym):
        if c.isdigit():
            underlying = sym[:i]
            break
    if not underlying:
        underlying = sym
    by_underlying[underlying].append(p)

# Classify each position
orphans = []
tracked_count = 0
total_orphan_upl = 0
total_orphan_mv = 0

print(f"\n{'Symbol':30s} {'Qty':>5s} {'Side':>6s} {'MktVal':>10s} {'uPnL':>10s} {'Owner':>12s}")
print("-" * 80)

for underlying in sorted(by_underlying.keys()):
    legs = by_underlying[underlying]
    for p in sorted(legs, key=lambda x: x["symbol"]):
        sym = p["symbol"]
        qty = p["qty"]
        side = p["side"]
        mv = float(p["market_value"])
        upl = float(p["unrealized_pl"])
        
        if sym in ps_tracked:
            owner = "PutSeller"
            tracked_count += 1
        elif sym in cb_tracked:
            owner = "CallBuyer"
            tracked_count += 1
        elif sym in ab_tracked:
            owner = "AlpacaBot"
            tracked_count += 1
        else:
            owner = "** ORPHAN **"
            orphans.append(p)
            total_orphan_upl += upl
            total_orphan_mv += mv
        
        print(f"{sym:30s} {qty:>5s} {side:>6s} ${mv:>9.2f} ${upl:>9.2f} {owner:>12s}")

print(f"\n{'='*80}")
print(f"TRACKED: {tracked_count} positions | ORPHANED: {len(orphans)} positions")
print(f"Orphan total market value: ${total_orphan_mv:,.2f}")
print(f"Orphan total unrealized PnL: ${total_orphan_upl:,.2f}")

# Group orphans by underlying for risk assessment
print(f"\n=== ORPHANED SPREAD ANALYSIS ===")
orphan_underlyings = defaultdict(list)
for o in orphans:
    underlying = ""
    for i, c in enumerate(o["symbol"]):
        if c.isdigit():
            underlying = o["symbol"][:i]
            break
    orphan_underlyings[underlying].append(o)

for underlying in sorted(orphan_underlyings.keys()):
    legs = orphan_underlyings[underlying]
    print(f"\n  {underlying}:")
    net_mv = 0
    net_upl = 0
    for leg in sorted(legs, key=lambda x: x["symbol"]):
        mv = float(leg["market_value"])
        upl = float(leg["unrealized_pl"])
        net_mv += mv
        net_upl += upl
        print(f"    {leg['symbol']} qty={leg['qty']} side={leg['side']} mv=${mv:.2f} uPnL=${upl:.2f}")
    print(f"    -> Net MV: ${net_mv:.2f}, Net uPnL: ${net_upl:.2f}")
