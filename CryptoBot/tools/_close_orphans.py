"""Close all orphaned Alpaca positions at next market open.
Uses limit orders with generous limits to ensure fill.
"""
import requests, json, os, time, sys
from datetime import datetime

API_KEY = "PKFYHFB2A7EJEXQUKOEHCYLNR2"
API_SECRET = "2397JkVebdYJrVaVUobiJAqFWzYXA3o94tBXEJC83Y5S"
BASE = "https://paper-api.alpaca.markets"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

# === Build tracked set from bot state files ===
tracked = set()

# PutSeller
ps_path = r"C:\PutSeller\data\state\positions.json"
if os.path.exists(ps_path):
    with open(ps_path) as f:
        ps = json.load(f)
    for k, v in ps.items():
        tracked.add(v.get("short_symbol", ""))
        tracked.add(v.get("long_symbol", ""))

# CallBuyer
cb_path = r"C:\CallBuyer\data\state\positions.json"
if os.path.exists(cb_path):
    with open(cb_path) as f:
        cb = json.load(f)
    for k, v in cb.items():
        tracked.add(v.get("contract", ""))

tracked.discard("")
print(f"[{datetime.now()}] Tracked by PutSeller/CallBuyer: {len(tracked)} symbols")

# === Get all Alpaca positions ===
pos = requests.get(f"{BASE}/v2/positions", headers=HEADERS).json()
print(f"Total Alpaca positions: {len(pos)}")

# Identify orphans
orphans = [p for p in pos if p["symbol"] not in tracked]
print(f"Orphans to close: {len(orphans)}")

if not orphans:
    print("Nothing to close!")
    sys.exit(0)

# === Close each orphan with limit orders ===
closed = 0
failed = 0
for p in sorted(orphans, key=lambda x: x["symbol"]):
    sym = p["symbol"]
    qty = int(p["qty"])
    side = p["side"]
    upl = float(p["unrealized_pl"])
    current_price = abs(float(p["current_price"]))

    close_side = "sell" if qty > 0 else "buy"
    abs_qty = abs(qty)

    # Generous limit to ensure fill at open
    if close_side == "sell":
        limit_price = round(max(current_price * 0.85, 0.01), 2)
    else:
        limit_price = round(current_price * 1.15 + 0.01, 2)

    order = {
        "symbol": sym,
        "qty": str(abs_qty),
        "side": close_side,
        "type": "limit",
        "limit_price": str(limit_price),
        "time_in_force": "day",
    }

    print(f"  Closing {sym} ({close_side} {abs_qty} @ limit ${limit_price})... ", end="")

    try:
        resp = requests.post(f"{BASE}/v2/orders", headers=HEADERS, json=order)
        if resp.status_code in (200, 201):
            oid = resp.json().get("id", "?")
            print(f"OK -> order {oid[:8]}... (uPnL was ${upl:.2f})")
            closed += 1
        else:
            error_msg = resp.text[:150]
            print(f"FAILED {resp.status_code}: {error_msg}")
            failed += 1
    except Exception as e:
        print(f"ERROR: {e}")
        failed += 1

    time.sleep(0.3)

print(f"\n=== SUMMARY ===")
print(f"Orders submitted: {closed} | Failed: {failed} | Total orphans: {len(orphans)}")
print(f"Orders are day orders — will attempt to fill at next market open (9:30 AM ET)")
print(f"Re-run after market open to verify fills.")
