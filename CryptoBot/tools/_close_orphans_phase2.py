"""Phase 2: Close remaining orphaned long legs after short legs filled.
Run this AFTER market open (9:35+ AM ET) once the Phase 1 buy orders have filled.
Also closes any remaining AlpacaBot positions.
"""
import requests, json, os, time, sys
from datetime import datetime

API_KEY = "PKFYHFB2A7EJEXQUKOEHCYLNR2"
API_SECRET = "2397JkVebdYJrVaVUobiJAqFWzYXA3o94tBXEJC83Y5S"
BASE = "https://paper-api.alpaca.markets"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

# === Build tracked set from bot state files ===
tracked = set()
for path, key_short, key_long, key_contract in [
    (r"C:\PutSeller\data\state\positions.json", "short_symbol", "long_symbol", None),
    (r"C:\CallBuyer\data\state\positions.json", None, None, "contract"),
]:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            if key_short: tracked.add(v.get(key_short, ""))
            if key_long: tracked.add(v.get(key_long, ""))
            if key_contract: tracked.add(v.get(key_contract, ""))
tracked.discard("")

print(f"[{datetime.now()}] Phase 2 — Closing remaining orphans")
print(f"Tracked by active bots: {len(tracked)} symbols")

# === Get current positions ===
pos = requests.get(f"{BASE}/v2/positions", headers=HEADERS).json()
orphans = [p for p in pos if p["symbol"] not in tracked]
print(f"Total positions: {len(pos)} | Remaining orphans: {len(orphans)}")

if not orphans:
    print("All orphans already closed!")
    sys.exit(0)

# === Check pending orders first ===
orders = requests.get(f"{BASE}/v2/orders?status=open", headers=HEADERS).json()
pending_syms = {o["symbol"] for o in orders}
print(f"Pending orders: {len(orders)}")

# === Close remaining orphans ===
closed = 0
failed = 0
skipped = 0
for p in sorted(orphans, key=lambda x: x["symbol"]):
    sym = p["symbol"]
    qty = int(p["qty"])
    upl = float(p["unrealized_pl"])

    if sym in pending_syms:
        print(f"  SKIP {sym} — has pending order")
        skipped += 1
        continue

    close_side = "sell" if qty > 0 else "buy"
    abs_qty = abs(qty)
    current_price = abs(float(p["current_price"]))

    # Use market orders during market hours
    order = {
        "symbol": sym,
        "qty": str(abs_qty),
        "side": close_side,
        "type": "market",
        "time_in_force": "day",
    }

    print(f"  Closing {sym} ({close_side} {abs_qty})... ", end="")
    try:
        resp = requests.post(f"{BASE}/v2/orders", headers=HEADERS, json=order)
        if resp.status_code in (200, 201):
            oid = resp.json().get("id", "?")
            print(f"OK -> order {oid[:8]}... (uPnL was ${upl:.2f})")
            closed += 1
        else:
            error_msg = resp.text[:200]
            print(f"FAILED {resp.status_code}: {error_msg}")
            failed += 1
    except Exception as e:
        print(f"ERROR: {e}")
        failed += 1
    time.sleep(0.3)

print(f"\n=== PHASE 2 SUMMARY ===")
print(f"Closed: {closed} | Failed: {failed} | Skipped: {skipped}")

# Final check
time.sleep(3)
remaining = requests.get(f"{BASE}/v2/positions", headers=HEADERS).json()
remaining_orphans = [p for p in remaining if p["symbol"] not in tracked]
print(f"After cleanup: {len(remaining)} positions total, {len(remaining_orphans)} orphans remain")
if remaining_orphans:
    print("Remaining orphans:")
    for p in remaining_orphans:
        print(f"  {p['symbol']} qty={p['qty']} uPnL=${float(p['unrealized_pl']):.2f}")
