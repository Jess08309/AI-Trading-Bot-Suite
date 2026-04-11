"""Automated orphan cleanup — runs unattended.
Waits for Phase 1 limit orders to fill, then closes remaining long legs.
Scheduled for 9:35 AM ET (7:35 AM MT) on April 8, 2026.
"""
import requests, json, os, time, sys
from datetime import datetime

LOG = r"C:\Bot\logs\orphan_cleanup.log"
os.makedirs(os.path.dirname(LOG), exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

API_KEY = "PKFYHFB2A7EJEXQUKOEHCYLNR2"
API_SECRET = "2397JkVebdYJrVaVUobiJAqFWzYXA3o94tBXEJC83Y5S"
BASE = "https://paper-api.alpaca.markets"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

# === Build tracked set ===
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

log("=" * 60)
log("ORPHAN AUTO-CLEANUP STARTED")
log(f"Tracked by active bots: {len(tracked)} symbols")

# === Phase 1: Wait for pending buy-to-close orders to fill ===
log("--- Phase 1: Waiting for pending limit orders to fill ---")
max_wait_min = 30  # wait up to 30 min for fills
check_interval = 60  # check every 60 seconds
waited = 0

while waited < max_wait_min * 60:
    try:
        orders = requests.get(f"{BASE}/v2/orders?status=open", headers=HEADERS).json()
    except Exception as e:
        log(f"  API error checking orders: {e}")
        time.sleep(check_interval)
        waited += check_interval
        continue

    pending = len(orders)
    if pending == 0:
        log(f"  All Phase 1 orders filled! (waited {waited//60}m {waited%60}s)")
        break
    
    log(f"  {pending} orders still pending... (waited {waited//60}m)")
    time.sleep(check_interval)
    waited += check_interval

if waited >= max_wait_min * 60:
    # Cancel any remaining unfilled orders and proceed anyway
    try:
        orders = requests.get(f"{BASE}/v2/orders?status=open", headers=HEADERS).json()
    except:
        orders = []
    if orders:
        log(f"  Timed out with {len(orders)} unfilled orders — cancelling them")
        for o in orders:
            try:
                requests.delete(f"{BASE}/v2/orders/{o['id']}", headers=HEADERS)
                log(f"    Cancelled {o['symbol']} {o['side']} qty={o['qty']}")
            except Exception as e:
                log(f"    Failed to cancel {o['symbol']}: {e}")
        time.sleep(5)

# === Phase 2: Close remaining orphans with market orders ===
log("--- Phase 2: Closing remaining orphan positions ---")
time.sleep(3)

try:
    pos = requests.get(f"{BASE}/v2/positions", headers=HEADERS).json()
except Exception as e:
    log(f"FATAL: Cannot get positions: {e}")
    sys.exit(1)

orphans = [p for p in pos if p["symbol"] not in tracked]
log(f"Total positions: {len(pos)} | Remaining orphans: {len(orphans)}")

if not orphans:
    log("All orphans already closed! Nothing to do.")
    log("CLEANUP COMPLETE")
    sys.exit(0)

closed = 0
failed = 0
for p in sorted(orphans, key=lambda x: x["symbol"]):
    sym = p["symbol"]
    qty = int(p["qty"])
    upl = float(p["unrealized_pl"])
    close_side = "sell" if qty > 0 else "buy"
    abs_qty = abs(qty)

    order = {
        "symbol": sym,
        "qty": str(abs_qty),
        "side": close_side,
        "type": "market",
        "time_in_force": "day",
    }

    try:
        resp = requests.post(f"{BASE}/v2/orders", headers=HEADERS, json=order)
        if resp.status_code in (200, 201):
            oid = resp.json().get("id", "?")
            log(f"  CLOSED {sym} ({close_side} {abs_qty}) -> order {oid[:8]}... (uPnL ${upl:.2f})")
            closed += 1
        else:
            error_msg = resp.text[:200]
            log(f"  FAILED {sym}: {resp.status_code} — {error_msg}")
            failed += 1
    except Exception as e:
        log(f"  ERROR {sym}: {e}")
        failed += 1
    time.sleep(0.3)

# === Final verification ===
time.sleep(5)
try:
    remaining = requests.get(f"{BASE}/v2/positions", headers=HEADERS).json()
    remaining_orphans = [p for p in remaining if p["symbol"] not in tracked]
    log(f"\n=== FINAL RESULT ===")
    log(f"Closed: {closed} | Failed: {failed}")
    log(f"Positions remaining: {len(remaining)} total, {len(remaining_orphans)} orphans")
    if remaining_orphans:
        log("Still open:")
        for p in remaining_orphans:
            log(f"  {p['symbol']} qty={p['qty']} uPnL=${float(p['unrealized_pl']):.2f}")
    else:
        log("ALL ORPHANS SUCCESSFULLY CLOSED!")
except Exception as e:
    log(f"Could not verify: {e}")

log("CLEANUP COMPLETE")
