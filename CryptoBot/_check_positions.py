import json, urllib.request

KEY = "PKO4HHAS2T2YYVSCQMWNIE4EA4"
SECRET = "4cjicUMnxdQb199af8s4mYTrmL64ZNXyRU7PzbrivrEn"

req = urllib.request.Request(
    "https://paper-api.alpaca.markets/v2/positions",
    headers={"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET},
)
resp = urllib.request.urlopen(req)
positions = json.loads(resp.read())

print(f"Open positions: {len(positions)}\n")
for p in positions:
    symbol = p["symbol"]
    qty = p["qty"]
    side = p["side"]
    entry = float(p["avg_entry_price"])
    current = float(p["current_price"])
    pnl = float(p["unrealized_pl"])
    pnl_pct = float(p["unrealized_plpc"]) * 100
    asset_class = p.get("asset_class", "?")
    print(f"  {symbol:8s} | {side:5s} | qty={qty:>6s} | entry=${entry:>10.2f} | now=${current:>10.2f} | pnl=${pnl:>8.2f} ({pnl_pct:>+.1f}%) | {asset_class}")
