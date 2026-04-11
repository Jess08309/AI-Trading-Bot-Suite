import json, urllib.request

# Alpaca account
try:
    with urllib.request.urlopen("http://127.0.0.1:8088/api/account", timeout=5) as r:
        acct = json.loads(r.read())
    print("=== ALPACA ACCOUNT ===")
    for k in ["equity","cash","buying_power","portfolio_value","last_equity","initial_margin","maintenance_margin"]:
        print(f"  {k}: {acct.get(k)}")
except Exception as e:
    print(f"Dashboard error: {e}")

# All bots
try:
    with urllib.request.urlopen("http://127.0.0.1:8088/api/all", timeout=5) as r:
        data = json.loads(r.read())
    for bot in data.get("bots", []):
        bal = bot.get("balances", {})
        positions = bot.get("positions", [])
        print(f"\n=== {bot['name']} ===")
        print(f"  Status: {bot.get('status')}")
        print(f"  Total PnL: {bal.get('total_pnl')}")
        print(f"  Return: {bal.get('return_pct')}")
        print(f"  Daily PnL: {bal.get('daily_pnl')}")
        print(f"  Trades: {bot.get('total_trades')}")
        print(f"  Win Rate: {bot.get('win_rate')}")
        print(f"  Consec Losses: {bot.get('consecutive_losses')}")
        print(f"  Positions: {len(positions)}")
        for pos in positions:
            sym = pos.get("symbol") or pos.get("underlying") or "?"
            pnl = pos.get("current_pnl_total") or pos.get("unrealized_pnl") or pos.get("pnl") or 0
            qty = pos.get("qty") or pos.get("quantity") or "?"
            side = pos.get("side") or pos.get("spread_type") or "?"
            print(f"    POS: {sym} side={side} qty={qty} pnl={pnl}")
except Exception as e:
    print(f"API error: {e}")
