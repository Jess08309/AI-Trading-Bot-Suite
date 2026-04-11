import json, urllib.request

r = urllib.request.urlopen("http://127.0.0.1:8088/api/all", timeout=10)
d = json.loads(r.read())

print("=" * 80)
print("ALPACA ACCOUNT DEEP DIVE")
print("=" * 80)

acct = d.get("account", {})
print(f"\nEquity:       ${float(acct.get('equity', 0)):>12,.2f}")
print(f"Cash:         ${float(acct.get('cash', 0)):>12,.2f}")
print(f"Buying Power: ${float(acct.get('buying_power', 0)):>12,.2f}")
print(f"Last Equity:  ${float(acct.get('last_equity', 0)):>12,.2f}")
print(f"Change Today: ${float(acct.get('equity', 0)) - float(acct.get('last_equity', 0)):>+12,.2f}")

# Each bot section
for key in ["putseller", "callbuyer", "alpaca", "crypto"]:
    bot = d.get(key, {})
    if not bot:
        continue
    bal = bot.get("balances", {})
    positions = bot.get("positions", [])
    print(f"\n{'─' * 80}")
    print(f"  {bot.get('name', key.upper())}")
    print(f"  Status: {bot.get('status')}  PID: {bot.get('pid')}  Mem: {bot.get('mem_mb')}MB")
    print(f"  Balance: ${bal.get('total', 0):,.2f}  Daily PnL: ${bal.get('daily_pnl', 0):+,.2f}  Total PnL: ${bal.get('total_pnl', 0):+,.2f}")
    print(f"  Trades: {bot.get('total_trades', 0)}  Wins: {bot.get('wins', 0)}  Losses: {bot.get('losses', 0)}  WR: {bot.get('win_rate', 'N/A')}")
    print(f"  Consec Losses: {bot.get('consecutive_losses', 0)}")
    if key == "putseller":
        print(f"  Puts: {bot.get('put_count', 0)}  Calls: {bot.get('call_count', 0)}")
        print(f"  Total Credit: ${bot.get('total_credit', 0):,.2f}  Total Risk: ${bot.get('total_risk', 0):,.2f}")
        print(f"  Unrealized: ${bot.get('total_unrealized', 0):+,.2f}")
    print(f"  Open Positions: {len(positions)}")
    
    if positions:
        # Sort by P&L
        for pos in sorted(positions, key=lambda p: float(p.get("current_pnl_total", 0) or p.get("unrealized_pnl", 0) or p.get("pnl", 0) or 0)):
            sym = pos.get("symbol") or pos.get("underlying") or "?"
            pnl = pos.get("current_pnl_total") or pos.get("unrealized_pnl") or pos.get("pnl") or 0
            side = pos.get("side") or pos.get("spread_type") or ""
            qty = pos.get("qty") or pos.get("quantity") or ""
            exp = pos.get("expiration") or pos.get("exp") or ""
            short = pos.get("short_strike", "")
            long_s = pos.get("long_strike", "")
            credit = pos.get("credit", "")
            
            if short and long_s:
                detail = f"{side} ${short}/${long_s} exp={exp} credit=${credit}"
            else:
                detail = f"{side} qty={qty}"
            
            pnl_f = float(pnl) if pnl else 0
            marker = "!!" if pnl_f < -100 else "  "
            print(f"  {marker} {sym:>6} PnL=${pnl_f:>+8.2f}  {detail}")

# Live positions from Alpaca
print(f"\n{'─' * 80}")
live = d.get("live_positions", [])
print(f"  ALPACA LIVE POSITIONS: {len(live)} total")
total_unrealized = 0
for pos in sorted(live, key=lambda p: float(p.get("unrealized_pl", 0) or 0)):
    sym = pos.get("symbol", "?")
    upl = float(pos.get("unrealized_pl", 0) or 0)
    qty = pos.get("qty", "?")
    side = pos.get("side", "?")
    mv = float(pos.get("market_value", 0) or 0)
    total_unrealized += upl
    marker = "!!" if upl < -100 else "  "
    print(f"  {marker} {sym:>30} {side:>5} qty={qty:>4} mkt_val=${mv:>10,.2f} uPnL=${upl:>+8.2f}")

print(f"\n  Total Unrealized P&L: ${total_unrealized:+,.2f}")
