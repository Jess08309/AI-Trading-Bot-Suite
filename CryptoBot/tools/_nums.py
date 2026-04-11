import urllib.request, json
d = json.loads(urllib.request.urlopen('http://127.0.0.1:8088/api/all', timeout=10).read())
a = d.get('account', {})
ps = d.get('putseller', {})
positions = ps.get('positions', [])
total_pnl = sum(p.get('current_pnl',0) for p in positions)
print(f"EQUITY={a.get('equity',0)}")
print(f"CASH={a.get('cash',0)}")
print(f"BP={a.get('buying_power',0)}")
print(f"PS_RISK={ps.get('total_risk',0)}")
print(f"PS_CREDIT={ps.get('total_credit',0)}")
print(f"PS_UNREAL={total_pnl}")
bal = ps.get('balances', {})
print(f"PS_BAL={bal.get('current_balance',0)}")
print(f"PS_RPNL={bal.get('total_pnl',0)}")
# Per-position detail
for p in sorted(positions, key=lambda x: x.get('current_pnl',0)):
    sym = p['underlying']
    st = p['spread_type']
    pnl = p.get('current_pnl',0)
    pct = p.get('current_pnl_pct',0)
    qty = p.get('qty', 0)
    risk = p.get('max_loss_total', 0)
    credit = p.get('total_credit', 0)
    exp = p.get('expiration', '?')
    print(f"POS|{sym}|{st}|qty={qty}|credit={credit:.0f}|risk={risk:.0f}|pnl={pnl:.0f}|pct={pct:.1f}%|exp={exp}")
