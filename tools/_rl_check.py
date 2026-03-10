import json

r = json.load(open("data/state/rl_shadow_report.json"))
b = r.get("books", {})
bl = b.get("baseline", {})
rl = b.get("rl", {})
d = r.get("delta", {})

print("=== RL SHADOW REPORT ===")
print(f"BASELINE: equity=${bl.get('equity',0):.2f}  pnl=${bl.get('realized_pnl',0):.2f}  wr={bl.get('win_rate',0):.1f}%  trades={bl.get('trades',0)}  max_dd={bl.get('max_drawdown_pct',0):.2f}%")
print(f"RL AGENT: equity=${rl.get('equity',0):.2f}  pnl=${rl.get('realized_pnl',0):.2f}  wr={rl.get('win_rate',0):.1f}%  trades={rl.get('trades',0)}  max_dd={rl.get('max_drawdown_pct',0):.2f}%")
print(f"DELTA:    equity=${d.get('equity',0):.2f}  pnl=${d.get('realized_pnl',0):.2f}  dd={d.get('max_drawdown_pct',0):.2f}%")

events = r.get("events", [])
print(f"\nShadow events logged: {len(events)}")
