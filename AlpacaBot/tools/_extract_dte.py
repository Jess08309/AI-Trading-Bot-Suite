import json

d = json.load(open("tools/backtests/dte_sweep_results.json"))

print("=== KEEP (PF >= 1.2) ===")
keeps = sorted(
    [(s, v) for s, v in d.items() if v["verdict"] == "KEEP"],
    key=lambda x: x[1]["best_pnl"], reverse=True
)
for s, v in keeps:
    print(f"  {s:6s}  DTE={v['best_dte']:2d}  PnL=${v['best_pnl']:>10,.0f}  PF={v['best_pf']:.2f}")

print(f"\n=== MAYBE (PF 1.0-1.2) ===")
maybes = sorted(
    [(s, v) for s, v in d.items() if v["verdict"] == "MAYBE"],
    key=lambda x: x[1]["best_pnl"], reverse=True
)
for s, v in maybes:
    print(f"  {s:6s}  DTE={v['best_dte']:2d}  PnL=${v['best_pnl']:>10,.0f}  PF={v['best_pf']:.2f}")

print(f"\n=== SYMBOL_DTE_MAP ===")
all_profitable = keeps + maybes
dte_map = {s: v["best_dte"] for s, v in all_profitable}
print("SYMBOL_DTE_MAP = {")
for s, dte in sorted(dte_map.items()):
    print(f'    "{s}": {dte},')
print("}")

print(f"\n=== WATCHLIST (top 2 KEEP by PnL) ===")
print(f'WATCHLIST = ["{keeps[0][0]}", "{keeps[1][0]}"]')

print(f"\n=== MAX DTE needed ===")
print(f"Max DTE in map: {max(v['best_dte'] for _, v in all_profitable)}")
