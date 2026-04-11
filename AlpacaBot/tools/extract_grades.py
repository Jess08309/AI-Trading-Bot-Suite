import json

with open("data/state/new_stock_backtest_results.json") as f:
    data = json.load(f)

for b in ["alpacabot", "putseller", "callbuyer"]:
    results = data["results"].get(b, {})
    print(f"=== {b.upper()} ===")
    for g in ["A", "B", "C", "D", "F"]:
        syms = sorted([s for s, i in results.items() if i.get("grade") == g])
        print(f"  {g} ({len(syms)}): {syms}")
    print()
