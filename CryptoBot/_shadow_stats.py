import json

rpt = json.load(open(r'c:\Bot\data\state\rl_shadow_report.json'))
print(f"Updated: {rpt['updated_at']}  Cycle: {rpt['cycle']}")
print()

for name, book in rpt['books'].items():
    wr = book['win_rate']
    print(f"=== {name.upper()} ===")
    print(f"  Balance:      ${book['balance']:.2f}")
    print(f"  Equity:       ${book['equity']:.2f}")
    print(f"  Peak Equity:  ${book['peak_equity']:.2f}")
    print(f"  Max Drawdown: {book['max_drawdown_pct']:.2f}%")
    print(f"  Trades:       {book['trades']}  ({book['wins']}W / {book['losses']}L)")
    print(f"  Win Rate:     {wr:.1f}%")
    print(f"  Realized PnL: ${book['realized_pnl']:.2f}")
    positions = book.get('positions', {})
    print(f"  Open Pos:     {len(positions)}")
    if book['trades'] > 0:
        avg = book['realized_pnl'] / book['trades']
        print(f"  Avg Trade:    ${avg:.3f}")
    print()

# Comparison
books = list(rpt['books'].values())
if len(books) >= 2:
    b = books[0]  # baseline
    r = books[1]  # rl
    print("--- COMPARISON ---")
    diff_pnl = r['realized_pnl'] - b['realized_pnl']
    print(f"  PnL difference (RL - Baseline): ${diff_pnl:.2f}")
    print(f"  WR difference: {r['win_rate'] - b['win_rate']:.1f}pp")
    print(f"  DD difference: {r['max_drawdown_pct'] - b['max_drawdown_pct']:.2f}pp")
