import csv, os, sys
from collections import defaultdict

trades_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trades.csv")
date_filter = sys.argv[1] if len(sys.argv) > 1 else "2026-02-25"

trades = []
with open(trades_file) as f:
    for row in csv.reader(f):
        if row[0].startswith(date_filter):
            trades.append(row)

if not trades:
    print(f"No trades found for {date_filter}")
    sys.exit(0)

wins = [t for t in trades if float(t[6]) > 0]
losses = [t for t in trades if float(t[6]) <= 0]
total_pnl = sum(float(t[6]) for t in trades)
avg_win = sum(float(t[6]) for t in wins) / max(1, len(wins))
avg_loss = sum(float(t[6]) for t in losses) / max(1, len(losses))

print(f"=== {date_filter} PERFORMANCE ===")
print(f"Trades: {len(trades)}  |  W: {len(wins)}  L: {len(losses)}  |  WR: {len(wins)/len(trades)*100:.0f}%")
print(f"Total P&L: ${total_pnl:+.2f}")
print(f"Avg Win: ${avg_win:+.2f}  |  Avg Loss: ${avg_loss:+.2f}")

# Direction
longs = [t for t in trades if t[2] == "LONG"]
shorts = [t for t in trades if t[2] == "SHORT"]
long_pnl = sum(float(t[6]) for t in longs)
short_pnl = sum(float(t[6]) for t in shorts)
print(f"\nLONG:  {len(longs)} trades  P&L ${long_pnl:+.2f}")
print(f"SHORT: {len(shorts)} trades  P&L ${short_pnl:+.2f}")

# Exit reasons
exits = defaultdict(lambda: [0, 0.0])
for t in trades:
    exits[t[8]][0] += 1
    exits[t[8]][1] += float(t[6])
print("\nEXIT REASONS:")
for e in sorted(exits, key=lambda x: exits[x][1], reverse=True):
    print(f"  {e:25s} {exits[e][0]:2d} trades  ${exits[e][1]:+.2f}")

# Symbol breakdown
syms = defaultdict(lambda: [0, 0.0])
for t in trades:
    syms[t[1]][0] += 1
    syms[t[1]][1] += float(t[6])
print("\nSYMBOL BREAKDOWN:")
for s in sorted(syms, key=lambda x: syms[x][1], reverse=True):
    print(f"  {s:15s} {syms[s][0]:2d} trades  ${syms[s][1]:+.2f}")
