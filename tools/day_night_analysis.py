"""Day vs Night performance analysis for the crypto bot."""
import csv
from collections import defaultdict
from datetime import datetime

trades = []
with open("data/trades.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        trades.append(row)

day = {"trades": 0, "wins": 0, "pnl": 0.0}
night = {"trades": 0, "wins": 0, "pnl": 0.0}
hourly = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0})

for t in trades:
    ts = t["timestamp"]
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        continue
    hour = dt.hour
    pnl = float(t.get("pnl_usd", 0))
    win = 1 if pnl > 0 else 0

    # Day = 6 AM - 8:59 PM, Night = 9 PM - 5:59 AM
    bucket = day if 6 <= hour <= 20 else night
    bucket["trades"] += 1
    bucket["wins"] += win
    bucket["pnl"] += pnl

    hourly[hour]["t"] += 1
    hourly[hour]["w"] += win
    hourly[hour]["pnl"] += pnl

print("=" * 50)
print("  DAY vs NIGHT PERFORMANCE ANALYSIS")
print("=" * 50)

for label, b in [("DAY (6 AM - 9 PM)", day), ("NIGHT (9 PM - 6 AM)", night)]:
    print(f"\n--- {label} ---")
    print(f"  Trades:    {b['trades']}")
    wr = (b["wins"] / b["trades"] * 100) if b["trades"] else 0
    print(f"  Win Rate:  {wr:.1f}%  ({b['wins']}W / {b['trades'] - b['wins']}L)")
    print(f"  Total PnL: ${b['pnl']:+.2f}")
    avg = b["pnl"] / b["trades"] if b["trades"] else 0
    print(f"  Avg Trade: ${avg:+.4f}")

print(f"\n{'='*50}")
print("  HOURLY BREAKDOWN")
print(f"{'='*50}")
print(f"  {'Hour':>5}  {'Trades':>6}  {'WR':>5}  {'PnL':>9}  {'Avg':>8}")
print(f"  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*9}  {'-'*8}")

for hr in sorted(hourly.keys()):
    s = hourly[hr]
    wr = (s["w"] / s["t"] * 100) if s["t"] else 0
    avg = s["pnl"] / s["t"] if s["t"] else 0
    bar = "#" * max(1, s["t"] // 2)
    label = f"{hr:02d}:00"
    marker = " <-- BEST" if s["pnl"] == max(h["pnl"] for h in hourly.values()) else ""
    marker = " <-- WORST" if s["pnl"] == min(h["pnl"] for h in hourly.values()) else marker
    print(f"  {label:>5}  {s['t']:>6}  {wr:>4.1f}%  ${s['pnl']:>+8.2f}  ${avg:>+7.4f}  {bar}{marker}")
