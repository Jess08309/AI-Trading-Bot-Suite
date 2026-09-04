"""Run the synthetic PutSeller backtest and print a summary.

Usage: python3 run_putseller.py [start] [end]
"""
import sys
from engine import run_backtest

start = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-01"

result = run_backtest(start, end)

print(f"Period: {start} -> {end}")
print(f"Trades: {result['trades']}")
print(f"Net Profit: {result['net_profit_pct']:.2f}%")
print(f"End Equity: ${result.get('end_equity', 0):,.2f}")
print(f"Win Rate: {result['win_rate']:.1f}%")
print(f"Max Drawdown: {result.get('max_drawdown_pct', 0):.1f}%")
print()
print(f"{'Exit Reason':20} {'Count':>7} {'Win%':>7} {'Total PnL':>15}")
for reason, d in sorted(result.get("by_reason", {}).items(), key=lambda kv: -kv[1]["pnl"]):
    winpct = 100 * d["wins"] / d["count"] if d["count"] else 0
    print(f"{reason:20} {d['count']:7} {winpct:6.1f}% {d['pnl']:15,.2f}")
