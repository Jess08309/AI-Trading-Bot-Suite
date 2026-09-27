"""Run the PutSeller Cash-Secured Put (CSP) backtest and print a summary.

This is the "PutSeller v1" naked-put strategy (20-25 delta, 30-45 DTE), NOT the
existing spread/iron-condor strategy tested by run_putseller.py/run_putseller_v2.py
in this same folder -- see csp_engine.py's module docstring for the full context
on why this is a separate file.

Usage: python3 run_csp_backtest.py [start] [end]
"""
import sys
from csp_engine import run_backtest, UNIVERSE

start = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
end = sys.argv[2] if len(sys.argv) > 2 else "2025-12-31"

result = run_backtest(start, end, verbose=True)

sectors = sorted(set(UNIVERSE.values()))
print(f"Cash-Secured Put backtest ({len(UNIVERSE)} symbols, {len(sectors)} sectors: {', '.join(sectors)})")
print(f"Period: {start} -> {end}")
print(f"Trades: {result['trades']}")
print(f"Net Profit: {result['net_profit_pct']:.2f}%")
print(f"End Equity: ${result.get('end_equity', 0):,.2f}")
print(f"Win Rate: {result['win_rate']:.1f}%")
print(f"Avg Win: ${result['avg_win']:,.2f}")
print(f"Avg Loss: ${result['avg_loss']:,.2f}")
print(f"Max Drawdown: {result.get('max_drawdown_pct', 0):.1f}%")
print(f"Sharpe Ratio: {result.get('sharpe', 0):.2f}")
print()
print(f"{'Exit Reason':22} {'Count':>7} {'Win%':>7} {'Total PnL':>15}")
for reason, d in sorted(result.get("by_reason", {}).items(), key=lambda kv: -kv[1]["pnl"]):
    winpct = 100 * d["wins"] / d["count"] if d["count"] else 0
    print(f"{reason:22} {d['count']:7} {winpct:6.1f}% {d['pnl']:15,.2f}")
print()
print(f"{'Sector':15} {'Count':>7} {'Win%':>7} {'Total PnL':>15}")
for sector, d in sorted(result.get("by_sector", {}).items(), key=lambda kv: -kv[1]["pnl"]):
    winpct = 100 * d["wins"] / d["count"] if d["count"] else 0
    print(f"{sector:15} {d['count']:7} {winpct:6.1f}% {d['pnl']:15,.2f}")
