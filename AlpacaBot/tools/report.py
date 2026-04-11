"""
AlpacaBot Quick Report - Show today's and all-time performance.
Run: python tools/report.py
"""
import csv
import os
import sys
from datetime import datetime, date
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRADE_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trades.csv")


def load_trades():
    """Load all trades from CSV."""
    if not os.path.exists(TRADE_LOG):
        print(f"No trade log found at {TRADE_LOG}")
        return []

    trades = []
    with open(TRADE_LOG, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["pnl"] = float(row.get("pnl", 0))
                row["pnl_pct"] = float(row.get("pnl_pct", 0))
                row["cost"] = float(row.get("cost", 0))
                row["qty"] = int(row.get("qty", 0))
                row["confidence"] = float(row.get("confidence", 0))
                trades.append(row)
            except (ValueError, KeyError):
                continue
    return trades


def report():
    trades = load_trades()
    if not trades:
        print("No trades yet.")
        return

    today_str = date.today().isoformat()

    # Split today vs all
    today_trades = [t for t in trades if t.get("timestamp", "").startswith(today_str)]
    all_trades = trades

    print("=" * 60)
    print("  AlpacaBot Performance Report")
    print("=" * 60)

    for label, subset in [("TODAY", today_trades), ("ALL TIME", all_trades)]:
        if not subset:
            print(f"\n── {label}: No trades ──")
            continue

        total_pnl = sum(t["pnl"] for t in subset)
        winners = [t for t in subset if t["pnl"] > 0]
        losers = [t for t in subset if t["pnl"] <= 0]
        win_rate = len(winners) / len(subset) * 100 if subset else 0

        gross_profit = sum(t["pnl"] for t in winners) if winners else 0
        gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0

        print(f"\n── {label} ({len(subset)} trades) ──")
        print(f"  P&L:           ${total_pnl:+.2f}")
        print(f"  Win Rate:      {win_rate:.0f}% ({len(winners)}W / {len(losers)}L)")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  Avg Win:       ${avg_win:.2f}")
        print(f"  Avg Loss:      ${avg_loss:.2f}")

        # By underlying
        by_sym = defaultdict(list)
        for t in subset:
            by_sym[t.get("underlying", "?")].append(t)

        print(f"\n  By Symbol:")
        for sym in sorted(by_sym.keys()):
            sym_trades = by_sym[sym]
            sym_pnl = sum(t["pnl"] for t in sym_trades)
            sym_wr = len([t for t in sym_trades if t["pnl"] > 0]) / len(sym_trades) * 100
            print(f"    {sym:6s}: ${sym_pnl:+8.2f} ({len(sym_trades)} trades, {sym_wr:.0f}% WR)")

        # By direction
        calls = [t for t in subset if t.get("option_type") == "call"]
        puts = [t for t in subset if t.get("option_type") == "put"]
        if calls:
            call_pnl = sum(t["pnl"] for t in calls)
            print(f"\n  Calls:  ${call_pnl:+.2f} ({len(calls)} trades)")
        if puts:
            put_pnl = sum(t["pnl"] for t in puts)
            print(f"  Puts:   ${put_pnl:+.2f} ({len(puts)} trades)")

        # Exit reasons
        reasons = defaultdict(int)
        for t in subset:
            reasons[t.get("exit_reason", "?")] += 1
        print(f"\n  Exit Reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:20s}: {count}")

    # Daily breakdown (all time)
    if len(all_trades) > 1:
        print(f"\n── DAILY BREAKDOWN ──")
        by_day = defaultdict(list)
        for t in all_trades:
            day = t.get("timestamp", "")[:10]
            by_day[day].append(t)

        cumulative = 0
        for day in sorted(by_day.keys()):
            day_trades = by_day[day]
            day_pnl = sum(t["pnl"] for t in day_trades)
            cumulative += day_pnl
            day_wr = len([t for t in day_trades if t["pnl"] > 0]) / len(day_trades) * 100
            print(f"  {day}: ${day_pnl:+8.2f} ({len(day_trades)} trades, {day_wr:.0f}% WR) "
                  f"| cumulative: ${cumulative:+.2f}")


if __name__ == "__main__":
    report()
