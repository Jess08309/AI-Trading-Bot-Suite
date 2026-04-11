import re, os
from pathlib import Path
from collections import defaultdict

log_dir = Path("C:/AlpacaBot/logs")
close_re = re.compile(
    r"(\d{4}-\d{2}-\d{2}).*CLOSED:\s+(\w+)\s+(PUT|CALL)\s+\|\s+PnL\s+\$([+-]?[\d,.]+)\s+\(([+-]?\d+)%\)\s+\|\s+(\w+)"
)

trades = []
for lf in sorted(log_dir.glob("alpacabot_*.log")):
    try:
        with open(lf, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = close_re.search(line)
                if m:
                    dt, sym, side, pnl_s, pct_s, reason = m.groups()
                    trades.append({
                        "date": dt, "symbol": sym, "side": side,
                        "pnl": float(pnl_s.replace(",", "")),
                        "pct": float(pct_s), "reason": reason,
                    })
    except Exception:
        pass

print(f"Total AlpacaBot trades: {len(trades)}")
wins = [t for t in trades if t["pnl"] > 0]
losses = [t for t in trades if t["pnl"] <= 0]
print(f"Wins: {len(wins)}  Losses: {len(losses)}  WR: {len(wins)/len(trades)*100:.1f}%")
total_pnl = sum(t["pnl"] for t in trades)
print(f"Total P&L: ${total_pnl:+,.2f}")
print(f"Avg Win: ${sum(t['pnl'] for t in wins)/len(wins):+,.2f}" if wins else "No wins")
print(f"Avg Loss: ${sum(t['pnl'] for t in losses)/len(losses):+,.2f}" if losses else "No losses")

# By date
print("\n--- Daily P&L ---")
by_date = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
for t in trades:
    by_date[t["date"]]["pnl"] += t["pnl"]
    by_date[t["date"]]["trades"] += 1
    if t["pnl"] > 0: by_date[t["date"]]["wins"] += 1
for dt in sorted(by_date):
    d = by_date[dt]
    wr = d["wins"]/d["trades"]*100 if d["trades"] else 0
    print(f"  {dt}: {d['trades']:>3} trades, {d['wins']:>2}W, PnL ${d['pnl']:>+10,.2f}  WR={wr:.0f}%")

# By side
print("\n--- By Side ---")
for side in ["PUT", "CALL"]:
    st = [t for t in trades if t["side"] == side]
    if st:
        sw = [t for t in st if t["pnl"] > 0]
        tp = sum(t["pnl"] for t in st)
        print(f"  {side}: {len(st)} trades, {len(sw)}W/{len(st)-len(sw)}L ({len(sw)/len(st)*100:.0f}% WR), P&L ${tp:+,.2f}")

# By exit reason
print("\n--- By Exit Reason ---")
by_reason = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0})
for t in trades:
    by_reason[t["reason"]]["count"] += 1
    by_reason[t["reason"]]["pnl"] += t["pnl"]
    if t["pnl"] > 0: by_reason[t["reason"]]["wins"] += 1
for r, d in sorted(by_reason.items(), key=lambda x: x[1]["pnl"]):
    wr = d["wins"]/d["count"]*100 if d["count"] else 0
    print(f"  {r:>20}: {d['count']:>4} trades, {d['wins']:>3}W, PnL ${d['pnl']:>+10,.2f}  WR={wr:.0f}%")

# Top losers by symbol
print("\n--- Worst Symbols ---")
by_sym = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
for t in trades:
    by_sym[t["symbol"]]["pnl"] += t["pnl"]
    by_sym[t["symbol"]]["trades"] += 1
    if t["pnl"] > 0: by_sym[t["symbol"]]["wins"] += 1
for sym, d in sorted(by_sym.items(), key=lambda x: x[1]["pnl"])[:15]:
    wr = d["wins"]/d["trades"]*100
    print(f"  {sym:>6}: {d['trades']:>3} trades, PnL ${d['pnl']:>+10,.2f}  WR={wr:.0f}%")
