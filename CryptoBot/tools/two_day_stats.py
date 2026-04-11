import csv, datetime

wins = losses = flat = 0
total_pnl = 0.0
cutoff = datetime.datetime(2026, 2, 22, 0, 0)

with open("data/trades.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ct = row.get("close_time") or row.get("timestamp", "")
        if not ct:
            continue
        try:
            ts = datetime.datetime.fromisoformat(ct)
        except:
            continue
        if ts < cutoff:
            continue
        try:
            pnl = float(row.get("pnl_usd", 0) or 0)
        except:
            pnl = 0.0
        total_pnl += pnl
        if pnl > 0.01:
            wins += 1
        elif pnl < -0.01:
            losses += 1
        else:
            flat += 1

total = wins + losses + flat
wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
print(f"Closed trades since Feb 22: {total}")
print(f"  Wins: {wins}  |  Losses: {losses}  |  Flat/breakeven: {flat}")
print(f"  Win rate: {wr:.1f}%")
print(f"  Net P&L from closed trades: ${total_pnl:+.2f}")
