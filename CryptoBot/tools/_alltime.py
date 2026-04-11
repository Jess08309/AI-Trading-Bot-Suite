import csv
from collections import defaultdict

trades = []
with open('C:/Bot/data/trades.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        trades.append(row)

# Daily breakdown
days = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0})
for t in trades:
    day = t[0][:10]
    pnl = float(t[6])
    days[day]['trades'] += 1
    days[day]['pnl'] += pnl
    if pnl > 0:
        days[day]['wins'] += 1
    else:
        days[day]['losses'] += 1

total_pnl = sum(d['pnl'] for d in days.values())
total_trades = sum(d['trades'] for d in days.values())
total_wins = sum(d['wins'] for d in days.values())
num_days = len(days)
winning_days = sum(1 for d in days.values() if d['pnl'] > 0)

print(f'=== ALL-TIME LIVE BOT PERFORMANCE (PAPER) ===')
print(f'Period: {min(days.keys())} to {max(days.keys())} ({num_days} days)')
print(f'Total Trades: {total_trades}')
print(f'Total P&L: ${total_pnl:+.2f}')
print(f'Win Rate: {total_wins/total_trades*100:.0f}% ({total_wins}W / {total_trades-total_wins}L)')
print(f'Winning Days: {winning_days}/{num_days} ({winning_days/num_days*100:.0f}%)')
print(f'Avg Daily P&L: ${total_pnl/num_days:+.2f}')
print(f'Avg Daily Return: {total_pnl/num_days/5000*100:+.3f}% (on $5K paper)')
print()
print('DAILY BREAKDOWN:')
for day in sorted(days.keys()):
    d = days[day]
    wr = d['wins']/d['trades']*100 if d['trades'] > 0 else 0
    print(f"  {day}: {d['trades']:3d} trades  {wr:4.0f}% WR  P&L ${d['pnl']:+8.2f}")

# Running balance
print()
bal = 5000
print('EQUITY CURVE:')
for day in sorted(days.keys()):
    bal += days[day]['pnl']
    print(f'  {day}: ${bal:,.2f}')

# Projection scenarios
print()
print('=' * 60)
print('PROJECTION SCENARIOS (based on actual avg daily return)')
avg_daily_pct = total_pnl / num_days / 5000
print(f'Actual avg daily return: {avg_daily_pct*100:+.4f}%')
print()

for starting in [1000, 2000, 5000]:
    print(f'--- Starting with ${starting:,} ---')
    bal = starting
    for months in [1, 3, 6, 12]:
        proj = starting * ((1 + avg_daily_pct) ** (months * 30))
        print(f'  {months:2d} months: ${proj:,.2f} (P&L ${proj - starting:+,.2f})')
    print()

# Position sizing analysis
print('POSITION SIZING ANALYSIS:')
sizes = [float(t[5]) for t in trades]
pnls = [float(t[6]) for t in trades]
print(f'  Avg position size: ${sum(sizes)/len(sizes):.2f}')
print(f'  Min position size: ${min(sizes):.2f}')
print(f'  Max position size: ${max(sizes):.2f}')
print(f'  Avg P&L per trade: ${sum(pnls)/len(pnls):+.2f}')
print(f'  Avg return per trade: {sum(pnls)/sum(sizes)*100:+.3f}%')
