import csv
trades = list(csv.DictReader(open(r'c:\Bot\data\trades.csv')))
print(f'Total trades: {len(trades)}')
wins = [t for t in trades if float(t['pnl_usd']) > 0]
losses = [t for t in trades if float(t['pnl_usd']) <= 0]
wr = len(wins)/len(trades)*100 if trades else 0
total_pnl = sum(float(t['pnl_usd']) for t in trades)
print(f'Wins: {len(wins)}, Losses: {len(losses)}, WR: {wr:.1f}%')
print(f'Total PnL: {total_pnl:.2f}')
from collections import Counter
reasons = Counter(t['exit_reason'] for t in trades)
for r, c in reasons.most_common():
    rpnl = sum(float(t['pnl_usd']) for t in trades if t['exit_reason']==r)
    print(f'  {r}: {c} trades, PnL={rpnl:.2f}')
for d in ['SHORT','LONG']:
    dt = [t for t in trades if t['direction']==d]
    if dt:
        dpnl = sum(float(t['pnl_usd']) for t in dt)
        dw = sum(1 for t in dt if float(t['pnl_usd'])>0)
        print(f'{d}: {len(dt)} trades, {dw}W/{len(dt)-dw}L, WR={dw/len(dt)*100:.0f}%, PnL={dpnl:.2f}')
for mkt in ['Spot','Futures']:
    pfx = 'PI_' if mkt=='Futures' else '-USD'
    mt = [t for t in trades if pfx in t['symbol']]
    if mt:
        mpnl = sum(float(t['pnl_usd']) for t in mt)
        mw = sum(1 for t in mt if float(t['pnl_usd'])>0)
        print(f'{mkt}: {len(mt)} trades, {mw}W/{len(mt)-mw}L, PnL={mpnl:.2f}')
avg = sum(float(t['pnl_usd']) for t in trades)/len(trades) if trades else 0
print(f'Avg trade: {avg:.3f}')
