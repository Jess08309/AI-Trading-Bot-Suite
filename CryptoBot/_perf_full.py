"""Full performance report across all trades."""
import csv, json, sys
from collections import defaultdict
from datetime import datetime

rows = []
with open('data/trades.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        r['pnl_usd'] = float(r['pnl_usd'])
        r['pnl_pct'] = float(r['pnl_pct'])
        r['size_usd'] = float(r['size_usd'])
        r['ts'] = datetime.fromisoformat(r['timestamp'])
        rows.append(r)

print(f'TOTAL TRADES: {len(rows)}')
print(f'DATE RANGE: {rows[0]["timestamp"][:10]} to {rows[-1]["timestamp"][:10]}')
print()

# Overall
wins = [r for r in rows if r['pnl_usd'] > 0]
losses = [r for r in rows if r['pnl_usd'] < 0]
flat = [r for r in rows if r['pnl_usd'] == 0]
total_pnl = sum(r['pnl_usd'] for r in rows)
wr = len(wins)/len(rows)*100 if rows else 0
avg_win = sum(r['pnl_usd'] for r in wins)/len(wins) if wins else 0
avg_loss = sum(r['pnl_usd'] for r in losses)/len(losses) if losses else 0

print('='*60)
print('OVERALL PERFORMANCE')
print('='*60)
print(f'Total P&L:       ${total_pnl:+.2f}')
print(f'Win Rate:        {wr:.1f}% ({len(wins)}W / {len(losses)}L / {len(flat)}F)')
print(f'Avg Win:         ${avg_win:+.2f}')
print(f'Avg Loss:        ${avg_loss:+.2f}')
if avg_loss:
    print(f'Win/Loss Ratio:  {abs(avg_win/avg_loss):.2f}x')
print(f'Expectancy:      ${total_pnl/len(rows):+.4f} per trade')
print(f'Total Volume:    ${sum(r["size_usd"] for r in rows):,.0f}')
print()

# By direction
for d in ['SHORT','LONG']:
    dr = [r for r in rows if r['direction']==d]
    if not dr: continue
    dw = [r for r in dr if r['pnl_usd']>0]
    dl = [r for r in dr if r['pnl_usd']<0]
    dp = sum(r['pnl_usd'] for r in dr)
    dwr = len(dw)/len(dr)*100
    davg_w = sum(r['pnl_usd'] for r in dw)/len(dw) if dw else 0
    davg_l = sum(r['pnl_usd'] for r in dl)/len(dl) if dl else 0
    print(f'{d}: {len(dr)} trades, {dwr:.1f}% WR, P&L ${dp:+.2f}, AvgW ${davg_w:+.2f}, AvgL ${davg_l:+.2f}')

print()

# By exit reason
print('EXIT REASONS:')
reasons = defaultdict(lambda: {'count':0,'pnl':0,'wins':0})
for r in rows:
    reasons[r['exit_reason']]['count'] += 1
    reasons[r['exit_reason']]['pnl'] += r['pnl_usd']
    if r['pnl_usd'] > 0: reasons[r['exit_reason']]['wins'] += 1
for reason, d in sorted(reasons.items(), key=lambda x: -x[1]['count']):
    wr2 = d['wins']/d['count']*100 if d['count'] else 0
    print(f'  {reason:25s} {d["count"]:4d} trades  {wr2:5.1f}% WR  P&L ${d["pnl"]:+8.2f}')

print()

# By day
print('DAILY BREAKDOWN:')
days = defaultdict(lambda: {'count':0,'pnl':0,'wins':0})
for r in rows:
    day = r['timestamp'][:10]
    days[day]['count'] += 1
    days[day]['pnl'] += r['pnl_usd']
    if r['pnl_usd'] > 0: days[day]['wins'] += 1
for day in sorted(days):
    d = days[day]
    wr2 = d['wins']/d['count']*100
    print(f'  {day}: {d["count"]:4d} trades  {wr2:5.1f}% WR  P&L ${d["pnl"]:+8.2f}')

print()

# Spot vs Futures
spot = [r for r in rows if 'PI_' not in r['symbol']]
futures = [r for r in rows if 'PI_' in r['symbol']]
sp = sum(r['pnl_usd'] for r in spot)
fp = sum(r['pnl_usd'] for r in futures)
swr = len([r for r in spot if r['pnl_usd']>0])/len(spot)*100 if spot else 0
fwr = len([r for r in futures if r['pnl_usd']>0])/len(futures)*100 if futures else 0
print(f'SPOT:    {len(spot):4d} trades  {swr:.1f}% WR  P&L ${sp:+.2f}')
print(f'FUTURES: {len(futures):4d} trades  {fwr:.1f}% WR  P&L ${fp:+.2f}')
print()

# Symbol-level
print('SYMBOL STATS (sorted by P&L):')
syms = defaultdict(lambda: {'count':0,'pnl':0,'wins':0})
for r in rows:
    syms[r['symbol']]['count'] += 1
    syms[r['symbol']]['pnl'] += r['pnl_usd']
    if r['pnl_usd'] > 0: syms[r['symbol']]['wins'] += 1
for sym, d in sorted(syms.items(), key=lambda x: -x[1]['pnl']):
    wr2 = d['wins']/d['count']*100
    print(f'  {sym:18s} {d["count"]:4d} trades  {wr2:5.1f}% WR  P&L ${d["pnl"]:+8.2f}')

# Biggest winners and losers
print()
print('TOP 5 WINS:')
for r in sorted(rows, key=lambda x: -x['pnl_usd'])[:5]:
    print(f'  {r["timestamp"][:16]} {r["direction"]:5s} {r["symbol"]:15s} ${r["pnl_usd"]:+.2f} ({r["pnl_pct"]:+.2f}%) {r["exit_reason"]}')

print()
print('TOP 5 LOSSES:')
for r in sorted(rows, key=lambda x: x['pnl_usd'])[:5]:
    print(f'  {r["timestamp"][:16]} {r["direction"]:5s} {r["symbol"]:15s} ${r["pnl_usd"]:+.2f} ({r["pnl_pct"]:+.2f}%) {r["exit_reason"]}')

# Drawdown
print()
cumulative = 0
peak = 0
max_dd = 0
for r in rows:
    cumulative += r['pnl_usd']
    if cumulative > peak:
        peak = cumulative
    dd = cumulative - peak
    if dd < max_dd:
        max_dd = dd
print(f'MAX DRAWDOWN: ${max_dd:+.2f}')
print(f'PEAK P&L:     ${peak:+.2f}')
print(f'CURRENT P&L:  ${cumulative:+.2f}')

# Post-fix analysis (after 14:19 today)
print()
print('='*60)
print('POST-FIX PERFORMANCE (after 2026-02-18 14:19)')
print('='*60)
cutoff = datetime(2026, 2, 18, 14, 19)
post = [r for r in rows if r['ts'] >= cutoff]
if post:
    pw = [r for r in post if r['pnl_usd'] > 0]
    pl = [r for r in post if r['pnl_usd'] < 0]
    pp = sum(r['pnl_usd'] for r in post)
    pwr = len(pw)/len(post)*100
    pavg_w = sum(r['pnl_usd'] for r in pw)/len(pw) if pw else 0
    pavg_l = sum(r['pnl_usd'] for r in pl)/len(pl) if pl else 0
    print(f'Trades:    {len(post)}')
    print(f'Win Rate:  {pwr:.1f}% ({len(pw)}W / {len(pl)}L)')
    print(f'P&L:       ${pp:+.2f}')
    print(f'Avg Win:   ${pavg_w:+.2f}')
    print(f'Avg Loss:  ${pavg_l:+.2f}')
    shorts = [r for r in post if r['direction']=='SHORT']
    longs = [r for r in post if r['direction']=='LONG']
    print(f'SHORT:     {len(shorts)} trades, P&L ${sum(r["pnl_usd"] for r in shorts):+.2f}')
    print(f'LONG:      {len(longs)} trades, P&L ${sum(r["pnl_usd"] for r in longs):+.2f}')

# Balance
print()
try:
    bal = json.load(open('data/state/paper_balances.json'))
    total = bal.get('spot',0)+bal.get('futures',0)
    print(f'CURRENT BALANCE: Spot ${bal.get("spot",0):.2f} + Futures ${bal.get("futures",0):.2f} = ${total:.2f}')
    print(f'ALL-TIME P&L:    ${total - 5000:+.2f} ({(total-5000)/5000*100:+.1f}%)')
except: pass

# RL shadow
print()
try:
    rl = json.load(open('data/state/rl_shadow_report.json'))
    print('RL SHADOW REPORT:')
    events = rl if isinstance(rl, list) else rl.get('events', [])
    agree = sum(1 for e in events if e.get('type')=='decision' and e.get('rl_trade',False))
    disagree = sum(1 for e in events if e.get('type')=='decision' and not e.get('rl_trade',False))
    print(f'  Decisions: {len(events)} total, {agree} agree, {disagree} disagree')
except Exception as e:
    print(f'RL shadow: {e}')
