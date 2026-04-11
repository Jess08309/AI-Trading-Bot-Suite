import pandas as pd

df = pd.read_csv('trade_history.csv', names=['timestamp','symbol','action','price','amount','pnl_percent'])
closed = df[df['action'].isin(['SELL','CLOSE','CLOSE_LONG','CLOSE_SHORT'])]

# Convert pnl_percent to numeric
closed['pnl_percent'] = pd.to_numeric(closed['pnl_percent'], errors='coerce')

wins = (closed['pnl_percent'] > 0).sum()
losses = (closed['pnl_percent'] < 0).sum()

print('\n=== OVERALL PERFORMANCE ===')
print(f'Total closed trades: {len(closed)}')
print(f'Wins: {wins} ({wins/len(closed)*100:.1f}%)')
print(f'Losses: {losses} ({losses/len(closed)*100:.1f}%)')
print(f'Avg P/L per trade: {closed["pnl_percent"].mean():.3f}%')
print(f'Total P/L sum: {closed["pnl_percent"].sum():.2f}%')

# Spot vs Futures
spot = closed[~closed['symbol'].str.contains('PI_')]
futures = closed[closed['symbol'].str.contains('PI_')]

print('\n=== SPOT vs FUTURES ===')
print(f'Spot: {len(spot)} trades, Avg: {spot["pnl_percent"].mean():.3f}%, Total: {spot["pnl_percent"].sum():.2f}%')
print(f'Futures: {len(futures)} trades, Avg: {futures["pnl_percent"].mean():.3f}%, Total: {futures["pnl_percent"].sum():.2f}%')

print('\n=== WORST 10 TRADES ===')
print(closed.nsmallest(10, 'pnl_percent')[['timestamp','symbol','pnl_percent']])

print('\n=== BEST 10 TRADES ===')
print(closed.nlargest(10, 'pnl_percent')[['timestamp','symbol','pnl_percent']])

# Recent performance (last 100 trades)
recent = closed.tail(100)
recent_wins = (recent['pnl_percent'] > 0).sum()
print(f'\n=== LAST 100 TRADES ===')
print(f'Win rate: {recent_wins/len(recent)*100:.1f}%')
print(f'Avg P/L: {recent["pnl_percent"].mean():.3f}%')
