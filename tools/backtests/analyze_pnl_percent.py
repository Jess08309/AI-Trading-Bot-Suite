import csv
from collections import defaultdict

# File path to your trade history
TRADE_HISTORY_FILE = 'trade_history.csv'

# Store per-symbol stats
stats = defaultdict(lambda: {'buy_qty': 0, 'buy_total': 0, 'sell_qty': 0, 'sell_total': 0})

with open(TRADE_HISTORY_FILE, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        symbol = row['symbol']
        side = row['side']
        price = float(row['price'])
        amount = float(row['amount'])
        if side == 'BUY':
            stats[symbol]['buy_qty'] += amount
            stats[symbol]['buy_total'] += price * amount
        elif side == 'SELL':
            stats[symbol]['sell_qty'] += amount
            stats[symbol]['sell_total'] += price * amount

print(f"{'Symbol':<12} {'Return %':>10}")
print('-' * 24)
for symbol, s in stats.items():
    # Only compute for closed positions (both buy and sell)
    if s['buy_qty'] > 0 and s['sell_qty'] > 0:
        avg_buy = s['buy_total'] / s['buy_qty']
        avg_sell = s['sell_total'] / s['sell_qty']
        # Use min(buy_qty, sell_qty) to avoid overcounting partial closes
        qty = min(s['buy_qty'], s['sell_qty'])
        pnl = (avg_sell - avg_buy) / avg_buy * 100
        print(f"{symbol:<12} {pnl:10.2f}")
    else:
        print(f"{symbol:<12} {'(open)':>10}")
