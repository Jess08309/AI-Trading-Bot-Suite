import json, os

state_dir = r'c:\Bot\data\state'

resets = {
    'paper_balances.json': {'spot': 2500.0, 'futures': 2500.0, 'peak_balance': 5000.0, 'daily_pnl': 0.0, 'consecutive_losses': 0, 'daily_pnl_date': '2026-02-22'},
    'positions.json': [],
    'futures_positions.json': [],
    'coin_performance.json': {},
    'symbol_performance.json': {},
    'performance_report.json': {},
    'trade_transparency.json': {},
    'circuit_breaker.json': {'tripped': False, 'reason': '', 'trip_time': None},
    'rl_shadow_report.json': {},
    'meta_learner.json': {},
    'correlations.json': {},
    'candidate_backtest_results.json': {},
    'candidate_winners.json': [],
    'futures_probation_state.json': {},
    'state_root.json': {},
    'strategy_backtest_report.json': {},
}

for fname, data in resets.items():
    path = os.path.join(state_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'  wrote {fname}')

for fname in ['rl_shadow_events.jsonl', 'runtime_fingerprint_history.jsonl']:
    path = os.path.join(state_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        pass
    print(f'  cleared {fname}')

trades_path = r'c:\Bot\data\trades.csv'
with open(trades_path, 'w', encoding='utf-8') as f:
    f.write('timestamp,symbol,direction,entry_price,exit_price,size_usd,pnl_usd,pnl_pct,exit_reason,hold_seconds,confidence,market_type\n')
print('  reset trades.csv')

print('\nAll state files written (UTF-8, no BOM)')
