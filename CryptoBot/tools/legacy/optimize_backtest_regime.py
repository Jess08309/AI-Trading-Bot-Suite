"""
ARCHIVED (backtest consolidation, see CryptoBot/backtest/README.md).

Reason: round-2 sweep on top of local_backtest_qc_port.py's rules-only
strategy (archived alongside this file). Closed investigation -- see
optimize_backtest.py / local_backtest_qc_port.py docstrings. Kept here
unmodified for reference/history. Not maintained, not run in CI.
============================================================================

Second-round sweep: adds the optional REGIME_SMA_PERIOD filter (price above
a longer SMA, to avoid buying momentum blips inside a broader downtrend) on
top of the best entry/exit combo found by the first optimize_backtest.py
sweep (MIN_RULE_SCORE=8.0, MIN_TREND_SLOPE=0.0015, TAKE_PROFIT_PCT=2.5,
STOP_LOSS_PCT=-1.5), which still lost -51.4% on its own.
"""
import itertools

from local_backtest_qc_port import load_events, run_backtest

events = load_events()

base = {
    "MIN_RULE_SCORE": 8.0,
    "MIN_TREND_SLOPE": 0.0015,
    "TAKE_PROFIT_PCT": 2.5,
    "STOP_LOSS_PCT": -1.5,
}

grid = {
    "REGIME_SMA_PERIOD": [0, 50, 100, 200, 400],
}

keys = list(grid.keys())
combos = list(itertools.product(*grid.values()))
print(f"Running {len(combos)} regime-filter combinations on top of the best prior combo...\n")

results = []
for combo in combos:
    params = dict(base)
    params.update(dict(zip(keys, combo)))
    r = run_backtest(events, params=params)
    results.append((params, r))

results.sort(key=lambda pr: pr[1]["net_profit_pct"], reverse=True)

print(f"{'REGIME_SMA':>10} {'trades':>7} {'win%':>6} {'net%':>9} {'maxDD%':>8}")
for params, r in results:
    print(f"{params['REGIME_SMA_PERIOD']:>10} {r['total_trades']:>7} {r['win_rate_pct']:>6.1f} "
          f"{r['net_profit_pct']:>9.2f} {r['max_drawdown_pct']:>8.2f}")

best_params, best_r = results[0]
print(f"\nBest combo: {best_params}")
print(f"Net Profit: {best_r['net_profit_pct']:.3f}%  |  Trades: {best_r['total_trades']}  |  "
      f"Win Rate: {best_r['win_rate_pct']:.1f}%  |  Max DD: {best_r['max_drawdown_pct']:.2f}%")
print("Exit reasons:", best_r["exit_reasons"])
