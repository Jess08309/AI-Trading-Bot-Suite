"""
ARCHIVED (backtest consolidation, see CryptoBot/backtest/README.md).

Reason: parameter sweep on top of local_backtest_qc_port.py's rules-only
strategy (archived alongside this file). Closed investigation -- see
local_backtest_qc_port.py's docstring: a rules-only mechanical strategy
without CryptoBot's live ML confidence gate does not have a positive edge
on this data, no matter how it's tuned (best found across all sweep rounds:
-15.5% net over 2021-2026, still a net loser). Kept here unmodified for
reference/history. Not maintained, not run in CI.
============================================================================

Parameter sweep over CryptoBot/tools/local_backtest_qc_port.py's rules-only
strategy, using the same cached real Alpaca hourly bars. Tests whether a
higher entry bar (fewer, higher-quality trades) and a better reward:risk
ratio can turn the baseline -88.5% net loss into something less bad or
profitable.

Usage:
  python3 CryptoBot/tools/optimize_backtest.py
"""
import itertools

from local_backtest_qc_port import load_events, run_backtest

events = load_events()

grid = {
    "MIN_RULE_SCORE": [5.0, 6.0, 7.0, 8.0],
    "MIN_TREND_SLOPE": [0.0005, 0.001, 0.0015],
    "TAKE_PROFIT_PCT": [1.5, 2.0, 2.5],
    "STOP_LOSS_PCT": [-1.5, -1.0],
}

keys = list(grid.keys())
combos = list(itertools.product(*grid.values()))
print(f"Running {len(combos)} parameter combinations over {len(events)} events...\n")

results = []
for combo in combos:
    params = dict(zip(keys, combo))
    r = run_backtest(events, params=params)
    results.append((params, r))

results.sort(key=lambda pr: pr[1]["net_profit_pct"], reverse=True)

print(f"{'MIN_SCORE':>9} {'SLOPE':>8} {'TP%':>6} {'SL%':>6} {'trades':>7} {'win%':>6} {'net%':>9} {'maxDD%':>8}")
for params, r in results[:20]:
    print(f"{params['MIN_RULE_SCORE']:>9.1f} {params['MIN_TREND_SLOPE']:>8.4f} "
          f"{params['TAKE_PROFIT_PCT']:>6.1f} {params['STOP_LOSS_PCT']:>6.1f} "
          f"{r['total_trades']:>7} {r['win_rate_pct']:>6.1f} {r['net_profit_pct']:>9.2f} {r['max_drawdown_pct']:>8.2f}")

best_params, best_r = results[0]
print(f"\nBest combo: {best_params}")
print(f"Net Profit: {best_r['net_profit_pct']:.3f}%  |  Trades: {best_r['total_trades']}  |  "
      f"Win Rate: {best_r['win_rate_pct']:.1f}%  |  Max DD: {best_r['max_drawdown_pct']:.2f}%")
print("Exit reasons:", best_r["exit_reasons"])
