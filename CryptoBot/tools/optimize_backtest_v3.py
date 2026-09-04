"""
Third-round sweep: push REGIME_SMA_PERIOD further out (the second-round
sweep showed net profit monotonically improving as the regime SMA lengthened:
0 -> -51.4%, 50 -> -50.5%, 100 -> -43.0%, 200 -> -36.6%, 400 -> -33.7%), and
jointly re-sweep MIN_RULE_SCORE/TAKE_PROFIT_PCT/STOP_LOSS_PCT now that the
regime filter changes which setups qualify.
"""
import itertools

from local_backtest_qc_port import load_events, run_backtest

events = load_events()

grid = {
    "REGIME_SMA_PERIOD": [400, 600, 800, 1000],
    "MIN_RULE_SCORE": [6.0, 7.0, 8.0],
    "TAKE_PROFIT_PCT": [2.0, 2.5, 3.0],
    "STOP_LOSS_PCT": [-1.0, -1.5],
    "MIN_TREND_SLOPE": [0.0015],
}

keys = list(grid.keys())
combos = list(itertools.product(*grid.values()))
print(f"Running {len(combos)} combinations...\n")

results = []
for combo in combos:
    params = dict(zip(keys, combo))
    r = run_backtest(events, params=params)
    results.append((params, r))

results.sort(key=lambda pr: pr[1]["net_profit_pct"], reverse=True)

print(f"{'REGIME':>7} {'SCORE':>6} {'TP%':>5} {'SL%':>5} {'trades':>7} {'win%':>6} {'net%':>9} {'maxDD%':>8}")
for params, r in results[:25]:
    print(f"{params['REGIME_SMA_PERIOD']:>7} {params['MIN_RULE_SCORE']:>6.1f} "
          f"{params['TAKE_PROFIT_PCT']:>5.1f} {params['STOP_LOSS_PCT']:>5.1f} "
          f"{r['total_trades']:>7} {r['win_rate_pct']:>6.1f} {r['net_profit_pct']:>9.2f} {r['max_drawdown_pct']:>8.2f}")

best_params, best_r = results[0]
print(f"\nBest combo: {best_params}")
print(f"Net Profit: {best_r['net_profit_pct']:.3f}%  |  Trades: {best_r['total_trades']}  |  "
      f"Win Rate: {best_r['win_rate_pct']:.1f}%  |  Avg Win: {best_r['avg_win_pct']:.3f}%  |  "
      f"Avg Loss: {best_r['avg_loss_pct']:.3f}%  |  Max DD: {best_r['max_drawdown_pct']:.2f}%")
print("Exit reasons:", best_r["exit_reasons"])
print("Trades per symbol:", best_r["trades_per_symbol"])
