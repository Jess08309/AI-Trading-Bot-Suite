"""
ARCHIVED (backtest consolidation, see CryptoBot/backtest/README.md).

Reason: round-4 (final) sweep on top of local_backtest_qc_port.py's
rules-only strategy (archived alongside this file). Closed investigation --
a rules-only mechanical strategy without CryptoBot's live ML confidence gate
does not have a positive edge on this data, no matter how it's tuned (best
found here: -15.5% net over 2021-2026, still a net loser). Kept here
unmodified for reference/history. Not maintained, not run in CI.
============================================================================

Fourth-round sweep: push REGIME_SMA_PERIOD even further (1000 was still the
best in round 3, -28.46% net) to see if the trend continues, plateaus, or
reverses. Fixed at the round-3 winning entry/exit combo.
"""
from local_backtest_qc_port import load_events, run_backtest

events = load_events()

base = {
    "MIN_RULE_SCORE": 8.0,
    "MIN_TREND_SLOPE": 0.0015,
    "TAKE_PROFIT_PCT": 2.5,
    "STOP_LOSS_PCT": -1.0,
}

print(f"{'REGIME':>7} {'trades':>7} {'win%':>6} {'net%':>9} {'maxDD%':>8}")
results = []
for regime in [1000, 1500, 2000, 3000, 4000]:
    params = dict(base)
    params["REGIME_SMA_PERIOD"] = regime
    r = run_backtest(events, params=params)
    results.append((regime, r))
    print(f"{regime:>7} {r['total_trades']:>7} {r['win_rate_pct']:>6.1f} "
          f"{r['net_profit_pct']:>9.2f} {r['max_drawdown_pct']:>8.2f}")

best_regime, best_r = max(results, key=lambda rr: rr[1]["net_profit_pct"])
print(f"\nBest regime period: {best_regime}")
print(f"Net Profit: {best_r['net_profit_pct']:.3f}%  |  Trades: {best_r['total_trades']}  |  "
      f"Win Rate: {best_r['win_rate_pct']:.1f}%  |  Avg Win: {best_r['avg_win_pct']:.3f}%  |  "
      f"Avg Loss: {best_r['avg_loss_pct']:.3f}%  |  Max DD: {best_r['max_drawdown_pct']:.2f}%")
print("Exit reasons:", best_r["exit_reasons"])
