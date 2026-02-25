"""
Threshold Tuning Script - Test new parameters against the backtester.

Compares BASELINE (current engine thresholds) vs TUNED (proposed improvements)
based on backtest analysis findings:
  - Stop losses fire 44% of spot exits (avg -2.2%) and 49% of futures exits
  - Win rate too low (40% spot, 51% futures)
  - Take-profits rarely trigger for spot (7%)
  - Risk/reward is negative (avg loss >> avg win)

PROPOSED FIXES:
  Spot: Raise buy score, widen SL, lower TP, tighter trailing
  Futures: Tighter entry thresholds, lower TP, add trailing stop
"""

import os
import sys
import csv
import time
from collections import defaultdict

# Must run from project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from cryptotrades.utils.backtester import (
    SpotBacktester, FuturesBacktester, aggregate_to_candles
)
from cryptotrades.utils.market_predictor import MarketPredictor


def load_prices(csv_path):
    """Load price history from CSV."""
    spot_prices = defaultdict(list)
    futures_prices = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["symbol"]
            price = float(row["price"])
            if symbol.startswith("PI_"):
                futures_prices[symbol].append(price)
            else:
                spot_prices[symbol].append(price)
    return dict(spot_prices), dict(futures_prices)


def run_spot_test(predictor, spot_prices, label, **kwargs):
    """Run spot backtest with given parameters."""
    bt = SpotBacktester(predictor, **kwargs)
    results = []
    for symbol, prices in sorted(spot_prices.items()):
        r = bt.run(symbol, prices)
        results.append(r)
    return results


def run_futures_test(predictor, futures_prices, label,
                     long_threshold=0.55, short_threshold=0.45,
                     take_profit=5.0, stop_loss=-3.0,
                     trailing_stop=0.6, trailing_activate=0.3,
                     **spot_kwargs):
    """Run futures backtest with given parameters."""
    bt = FuturesBacktester(
        predictor,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        take_profit=take_profit,
        stop_loss=stop_loss,
        trailing_stop=trailing_stop,
        trailing_activate=trailing_activate,
        **spot_kwargs,
    )
    results = []
    for symbol, prices in sorted(futures_prices.items()):
        r = bt.run(symbol, prices)
        results.append(r)
    return results


def print_comparison(label_a, results_a, label_b, results_b, side="spot"):
    """Print side-by-side comparison of two backtest runs."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: {label_a} vs {label_b}  ({side.upper()})")
    print(f"{'='*80}")
    print(f"{'Symbol':<20} {'Trades':>7} {'Win%':>7} {'Return%':>9} | "
          f"{'Trades':>7} {'Win%':>7} {'Return%':>9} | {'Delta':>8}")
    print(f"{'':<20} {label_a:>25} | {label_b:>25} |")
    print("-" * 80)

    total_a = 0.0
    total_b = 0.0
    for ra, rb in zip(results_a, results_b):
        sym = ra.symbol[:18]
        total_a += ra.total_return_usd
        total_b += rb.total_return_usd
        delta = rb.total_return_pct - ra.total_return_pct
        print(f"{sym:<20} {ra.num_trades:>7} {ra.win_rate:>6.1%} {ra.total_return_pct:>+8.2f}% | "
              f"{rb.num_trades:>7} {rb.win_rate:>6.1%} {rb.total_return_pct:>+8.2f}% | "
              f"{delta:>+7.2f}%")

    print("-" * 80)
    delta_total = total_b - total_a
    print(f"{'TOTAL':<20} {'':>7} {'':>7} ${total_a:>+8.2f} | "
          f"{'':>7} {'':>7} ${total_b:>+8.2f} | ${delta_total:>+7.2f}")
    print(f"{'='*80}")

    # Exit reason comparison
    print(f"\n  Exit Breakdown Comparison:")
    reasons_a = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    reasons_b = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for r in results_a:
        for t in r.trades:
            reasons_a[t.exit_reason]["count"] += 1
            reasons_a[t.exit_reason]["pnl"] += t.pnl_pct
    for r in results_b:
        for t in r.trades:
            reasons_b[t.exit_reason]["count"] += 1
            reasons_b[t.exit_reason]["pnl"] += t.pnl_pct

    all_reasons = sorted(set(list(reasons_a.keys()) + list(reasons_b.keys())))
    print(f"  {'Reason':<22} {label_a:>20} | {label_b:>20}")
    print(f"  {'':<22} {'count':>8} {'avg%':>10} | {'count':>8} {'avg%':>10}")
    for reason in all_reasons:
        ca = reasons_a[reason]["count"]
        cb = reasons_b[reason]["count"]
        pa = reasons_a[reason]["pnl"] / ca if ca else 0
        pb = reasons_b[reason]["pnl"] / cb if cb else 0
        print(f"  {reason:<22} {ca:>8} {pa:>+9.2f}% | {cb:>8} {pb:>+9.2f}%")
    print()


def main():
    print("Loading price data...")
    csv_path = os.path.join("cryptotrades", "price_history.csv")
    spot_prices, futures_prices = load_prices(csv_path)
    print(f"  Spot: {len(spot_prices)} symbols")
    print(f"  Futures: {len(futures_prices)} symbols")

    print("Loading ML model...")
    predictor = MarketPredictor()
    predictor.load_model()

    # ============================================================
    # SPOT TESTS
    # ============================================================
    print("\n" + "="*60)
    print("  RUNNING SPOT BACKTESTS")
    print("="*60)

    # BASELINE (current engine values)
    print("\n[1/2] Running BASELINE spot...")
    t0 = time.time()
    baseline_spot = run_spot_test(
        predictor, spot_prices, "BASELINE",
        candle_size=10,
        buy_score_threshold=4,
        take_profit_high=3.0,
        take_profit_low=2.0,
        stop_loss_high=-2.0,
        stop_loss_low=-1.5,
        trailing_stop=0.8,
        trailing_activate=0.5,
        ensemble_buy_threshold=0.52,
        ensemble_sell_threshold=0.40,
    )
    print(f"  Done in {time.time()-t0:.0f}s")

    # TUNED (proposed improvements)
    print("\n[2/2] Running TUNED spot...")
    t0 = time.time()
    tuned_spot = run_spot_test(
        predictor, spot_prices, "TUNED",
        candle_size=10,
        buy_score_threshold=6,        # was 4: reject weak entries
        take_profit_high=2.0,         # was 3.0: capture profits earlier
        take_profit_low=1.5,          # was 2.0: capture profits earlier
        stop_loss_high=-3.0,          # was -2.0: give strong signals room
        stop_loss_low=-2.5,           # was -1.5: give trades room
        trailing_stop=0.5,            # was 0.8: tighter trailing
        trailing_activate=0.3,        # was 0.5: activate protection sooner
        ensemble_buy_threshold=0.55,  # was 0.52: slightly stricter
        ensemble_sell_threshold=0.42, # was 0.40: slightly stricter
    )
    print(f"  Done in {time.time()-t0:.0f}s")

    print_comparison("BASELINE", baseline_spot, "TUNED", tuned_spot, "spot")

    # ============================================================
    # FUTURES TESTS
    # ============================================================
    print("\n" + "="*60)
    print("  RUNNING FUTURES BACKTESTS")
    print("="*60)

    # BASELINE (matching live engine defaults)
    print("\n[1/2] Running BASELINE futures...")
    t0 = time.time()
    baseline_futures = run_futures_test(
        predictor, futures_prices, "BASELINE",
        candle_size=10,
        long_threshold=0.55,
        short_threshold=0.45,
        take_profit=5.0,
        stop_loss=-3.0,
        trailing_stop=0.6,
        trailing_activate=0.3,
    )
    print(f"  Done in {time.time()-t0:.0f}s")

    # TUNED
    print("\n[2/2] Running TUNED futures...")
    t0 = time.time()
    tuned_futures = run_futures_test(
        predictor, futures_prices, "TUNED",
        candle_size=10,
        long_threshold=0.60,          # was 0.55: more selective entries
        short_threshold=0.40,         # was 0.45: more selective entries
        take_profit=4.0,              # was 5.0: capture profits earlier
        stop_loss=-2.5,               # was -3.0: tighter to limit damage
        trailing_stop=0.5,            # was 0.6: tighter trailing
        trailing_activate=0.2,        # was 0.3: protect sooner
    )
    print(f"  Done in {time.time()-t0:.0f}s")

    print_comparison("BASELINE", baseline_futures, "TUNED", tuned_futures, "futures")

    # ============================================================
    # COMBINED SUMMARY
    # ============================================================
    base_spot_pnl = sum(r.total_return_usd for r in baseline_spot)
    tune_spot_pnl = sum(r.total_return_usd for r in tuned_spot)
    base_fut_pnl = sum(r.total_return_usd for r in baseline_futures)
    tune_fut_pnl = sum(r.total_return_usd for r in tuned_futures)

    print(f"\n{'='*60}")
    print(f"  COMBINED IMPACT")
    print(f"{'='*60}")
    print(f"  {'':20} {'BASELINE':>12} {'TUNED':>12} {'DELTA':>12}")
    print(f"  {'Spot P/L':20} ${base_spot_pnl:>+10.2f} ${tune_spot_pnl:>+10.2f} ${tune_spot_pnl-base_spot_pnl:>+10.2f}")
    print(f"  {'Futures P/L':20} ${base_fut_pnl:>+10.2f} ${tune_fut_pnl:>+10.2f} ${tune_fut_pnl-base_fut_pnl:>+10.2f}")
    print(f"  {'TOTAL':20} ${base_spot_pnl+base_fut_pnl:>+10.2f} ${tune_spot_pnl+tune_fut_pnl:>+10.2f} ${(tune_spot_pnl+tune_fut_pnl)-(base_spot_pnl+base_fut_pnl):>+10.2f}")
    print(f"{'='*60}")

    if tune_spot_pnl + tune_fut_pnl > base_spot_pnl + base_fut_pnl:
        print("  [OK] TUNED thresholds IMPROVE overall performance!")
        print("  Apply these to trading_engine.py and restart bot.")
    else:
        print("  [!!] TUNED thresholds did NOT improve. Need different values.")

    # Write results to file for easy reading
    with open("tune_results.txt", "w") as f:
        f.write("TUNING COMPLETE\n")
        f.write(f"Spot:    BASELINE ${base_spot_pnl:+.2f} -> TUNED ${tune_spot_pnl:+.2f} (delta ${tune_spot_pnl-base_spot_pnl:+.2f})\n")
        f.write(f"Futures: BASELINE ${base_fut_pnl:+.2f} -> TUNED ${tune_fut_pnl:+.2f} (delta ${tune_fut_pnl-base_fut_pnl:+.2f})\n")
        f.write(f"Total:   BASELINE ${base_spot_pnl+base_fut_pnl:+.2f} -> TUNED ${tune_spot_pnl+tune_fut_pnl:+.2f} (delta ${(tune_spot_pnl+tune_fut_pnl)-(base_spot_pnl+base_fut_pnl):+.2f})\n")
        f.write(f"\nSpot per symbol:\n")
        for rb, rt in zip(baseline_spot, tuned_spot):
            f.write(f"  {rb.symbol:16s}  BASE: {rb.num_trades:3d}t {rb.win_rate:.0%} {rb.total_return_pct:+.2f}%  TUNED: {rt.num_trades:3d}t {rt.win_rate:.0%} {rt.total_return_pct:+.2f}%  delta: {rt.total_return_pct-rb.total_return_pct:+.2f}%\n")
        f.write(f"\nFutures per symbol:\n")
        for rb, rt in zip(baseline_futures, tuned_futures):
            f.write(f"  {rb.symbol:16s}  BASE: {rb.num_trades:3d}t {rb.win_rate:.0%} {rb.total_return_pct:+.2f}%  TUNED: {rt.num_trades:3d}t {rt.win_rate:.0%} {rt.total_return_pct:+.2f}%  delta: {rt.total_return_pct-rb.total_return_pct:+.2f}%\n")
    print("\nResults written to tune_results.txt")


if __name__ == "__main__":
    main()
