"""
Strategy backtest harness aligned with the live engine's core risk semantics.

Usage:
  python backtest_harness.py
  python backtest_harness.py --csv data/backtest/candidate_prices_daily.csv --symbol BTC-USD
  python backtest_harness.py --instrument futures
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime, timedelta


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "data", "backtest", "candidate_prices_daily.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "state", "strategy_backtest_report.json")


def _rsi(values, period=14):
    if len(values) < period + 1:
        return 50.0
    deltas = [values[i] - values[i - 1] for i in range(len(values) - period, len(values))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 1e-12
    rs = avg_gain / max(avg_loss, 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (value / peak) - 1.0 if peak > 0 else 0.0
        if drawdown < worst:
            worst = drawdown
    return worst * 100.0


def _slippage_multiplier(direction, is_entry, slippage_bps):
    slip = max(0.0, slippage_bps) / 10000.0
    if direction == "LONG":
        return 1 + slip if is_entry else 1 - slip
    return 1 - slip if is_entry else 1 + slip


def _load_price_series(csv_path, target_symbol=None):
    by_symbol = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get("symbol")
            if not symbol:
                continue
            if target_symbol and symbol != target_symbol:
                continue

            try:
                ts = row.get("timestamp", "")
                close = float(row.get("close", 0.0))
            except (TypeError, ValueError):
                continue

            by_symbol.setdefault(symbol, []).append((ts, close))

    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda x: x[0])

    return by_symbol


def _parse_iso_like_timestamp(ts: str):
    if not ts:
        return None
    value = ts.strip().replace("Z", "")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _filter_prices_by_date(prices, start_dt=None, end_dt=None):
    if not start_dt and not end_dt:
        return prices

    filtered = []
    for ts, close in prices:
        dt = _parse_iso_like_timestamp(ts)
        if dt is None:
            continue
        if start_dt and dt < start_dt:
            continue
        if end_dt and dt > end_dt:
            continue
        filtered.append((ts, close))
    return filtered


def _simulate_symbol(prices, initial_balance, max_position_pct, stop_loss_pct, take_profit_pct,
                     trailing_stop_pct, fee_rate, slippage_bps):
    equity = initial_balance
    equity_curve = [equity]
    trades = []
    position = None

    closes = [p[1] for p in prices]
    for idx in range(50, len(prices)):
        timestamp, raw_price = prices[idx]
        history = closes[: idx + 1]

        if position:
            fill_exit = raw_price * _slippage_multiplier(position["direction"], False, slippage_bps)
            if position["direction"] == "LONG":
                pnl_pct = (fill_exit - position["entry_price"]) / position["entry_price"] * 100.0
                if fill_exit > position["max_price"]:
                    position["max_price"] = fill_exit
                drawdown_from_peak = (position["max_price"] - fill_exit) / position["max_price"] * 100.0
                trailing_hit = drawdown_from_peak >= trailing_stop_pct and pnl_pct > 1.0
            else:
                pnl_pct = (position["entry_price"] - fill_exit) / position["entry_price"] * 100.0
                trailing_hit = False

            should_exit = pnl_pct <= stop_loss_pct or pnl_pct >= take_profit_pct or trailing_hit
            if should_exit:
                pnl_value = position["size"] * pnl_pct / 100.0
                exit_fee = position["size"] * fee_rate
                equity += pnl_value - exit_fee
                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": fill_exit,
                    "size": position["size"],
                    "pnl": pnl_value - exit_fee,
                    "pnl_pct": pnl_pct,
                })
                position = None
                equity_curve.append(equity)
                continue

        if not position:
            rsi_val = _rsi(history, period=14)
            trend = (history[-1] - history[-20]) / max(history[-20], 1e-12)

            direction = None
            if trend > 0.01 and rsi_val < 70:
                direction = "LONG"
            elif trend < -0.01 and rsi_val > 30:
                direction = "SHORT"

            if direction:
                size = equity * max_position_pct
                if size <= 0:
                    continue
                entry_price = raw_price * _slippage_multiplier(direction, True, slippage_bps)
                entry_fee = size * fee_rate
                equity -= entry_fee
                position = {
                    "direction": direction,
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "size": size,
                    "max_price": entry_price,
                }
                equity_curve.append(equity)

    if position:
        timestamp, raw_price = prices[-1]
        fill_exit = raw_price * _slippage_multiplier(position["direction"], False, slippage_bps)
        if position["direction"] == "LONG":
            pnl_pct = (fill_exit - position["entry_price"]) / position["entry_price"] * 100.0
        else:
            pnl_pct = (position["entry_price"] - fill_exit) / position["entry_price"] * 100.0
        pnl_value = position["size"] * pnl_pct / 100.0
        exit_fee = position["size"] * fee_rate
        equity += pnl_value - exit_fee
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": fill_exit,
            "size": position["size"],
            "pnl": pnl_value - exit_fee,
            "pnl_pct": pnl_pct,
        })
        equity_curve.append(equity)

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    returns = []
    running = initial_balance
    for pnl in pnls:
        returns.append(pnl / max(running, 1e-12))
        running += pnl

    mean_ret = sum(returns) / len(returns) if returns else 0.0
    stdev = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / len(returns)) if len(returns) > 1 else 0.0
    sharpe = (mean_ret / stdev) * math.sqrt(252) if stdev > 0 else 0.0

    return {
        "trades": len(trades),
        "total_pnl": sum(pnls),
        "win_rate_pct": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) < 0 else 0.0,
        "sharpe": sharpe,
        "max_drawdown_pct": _max_drawdown(equity_curve),
        "ending_balance": equity,
        "sample_trades": trades[-10:],
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest harness using engine-like risk semantics")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV with timestamp,symbol,close columns")
    parser.add_argument("--symbol", help="Run for one symbol")
    parser.add_argument("--initial-balance", type=float, default=5000.0)
    parser.add_argument("--instrument", choices=["spot", "futures"], default="spot")
    parser.add_argument("--max-position-pct", type=float, default=0.10)
    parser.add_argument("--stop-loss", type=float, default=-2.0)
    parser.add_argument("--take-profit", type=float, default=4.0)
    parser.add_argument("--trailing-stop", type=float, default=1.5)
    parser.add_argument("--start-date", help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date inclusive (YYYY-MM-DD)")
    parser.add_argument("--months", type=int, help="Use only the last N months from input data")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Input CSV not found: {args.csv}")
        return

    slippage_bps = 8.0 if args.instrument == "futures" else 5.0
    stop_loss = -3.0 if args.instrument == "futures" else args.stop_loss

    series = _load_price_series(args.csv, args.symbol)
    if not series:
        print("No symbol data found in CSV")
        return

    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    if args.months and args.months > 0:
        reference_end = end_dt
        if reference_end is None:
            all_dates = []
            for prices in series.values():
                for ts, _ in prices:
                    dt = _parse_iso_like_timestamp(ts)
                    if dt:
                        all_dates.append(dt)
            reference_end = max(all_dates) if all_dates else None
        if reference_end:
            start_dt = reference_end - timedelta(days=30 * args.months)

    filtered_series = {}
    for symbol, prices in series.items():
        filtered_prices = _filter_prices_by_date(prices, start_dt=start_dt, end_dt=end_dt)
        if filtered_prices:
            filtered_series[symbol] = filtered_prices

    series = filtered_series
    if not series:
        print("No symbol data left after date filtering")
        return

    report = {
        "generated_at": datetime.now().isoformat(),
        "input_csv": args.csv,
        "instrument": args.instrument,
        "window": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "months": args.months,
        },
        "symbols": {},
    }

    for symbol, prices in series.items():
        if len(prices) < 60:
            continue
        metrics = _simulate_symbol(
            prices,
            initial_balance=args.initial_balance,
            max_position_pct=args.max_position_pct,
            stop_loss_pct=stop_loss,
            take_profit_pct=args.take_profit,
            trailing_stop_pct=args.trailing_stop,
            fee_rate=0.001,
            slippage_bps=slippage_bps,
        )
        report["symbols"][symbol] = metrics

    aggregate_pnl = sum(m["total_pnl"] for m in report["symbols"].values())
    aggregate_trades = sum(m["trades"] for m in report["symbols"].values())
    report["aggregate"] = {
        "symbols_tested": len(report["symbols"]),
        "trades": aggregate_trades,
        "total_pnl": aggregate_pnl,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 64)
    print("STRATEGY BACKTEST HARNESS")
    print("=" * 64)
    print(f"Symbols tested: {len(report['symbols'])}")
    print(f"Trades:         {aggregate_trades}")
    print(f"Aggregate P/L:  ${aggregate_pnl:+,.2f}")
    print(f"Saved report:   {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
