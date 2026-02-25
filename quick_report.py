import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRADES_FILES = [
    os.path.join(BASE_DIR, "data", "trades.csv"),
    os.path.join(BASE_DIR, "data", "history", "trade_history.csv"),
]
REPORT_PATH = os.path.join(BASE_DIR, "data", "state", "performance_report.json")


def _read_trades(path: str):
    trades = []
    with open(path, "r", encoding="utf-8") as f:
        sample = f.readline()
        f.seek(0)

        if "timestamp" in sample and "pnl_usd" in sample:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    trades.append(
                        {
                            "time": row.get("timestamp", ""),
                            "symbol": row.get("symbol", ""),
                            "side": row.get("direction", ""),
                            "pnl": float(row.get("pnl_usd", 0.0)),
                        }
                    )
                except (TypeError, ValueError):
                    continue
        else:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                try:
                    trades.append(
                        {
                            "time": parts[0],
                            "symbol": parts[1],
                            "side": parts[2],
                            "pnl": float(parts[5]),
                        }
                    )
                except ValueError:
                    continue

    return sorted(trades, key=lambda x: x["time"])


def _safe_std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _max_drawdown_pct(equity_curve):
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (equity / peak) - 1.0 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd * 100.0


def build_report(trades, initial_balance):
    closes = [t for t in trades if t["side"] in ("LONG", "SHORT", "SELL", "CLOSE")]
    pnls = [t["pnl"] for t in closes]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    total_trades = len(pnls)
    win_rate = (len(wins) / total_trades * 100.0) if total_trades else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = (win_rate / 100.0 * avg_win) + ((1 - win_rate / 100.0) * avg_loss)
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) < 0 else 0.0

    equity = [initial_balance]
    for p in pnls:
        equity.append(equity[-1] + p)
    returns = [pnls[i] / max(equity[i], 1e-12) for i in range(len(pnls))] if pnls else []

    mean_ret = sum(returns) / len(returns) if returns else 0.0
    std_ret = _safe_std(returns)
    sharpe = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0.0

    rolling_windows = [20, 50]
    rolling = {}
    for window in rolling_windows:
        if len(pnls) >= window:
            sample = pnls[-window:]
            rolling[f"last_{window}"] = {
                "pnl": sum(sample),
                "win_rate_pct": (sum(1 for p in sample if p > 0) / window) * 100.0,
                "avg_pnl": sum(sample) / window,
            }

    by_day = defaultdict(float)
    by_month = defaultdict(float)
    by_symbol = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0})
    for t in closes:
        by_day[t["time"][:10]] += t["pnl"]
        by_month[t["time"][:7]] += t["pnl"]
        symbol = t["symbol"] or "UNKNOWN"
        by_symbol[symbol]["trades"] += 1
        by_symbol[symbol]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_symbol[symbol]["wins"] += 1
        else:
            by_symbol[symbol]["losses"] += 1

    symbol_summary = {}
    for symbol, stats in by_symbol.items():
        trades_n = stats["trades"]
        symbol_summary[symbol] = {
            "trades": trades_n,
            "pnl": stats["pnl"],
            "win_rate_pct": (stats["wins"] / trades_n * 100.0) if trades_n else 0.0,
        }

    best_symbol = None
    worst_symbol = None
    if symbol_summary:
        ranked = sorted(symbol_summary.items(), key=lambda x: x[1]["pnl"], reverse=True)
        best_symbol = {"symbol": ranked[0][0], **ranked[0][1]}
        worst_symbol = {"symbol": ranked[-1][0], **ranked[-1][1]}

    return {
        "generated_at": datetime.now().isoformat(),
        "initial_balance": initial_balance,
        "ending_balance": equity[-1] if equity else initial_balance,
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown_pct": _max_drawdown_pct(equity),
        "rolling": rolling,
        "daily_pnl": dict(sorted(by_day.items())),
        "monthly_pnl": dict(sorted(by_month.items())),
        "symbols": symbol_summary,
        "best_symbol": best_symbol,
        "worst_symbol": worst_symbol,
    }


def _resolve_trades_path(explicit):
    if explicit:
        return explicit
    for candidate in DEFAULT_TRADES_FILES:
        if os.path.exists(candidate):
            return candidate
    return ""


def main():
    parser = argparse.ArgumentParser(description="Generate strategy performance dashboard metrics")
    parser.add_argument("--trades-file", help="Path to trades csv")
    parser.add_argument("--initial-balance", type=float, default=5000.0)
    args = parser.parse_args()

    trades_path = _resolve_trades_path(args.trades_file)
    if not trades_path or not os.path.exists(trades_path):
        print("No trades file found. Checked data/trades.csv and data/history/trade_history.csv")
        return

    trades = _read_trades(trades_path)
    report = build_report(trades, args.initial_balance)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 64)
    print("TRADING PERFORMANCE DASHBOARD")
    print("=" * 64)
    print(f"Trades file:   {trades_path}")
    print(f"Closed trades: {report['total_trades']}")
    print(f"Total P/L:     ${report['total_pnl']:+,.2f}")
    print(f"Win rate:      {report['win_rate_pct']:.1f}%")
    print(f"Expectancy:    ${report['expectancy']:+.4f} per trade")
    print(f"Profit factor: {report['profit_factor']:.2f}")
    print(f"Sharpe:        {report['sharpe']:.3f}")
    print(f"Max drawdown:  {report['max_drawdown_pct']:.2f}%")
    print(f"Ending bal:    ${report['ending_balance']:,.2f}")
    if report.get("best_symbol"):
        print(f"Best symbol:   {report['best_symbol']['symbol']} (${report['best_symbol']['pnl']:+,.2f})")
    if report.get("worst_symbol"):
        print(f"Worst symbol:  {report['worst_symbol']['symbol']} (${report['worst_symbol']['pnl']:+,.2f})")
    print(f"Saved report:  {REPORT_PATH}")


if __name__ == "__main__":
    main()
