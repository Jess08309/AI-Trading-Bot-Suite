import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

CLOSE_SIDES = {"SELL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT"}


def resolve_data_root() -> Path:
    """Pick the data root to watch: prefer local, else known external copy."""
    local = Path(__file__).parent / "data"
    external = Path("D:/042021/CryptoBot/data")
    if (local / "history" / "trade_history.csv").exists():
        return local
    if (external / "history" / "trade_history.csv").exists():
        return external
    raise FileNotFoundError("trade_history.csv not found in local or external data paths")


def load_trades(data_root: Path) -> List[Dict]:
    trades = []
    trade_file = data_root / "history" / "trade_history.csv"
    if not trade_file.exists():
        return trades
    with trade_file.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            time_str, symbol, side, price, amount, pnl = row[:6]
            try:
                ts = datetime.fromisoformat(time_str)
                trades.append(
                    {
                        "time": ts,
                        "symbol": symbol,
                        "side": side,
                        "price": float(price),
                        "amount": float(amount),
                        "pnl": float(pnl),
                    }
                )
            except Exception:
                continue
    return trades


def load_balances(data_root: Path) -> Tuple[float, float]:
    bal_file = data_root / "state" / "paper_balances.json"
    if not bal_file.exists():
        return 0.0, 0.0
    try:
        with bal_file.open() as f:
            data = json.load(f)
            return float(data.get("spot", 0.0)), float(data.get("futures", 0.0))
    except Exception:
        return 0.0, 0.0


def summarize(trades: List[Dict], last_hours: int = 24) -> Dict:
    closing = [t for t in trades if t["side"] in CLOSE_SIDES]
    wins = [t for t in closing if t["pnl"] > 0]
    losses = [t for t in closing if t["pnl"] < 0]
    neutrals = [t for t in closing if t["pnl"] == 0]

    now = datetime.utcnow()
    cutoff = now - timedelta(hours=last_hours)
    recent = [t for t in closing if t["time"] >= cutoff]
    recent_pnl = sum(t["pnl"] for t in recent)
    recent_wins = sum(1 for t in recent if t["pnl"] > 0)
    recent_losses = sum(1 for t in recent if t["pnl"] < 0)

    return {
        "total_closed": len(closing),
        "wins": len(wins),
        "losses": len(losses),
        "neutrals": len(neutrals),
        "win_rate": (len(wins) / len(closing) * 100) if closing else 0.0,
        "avg_pnl": (sum(t["pnl"] for t in closing) / len(closing)) if closing else 0.0,
        "realized": sum(t["pnl"] for t in closing),
        "recent_closed": len(recent),
        "recent_pnl": recent_pnl,
        "recent_win_rate": (recent_wins / len(recent) * 100) if recent else 0.0,
        "last10": closing[-10:],
    }


def print_summary(data_root: Path, last_hours: int = 24) -> None:
    trades = load_trades(data_root)
    if not trades:
        print("[summary] No trades found yet")
        return
    spot_cash, fut_cash = load_balances(data_root)
    stats = summarize(trades, last_hours=last_hours)
    print("=" * 72)
    print(f"[cycle] {datetime.now().isoformat(timespec='seconds')} | data_root={data_root}")
    print(
        f"Closed: {stats['total_closed']} | W {stats['wins']} L {stats['losses']} N {stats['neutrals']} | "
        f"Win% {stats['win_rate']:.1f} | Avg P/L ${stats['avg_pnl']:.2f} | Realized ${stats['realized']:+.2f}"
    )
    print(
        f"Recent {last_hours}h: closed {stats['recent_closed']} | Win% {stats['recent_win_rate']:.1f} | "
        f"P/L ${stats['recent_pnl']:+.2f}"
    )
    if spot_cash or fut_cash:
        print(f"Balances: spot ${spot_cash:,.2f} | futures ${fut_cash:,.2f}")
    print("Last 10 closes:")
    for t in stats["last10"]:
        tag = "WIN " if t["pnl"] > 0 else ("LOSS" if t["pnl"] < 0 else "FLAT")
        print(
            f"  {t['time'].isoformat(timespec='seconds')} {t['symbol']:12s} "
            f"{t['side']:10s} P/L ${t['pnl']:+.2f} [{tag}]"
        )
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Watch bot progress and print a summary every N cycles (10 by default)."
    )
    parser.add_argument(
        "--cycle-seconds",
        type=int,
        default=60,
        help="Seconds per bot cycle (default 60).",
    )
    parser.add_argument(
        "--interval-cycles",
        type=int,
        default=10,
        help="Print summary every N cycles (default 10).",
    )
    parser.add_argument(
        "--recent-hours",
        type=int,
        default=24,
        help="Window for recent P/L stats (default 24h).",
    )
    args = parser.parse_args()

    data_root = resolve_data_root()
    tick = 0
    try:
        while True:
            if tick % args.interval_cycles == 0:
                print_summary(data_root, last_hours=args.recent_hours)
            tick += 1
            time.sleep(args.cycle_seconds)
    except KeyboardInterrupt:
        print("\n[summary] watcher stopped")


if __name__ == "__main__":
    main()
