from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _find_startup_timestamp(log_path: Path) -> datetime:
    startup_ts = None
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Paper trading started" in line:
                    try:
                        ts_str = line.split(" ")[0] + " " + line.split(" ")[1]
                        startup_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    except Exception:
                        continue
    return startup_ts or datetime.fromtimestamp(0, tz=timezone.utc)


def _compute_pnl_since_startup(trades: pd.DataFrame) -> pd.DataFrame:
    pnl_records = []
    for symbol in trades["symbol"].unique():
        sym_trades = trades[trades["symbol"] == symbol]
        buys = []
        for _, row in sym_trades.iterrows():
            if row["side"] == "BUY":
                buys.append({"amount": float(row["amount"]), "price": float(row["price"])})
            elif row["side"] == "SELL":
                sell_amt = float(row["amount"])
                sell_price = float(row["price"])
                realized = 0.0
                while sell_amt > 0 and buys:
                    buy = buys[0]
                    match_amt = min(sell_amt, buy["amount"])
                    realized += (sell_price - buy["price"]) * match_amt
                    buy["amount"] -= match_amt
                    sell_amt -= match_amt
                    if buy["amount"] <= 0:
                        buys.pop(0)
                pnl_records.append(
                    {
                        "symbol": symbol,
                        "market": "FUTURES" if symbol.startswith("PI_") else "SPOT",
                        "pnl": realized,
                        "timestamp": row["timestamp"],
                    }
                )
    return pd.DataFrame(pnl_records)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    bot_dir = base_dir / "cryptotrades"
    trade_path = bot_dir / "trade_history.csv"
    log_path = bot_dir / "bot.log"

    startup_ts = _find_startup_timestamp(log_path)

    trades = pd.read_csv(trade_path)
    trades = trades[trades["side"].isin(["BUY", "SELL"])]
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    trades = trades[trades["timestamp"] >= startup_ts].sort_values("timestamp")
    trades["market"] = trades["symbol"].astype(str).str.startswith("PI_").map({True: "FUTURES", False: "SPOT"})

    pnl_df = _compute_pnl_since_startup(trades)

    total_trades = len(trades)
    total_sells = len(pnl_df)
    wins = int((pnl_df["pnl"] > 0).sum()) if not pnl_df.empty else 0
    losses = int((pnl_df["pnl"] < 0).sum()) if not pnl_df.empty else 0
    total_pnl = float(pnl_df["pnl"].sum()) if not pnl_df.empty else 0.0

    if len(trades) > 0:
        last_time = trades["timestamp"].max()
        first_time = trades["timestamp"].min()
        duration_hours = (last_time - first_time).total_seconds() / 3600
        trades_per_hour = total_trades / duration_hours if duration_hours > 0 else float(total_trades)
    else:
        trades_per_hour = 0.0

    print(f"Startup timestamp (UTC): {startup_ts.isoformat()}")
    print(f"Total trades since startup: {total_trades}")
    print(f"Total sells since startup: {total_sells}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Total PNL: ${total_pnl:.2f}")
    print(f"Trading frequency: {trades_per_hour:.2f} trades/hour")

    if not trades.empty:
        market_counts = trades.groupby(["market", "side"]).size().unstack(fill_value=0)
        market_counts["total"] = market_counts.sum(axis=1)
        print("\nTrades by market:")
        print(market_counts.to_string())

    if not pnl_df.empty:
        market_pnl = pnl_df.groupby("market")["pnl"].agg(["count", "sum", "mean", "median", "max", "min"])
        print("\nPNL by market:")
        print(market_pnl.to_string())

        symbol_pnl = pnl_df.groupby("symbol")["pnl"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)
        print("\nTop symbols by PNL:")
        print(symbol_pnl.head(10).to_string())


if __name__ == "__main__":
    main()
