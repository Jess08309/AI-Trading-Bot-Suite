import json
import os
import time
from prometheus_client import Gauge, start_http_server


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(BASE_DIR, "data", "state")
PERF_REPORT = os.path.join(STATE_DIR, "performance_report.json")
BACKTEST_REPORT = os.path.join(STATE_DIR, "strategy_backtest_report.json")
BALANCES_FILE = os.path.join(STATE_DIR, "paper_balances.json")
POSITIONS_FILE = os.path.join(STATE_DIR, "positions.json")


g_total_pnl = Gauge("trading_total_pnl_usd", "Total PnL from performance report (USD)")
g_win_rate = Gauge("trading_win_rate_pct", "Win rate percentage")
g_sharpe = Gauge("trading_sharpe_ratio", "Sharpe ratio")
g_drawdown = Gauge("trading_max_drawdown_pct", "Max drawdown percentage")
g_backtest_pnl = Gauge("trading_backtest_total_pnl_usd", "Backtest aggregate PnL (USD)")
g_backtest_trades = Gauge("trading_backtest_trades_total", "Backtest aggregate trades")
g_balance_spot = Gauge("trading_balance_spot_usd", "Spot paper balance")
g_balance_futures = Gauge("trading_balance_futures_usd", "Futures paper balance")
g_open_positions = Gauge("trading_open_positions", "Current open positions")



def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}



def collect_once():
    perf = _load_json(PERF_REPORT)
    backtest = _load_json(BACKTEST_REPORT)
    balances = _load_json(BALANCES_FILE)
    positions = _load_json(POSITIONS_FILE)

    g_total_pnl.set(float(perf.get("total_pnl", 0.0) or 0.0))
    g_win_rate.set(float(perf.get("win_rate_pct", 0.0) or 0.0))
    g_sharpe.set(float(perf.get("sharpe", 0.0) or 0.0))
    g_drawdown.set(float(perf.get("max_drawdown_pct", 0.0) or 0.0))

    agg = backtest.get("aggregate", {}) if isinstance(backtest, dict) else {}
    g_backtest_pnl.set(float(agg.get("total_pnl", 0.0) or 0.0))
    g_backtest_trades.set(float(agg.get("trades", 0.0) or 0.0))

    g_balance_spot.set(float(balances.get("spot", 0.0) or 0.0))
    g_balance_futures.set(float(balances.get("futures", 0.0) or 0.0))

    open_positions = len(positions) if isinstance(positions, dict) else 0
    g_open_positions.set(float(open_positions))



def main():
    port = int(os.getenv("EXPORTER_PORT", "9108"))
    interval = int(os.getenv("EXPORTER_INTERVAL_SEC", "15"))
    start_http_server(port)
    print(f"state_exporter listening on :{port}")

    while True:
        collect_once()
        time.sleep(interval)


if __name__ == "__main__":
    main()
