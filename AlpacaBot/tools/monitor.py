"""AlpacaBot monitoring script

Runs once, snapshots account and positions, writes JSON to `data/monitor/` and
alerts if thresholds are hit. Designed to be run from cron or systemd timer.

Usage:
  python tools/monitor.py

Environment:
  ALPACA_API_KEY, ALPACA_API_SECRET (or .env in bot dir)
  ALPACA_PAPER (optional)
  ALPACA_MONITOR_WEBHOOK (optional) - HTTP webhook to POST alert JSON
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.api_client import AlpacaAPI

log = logging.getLogger("alpacabot.monitor")


def ensure_dirs(base: Path):
    (base / "data" / "monitor").mkdir(parents=True, exist_ok=True)


def write_snapshot(base: Path, data: Dict[str, Any]):
    path = base / "data" / "monitor"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = path / f"snapshot_{ts}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    # also update latest
    latest = path / "latest.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return out


def load_latest(base: Path) -> Dict[str, Any]:
    p = base / "data" / "monitor" / "latest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compute_allocation_usage(account: Dict[str, Any], positions: list, cfg: Config) -> Dict[str, Any]:
    portfolio = account.get("portfolio_value", 0.0)
    allocated = portfolio * cfg.ALLOCATION_PCT
    pos_value = sum(p.get("market_value", 0.0) for p in positions)
    usage_pct = pos_value / allocated if allocated > 0 else 0.0
    return {
        "portfolio_value": portfolio,
        "allocated_slice": allocated,
        "positions_market_value": pos_value,
        "usage_pct": usage_pct,
    }


def send_webhook(url: str, payload: Dict[str, Any]):
    try:
        import requests
        resp = requests.post(url, json=payload, timeout=10)
        log.info(f"Webhook sent, status={resp.status_code}")
        return resp.status_code
    except Exception as e:
        log.warning(f"Failed to send webhook: {e}")
        return None


def main():
    base = Path(os.getcwd())
    cfg = Config()
    ensure_dirs(base)

    # init logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    api = AlpacaAPI(cfg)
    if not api.connect():
        log.error("Alpaca connect failed — aborting monitor run")
        return 2

    account = api.get_account()
    positions = api.get_positions()

    latest = load_latest(base)
    prev_equity = latest.get("account", {}).get("equity") if latest else None
    equity = account.get("equity")
    pct_change = None
    if prev_equity and prev_equity > 0:
        pct_change = (equity - prev_equity) / prev_equity

    alloc = compute_allocation_usage(account, positions, cfg)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account": account,
        "positions": positions,
        "allocation": alloc,
        "pct_change_from_prev_snapshot": pct_change,
        "config_snapshot": {
            "ALLOCATION_PCT": cfg.ALLOCATION_PCT,
            "MAX_POSITION_PCT": cfg.MAX_POSITION_PCT,
            "MAX_DAILY_LOSS_PCT": cfg.MAX_DAILY_LOSS_PCT,
        }
    }

    out = write_snapshot(base, snapshot)
    log.info(f"Wrote monitor snapshot: {out}")

    # Alerting rules
    alerts = []
    # Daily loss cap check (compare pct_change to MAX_DAILY_LOSS_PCT)
    if pct_change is not None and pct_change <= cfg.MAX_DAILY_LOSS_PCT:
        alerts.append({
            "type": "daily_loss",
            "msg": f"Equity change {pct_change:.2%} <= MAX_DAILY_LOSS_PCT {cfg.MAX_DAILY_LOSS_PCT:.2%}",
        })

    # Allocation usage high
    if alloc.get("usage_pct", 0) >= 0.9:
        alerts.append({
            "type": "allocation_high",
            "msg": f"Allocation usage {alloc['usage_pct']:.0%} >= 90%",
        })

    # Open positions heavy single-symbol concentration
    by_symbol = {}
    for p in positions:
        s = p.get("symbol")
        by_symbol[s] = by_symbol.get(s, 0.0) + p.get("market_value", 0.0)
    for s, mv in by_symbol.items():
        if alloc["allocated_slice"] > 0 and mv / alloc["allocated_slice"] >= 0.5:
            alerts.append({
                "type": "concentration",
                "msg": f"Symbol {s} represents {mv/alloc['allocated_slice']:.0%} of allocated slice",
            })

    if alerts:
        payload = {"snapshot": snapshot, "alerts": alerts}
        log.warning(f"Monitor alerts: {alerts}")
        webhook = os.getenv("ALPACA_MONITOR_WEBHOOK")
        if webhook:
            send_webhook(webhook, payload)
        # also write alerts file
        alert_file = Path(base / "data" / "monitor") / f"alerts_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        alert_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info(f"Wrote alerts: {alert_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
