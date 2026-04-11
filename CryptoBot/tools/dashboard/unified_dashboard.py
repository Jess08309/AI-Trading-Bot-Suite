"""
Unified Quad-Bot Dashboard
Shows CryptoBot + AlpacaBot + IronCondor + CallBuyer in a single web UI.
Reads state files from all three bots and streams updates via polling.

Run:  python tools/dashboard/unified_dashboard.py
Open:  http://127.0.0.1:8088
"""

import csv
import json
import os
import re
import ssl
import subprocess
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

# ── Paths ──────────────────────────────────────────────────────────────
CRYPTO_ROOT = Path(r"C:\Bot")
ALPACA_ROOT = Path(r"C:\AlpacaBot")
PUTSELLER_ROOT = Path(r"C:\PutSeller")

CRYPTO_STATE = CRYPTO_ROOT / "data" / "state"
CRYPTO_LOGS = CRYPTO_ROOT / "logs"
CRYPTO_HISTORY = CRYPTO_ROOT / "data" / "history"
CRYPTO_TRADES_CSV = CRYPTO_ROOT / "data" / "trades.csv"

ALPACA_STATE = ALPACA_ROOT / "data" / "state"
ALPACA_LOGS = ALPACA_ROOT / "logs"
ALPACA_TRADES_CSV = ALPACA_ROOT / "data" / "trades.csv"

PUTSELLER_STATE = PUTSELLER_ROOT / "data" / "state"
PUTSELLER_LOGS = PUTSELLER_ROOT / "logs"
PUTSELLER_TRADES_CSV = PUTSELLER_ROOT / "data" / "trades.csv"

CALLBUYER_ROOT = Path(r"C:\CallBuyer")
CALLBUYER_STATE = CALLBUYER_ROOT / "data" / "state"
CALLBUYER_LOGS = CALLBUYER_ROOT / "logs"
CALLBUYER_TRADES_CSV = CALLBUYER_ROOT / "data" / "trades.csv"

# ── Alpaca API credentials (loaded from AlpacaBot .env) ────────────────
def _load_alpaca_keys() -> tuple:
    env_file = ALPACA_ROOT / ".env"
    key, secret = "", ""
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ALPACA_API_KEY=") and not line.startswith("#"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("ALPACA_API_SECRET=") and not line.startswith("#"):
                secret = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key, secret

_alpaca_cache = {"data": None, "ts": 0}

def _alpaca_account_live() -> Dict[str, Any]:
    """Fetch live Alpaca account data, cached for 30s."""
    now = datetime.now().timestamp()
    if _alpaca_cache["data"] and (now - _alpaca_cache["ts"]) < 30:
        return _alpaca_cache["data"]
    key, secret = _load_alpaca_keys()
    if not key or not secret:
        return {}
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            "https://paper-api.alpaca.markets/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        )
        resp = json.loads(urllib.request.urlopen(req, context=ctx, timeout=5).read())
        result = {
            "equity": float(resp.get("equity", 0)),
            "cash": float(resp.get("cash", 0)),
            "buying_power": float(resp.get("buying_power", 0)),
            "portfolio_value": float(resp.get("portfolio_value", 0)),
            "last_equity": float(resp.get("last_equity", 0)),
            "status": resp.get("status", "UNKNOWN"),
        }
        result["daily_change"] = result["equity"] - result["last_equity"]
        _alpaca_cache["data"] = result
        _alpaca_cache["ts"] = now
        return result
    except Exception:
        return _alpaca_cache.get("data") or {}

_positions_cache = {"data": None, "ts": 0}


def _parse_occ_symbol(sym: str) -> Optional[Dict[str, Any]]:
    """Parse OCC option symbol like AAPL260501C00270000."""
    import re
    m = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', sym)
    if not m:
        return None
    underlying, datestr, cp, strike_raw = m.groups()
    exp = f"20{datestr[:2]}-{datestr[2:4]}-{datestr[4:6]}"
    strike = int(strike_raw) / 1000.0
    return {"underlying": underlying, "expiration": exp,
            "type": "call" if cp == "C" else "put", "strike": strike}


def _pair_alpaca_spreads(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group live Alpaca option positions into credit spreads."""
    options = []
    for p in positions:
        if p.get("asset_class") != "us_option":
            continue
        parsed = _parse_occ_symbol(p["symbol"])
        if not parsed:
            continue
        parsed.update({"qty": p["qty"], "avg_entry_price": p["avg_entry_price"],
                        "current_price": p["current_price"], "market_value": p["market_value"],
                        "unrealized_pl": p["unrealized_pl"], "symbol": p["symbol"]})
        options.append(parsed)

    # Group by (underlying, expiration, type)
    from collections import defaultdict
    groups = defaultdict(list)
    for o in options:
        groups[(o["underlying"], o["expiration"], o["type"])].append(o)

    spreads = []
    for (underlying, exp, spread_type), legs in groups.items():
        if len(legs) < 2:
            continue
        # Sort by strike — short leg (negative qty) and long leg (positive qty)
        shorts = sorted([l for l in legs if l["qty"] < 0], key=lambda x: x["strike"])
        longs = sorted([l for l in legs if l["qty"] > 0], key=lambda x: x["strike"])
        while shorts and longs:
            s = shorts.pop(0)
            lo = longs.pop(0)
            short_strike = s["strike"]
            long_strike = lo["strike"]
            width = abs(short_strike - long_strike)
            qty = abs(s["qty"])
            credit_per = s["avg_entry_price"] - lo["avg_entry_price"]
            total_credit = credit_per * qty * 100
            max_loss = (width - credit_per) * qty * 100
            pnl = s["unrealized_pl"] + lo["unrealized_pl"]
            pnl_pct = (pnl / total_credit * 100) if total_credit else 0
            dte = max(0, (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days)
            days_held = 1  # approximate
            roc = (credit_per / width * 365 / max(dte, 1) * 100) if width else 0
            spreads.append({
                "underlying": underlying, "spread_type": spread_type,
                "short_strike": short_strike, "long_strike": long_strike,
                "spread_width": width, "expiration": exp, "qty": qty,
                "credit_per_share": credit_per, "total_credit": total_credit,
                "max_loss_total": max_loss, "current_pnl": pnl,
                "current_pnl_pct": pnl_pct, "roc_annual": roc,
                "open_date": "", "dte_at_open": dte,
            })
    return spreads


def _alpaca_positions_live() -> List[Dict[str, Any]]:
    """Fetch live Alpaca positions, cached for 30s."""
    now = datetime.now().timestamp()
    if _positions_cache["data"] is not None and (now - _positions_cache["ts"]) < 30:
        return _positions_cache["data"]
    key, secret = _load_alpaca_keys()
    if not key or not secret:
        return []
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            "https://paper-api.alpaca.markets/v2/positions",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        )
        raw = json.loads(urllib.request.urlopen(req, context=ctx, timeout=5).read())
        positions = []
        for p in raw:
            positions.append({
                "symbol": p.get("symbol", ""),
                "qty": int(p.get("qty", 0)),
                "side": p.get("side", ""),
                "avg_entry_price": float(p.get("avg_entry_price", 0)),
                "current_price": float(p.get("current_price", 0)),
                "market_value": float(p.get("market_value", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0)),
                "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
                "asset_class": p.get("asset_class", ""),
            })
        _positions_cache["data"] = positions
        _positions_cache["ts"] = now
        return positions
    except Exception:
        return _positions_cache.get("data") or []

app = Flask(__name__)

# ── Helpers ────────────────────────────────────────────────────────────

def _safe_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def _latest_log(log_dir: Path, prefix: str) -> Optional[Path]:
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob(f"{prefix}*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _parse_log_tail(log_path: Optional[Path], limit: int = 120, keywords: tuple = None) -> List[Dict[str, Any]]:
    if not log_path or not log_path.exists():
        return []
    lines: deque = deque(maxlen=3000)
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                clean = line.strip()
                if clean:
                    lines.append(clean)
    except Exception:
        return []

    # Two log formats: "TIMESTAMP | LEVEL | msg" or "TIMESTAMP [LEVEL] name: msg"
    pattern1 = re.compile(r"^([0-9\-: ]+) \| ([A-Z]+)\s+\| (.*)$")
    pattern2 = re.compile(r"^([0-9\-: ,]+) \[([A-Z]+)\] [^:]+: (.*)$")

    if keywords is None:
        keywords = (
            "Signal filters", "Counter-trend", "Sentiment", "OPEN ", "CLOSE ",
            "ALERT[", "Cycle", "Model", "Skip ", "Direction", "Total:",
            "whale", "liquidat", "flow", "depth", "liquidity",
            "Futures data", "Scanning", "position", "ensemble", "ML ",
        )

    entries = []
    for line in reversed(lines):
        if not any(k.lower() in line.lower() for k in keywords):
            continue
        match = pattern1.match(line) or pattern2.match(line)
        if match:
            ts, level, message = match.groups()
        else:
            ts, level, message = "", "INFO", line

        kind = "info"
        if "OPEN " in message or "CLOSE " in message or "OPENED" in message:
            kind = "action"
        elif "ALERT[" in message or level == "WARNING" or level == "ERROR":
            kind = "warning"
        elif any(k in message for k in ("Signal filters", "Counter-trend", "Sentiment", "Direction", "FOUND")):
            kind = "reasoning"

        entries.append({"timestamp": ts, "level": level, "kind": kind, "message": message})
        if len(entries) >= limit:
            break
    return entries


def _parse_csv_trades(csv_path: Path, limit: int = 100) -> List[Dict]:
    if not csv_path.exists():
        return []
    header_line = None
    rows: deque = deque(maxlen=max(400, limit * 4))
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if header_line is None:
                    first_cols = next(csv.reader([line]))
                    if first_cols and first_cols[0].lower() in {"timestamp", "symbol", "date"}:
                        header_line = line
                        continue
                rows.append(line)
    except Exception:
        return []

    if not rows:
        return []

    header_map: Dict[str, int] = {}
    if header_line:
        first_cols = next(csv.reader([header_line]))
        header_map = {c.lower(): i for i, c in enumerate(first_cols)}

    parsed = []
    for line in rows:
        cols = next(csv.reader([line]))
        if len(cols) < 5:
            continue
        if cols[0].lower() in {"timestamp", "symbol", "date"}:
            continue
        try:
            ts = cols[0]
        except Exception:
            continue

        if header_map:
            sym_raw = cols[header_map["symbol"]] if "symbol" in header_map else ""
            underlying = cols[header_map["underlying"]] if "underlying" in header_map else ""
            display_sym = underlying if underlying else sym_raw

            pnl_col = header_map.get("pnl", header_map.get("pnl_usd"))
            pct_col = header_map.get("pnl_pct")
            side_col = header_map.get("direction", header_map.get("side", header_map.get("spread_type")))
            price_col = header_map.get("exit_price", header_map.get("price"))
            amount_col = header_map.get("cost", header_map.get("size_usd", header_map.get("amount")))
            exit_col = header_map.get("exit_reason")

            open_date_col = header_map.get("open_date")
            parsed.append({
                "timestamp": ts,
                "symbol": display_sym,
                "side": cols[side_col] if side_col is not None and side_col < len(cols) else "",
                "price": _safe_float(cols[price_col]) if price_col is not None and price_col < len(cols) else 0,
                "amount": _safe_float(cols[amount_col]) if amount_col is not None and amount_col < len(cols) else 0,
                "pnl": _safe_float(cols[pnl_col]) if pnl_col is not None and pnl_col < len(cols) else None,
                "pnl_pct": _safe_float(cols[pct_col]) if pct_col is not None and pct_col < len(cols) else None,
                "exit_reason": cols[exit_col] if exit_col is not None and exit_col < len(cols) else "",
                "open_date": cols[open_date_col] if open_date_col is not None and open_date_col < len(cols) else "",
            })
        else:
            parsed.append({
                "timestamp": ts,
                "symbol": cols[1] if len(cols) > 1 else "",
                "side": cols[2] if len(cols) > 2 else "",
                "price": _safe_float(cols[3]) if len(cols) > 3 else 0,
                "amount": _safe_float(cols[4]) if len(cols) > 4 else 0,
                "pnl": _safe_float(cols[5]) if len(cols) > 5 else None,
                "pnl_pct": None,
                "exit_reason": "",
            })
    parsed.sort(key=lambda x: x["timestamp"], reverse=True)
    return parsed[:limit]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        s = str(v).strip().rstrip('%')
        return float(s)
    except (ValueError, TypeError):
        return default


def _get_process_info(venv_fragment: str) -> Optional[Dict]:
    """Check if a bot process is alive by matching venv path."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-c",
             f"Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | "
             f"Where-Object {{ $_.CommandLine -like '*{venv_fragment}*' -and $_.WorkingSetSize -gt 50MB }} | "
             f"Select-Object ProcessId, @{{N='MB';E={{[math]::Round($_.WorkingSetSize/1MB,1)}}}} | "
             f"ConvertTo-Json"],
            capture_output=True, text=True, timeout=8,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        data = json.loads(result.stdout.strip()) if result.stdout.strip() else None
        if isinstance(data, list):
            data = data[0] if data else None
        if data:
            return {"pid": data.get("ProcessId"), "mem_mb": data.get("MB", 0)}
    except Exception:
        pass
    return None


def _equity_from_log(log_path: Optional[Path]) -> List[Dict]:
    if not log_path or not log_path.exists():
        return []
    total_re = re.compile(r"^([0-9\-: ]+) \| [A-Z]+\s+\| Total: \$([0-9,]+(?:\.[0-9]+)?)")
    points = []
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = total_re.match(line.strip())
                if m:
                    ts, val = m.groups()
                    points.append({"timestamp": ts, "total": float(val.replace(",", ""))})
    except Exception:
        pass
    return points[-200:]


# ── API Payloads ───────────────────────────────────────────────────────

def _crypto_payload() -> Dict[str, Any]:
    balances = _safe_json(CRYPTO_STATE / "paper_balances.json")
    positions = _safe_json(CRYPTO_STATE / "positions.json")
    perf = _safe_json(CRYPTO_STATE / "performance_report.json")
    fingerprint = _safe_json(CRYPTO_STATE / "runtime_fingerprint_latest.json")
    circuit = _safe_json(CRYPTO_STATE / "circuit_breaker.json")
    rl_shadow = _safe_json(CRYPTO_STATE / "rl_shadow_report.json")

    spot = _safe_float(balances.get("spot"))
    futures = _safe_float(balances.get("futures"))
    total = spot + futures
    daily_pnl = _safe_float(balances.get("daily_pnl"))
    consec_losses = int(balances.get("consecutive_losses", 0) or 0)
    peak = _safe_float(balances.get("peak_balance", 5000))

    cfg = fingerprint.get("config", {}) if isinstance(fingerprint, dict) else {}
    initial = _safe_float(cfg.get("INITIAL_SPOT_BALANCE", 2500)) + _safe_float(cfg.get("INITIAL_FUTURES_BALANCE", 2500))
    if initial == 0:
        initial = 5000
    total_pnl = total - initial
    return_pct = (total_pnl / initial * 100) if initial else 0

    open_pos = []
    if isinstance(positions, dict):
        for key, val in positions.items():
            if not isinstance(val, dict):
                continue
            open_pos.append({
                "symbol": val.get("symbol", key),
                "direction": val.get("direction", "?"),
                "size": _safe_float(val.get("size")),
                "entry_price": _safe_float(val.get("entry_price")),
                "entry_time": val.get("entry_time", ""),
                "ml_confidence": _safe_float(val.get("ml_confidence")),
                "stop_loss": _safe_float(val.get("stop_loss")),
                "take_profit": _safe_float(val.get("take_profit")),
                "entry_reason": val.get("entry_reason", ""),
            })

    log_path = _latest_log(CRYPTO_LOGS, "trading_")
    log_entries = _parse_log_tail(log_path, limit=100)

    shadow = {}
    if isinstance(rl_shadow, dict) and rl_shadow.get("books"):
        baseline = rl_shadow["books"].get("baseline", {})
        rl_book = rl_shadow["books"].get("rl", {})
        delta = rl_shadow.get("delta", {})
        shadow = {
            "cycle": rl_shadow.get("cycle"),
            "baseline_trades": baseline.get("trades", 0),
            "baseline_wr": baseline.get("win_rate", 0),
            "baseline_pnl": baseline.get("realized_pnl", 0),
            "rl_trades": rl_book.get("trades", 0),
            "rl_wr": rl_book.get("win_rate", 0),
            "rl_pnl": rl_book.get("realized_pnl", 0),
            "delta_equity": delta.get("equity", 0),
        }

    process = _get_process_info("Bot\\.venv")
    equity_curve = _equity_from_log(log_path)

    trade_history = _parse_csv_trades(CRYPTO_TRADES_CSV, limit=60)
    if not trade_history:
        history_csv = CRYPTO_HISTORY / "trade_history.csv"
        if history_csv.exists():
            trade_history = _parse_csv_trades(history_csv, limit=60)

    return {
        "name": "CryptoBot",
        "appendage": "Left Arm",
        "status": "RUNNING" if process else "STOPPED",
        "pid": process["pid"] if process else None,
        "mem_mb": process["mem_mb"] if process else 0,
        "balances": {
            "spot": spot, "futures": futures, "total": total,
            "daily_pnl": daily_pnl, "total_pnl": total_pnl,
            "return_pct": return_pct, "peak": peak,
        },
        "consecutive_losses": consec_losses,
        "circuit_breaker": circuit.get("tripped", False),
        "positions": open_pos,
        "trade_history": trade_history,
        "log_entries": log_entries,
        "shadow": shadow,
        "equity_curve": equity_curve,
        "win_rate": _safe_float(perf.get("win_rate_pct")),
        "total_trades": int(perf.get("total_trades", 0) or 0),
        "profit_factor": _safe_float(perf.get("profit_factor")),
    }


def _alpaca_payload() -> Dict[str, Any]:
    bot_state = _safe_json(ALPACA_STATE / "bot_state.json")
    positions = _safe_json(ALPACA_STATE / "positions.json")
    fingerprint = _safe_json(ALPACA_STATE / "runtime_fingerprint_latest.json")
    rl_shadow = _safe_json(ALPACA_STATE / "rl_shadow_report.json")

    current_bal = _safe_float(bot_state.get("current_balance", 0))
    peak_bal = _safe_float(bot_state.get("peak_balance", 0))
    daily_pnl = _safe_float(bot_state.get("daily_pnl", 0))
    consec_losses = int(bot_state.get("consecutive_losses", 0) or 0)
    daily_trades = int(bot_state.get("daily_trades", 0) or 0)
    breaker = bool(bot_state.get("breaker_active", False))

    cfg = fingerprint.get("config", {}) if isinstance(fingerprint, dict) else {}
    initial = _safe_float(cfg.get("INITIAL_CAPITAL", 50000))
    if initial == 0:
        initial = 50000
    total_pnl = current_bal - initial
    return_pct = (total_pnl / initial * 100) if initial else 0

    open_pos = []
    if isinstance(positions, dict):
        for key, val in positions.items():
            if not isinstance(val, dict):
                continue
            open_pos.append({
                "symbol": key,
                "underlying": val.get("underlying", key),
                "direction": val.get("direction", "?"),
                "option_type": val.get("option_type", "?"),
                "strike": _safe_float(val.get("strike")),
                "expiration": val.get("expiration", ""),
                "entry_price": _safe_float(val.get("entry_price")),
                "current_price": _safe_float(val.get("current_price")),
                "qty": int(val.get("qty", 0) or 0),
                "cost": _safe_float(val.get("cost")),
                "entry_time": val.get("entry_time", ""),
                "ml_confidence": _safe_float(val.get("ml_confidence")),
                "ensemble_score": _safe_float(val.get("ensemble_score")),
                "score": _safe_float(val.get("score")),
            })

    log_path = _latest_log(ALPACA_LOGS, "alpacabot_")
    log_entries = _parse_log_tail(log_path, limit=100)

    shadow = {}
    if isinstance(rl_shadow, dict) and rl_shadow.get("books"):
        baseline = rl_shadow["books"].get("baseline", {})
        rl_book = rl_shadow["books"].get("rl", {})
        delta = rl_shadow.get("delta", {})
        shadow = {
            "cycle": rl_shadow.get("cycle"),
            "baseline_trades": baseline.get("trades", 0),
            "baseline_wr": baseline.get("win_rate", 0),
            "rl_trades": rl_book.get("trades", 0),
            "rl_wr": rl_book.get("win_rate", 0),
            "delta_equity": delta.get("equity", 0),
        }

    process = _get_process_info("AlpacaBot\\.venv")
    trade_history = _parse_csv_trades(ALPACA_TRADES_CSV, limit=60)

    return {
        "name": "AlpacaBot",
        "appendage": "Right Arm",
        "status": "RUNNING" if process else "STOPPED",
        "pid": process["pid"] if process else None,
        "mem_mb": process["mem_mb"] if process else 0,
        "balances": {
            "total": current_bal, "daily_pnl": daily_pnl,
            "total_pnl": total_pnl, "return_pct": return_pct,
            "peak": peak_bal,
        },
        "daily_trades": daily_trades,
        "consecutive_losses": consec_losses,
        "circuit_breaker": breaker,
        "positions": open_pos,
        "trade_history": trade_history,
        "log_entries": log_entries,
        "shadow": shadow,
        "equity_curve": [],
    }


def _putseller_payload() -> Dict[str, Any]:
    bot_state = _safe_json(PUTSELLER_STATE / "bot_state.json")
    positions = _safe_json(PUTSELLER_STATE / "positions.json")

    current_bal = _safe_float(bot_state.get("current_balance", 50000))
    peak_bal = _safe_float(bot_state.get("peak_balance", 50000))
    daily_pnl = _safe_float(bot_state.get("daily_pnl", 0))
    total_pnl = _safe_float(bot_state.get("total_pnl", 0))

    # Override with live Alpaca data if available
    acct = _alpaca_account_live()
    if acct.get("equity"):
        current_bal = acct["equity"]
        peak_bal = max(peak_bal, current_bal)
        daily_pnl = acct.get("daily_change", daily_pnl)
    consec_losses = int(bot_state.get("consecutive_losses", 0) or 0)
    total_trades = int(bot_state.get("total_trades", 0) or 0)
    wins = int(bot_state.get("wins", 0) or 0)
    losses = int(bot_state.get("losses", 0) or 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    # Calculate return %
    initial = 50000.0
    return_pct = (total_pnl / initial * 100) if initial else 0

    # Build position list from positions.json, fall back to live Alpaca API
    open_pos = []
    total_credit = 0.0
    total_risk = 0.0
    total_unrealized = 0.0
    put_count = 0
    call_count = 0

    # Try state file first, then live Alpaca spreads
    if isinstance(positions, dict) and positions:
        for pos_id, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            credit = _safe_float(pos.get("total_credit", 0))
            max_loss = _safe_float(pos.get("max_loss_total", 0))
            pnl = _safe_float(pos.get("current_pnl_total", 0))
            pnl_pct = _safe_float(pos.get("current_pnl_pct", 0))
            spread_type = pos.get("spread_type", "put")
            total_credit += credit
            total_risk += max_loss
            total_unrealized += pnl
            if spread_type == "call":
                call_count += 1
            else:
                put_count += 1

            open_pos.append({
                "pos_id": pos_id,
                "underlying": pos.get("underlying", ""),
                "short_strike": _safe_float(pos.get("short_strike")),
                "long_strike": _safe_float(pos.get("long_strike")),
                "spread_width": _safe_float(pos.get("spread_width")),
                "expiration": pos.get("expiration", ""),
                "dte_at_open": int(pos.get("dte_at_open", 0) or 0),
                "qty": int(pos.get("qty", 0) or 0),
                "credit_per_share": _safe_float(pos.get("credit_per_share")),
                "total_credit": credit,
                "max_loss_total": max_loss,
                "current_pnl": pnl,
                "current_pnl_pct": pnl_pct,
                "roc_annual": _safe_float(pos.get("roc_annual")),
                "short_delta": pos.get("short_delta"),
                "iv_premium": pos.get("iv_premium"),
                "open_date": pos.get("open_date", ""),
                "spread_type": spread_type,
            })
    else:
        # State file empty — reconstruct spreads from live Alpaca positions
        live_spreads = _pair_alpaca_spreads(_alpaca_positions_live())
        for sp in live_spreads:
            total_credit += sp["total_credit"]
            total_risk += sp["max_loss_total"]
            total_unrealized += sp["current_pnl"]
            if sp["spread_type"] == "call":
                call_count += 1
            else:
                put_count += 1
            open_pos.append(sp)

    # Log entries with IronCondor-specific keywords
    log_path = _latest_log(PUTSELLER_LOGS, "putseller_")
    ps_keywords = (
        "FOUND", "OPENING", "OPENED", "CLOSING", "CLOSED", "TAKE_PROFIT",
        "STOP_LOSS", "DTE_EXIT", "EMERGENCY", "Scan", "Cycle", "Pos:",
        "Allocation", "ERROR", "WARNING", "credit", "spread",
    )
    log_entries = _parse_log_tail(log_path, limit=100, keywords=ps_keywords)

    # Trade history
    trade_history = _parse_csv_trades(PUTSELLER_TRADES_CSV, limit=60)

    process = _get_process_info("PutSeller\\.venv")

    return {
        "name": "IronCondor",
        "appendage": "Right Leg",
        "status": "RUNNING" if process else "STOPPED",
        "pid": process["pid"] if process else None,
        "mem_mb": process["mem_mb"] if process else 0,
        "balances": {
            "total": current_bal, "daily_pnl": daily_pnl,
            "total_pnl": total_pnl, "return_pct": return_pct,
            "peak": peak_bal,
        },
        "positions": open_pos,
        "position_count": len(open_pos),
        "total_credit": total_credit,
        "total_risk": total_risk,
        "total_unrealized": total_unrealized,
        "put_count": put_count,
        "call_count": call_count,
        "consecutive_losses": consec_losses,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "trade_history": trade_history,
        "log_entries": log_entries,
    }


def _callbuyer_payload() -> Dict[str, Any]:
    bot_state = _safe_json(CALLBUYER_STATE / "bot_state.json")
    positions = _safe_json(CALLBUYER_STATE / "positions.json")

    current_bal = _safe_float(bot_state.get("current_balance", 0))
    peak_bal = _safe_float(bot_state.get("peak_balance", 0))
    daily_pnl = _safe_float(bot_state.get("daily_pnl", 0))
    total_pnl = _safe_float(bot_state.get("total_pnl", 0))
    consec_losses = int(bot_state.get("consecutive_losses", 0) or 0)
    total_trades = int(bot_state.get("total_trades", 0) or 0)
    wins = int(bot_state.get("wins", 0) or 0)
    losses = int(bot_state.get("losses", 0) or 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    initial = _safe_float(bot_state.get("initial_capital", 8250))
    if initial == 0:
        initial = 8250
    return_pct = (total_pnl / initial * 100) if initial else 0

    open_pos = []
    total_invested = 0.0
    total_unrealized = 0.0
    if isinstance(positions, dict):
        for pos_id, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            entry_total = _safe_float(pos.get("entry_total", 0))
            current_val = _safe_float(pos.get("current_value", entry_total))
            pnl = current_val - entry_total if current_val else 0
            pnl_pct = (pnl / entry_total * 100) if entry_total else 0
            total_invested += entry_total
            total_unrealized += pnl
            open_pos.append({
                "pos_id": pos_id,
                "symbol": pos.get("symbol", pos_id),
                "contract": pos.get("contract", ""),
                "strike": _safe_float(pos.get("strike")),
                "expiration": pos.get("expiration", ""),
                "dte_at_entry": int(pos.get("dte_at_entry", 0) or 0),
                "qty": int(pos.get("qty", 0) or 0),
                "entry_price": _safe_float(pos.get("entry_price")),
                "entry_total": entry_total,
                "current_value": current_val,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "confidence": _safe_float(pos.get("confidence")),
                "rule_score": _safe_float(pos.get("rule_score")),
                "ml_proba": _safe_float(pos.get("ml_proba")),
                "entry_time": pos.get("entry_time", ""),
            })

    log_path = _latest_log(CALLBUYER_LOGS, "callbuyer_")
    cb_keywords = (
        "FOUND", "OPENING", "OPENED", "CLOSING", "CLOSED", "TAKE_PROFIT",
        "STOP_LOSS", "DTE_EXIT", "Scan", "Cycle", "Earnings",
        "ERROR", "WARNING", "confidence", "breakout", "regime",
    )
    log_entries = _parse_log_tail(log_path, limit=100, keywords=cb_keywords)

    trade_history = _parse_csv_trades(CALLBUYER_TRADES_CSV, limit=60)
    process = _get_process_info("CallBuyer\\.venv")

    return {
        "name": "CallBuyer",
        "appendage": "Left Leg",
        "status": "RUNNING" if process else "STOPPED",
        "pid": process["pid"] if process else None,
        "mem_mb": process["mem_mb"] if process else 0,
        "balances": {
            "total": current_bal, "daily_pnl": daily_pnl,
            "total_pnl": total_pnl, "return_pct": return_pct,
            "peak": peak_bal,
        },
        "positions": open_pos,
        "position_count": len(open_pos),
        "total_invested": total_invested,
        "total_unrealized": total_unrealized,
        "consecutive_losses": consec_losses,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "trade_history": trade_history,
        "log_entries": log_entries,
    }


def _guardrails_payload() -> Dict[str, Any]:
    """Aggregate cross-bot portfolio exposure and guardrail status."""
    ps_positions = _safe_json(PUTSELLER_STATE / "positions.json")
    cb_positions = _safe_json(CALLBUYER_STATE / "positions.json")
    ab_positions = _safe_json(ALPACA_STATE / "positions.json")

    ps_risk = sum(_safe_float(p.get("max_loss_total", 0)) for p in ps_positions.values() if isinstance(p, dict))
    cb_risk = sum(_safe_float(p.get("entry_total", 0)) for p in cb_positions.values() if isinstance(p, dict))
    ab_risk = sum(_safe_float(p.get("cost", 0)) for p in ab_positions.values() if isinstance(p, dict))
    total_risk = ps_risk + cb_risk + ab_risk

    ps_state = _safe_json(PUTSELLER_STATE / "bot_state.json")
    acct = _alpaca_account_live()
    equity = acct.get("equity", 0) or _safe_float(ps_state.get("current_balance", 55000)) or 55000
    pct_used = (total_risk / equity * 100) if equity else 0
    cap_pct = 25.0

    guardrails = [
        {"name": "Portfolio Exposure Cap", "limit": f"{cap_pct:.0f}%", "current": f"{pct_used:.1f}%",
         "status": "OK" if pct_used < cap_pct else "BLOCKED", "bot": "All"},
        {"name": "SPY Crash Filter", "limit": "SPY >-1.5%", "current": "Active",
         "status": "OK", "bot": "IronCondor"},
        {"name": "Regime Position Caps", "limit": "5P/3C in stress", "current": "Active",
         "status": "OK", "bot": "IronCondor"},
        {"name": "Earnings Guard", "limit": "7-day window", "current": "Active",
         "status": "OK", "bot": "CallBuyer"},
    ]

    return {
        "equity": equity,
        "total_risk": total_risk,
        "pct_used": pct_used,
        "cap_pct": cap_pct,
        "risk_breakdown": {
            "ironcondor": ps_risk,
            "callbuyer": cb_risk,
            "alpacabot": ab_risk,
        },
        "guardrails": guardrails,
    }


# ── Flask Routes ───────────────────────────────────────────────────────

@app.get("/api/crypto")
def api_crypto():
    return jsonify(_crypto_payload())


@app.get("/api/alpaca")
def api_alpaca():
    return jsonify(_alpaca_payload())


@app.get("/api/putseller")
@app.get("/api/ironcondor")
def api_putseller():
    return jsonify(_putseller_payload())


@app.get("/api/callbuyer")
def api_callbuyer():
    return jsonify(_callbuyer_payload())


@app.get("/api/guardrails")
def api_guardrails():
    return jsonify(_guardrails_payload())


@app.get("/api/account")
def api_account():
    return jsonify(_alpaca_account_live())


@app.get("/api/positions")
def api_positions():
    return jsonify(_alpaca_positions_live())


@app.get("/api/all")
def api_all():
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "account": _alpaca_account_live(),
        "live_positions": _alpaca_positions_live(),
        "crypto": _crypto_payload(),
        "alpaca": _alpaca_payload(),
        "putseller": _putseller_payload(),
        "callbuyer": _callbuyer_payload(),
        "guardrails": _guardrails_payload(),
    })


@app.get("/")
def index():
    return INDEX_HTML


# ── HTML / JS ──────────────────────────────────────────────────────────

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Command Center</title>
<style>
:root {
  --bg: #0b0e13;
  --panel: #12171f;
  --border: #1e2a3a;
  --text: #d8dfe8;
  --muted: #6b7f96;
  --green: #4ade80;
  --red: #f87171;
  --blue: #60a5fa;
  --purple: #a78bfa;
  --orange: #fb923c;
  --cyan: #22d3ee;
  --yellow: #facc15;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); }
a { color: var(--blue); text-decoration: none; }

.header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 20px; border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #141a24 0%, var(--bg) 100%);
}
.header h1 { font-size: 18px; font-weight: 700; }
.header h1 span { color: var(--green); }
.header-controls { display: flex; align-items: center; gap: 10px; font-size: 12px; }
.header-controls select, .header-controls button {
  background: var(--panel); border: 1px solid var(--border); color: var(--text);
  border-radius: 6px; padding: 5px 8px; cursor: pointer; font-size: 11px;
}
.header-controls button:hover { border-color: var(--green); }

/* Summary bar */
.summary-bar {
  display: flex; gap: 16px; padding: 8px 20px; border-bottom: 1px solid var(--border);
  background: var(--panel); font-size: 12px; flex-wrap: wrap; align-items: center;
}
.summary-item { display: flex; align-items: center; gap: 5px; }
.summary-item .lbl { color: var(--muted); }
.summary-item .val { font-weight: 700; }
.summary-sep { width: 1px; height: 18px; background: var(--border); }

.main { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; padding: 12px 16px; }
@media (max-width: 1800px) { .main { grid-template-columns: 1fr 1fr; } }
@media (max-width: 1000px) { .main { grid-template-columns: 1fr; } }

.bot-col { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.bot-label {
  font-size: 14px; font-weight: 700; padding: 7px 12px;
  border-radius: 8px; display: flex; align-items: center; gap: 8px;
  border: 1px solid var(--border); background: var(--panel);
}
.bot-label .appendage { font-size: 10px; color: var(--muted); font-weight: 400; }
.bot-label .tag { font-size: 10px; padding: 2px 7px; border-radius: 999px; font-weight: 500; }
.tag.running { background: #0d3320; color: var(--green); }
.tag.stopped { background: #3b1515; color: var(--red); }
.tag.pid { background: #1e293b; color: var(--muted); }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 5px; }
.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 7px;
  padding: 8px 10px; transition: border-color .15s;
}
.card:hover { border-color: #2e4560; }
.card .lbl { font-size: 9px; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); margin-bottom: 2px; }
.card .val { font-size: 16px; font-weight: 700; }
.card .sub { font-size: 10px; color: var(--muted); margin-top: 1px; }
.good { color: var(--green); }
.bad { color: var(--red); }
.neutral { color: var(--text); }

.section {
  background: var(--panel); border: 1px solid var(--border); border-radius: 7px;
  overflow: hidden;
}
.section-head {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px;
  padding: 6px 10px; border-bottom: 1px solid var(--border); color: var(--muted);
  display: flex; align-items: center; justify-content: space-between;
}
.section-body { max-height: 280px; overflow-y: auto; }
.section-body.tall { max-height: 400px; }

table { width: 100%; border-collapse: collapse; font-size: 11px; }
th { text-align: left; padding: 5px 8px; color: var(--muted); font-weight: 500;
     position: sticky; top: 0; background: var(--panel); border-bottom: 1px solid var(--border); font-size: 10px; }
td { padding: 4px 8px; border-bottom: 1px solid #151d28; }
tr:hover td { background: #151d28; }

.pill { display: inline-block; padding: 1px 6px; border-radius: 999px; font-size: 9px; font-weight: 600; }
.pill.long { background: #0d3320; color: var(--green); }
.pill.short, .pill.put, .pill.sell { background: #3b1515; color: var(--red); }
.pill.call, .pill.buy { background: #1a2f48; color: var(--blue); }
.pill.credit { background: #1a2f1a; color: var(--green); }

.log-line { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 10px; padding: 2px 8px; border-bottom: 1px solid #151d28; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.log-line .ts { color: var(--muted); margin-right: 5px; }
.log-line.action { color: var(--blue); }
.log-line.warning { color: var(--orange); }
.log-line.reasoning { color: var(--purple); }

.chart-wrap { padding: 6px; }
.chart-wrap svg { width: 100%; height: 120px; display: block; }

.footer { text-align: center; padding: 8px; font-size: 10px; color: var(--muted); border-top: 1px solid var(--border); }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2e4560; }

/* Guardrails panel */
.guardrail-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; border-bottom: 1px solid #151d28; font-size: 11px; }
.guardrail-row:last-child { border-bottom: none; }
.guardrail-ok { color: var(--green); }
.guardrail-blocked { color: var(--red); font-weight: 700; }
.risk-bar { height: 8px; background: #1e2a3a; border-radius: 4px; overflow: hidden; margin: 4px 10px 8px; }
.risk-bar .fill { height: 100%; border-radius: 4px; transition: width 0.5s; }

.filter-bar { display: flex; gap: 3px; }
.filter-bar button {
  background: none; border: 1px solid var(--border); color: var(--muted);
  border-radius: 4px; padding: 1px 6px; font-size: 9px; cursor: pointer;
}
.filter-bar button.active { border-color: var(--green); color: var(--green); }

/* IronCondor spread progress bar */
.spread-bar { display: flex; align-items: center; gap: 4px; }
.spread-bar .bar { flex: 1; height: 6px; background: #1e2a3a; border-radius: 3px; overflow: hidden; }
.spread-bar .fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
.spread-bar .fill.profit { background: var(--green); }
.spread-bar .fill.loss { background: var(--red); }

/* Music player */
.music-ctrl { display: flex; align-items: center; gap: 6px; }
.music-ctrl select { background: var(--panel); border: 1px solid var(--border); color: var(--text);
  border-radius: 6px; padding: 3px 6px; font-size: 10px; cursor: pointer; max-width: 120px; }
.music-ctrl button { background: var(--panel); border: 1px solid var(--border); color: var(--text);
  border-radius: 6px; padding: 3px 8px; font-size: 12px; cursor: pointer; line-height: 1; }
.music-ctrl button:hover { border-color: var(--green); }
.music-ctrl .vol { width: 60px; accent-color: var(--green); cursor: pointer; }
.music-ctrl .now { font-size: 9px; color: var(--muted); max-width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
</head>
<body>

<div class="header">
  <h1><span>&#9670;</span> Trading Command Center</h1>
  <div class="header-controls">
    <label style="display:flex;align-items:center;gap:4px;color:var(--muted);">
      <input type="checkbox" id="liveToggle" checked> Live
    </label>
    <select id="refreshRate">
      <option value="3000">3s</option>
      <option value="5000" selected>5s</option>
      <option value="10000">10s</option>
      <option value="30000">30s</option>
    </select>
    <button id="refreshBtn">&#8635; Refresh</button>
    <span id="lastUpdate" style="color:var(--muted);font-size:10px;"></span>
    <span style="width:1px;height:18px;background:var(--border);display:inline-block;"></span>
    <div class="music-ctrl">
      <button id="musicToggle" title="Play/Pause">&#9654;</button>
      <select id="musicChannel">
        <option value="dronezone">Drone Zone</option>
        <option value="spacestation">Space Station</option>
        <option value="groovesalad">Groove Salad</option>
        <option value="deepspaceone">Deep Space One</option>
        <option value="lush">Lush</option>
        <option value="seventies">Left Coast 70s</option>
      </select>
      <input type="range" class="vol" id="musicVol" min="0" max="100" value="30" title="Volume">
      <span class="now" id="musicNow">&#127925;</span>
    </div>
  </div>
</div>

<div class="summary-bar" id="summaryBar"></div>

<div class="section" style="margin:8px 16px;background:var(--card-bg);border-radius:8px;padding:10px 12px;">
  <div class="section-head" style="display:flex;justify-content:space-between;align-items:center;">
    <span>&#128200; Live Alpaca Positions <span id="livePositionCount"></span></span>
    <span id="livePositionPnl" style="font-size:12px;"></span>
  </div>
  <div class="section-body" style="max-height:220px;overflow-y:auto;">
    <table id="livePositionTable" style="width:100%">
      <thead><tr>
        <th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Current</th><th>Value</th><th>P&L</th><th>P&L%</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div class="main">
  <!-- CryptoBot Column -->
  <div class="bot-col" id="cryptoCol">
    <div class="bot-label">
      &#129518; CryptoBot <span class="appendage">Left Arm</span>
      <span class="tag" id="cryptoStatus">&mdash;</span>
      <span class="tag pid" id="cryptoPid"></span>
    </div>
    <div class="cards" id="cryptoCards"></div>
    <div class="section">
      <div class="section-head">Open Positions <span id="cryptoPosCount"></span></div>
      <div class="section-body"><table id="cryptoPosTable"><thead><tr>
        <th>Symbol</th><th>Dir</th><th>Size</th><th>Entry</th><th>SL / TP</th><th>Conf</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="section">
      <div class="section-head">
        Recent Log
        <div class="filter-bar">
          <button class="active" data-bot="crypto" data-f="all">All</button>
          <button data-bot="crypto" data-f="action">Trades</button>
          <button data-bot="crypto" data-f="reasoning">Signals</button>
          <button data-bot="crypto" data-f="warning">Warn</button>
        </div>
      </div>
      <div class="section-body tall" id="cryptoLog"></div>
    </div>
    <div class="section">
      <div class="section-head">Equity Curve</div>
      <div class="chart-wrap"><svg id="cryptoEquity"></svg></div>
    </div>
    <div class="section">
      <div class="section-head">Trade History <span id="cryptoTradeCount"></span></div>
      <div class="section-body"><table id="cryptoTradeTable"><thead><tr>
        <th>Time</th><th>Symbol</th><th>Side</th><th>Exit</th><th>P&amp;L</th><th>%</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="section">
      <div class="section-head">RL Shadow</div>
      <div class="section-body" id="cryptoShadow" style="padding:8px;font-size:11px;"></div>
    </div>
  </div>

  <!-- AlpacaBot Column -->
  <div class="bot-col" id="alpacaCol">
    <div class="bot-label">
      &#129305; AlpacaBot <span class="appendage">Right Arm</span>
      <span class="tag" id="alpacaStatus">&mdash;</span>
      <span class="tag pid" id="alpacaPid"></span>
    </div>
    <div class="cards" id="alpacaCards"></div>
    <div class="section">
      <div class="section-head">Open Positions <span id="alpacaPosCount"></span></div>
      <div class="section-body"><table id="alpacaPosTable"><thead><tr>
        <th>Underlying</th><th>Type</th><th>Strike</th><th>Exp</th><th>Qty</th><th>Entry</th><th>Current</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="section">
      <div class="section-head">
        Recent Log
        <div class="filter-bar">
          <button class="active" data-bot="alpaca" data-f="all">All</button>
          <button data-bot="alpaca" data-f="action">Trades</button>
          <button data-bot="alpaca" data-f="reasoning">Signals</button>
          <button data-bot="alpaca" data-f="warning">Warn</button>
        </div>
      </div>
      <div class="section-body tall" id="alpacaLog"></div>
    </div>
    <div class="section">
      <div class="section-head">Trade History <span id="alpacaTradeCount"></span></div>
      <div class="section-body"><table id="alpacaTradeTable"><thead><tr>
        <th>Time</th><th>Symbol</th><th>Side</th><th>Exit</th><th>P&amp;L</th><th>%</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="section">
      <div class="section-head">RL Shadow</div>
      <div class="section-body" id="alpacaShadow" style="padding:8px;font-size:11px;"></div>
    </div>
  </div>

  <!-- PutSeller Column -->
  <div class="bot-col" id="putsellerCol">
    <div class="bot-label">
      &#129470; IronCondor <span class="appendage">Right Leg</span>
      <span class="tag" id="putsellerStatus">&mdash;</span>
      <span class="tag pid" id="putsellerPid"></span>
    </div>
    <div class="cards" id="putsellerCards"></div>
    <div class="section">
      <div class="section-head">Open Spreads <span id="putsellerPosCount"></span></div>
      <div class="section-body"><table id="putsellerPosTable"><thead><tr>
        <th>Symbol</th><th>Type</th><th>Spread</th><th>Opened</th><th>Exp</th><th>Qty</th><th>Credit</th><th>P&amp;L</th><th>ROC/yr</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="section">
      <div class="section-head">
        Recent Log
        <div class="filter-bar">
          <button class="active" data-bot="putseller" data-f="all">All</button>
          <button data-bot="putseller" data-f="action">Trades</button>
          <button data-bot="putseller" data-f="reasoning">Signals</button>
          <button data-bot="putseller" data-f="warning">Warn</button>
        </div>
      </div>
      <div class="section-body tall" id="putsellerLog"></div>
    </div>
    <div class="section">
      <div class="section-head">Trade History <span id="putsellerTradeCount"></span></div>
      <div class="section-body"><table id="putsellerTradeTable"><thead><tr>
        <th>Opened</th><th>Closed</th><th>Symbol</th><th>Spread</th><th>Exit</th><th>P&amp;L</th><th>%</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="section">
      <div class="section-head">Spread Risk Map</div>
      <div class="section-body" id="putsellerRiskMap" style="padding:8px;font-size:11px;"></div>
    </div>
  </div>

  <!-- CallBuyer Column -->
  <div class="bot-col" id="callbuyerCol">
    <div class="bot-label">
      &#129470; CallBuyer <span class="appendage">Left Leg</span>
      <span class="tag" id="callbuyerStatus">&mdash;</span>
      <span class="tag pid" id="callbuyerPid"></span>
    </div>
    <div class="cards" id="callbuyerCards"></div>
    <div class="section">
      <div class="section-head">Open Calls <span id="callbuyerPosCount"></span></div>
      <div class="section-body"><table id="callbuyerPosTable"><thead><tr>
        <th>Symbol</th><th>Strike</th><th>Exp</th><th>Qty</th><th>Entry</th><th>P&amp;L</th><th>Conf</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="section">
      <div class="section-head">
        Recent Log
        <div class="filter-bar">
          <button class="active" data-bot="callbuyer" data-f="all">All</button>
          <button data-bot="callbuyer" data-f="action">Trades</button>
          <button data-bot="callbuyer" data-f="reasoning">Signals</button>
          <button data-bot="callbuyer" data-f="warning">Warn</button>
        </div>
      </div>
      <div class="section-body tall" id="callbuyerLog"></div>
    </div>
    <div class="section">
      <div class="section-head">Trade History <span id="callbuyerTradeCount"></span></div>
      <div class="section-body"><table id="callbuyerTradeTable"><thead><tr>
        <th>Time</th><th>Symbol</th><th>Strike</th><th>Exit</th><th>P&amp;L</th><th>%</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="section">
      <div class="section-head">&#128737; Portfolio Guardrails</div>
      <div class="section-body" id="guardrailsPanel" style="padding:0;"></div>
    </div>
  </div>
</div>

<div class="footer" id="footer">Dashboard loaded</div>

<script>
let data = null;
let filters = { crypto: 'all', alpaca: 'all', putseller: 'all', callbuyer: 'all' };
let refreshTimer = null;
let liveMode = true;
let refreshMs = 5000;

const $ = s => document.querySelector(s);

// ── SomaFM Music Player ─────────────────────────────────────
const SOMA_CHANNELS = {
  dronezone:    'https://ice2.somafm.com/dronezone-128-mp3',
  spacestation: 'https://ice2.somafm.com/spacestation-128-mp3',
  groovesalad:  'https://ice2.somafm.com/groovesalad-128-mp3',
  deepspaceone: 'https://ice2.somafm.com/deepspaceone-128-mp3',
  lush:         'https://ice2.somafm.com/lush-128-mp3',
  seventies:    'https://ice2.somafm.com/seventies-128-mp3'
};
const AMBIENT_KEYS = ['dronezone','spacestation','groovesalad','deepspaceone','lush'];
let somaAudio = null;
let musicPlaying = false;

function initMusic() {
  // Pick random ambient channel on load
  const pick = AMBIENT_KEYS[Math.floor(Math.random() * AMBIENT_KEYS.length)];
  document.getElementById('musicChannel').value = pick;
  somaAudio = new Audio(SOMA_CHANNELS[pick]);
  somaAudio.volume = 0.30;
  somaAudio.crossOrigin = 'anonymous';
  document.getElementById('musicNow').textContent = pick.replace(/([a-z])([A-Z])/g,'$1 $2');

  document.getElementById('musicToggle').addEventListener('click', () => {
    if (musicPlaying) { somaAudio.pause(); musicPlaying = false; document.getElementById('musicToggle').innerHTML = '&#9654;'; }
    else { somaAudio.play().catch(()=>{}); musicPlaying = true; document.getElementById('musicToggle').innerHTML = '&#10074;&#10074;'; }
  });

  document.getElementById('musicChannel').addEventListener('change', (e) => {
    const ch = e.target.value;
    const wasPlaying = musicPlaying;
    somaAudio.pause();
    somaAudio.src = SOMA_CHANNELS[ch];
    document.getElementById('musicNow').textContent = ch.replace(/([a-z])([A-Z])/g,'$1 $2');
    if (wasPlaying) { somaAudio.play().catch(()=>{}); }
  });

  document.getElementById('musicVol').addEventListener('input', (e) => {
    somaAudio.volume = e.target.value / 100;
  });
}
initMusic();
const $$ = s => document.querySelectorAll(s);
const fmt = (n, d=2) => Number(n||0).toLocaleString(undefined, {maximumFractionDigits:d, minimumFractionDigits:d});
const fmtUsd = n => '$' + fmt(n);
const pnlCls = v => Number(v) > 0.001 ? 'good' : Number(v) < -0.001 ? 'bad' : 'neutral';
const esc = v => String(v??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const shortTime = ts => { try { const d=new Date(ts); return isNaN(d)?String(ts).slice(11,19):d.toLocaleTimeString(); } catch(e) { return String(ts).slice(11,19); }};

function renderCards(containerId, cards) {
  document.getElementById(containerId).innerHTML = cards.map(([lbl,val,cls,sub]) =>
    `<div class="card"><div class="lbl">${lbl}</div><div class="val ${cls||''}">${val}</div>${sub?`<div class="sub">${sub}</div>`:''}</div>`
  ).join('');
}

function renderSummary(c, a, p, cb, g, acct) {
  const acctEquity = acct?.equity || 0;
  const acctDaily = acct?.daily_change || 0;
  const cryptoTotal = c.balances?.total || 0;
  const alpacaTotal = a.balances?.total || 0;
  const putsellerTotal = p.balances?.total || 0;
  const callbuyerTotal = cb.balances?.total || 0;
  const grandTotal = cryptoTotal + acctEquity;

  const cryptoPnl = c.balances?.total_pnl || 0;
  const alpacaPnl = acctEquity > 0 ? acctEquity - 100000 : 0;
  const grandPnl = cryptoPnl + alpacaPnl;

  const cryptoDaily = c.balances?.daily_pnl || 0;
  const grandDaily = cryptoDaily + acctDaily;

  const bots = [c, a, p, cb];
  const running = bots.filter(b => b.status === 'RUNNING').length;

  const riskPct = g.pct_used || 0;

  const items = [
    ['Bots', `${running}/4 running`, running === 4 ? 'good' : running > 0 ? 'neutral' : 'bad'],
    null,
    ['Alpaca Equity', fmtUsd(acctEquity), acctEquity > 0 ? 'good' : 'neutral', `Daily: ${fmtUsd(acctDaily)}`],
    ['Crypto', fmtUsd(cryptoTotal), pnlCls(cryptoPnl)],
    null,
    ['Combined', fmtUsd(grandTotal), pnlCls(grandPnl)],
    ['Daily P&L', fmtUsd(grandDaily), pnlCls(grandDaily)],
    ['Total P&L', fmtUsd(grandPnl), pnlCls(grandPnl)],
    null,
    ['IronCondor', `P:${p.put_count||0} C:${p.call_count||0}`, 'neutral'],
    ['CallBuyer', `${cb.position_count||0} calls`, 'neutral'],
    ['Credit', fmtUsd(p.total_credit || 0), 'good'],
    ['Portfolio Risk', `${fmt(riskPct,1)}%`, riskPct > 20 ? 'bad' : 'good'],
  ];

  $('#summaryBar').innerHTML = items.map(item => {
    if (!item) return '<div class="summary-sep"></div>';
    return `<div class="summary-item"><span class="lbl">${item[0]}:</span><span class="val ${item[2]}">${item[1]}</span></div>`;
  }).join('');
}

function renderCrypto(c) {
  const statusEl = $('#cryptoStatus');
  statusEl.textContent = c.status;
  statusEl.className = 'tag ' + (c.status==='RUNNING'?'running':'stopped');
  $('#cryptoPid').textContent = c.pid ? `PID ${c.pid}` : '';

  const b = c.balances;
  renderCards('cryptoCards', [
    ['Balance', fmtUsd(b.total), pnlCls(b.total_pnl), `Peak: ${fmtUsd(b.peak)}`],
    ['Total P&L', fmtUsd(b.total_pnl), pnlCls(b.total_pnl), `${fmt(b.return_pct)}%`],
    ['Daily P&L', fmtUsd(b.daily_pnl), pnlCls(b.daily_pnl)],
    ['Spot', fmtUsd(b.spot), ''],
    ['Futures', fmtUsd(b.futures), ''],
    ['Positions', c.positions.length, ''],
  ]);

  $('#cryptoPosCount').textContent = `(${c.positions.length})`;
  const ptbody = document.querySelector('#cryptoPosTable tbody');
  ptbody.innerHTML = c.positions.map(p => {
    const dirCls = p.direction.toLowerCase()==='long'?'long':'short';
    return `<tr>
      <td>${esc(p.symbol)}</td>
      <td><span class="pill ${dirCls}">${p.direction}</span></td>
      <td>${fmtUsd(p.size)}</td>
      <td>${fmt(p.entry_price,6)}</td>
      <td style="font-size:10px">${fmt(p.stop_loss,4)} / ${fmt(p.take_profit,4)}</td>
      <td>${fmt(p.ml_confidence*100,0)}%</td>
    </tr>`;
  }).join('') || '<tr><td colspan="6" style="color:var(--muted)">No positions</td></tr>';

  renderLog('cryptoLog', c.log_entries, 'crypto');
  drawLine($('#cryptoEquity'), c.equity_curve, '#60a5fa', 'total');

  $('#cryptoTradeCount').textContent = `(${c.trade_history.length})`;
  const ttbody = document.querySelector('#cryptoTradeTable tbody');
  ttbody.innerHTML = c.trade_history.slice(0, 60).map(t => {
    const pnlVal = t.pnl != null ? fmtUsd(t.pnl) : '-';
    const pctVal = t.pnl_pct != null ? (t.pnl_pct >= 0 ? '+' : '') + (t.pnl_pct * 100).toFixed(1) + '%' : '';
    const exitShort = (t.exit_reason || '').replace(/\s*\([^)]*\)/g, '');
    return `<tr style="background:${t.pnl >= 0 ? 'rgba(52,211,153,0.07)' : 'rgba(248,113,113,0.07)'}">
      <td style="font-size:10px">${shortTime(t.timestamp)}</td>
      <td>${esc(t.symbol)}</td><td>${esc(t.side)}</td>
      <td style="font-size:10px">${esc(exitShort)}</td>
      <td class="${pnlCls(t.pnl)}">${pnlVal}</td>
      <td class="${pnlCls(t.pnl)}" style="font-size:10px">${pctVal}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="6" style="color:var(--muted)">No trades</td></tr>';

  renderShadow('cryptoShadow', c.shadow);
}

function renderAlpaca(a) {
  const statusEl = $('#alpacaStatus');
  statusEl.textContent = a.status;
  statusEl.className = 'tag ' + (a.status==='RUNNING'?'running':'stopped');
  $('#alpacaPid').textContent = a.pid ? `PID ${a.pid}` : '';

  const b = a.balances;
  renderCards('alpacaCards', [
    ['Equity', fmtUsd(b.total), pnlCls(b.total_pnl), `Peak: ${fmtUsd(b.peak)}`],
    ['Total P&L', fmtUsd(b.total_pnl), pnlCls(b.total_pnl), `${fmt(b.return_pct)}%`],
    ['Daily P&L', fmtUsd(b.daily_pnl), pnlCls(b.daily_pnl)],
    ['Daily Trades', a.daily_trades, ''],
    ['Positions', a.positions.length, ''],
    ['Consec L', a.consecutive_losses, a.consecutive_losses>2?'bad':'good'],
  ]);

  $('#alpacaPosCount').textContent = `(${a.positions.length})`;
  const ptbody = document.querySelector('#alpacaPosTable tbody');
  ptbody.innerHTML = a.positions.map(p => {
    const typeCls = p.option_type==='call'?'call':'put';
    const unreal = p.current_price && p.entry_price ? ((p.current_price - p.entry_price) * p.qty * 100) : null;
    const unrStr = unreal != null ? `<span class="${pnlCls(unreal)}" style="font-size:10px">${fmtUsd(unreal)}</span>` : '';
    return `<tr>
      <td>${esc(p.underlying)}</td>
      <td><span class="pill ${typeCls}">${p.option_type}</span></td>
      <td>$${fmt(p.strike,2)}</td>
      <td style="font-size:10px">${esc(p.expiration)}</td>
      <td>${p.qty}</td>
      <td>${fmt(p.entry_price,2)}</td>
      <td>${fmt(p.current_price,2)} ${unrStr}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" style="color:var(--muted)">No positions</td></tr>';

  renderLog('alpacaLog', a.log_entries, 'alpaca');

  $('#alpacaTradeCount').textContent = `(${a.trade_history.length})`;
  const ttbody = document.querySelector('#alpacaTradeTable tbody');
  ttbody.innerHTML = a.trade_history.slice(0, 60).map(t => {
    const pnlVal = t.pnl != null ? fmtUsd(t.pnl) : '-';
    const pctVal = t.pnl_pct != null ? (t.pnl_pct >= 0 ? '+' : '') + (t.pnl_pct * 100).toFixed(1) + '%' : '';
    const exitShort = (t.exit_reason || '').replace(/\s*\([^)]*\)/g, '');
    return `<tr style="background:${t.pnl >= 0 ? 'rgba(52,211,153,0.07)' : 'rgba(248,113,113,0.07)'}">
      <td style="font-size:10px">${shortTime(t.timestamp)}</td>
      <td>${esc(t.symbol)}</td><td>${esc(t.side)}</td>
      <td style="font-size:10px">${esc(exitShort)}</td>
      <td class="${pnlCls(t.pnl)}">${pnlVal}</td>
      <td class="${pnlCls(t.pnl)}" style="font-size:10px">${pctVal}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="6" style="color:var(--muted)">No trades</td></tr>';

  renderShadow('alpacaShadow', a.shadow);
}

function renderPutSeller(p) {
  const statusEl = $('#putsellerStatus');
  statusEl.textContent = p.status;
  statusEl.className = 'tag ' + (p.status==='RUNNING'?'running':'stopped');
  $('#putsellerPid').textContent = p.pid ? `PID ${p.pid}` : '';

  const b = p.balances;
  renderCards('putsellerCards', [
    ['Allocation', fmtUsd(b.total), 'neutral', `Peak: ${fmtUsd(b.peak)}`],
    ['Credit', fmtUsd(p.total_credit), 'good', `P:${p.put_count||0} C:${p.call_count||0}`],
    ['Risk', fmtUsd(p.total_risk), 'neutral'],
    ['Unrealized', fmtUsd(p.total_unrealized), pnlCls(p.total_unrealized)],
    ['Daily P&L', fmtUsd(b.daily_pnl), pnlCls(b.daily_pnl)],
    ['W/L', `${p.wins}/${p.losses}`, p.win_rate >= 50 ? 'good' : p.total_trades > 0 ? 'bad' : 'neutral',
      p.total_trades > 0 ? `${fmt(p.win_rate,0)}% WR` : ''],
  ]);

  // Spread positions table
  $('#putsellerPosCount').textContent = `(P:${p.put_count||0} C:${p.call_count||0})`;
  const ptbody = document.querySelector('#putsellerPosTable tbody');
  ptbody.innerHTML = p.positions.map(pos => {
    const pnlVal = pos.current_pnl || 0;
    const pnlPct = pos.current_pnl_pct || 0;
    const rocStr = pos.roc_annual ? `${(pos.roc_annual * 100).toFixed(0)}%` : '-';
    const isCall = pos.spread_type === 'call';
    const typeBadge = isCall ? '<span class="pill call">CALL</span>' : '<span class="pill put">PUT</span>';
    const strikeSuffix = isCall ? 'C' : 'P';
    const openDt = pos.open_date ? pos.open_date.slice(5) : '-';
    return `<tr>
      <td><strong>${esc(pos.underlying)}</strong></td>
      <td>${typeBadge}</td>
      <td style="font-size:10px">$${fmt(pos.short_strike,0)}/$${fmt(pos.long_strike,0)}${strikeSuffix}</td>
      <td style="font-size:10px">${openDt}</td>
      <td style="font-size:10px">${esc(pos.expiration)}</td>
      <td>${pos.qty}</td>
      <td class="good">$${fmt(pos.total_credit,0)}</td>
      <td class="${pnlCls(pnlVal)}">$${fmt(pnlVal,0)} <span style="font-size:9px">(${pnlPct>=0?'+':''}${fmt(pnlPct,0)}%)</span></td>
      <td style="color:var(--cyan)">${rocStr}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="9" style="color:var(--muted)">No spreads open</td></tr>';

  renderLog('putsellerLog', p.log_entries, 'putseller');

  // Trade history
  $('#putsellerTradeCount').textContent = `(${p.trade_history.length})`;
  const ttbody = document.querySelector('#putsellerTradeTable tbody');
  ttbody.innerHTML = p.trade_history.slice(0, 60).map(t => {
    const pnlVal = t.pnl != null ? fmtUsd(t.pnl) : '-';
    const pctVal = t.pnl_pct != null ? (t.pnl_pct >= 0 ? '+' : '') + fmt(t.pnl_pct, 1) + '%' : '';
    const exitShort = (t.exit_reason || '').replace(/\s*\([^)]*\)/g, '');
    const openDt = t.open_date ? t.open_date.slice(5) : '-';
    return `<tr style="background:${t.pnl >= 0 ? 'rgba(52,211,153,0.07)' : 'rgba(248,113,113,0.07)'}">
      <td style="font-size:10px">${openDt}</td>
      <td style="font-size:10px">${shortTime(t.timestamp)}</td>
      <td>${esc(t.symbol)}</td><td>${esc(t.side)}</td>
      <td style="font-size:10px">${esc(exitShort)}</td>
      <td class="${pnlCls(t.pnl)}">${pnlVal}</td>
      <td class="${pnlCls(t.pnl)}" style="font-size:10px">${pctVal}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" style="color:var(--muted)">No closed trades yet</td></tr>';

  // Risk map
  renderRiskMap(p);
}

function renderRiskMap(p) {
  const el = $('#putsellerRiskMap');
  if (!p.positions || p.positions.length === 0) {
    el.innerHTML = '<span style="color:var(--muted)">No positions to display</span>';
    return;
  }
  const totalRisk = p.total_risk || 1;
  const alloc = p.balances?.total || 50000;

  el.innerHTML = `
    <div style="margin-bottom:8px;">
      <span style="color:var(--muted)">Risk utilization:</span>
      <strong class="${totalRisk/alloc > 0.5 ? 'bad' : 'good'}">${fmt(totalRisk/alloc*100,1)}%</strong>
      <span style="color:var(--muted)">of allocation</span>
    </div>
    ${p.positions.map(pos => {
      const pct = (pos.max_loss_total / totalRisk * 100);
      const pnlPct = pos.current_pnl_pct || 0;
      const fillClass = pnlPct >= 0 ? 'profit' : 'loss';
      const fillWidth = Math.min(100, Math.abs(pnlPct));
      const riskSuffix = pos.spread_type === 'call' ? 'C' : 'P';
      return `<div style="margin-bottom:6px;">
        <div style="display:flex;justify-content:space-between;font-size:10px;margin-bottom:2px;">
          <span><strong>${pos.underlying}</strong> $${fmt(pos.short_strike,0)}/$${fmt(pos.long_strike,0)}${riskSuffix}</span>
          <span class="${pnlCls(pos.current_pnl)}">${pnlPct>=0?'+':''}${fmt(pnlPct,0)}%</span>
        </div>
        <div class="spread-bar">
          <span style="font-size:9px;color:var(--muted);width:30px">${fmt(pct,0)}%</span>
          <div class="bar"><div class="fill ${fillClass}" style="width:${fillWidth}%"></div></div>
        </div>
      </div>`;
    }).join('')}
    <div style="margin-top:8px;padding-top:6px;border-top:1px solid var(--border);font-size:10px;display:grid;grid-template-columns:1fr 1fr;gap:4px;">
      <div><span style="color:var(--muted)">Total Credit:</span> <span class="good">${fmtUsd(p.total_credit)}</span></div>
      <div><span style="color:var(--muted)">Max Risk:</span> <span>${fmtUsd(p.total_risk)}</span></div>
      <div><span style="color:var(--muted)">Positions:</span> P:${p.put_count||0} C:${p.call_count||0}</div>
      <div><span style="color:var(--muted)">Unrealized:</span> <span class="${pnlCls(p.total_unrealized)}">${fmtUsd(p.total_unrealized)}</span></div>
    </div>
  `;
}

function renderCallBuyer(cb) {
  const statusEl = $('#callbuyerStatus');
  statusEl.textContent = cb.status;
  statusEl.className = 'tag ' + (cb.status==='RUNNING'?'running':'stopped');
  $('#callbuyerPid').textContent = cb.pid ? `PID ${cb.pid}` : '';

  const b = cb.balances;
  renderCards('callbuyerCards', [
    ['Allocation', fmtUsd(b.total), 'neutral', `Peak: ${fmtUsd(b.peak)}`],
    ['Total P&L', fmtUsd(b.total_pnl), pnlCls(b.total_pnl), `${fmt(b.return_pct)}%`],
    ['Daily P&L', fmtUsd(b.daily_pnl), pnlCls(b.daily_pnl)],
    ['Invested', fmtUsd(cb.total_invested), 'neutral'],
    ['Unrealized', fmtUsd(cb.total_unrealized), pnlCls(cb.total_unrealized)],
    ['W/L', `${cb.wins}/${cb.losses}`, cb.win_rate >= 50 ? 'good' : cb.total_trades > 0 ? 'bad' : 'neutral',
      cb.total_trades > 0 ? `${fmt(cb.win_rate,0)}% WR` : ''],
  ]);

  $('#callbuyerPosCount').textContent = `(${cb.position_count||0})`;
  const ptbody = document.querySelector('#callbuyerPosTable tbody');
  ptbody.innerHTML = cb.positions.map(pos => {
    const pnlVal = pos.pnl || 0;
    const pnlPct = pos.pnl_pct || 0;
    const confStr = pos.confidence ? `${fmt(pos.confidence*100,0)}%` : '-';
    return `<tr>
      <td><strong>${esc(pos.symbol)}</strong></td>
      <td>$${fmt(pos.strike,0)}</td>
      <td style="font-size:10px">${esc(pos.expiration)}</td>
      <td>${pos.qty}</td>
      <td>$${fmt(pos.entry_price,2)}</td>
      <td class="${pnlCls(pnlVal)}">$${fmt(pnlVal,0)} <span style="font-size:9px">(${pnlPct>=0?'+':''}${fmt(pnlPct,0)}%)</span></td>
      <td>${confStr}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" style="color:var(--muted)">No calls open</td></tr>';

  renderLog('callbuyerLog', cb.log_entries, 'callbuyer');

  $('#callbuyerTradeCount').textContent = `(${cb.trade_history.length})`;
  const ttbody = document.querySelector('#callbuyerTradeTable tbody');
  ttbody.innerHTML = cb.trade_history.slice(0, 60).map(t => {
    const pnlVal = t.pnl != null ? fmtUsd(t.pnl) : '-';
    const pctVal = t.pnl_pct != null ? (t.pnl_pct >= 0 ? '+' : '') + fmt(t.pnl_pct, 1) + '%' : '';
    const exitShort = (t.exit_reason || '').replace(/\s*\([^)]*\)/g, '');
    return `<tr style="background:${t.pnl >= 0 ? 'rgba(52,211,153,0.07)' : 'rgba(248,113,113,0.07)'}">
      <td style="font-size:10px">${shortTime(t.timestamp)}</td>
      <td>${esc(t.symbol)}</td><td>$${fmt(t.amount,0)}</td>
      <td style="font-size:10px">${esc(exitShort)}</td>
      <td class="${pnlCls(t.pnl)}">${pnlVal}</td>
      <td class="${pnlCls(t.pnl)}" style="font-size:10px">${pctVal}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="6" style="color:var(--muted)">No closed trades yet</td></tr>';
}

function renderGuardrails(g) {
  const el = $('#guardrailsPanel');
  if (!g) { el.innerHTML = '<div style="padding:8px;color:var(--muted)">No data</div>'; return; }

  const pct = g.pct_used || 0;
  const barColor = pct > 20 ? 'var(--orange)' : pct > 15 ? 'var(--yellow)' : 'var(--green)';
  const barWidth = Math.min(100, pct / (g.cap_pct || 25) * 100);

  let html = `
    <div style="padding:8px 10px 0;">
      <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px;">
        <span style="color:var(--muted)">Portfolio Risk</span>
        <span class="${pct > 20 ? 'bad' : 'good'}">${fmt(pct,1)}% / ${g.cap_pct||25}%</span>
      </div>
      <div class="risk-bar"><div class="fill" style="width:${barWidth}%;background:${barColor}"></div></div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;font-size:10px;margin-bottom:6px;">
        <div><span style="color:var(--muted)">IC:</span> ${fmtUsd(g.risk_breakdown?.ironcondor||0)}</div>
        <div><span style="color:var(--muted)">CB:</span> ${fmtUsd(g.risk_breakdown?.callbuyer||0)}</div>
        <div><span style="color:var(--muted)">AB:</span> ${fmtUsd(g.risk_breakdown?.alpacabot||0)}</div>
      </div>
    </div>`;

  html += (g.guardrails || []).map(r => {
    const statusCls = r.status === 'OK' ? 'guardrail-ok' : 'guardrail-blocked';
    const icon = r.status === 'OK' ? '&#10003;' : '&#10007;';
    return `<div class="guardrail-row">
      <div><strong>${esc(r.name)}</strong><br><span style="color:var(--muted);font-size:9px">${esc(r.bot)} &mdash; ${esc(r.limit)}</span></div>
      <span class="${statusCls}">${icon} ${r.status}</span>
    </div>`;
  }).join('');

  el.innerHTML = html;
}

function renderLog(containerId, entries, bot) {
  const f = filters[bot];
  const filtered = entries.filter(e => f==='all' || e.kind===f);
  document.getElementById(containerId).innerHTML = filtered.slice(0, 80).map(e =>
    `<div class="log-line ${e.kind}"><span class="ts">${esc(e.timestamp)}</span>${esc(e.message)}</div>`
  ).join('') || '<div class="log-line" style="color:var(--muted)">No log entries</div>';
}

function renderShadow(containerId, s) {
  const el = document.getElementById(containerId);
  if (!s || !s.cycle) {
    el.innerHTML = '<span style="color:var(--muted)">No shadow data available</span>';
    return;
  }
  el.innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
      <div><span style="color:var(--muted)">Cycle:</span> ${s.cycle}</div>
      <div><span style="color:var(--muted)">Delta Equity:</span> <span class="${pnlCls(s.delta_equity)}">${fmtUsd(s.delta_equity)}</span></div>
      <div><span style="color:var(--muted)">Baseline:</span> ${s.baseline_trades||0} trades, ${fmt(s.baseline_wr||0)}% WR</div>
      <div><span style="color:var(--muted)">RL Shadow:</span> ${s.rl_trades||0} trades, ${fmt(s.rl_wr||0)}% WR</div>
    </div>`;
}

function drawLine(svg, points, color, yKey) {
  const w = 600, h = 120;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  if (!points || points.length < 2) {
    svg.innerHTML = `<text x="10" y="20" fill="var(--muted)" font-size="11">Waiting for data...</text>`;
    return;
  }
  const vals = points.map(p => Number(p[yKey]||0));
  let minY = Math.min(...vals), maxY = Math.max(...vals);
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const pad = 14;
  const xStep = (w - pad*2) / (points.length - 1);
  const yScale = (h - pad*2) / (maxY - minY);
  const coords = points.map((p,i) => {
    const x = pad + i * xStep;
    const y = h - pad - ((Number(p[yKey]||0) - minY) * yScale);
    return `${x},${y}`;
  }).join(' ');

  const firstX = pad, lastX = pad + (points.length-1)*xStep;
  const fillCoords = `${firstX},${h-pad} ${coords} ${lastX},${h-pad}`;

  svg.innerHTML = `
    <defs><linearGradient id="grad_${yKey}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${color}" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="${color}" stop-opacity="0.02"/>
    </linearGradient></defs>
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h-pad}" stroke="var(--border)"/>
    <line x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}" stroke="var(--border)"/>
    <polygon fill="url(#grad_${yKey})" points="${fillCoords}" />
    <polyline fill="none" stroke="${color}" stroke-width="1.5" points="${coords}" />
    <text x="${pad+2}" y="11" fill="var(--muted)" font-size="9">${fmt(maxY)}</text>
    <text x="${pad+2}" y="${h-3}" fill="var(--muted)" font-size="9">${fmt(minY)}</text>
  `;
}

// Filter buttons
$$('.filter-bar button').forEach(btn => {
  btn.onclick = () => {
    const bot = btn.dataset.bot;
    filters[bot] = btn.dataset.f;
    btn.parentElement.querySelectorAll('button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (data) {
      if (bot === 'crypto') renderLog('cryptoLog', data.crypto.log_entries, 'crypto');
      else if (bot === 'alpaca') renderLog('alpacaLog', data.alpaca.log_entries, 'alpaca');
      else if (bot === 'putseller') renderLog('putsellerLog', data.putseller.log_entries, 'putseller');
      else if (bot === 'callbuyer') renderLog('callbuyerLog', data.callbuyer.log_entries, 'callbuyer');
    }
  };
});

function renderLivePositions(positions) {
  const tbody = document.querySelector('#livePositionTable tbody');
  const countEl = $('#livePositionCount');
  const pnlEl = $('#livePositionPnl');
  if (!positions || !positions.length) {
    tbody.innerHTML = '<tr><td colspan="8" style="color:var(--muted);text-align:center;padding:12px;">No live positions</td></tr>';
    countEl.textContent = '(0)';
    pnlEl.textContent = '';
    return;
  }
  let totalPnl = 0;
  tbody.innerHTML = positions.map(p => {
    const pnl = Number(p.unrealized_pl || 0);
    const pnlPct = Number(p.unrealized_plpc || 0) * 100;
    totalPnl += pnl;
    const cls = pnl >= 0 ? 'good' : 'bad';
    const side = Number(p.qty) >= 0 ? 'Long' : 'Short';
    return `<tr>
      <td><strong>${esc(p.symbol)}</strong></td>
      <td>${side}</td>
      <td>${Math.abs(Number(p.qty))}</td>
      <td>${fmtUsd(p.avg_entry_price)}</td>
      <td>${fmtUsd(p.current_price)}</td>
      <td>${fmtUsd(p.market_value)}</td>
      <td class="${cls}">${fmtUsd(pnl)}</td>
      <td class="${cls}">${fmt(pnlPct,1)}%</td>
    </tr>`;
  }).join('');
  countEl.textContent = `(${positions.length})`;
  pnlEl.innerHTML = `Total P&L: <span class="${totalPnl >= 0 ? 'good' : 'bad'}">${fmtUsd(totalPnl)}</span>`;
}

function scheduleRefresh() {
  if (refreshTimer) clearTimeout(refreshTimer);
  if (!liveMode) return;
  refreshTimer = setTimeout(fetchAll, refreshMs);
}

async function fetchAll() {
  try {
    const res = await fetch('/api/all');
    data = await res.json();
    renderSummary(data.crypto, data.alpaca, data.putseller, data.callbuyer, data.guardrails, data.account);
    renderLivePositions(data.live_positions);
    renderCrypto(data.crypto);
    renderAlpaca(data.alpaca);
    renderPutSeller(data.putseller);
    renderCallBuyer(data.callbuyer);
    renderGuardrails(data.guardrails);
    $('#lastUpdate').textContent = new Date().toLocaleTimeString();
    $('#footer').textContent = `Last update: ${new Date().toLocaleString()} \u00b7 CryptoBot ${data.crypto.status} \u00b7 AlpacaBot ${data.alpaca.status} \u00b7 IronCondor ${data.putseller.status} \u00b7 CallBuyer ${data.callbuyer.status}`;
  } catch (e) {
    $('#footer').textContent = 'Fetch error: ' + e.message;
  }
  scheduleRefresh();
}

$('#refreshBtn').onclick = fetchAll;
$('#liveToggle').onchange = e => { liveMode = e.target.checked; scheduleRefresh(); };
$('#refreshRate').onchange = e => { refreshMs = Number(e.target.value); scheduleRefresh(); };

fetchAll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("DASH_PORT", "8088"))
    print(f"\n  Trading Command Center (Quad-Bot) -> http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False)
