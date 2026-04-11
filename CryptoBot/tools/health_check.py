#!/usr/bin/env python3
"""
Bot Health Check & Performance Report
======================================
Automated daily health audit + performance summary for all trading bots.

Schedules (via Windows Task Scheduler):
  - Options bots: Mon-Fri 3:30 PM MT (market close)
  - CryptoBot:    Daily 8:00 PM MT

Usage:
  python health_check.py                  # Full report (all bots)
  python health_check.py --options-only   # PutSeller + CallBuyer + AlpacaBot only
  python health_check.py --crypto-only    # CryptoBot only

Output:
  C:/Bot/reports/health_YYYYMMDD_HHMMSS.html   (full report)
  C:/Bot/reports/health_YYYYMMDD_HHMMSS.txt    (plain text summary)
  C:/Bot/reports/latest.html                    (symlink to most recent)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from io import FileIO

# ── Bot Definitions ──────────────────────────────────────────────────────────

BOTS = {
    "CryptoBot": {
        "root": Path("C:/Bot"),
        "log_dir": Path("C:/Bot/logs"),
        "log_prefix": "trading_",
        "state_dir": Path("C:/Bot/data/state"),
        "state_file": "paper_balances.json",
        "positions_file": "positions.json",
        "venv": Path("C:/Bot/.venv"),
        "category": "crypto",
        "initial_balance": 1000.0,
        "dormant": False,
    },
    "AlpacaBot": {
        "root": Path("C:/AlpacaBot"),
        "log_dir": Path("C:/AlpacaBot/logs"),
        "log_prefix": "alpacabot_",
        "state_dir": Path("C:/AlpacaBot/data/state"),
        "state_file": "bot_state.json",
        "positions_file": "positions.json",
        "venv": Path("C:/AlpacaBot/.venv"),
        "category": "options",
        "initial_balance": 50000.0,
        "dormant": True,  # allocation=0%, not trading
    },
    "PutSeller": {
        "root": Path("C:/PutSeller"),
        "log_dir": Path("C:/PutSeller/logs"),
        "log_prefix": "putseller_",
        "state_dir": Path("C:/PutSeller/data/state"),
        "state_file": "bot_state.json",
        "positions_file": "positions.json",
        "venv": Path("C:/PutSeller/.venv"),
        "category": "options",
        "initial_balance": 35000.0,
        "dormant": False,
    },
    "CallBuyer": {
        "root": Path("C:/CallBuyer"),
        "log_dir": Path("C:/CallBuyer/logs"),
        "log_prefix": "callbuyer_",
        "state_dir": Path("C:/CallBuyer/data/state"),
        "state_file": "bot_state.json",
        "positions_file": "positions.json",
        "venv": Path("C:/CallBuyer/.venv"),
        "category": "options",
        "initial_balance": 15000.0,
        "dormant": False,
    },
}

DASHBOARD_URL = "http://127.0.0.1:8088"
REPORTS_DIR = Path("C:/Bot/reports")

# ── Alert Thresholds ─────────────────────────────────────────────────────────

THRESHOLDS = {
    "log_stale_minutes": 10,
    "drawdown_warn_pct": -5.0,
    "drawdown_critical_pct": -10.0,
    "win_rate_warn": 0.40,
    "consecutive_loss_warn": 4,
    "position_age_warn_hours": 6,  # options
    "position_age_warn_hours_crypto": 3,
    "error_scan_lines": 200,
}


# ── Utility Functions ────────────────────────────────────────────────────────

def read_json_safe(path: Path) -> dict | None:
    """Read JSON file, return None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_log_tail(log_path: Path, lines: int = 200) -> list[str]:
    """Read last N lines from a potentially locked log file."""
    try:
        fio = FileIO(str(log_path), "r")
        size = fio.seek(0, 2)
        chunk_size = min(size, lines * 300)  # ~300 bytes per line estimate
        fio.seek(max(0, size - chunk_size))
        data = fio.read().decode("utf-8", errors="replace")
        fio.close()
        all_lines = data.split("\n")
        return all_lines[-lines:]
    except Exception:
        return []


def find_latest_log(log_dir: Path, prefix: str) -> Path | None:
    """Find the most recently modified log file matching prefix."""
    if not log_dir.exists():
        return None
    candidates = sorted(
        log_dir.glob(f"{prefix}*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def get_pythonw_processes() -> list[dict]:
    """Get all pythonw.exe processes with PID, memory, start time, command line."""
    processes = []
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process pythonw -ErrorAction SilentlyContinue | "
             "Select-Object Id, @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB)}}, "
             "StartTime, @{N='CmdLine';E={(Get-CimInstance Win32_Process -Filter "
             "\"ProcessId=$($_.Id)\").CommandLine}} | ConvertTo-Json"],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.stdout.strip():
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                data = [data]
            processes = data
    except Exception:
        pass
    return processes


def identify_bot_process(processes: list[dict], bot_name: str, bot_info: dict) -> dict | None:
    """Match a pythonw process to a bot by its command line."""
    root_str = str(bot_info["root"]).lower()
    for proc in processes:
        cmd = (proc.get("CmdLine") or "").lower()
        if root_str.replace("\\", "/") in cmd.replace("\\", "/") or root_str in cmd:
            # Skip tiny shim processes (< 10MB)
            if proc.get("WS_MB", 0) > 8:
                return proc
    return None


def parse_log_timestamp(line: str) -> datetime | None:
    """Extract timestamp from a log line."""
    # Format 1: 2026-03-25 19:24:32 | INFO | ...
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    # Format 2: 19:24:31,492 [INFO] ...
    m = re.match(r"(\d{2}:\d{2}:\d{2}),?\d*\s*[\[|]", line)
    if m:
        try:
            t = datetime.strptime(m.group(1), "%H:%M:%S")
            return datetime.now().replace(hour=t.hour, minute=t.minute, second=t.second)
        except ValueError:
            pass
    return None


# ── Health Check Functions ───────────────────────────────────────────────────

def check_bot_health(bot_name: str, bot_info: dict, processes: list[dict]) -> dict:
    """Run full health check for a single bot."""
    report = {
        "name": bot_name,
        "category": bot_info["category"],
        "status": "UNKNOWN",
        "alerts": [],      # (level, message) tuples
        "metrics": {},
        "positions": [],
        "log_errors": [],
        "log_warnings": [],
        "last_log_time": None,
    }

    # ── 1. Process Check ─────────────────────────────────────────────────
    proc = identify_bot_process(processes, bot_name, bot_info)
    if proc:
        report["metrics"]["pid"] = proc.get("Id")
        report["metrics"]["memory_mb"] = proc.get("WS_MB", 0)
        report["metrics"]["start_time"] = proc.get("StartTime")
        report["metrics"]["process_alive"] = True
    else:
        report["metrics"]["process_alive"] = False
        report["alerts"].append(("CRITICAL", f"{bot_name} process NOT FOUND"))

    # ── 2. State File Check ──────────────────────────────────────────────
    state_path = bot_info["state_dir"] / bot_info["state_file"]
    state = read_json_safe(state_path)
    if state:
        report["metrics"]["state_loaded"] = True

        if bot_name == "CryptoBot":
            spot = state.get("spot", 0)
            futures = state.get("futures", 0)
            total = spot + futures
            daily_pnl = state.get("daily_pnl", 0)
            peak = state.get("peak_balance", total)
            consec = state.get("consecutive_losses", 0)
            report["metrics"]["balance"] = total
            report["metrics"]["spot_balance"] = spot
            report["metrics"]["futures_balance"] = futures
            report["metrics"]["daily_pnl"] = daily_pnl
            report["metrics"]["peak_balance"] = peak
            report["metrics"]["consecutive_losses"] = consec
        else:
            balance = state.get("current_balance", 0)
            peak = state.get("peak_balance", balance)
            daily_pnl = state.get("daily_pnl", 0)
            total_pnl = state.get("total_pnl", 0)
            wins = state.get("wins", 0)
            losses = state.get("losses", 0)
            total_trades = state.get("total_trades", wins + losses)
            consec = state.get("consecutive_losses", 0)
            report["metrics"]["balance"] = balance
            report["metrics"]["daily_pnl"] = daily_pnl
            report["metrics"]["total_pnl"] = total_pnl
            report["metrics"]["peak_balance"] = peak
            report["metrics"]["wins"] = wins
            report["metrics"]["losses"] = losses
            report["metrics"]["total_trades"] = total_trades
            report["metrics"]["consecutive_losses"] = consec
            if total_trades > 0:
                report["metrics"]["win_rate"] = wins / total_trades
            # PutSeller per-side tracking
            if "consecutive_losses_put" in state:
                report["metrics"]["consec_losses_put"] = state["consecutive_losses_put"]
            if "consecutive_losses_call" in state:
                report["metrics"]["consec_losses_call"] = state["consecutive_losses_call"]

        # Drawdown calculation
        initial = bot_info["initial_balance"]
        balance = report["metrics"].get("balance", 0)
        total_trades = report["metrics"].get("total_trades", 0)
        if initial > 0 and (balance > 0 or total_trades > 0):
            report["metrics"]["return_pct"] = ((balance - initial) / initial) * 100
        if peak and peak > 0 and (balance > 0 or total_trades > 0):
            report["metrics"]["drawdown_pct"] = ((balance - peak) / peak) * 100
    else:
        report["metrics"]["state_loaded"] = False
        report["alerts"].append(("WARNING", f"State file not found: {state_path}"))

    # ── 3. Positions Check ───────────────────────────────────────────────
    pos_path = bot_info["state_dir"] / bot_info["positions_file"]
    positions = read_json_safe(pos_path)
    if positions and isinstance(positions, dict):
        now = datetime.now()
        age_limit = (THRESHOLDS["position_age_warn_hours_crypto"]
                     if bot_info["category"] == "crypto"
                     else THRESHOLDS["position_age_warn_hours"])

        for key, pos in positions.items():
            pos_info = {"id": key}
            pos_info.update(pos)

            # Check position age
            entry_time_str = pos.get("entry_time") or pos.get("open_date")
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    age_hours = (now - entry_time).total_seconds() / 3600
                    pos_info["age_hours"] = round(age_hours, 1)
                    if age_hours > age_limit:
                        report["alerts"].append(("WARNING",
                            f"Stale position: {key} ({age_hours:.1f}h old, limit={age_limit}h)"))
                except (ValueError, TypeError):
                    pass

            report["positions"].append(pos_info)

        report["metrics"]["open_positions"] = len(positions)
    else:
        report["metrics"]["open_positions"] = 0

    # ── 4. Log Freshness & Error Scan ────────────────────────────────────
    log_file = find_latest_log(bot_info["log_dir"], bot_info["log_prefix"])
    if log_file:
        report["metrics"]["log_file"] = str(log_file)
        mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
        stale_minutes = (datetime.now() - mod_time).total_seconds() / 60
        report["metrics"]["log_age_minutes"] = round(stale_minutes, 1)
        report["metrics"]["log_size_kb"] = round(log_file.stat().st_size / 1024, 1)

        if stale_minutes > THRESHOLDS["log_stale_minutes"]:
            report["alerts"].append(("WARNING",
                f"Log stale: {stale_minutes:.0f} min since last write"))

        # Scan tail for errors
        tail_lines = read_log_tail(log_file, THRESHOLDS["error_scan_lines"])
        for line in tail_lines:
            if re.search(r"\bERROR\b|\bCRITICAL\b|Traceback|CRASH", line, re.IGNORECASE):
                report["log_errors"].append(line.strip())
            elif re.search(r"\bWARNING\b", line, re.IGNORECASE):
                report["log_warnings"].append(line.strip())

        # Get last timestamp
        for line in reversed(tail_lines):
            ts = parse_log_timestamp(line)
            if ts:
                report["last_log_time"] = ts.strftime("%Y-%m-%d %H:%M:%S")
                break

        if report["log_errors"]:
            report["alerts"].append(("WARNING",
                f"{len(report['log_errors'])} errors in last {THRESHOLDS['error_scan_lines']} log lines"))
    else:
        report["alerts"].append(("WARNING", f"No log file found in {bot_info['log_dir']}"))

    # ── 5. Circuit Breaker (CryptoBot) ───────────────────────────────────
    if bot_name == "CryptoBot":
        cb_path = bot_info["state_dir"] / "circuit_breaker.json"
        cb = read_json_safe(cb_path)
        if cb and cb.get("tripped"):
            report["alerts"].append(("CRITICAL", "Circuit breaker TRIPPED"))
            report["metrics"]["circuit_breaker"] = True
        else:
            report["metrics"]["circuit_breaker"] = False

    # ── 6. Alert Threshold Checks ────────────────────────────────────────
    # Skip drawdown/WR alerts for dormant bots (allocation=0%)
    is_dormant = bot_info.get("dormant", False)
    dd = report["metrics"].get("drawdown_pct", 0)
    if not is_dormant:
        if dd <= THRESHOLDS["drawdown_critical_pct"]:
            report["alerts"].append(("CRITICAL", f"Drawdown {dd:.1f}% (critical threshold: {THRESHOLDS['drawdown_critical_pct']}%)"))
        elif dd <= THRESHOLDS["drawdown_warn_pct"]:
            report["alerts"].append(("WARNING", f"Drawdown {dd:.1f}% (warn threshold: {THRESHOLDS['drawdown_warn_pct']}%)"))

    consec = report["metrics"].get("consecutive_losses", 0)
    if consec >= THRESHOLDS["consecutive_loss_warn"]:
        report["alerts"].append(("WARNING", f"Consecutive losses: {consec}"))

    wr = report["metrics"].get("win_rate")
    if wr is not None and wr < THRESHOLDS["win_rate_warn"] and not is_dormant:
        report["alerts"].append(("WARNING", f"Win rate {wr:.0%} (below {THRESHOLDS['win_rate_warn']:.0%})"))

    if is_dormant:
        report["status"] = "DORMANT"
        report["alerts"] = [(l, m) for l, m in report["alerts"] if l == "CRITICAL" and "process" in m.lower()]

    # ── Determine overall status ─────────────────────────────────────────
    levels = [a[0] for a in report["alerts"]]
    if report["status"] != "DORMANT":  # dormant already set above
        if "CRITICAL" in levels:
            report["status"] = "CRITICAL"
        elif "WARNING" in levels:
            report["status"] = "WARNING"
        elif report["metrics"].get("process_alive"):
            report["status"] = "HEALTHY"
        else:
            report["status"] = "UNKNOWN"

    return report


# ── Dashboard API Check ──────────────────────────────────────────────────────

def check_dashboard() -> dict | None:
    """Try to fetch /api/account from the dashboard."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{DASHBOARD_URL}/api/account", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# ── Report Generation ────────────────────────────────────────────────────────

def generate_text_report(reports: list[dict], account: dict | None) -> str:
    """Generate plain text summary."""
    now = datetime.now()
    lines = [
        "=" * 70,
        f"  BOT HEALTH CHECK & PERFORMANCE REPORT",
        f"  {now.strftime('%A, %B %d, %Y at %I:%M %p MT')}",
        "=" * 70,
        "",
    ]

    # Account summary
    if account:
        equity = account.get("equity", 0)
        cash = account.get("cash", 0)
        bp = account.get("buying_power", 0)
        lines.append(f"  ALPACA ACCOUNT: ${equity:,.2f} equity | ${cash:,.2f} cash | ${bp:,.2f} buying power")
        lines.append(f"  RETURN: ${equity - 100000:+,.2f} ({(equity - 100000) / 100000:+.1%} from $100K)")
        lines.append("")

    # Per-bot summaries
    for r in reports:
        status_icon = {"HEALTHY": "[OK]", "WARNING": "[!!]", "CRITICAL": "[XX]", "DORMANT": "[ZZ]"}.get(r["status"], "[??]")
        m = r["metrics"]

        lines.append(f"  {status_icon} {r['name']}")
        lines.append(f"  {'─' * 50}")

        if m.get("process_alive"):
            lines.append(f"    Process: PID {m.get('pid')} | {m.get('memory_mb', 0)} MB")
        else:
            lines.append(f"    Process: NOT RUNNING")

        bal = m.get("balance", 0)
        lines.append(f"    Balance: ${bal:,.2f}")

        if "daily_pnl" in m:
            lines.append(f"    Daily P&L: ${m['daily_pnl']:+,.2f}")

        if "total_pnl" in m:
            lines.append(f"    Total P&L: ${m['total_pnl']:+,.2f}")

        if "return_pct" in m:
            lines.append(f"    Return: {m['return_pct']:+.1f}%")

        if "drawdown_pct" in m:
            lines.append(f"    Drawdown from peak: {m['drawdown_pct']:.1f}%")

        if "win_rate" in m:
            lines.append(f"    Win Rate: {m['win_rate']:.0%} ({m.get('wins',0)}W / {m.get('losses',0)}L / {m.get('total_trades',0)} total)")

        if "consecutive_losses" in m:
            lines.append(f"    Consecutive Losses: {m['consecutive_losses']}")

        lines.append(f"    Open Positions: {m.get('open_positions', 0)}")

        if m.get("log_age_minutes") is not None:
            lines.append(f"    Log: {m.get('log_size_kb', 0)} KB | last write {m['log_age_minutes']:.0f} min ago")

        # Alerts
        if r["alerts"]:
            lines.append(f"    ALERTS:")
            for level, msg in r["alerts"]:
                lines.append(f"      [{level}] {msg}")

        lines.append("")

    # Positions summary
    total_positions = sum(len(r["positions"]) for r in reports)
    if total_positions > 0:
        lines.append(f"  OPEN POSITIONS ({total_positions} total)")
        lines.append(f"  {'─' * 50}")
        for r in reports:
            for pos in r["positions"]:
                pid = pos.get("id", "?")
                age = pos.get("age_hours", "?")
                pnl = pos.get("current_pnl") or pos.get("pnl") or pos.get("peak_pnl_pct") or 0
                lines.append(f"    {r['name']:12s} | {pid:20s} | age={age}h | pnl=${pnl:+.2f}" if isinstance(pnl, (int, float)) else f"    {r['name']:12s} | {pid}")
        lines.append("")

    # Error summary
    total_errors = sum(len(r["log_errors"]) for r in reports)
    if total_errors > 0:
        lines.append(f"  RECENT LOG ERRORS ({total_errors} total)")
        lines.append(f"  {'─' * 50}")
        for r in reports:
            for err in r["log_errors"][:5]:  # Max 5 per bot
                lines.append(f"    [{r['name']}] {err[:100]}")
        lines.append("")

    # Overall verdict
    active_statuses = [r["status"] for r in reports if r["status"] != "DORMANT"]
    if "CRITICAL" in active_statuses:
        verdict = "CRITICAL — Immediate attention needed"
    elif "WARNING" in active_statuses:
        verdict = "WARNING — Review recommended"
    elif all(s == "HEALTHY" for s in active_statuses):
        verdict = "ALL HEALTHY — No issues detected"
    else:
        verdict = "MIXED — Some bots have unknown status"

    lines.append(f"  VERDICT: {verdict}")
    lines.append("=" * 70)
    return "\n".join(lines)


def generate_html_report(reports: list[dict], account: dict | None) -> str:
    """Generate styled HTML report."""
    now = datetime.now()

    status_colors = {
        "HEALTHY": "#22c55e",
        "WARNING": "#f59e0b",
        "CRITICAL": "#ef4444",
        "UNKNOWN": "#6b7280",
        "DORMANT": "#64748b",
    }

    # Build bot cards
    bot_cards = ""
    for r in reports:
        m = r["metrics"]
        color = status_colors.get(r["status"], "#6b7280")

        # Metrics rows
        metrics_html = ""
        if m.get("process_alive"):
            metrics_html += f'<div class="metric"><span class="label">Process</span><span class="value ok">PID {m.get("pid")} ({m.get("memory_mb",0)} MB)</span></div>'
        else:
            metrics_html += '<div class="metric"><span class="label">Process</span><span class="value critical">NOT RUNNING</span></div>'

        bal = m.get("balance", 0)
        metrics_html += f'<div class="metric"><span class="label">Balance</span><span class="value">${bal:,.2f}</span></div>'

        if "daily_pnl" in m:
            dpnl = m["daily_pnl"]
            dpnl_class = "ok" if dpnl >= 0 else "critical"
            metrics_html += f'<div class="metric"><span class="label">Daily P&L</span><span class="value {dpnl_class}">${dpnl:+,.2f}</span></div>'

        if "total_pnl" in m:
            tpnl = m["total_pnl"]
            tpnl_class = "ok" if tpnl >= 0 else "critical"
            metrics_html += f'<div class="metric"><span class="label">Total P&L</span><span class="value {tpnl_class}">${tpnl:+,.2f}</span></div>'

        if "return_pct" in m:
            ret = m["return_pct"]
            ret_class = "ok" if ret >= 0 else "critical"
            metrics_html += f'<div class="metric"><span class="label">Return</span><span class="value {ret_class}">{ret:+.1f}%</span></div>'

        if "drawdown_pct" in m:
            dd = m["drawdown_pct"]
            dd_class = "ok" if dd > -5 else ("warning" if dd > -10 else "critical")
            metrics_html += f'<div class="metric"><span class="label">Drawdown</span><span class="value {dd_class}">{dd:.1f}%</span></div>'

        if "win_rate" in m:
            wr = m["win_rate"]
            wr_class = "ok" if wr >= 0.5 else ("warning" if wr >= 0.4 else "critical")
            metrics_html += f'<div class="metric"><span class="label">Win Rate</span><span class="value {wr_class}">{wr:.0%} ({m.get("wins",0)}W/{m.get("losses",0)}L)</span></div>'

        if "consecutive_losses" in m:
            cl = m["consecutive_losses"]
            cl_class = "ok" if cl < 3 else ("warning" if cl < 5 else "critical")
            metrics_html += f'<div class="metric"><span class="label">Loss Streak</span><span class="value {cl_class}">{cl}</span></div>'

        metrics_html += f'<div class="metric"><span class="label">Positions</span><span class="value">{m.get("open_positions", 0)}</span></div>'

        if m.get("log_age_minutes") is not None:
            la = m["log_age_minutes"]
            la_class = "ok" if la < 10 else "warning"
            metrics_html += f'<div class="metric"><span class="label">Log Age</span><span class="value {la_class}">{la:.0f} min</span></div>'

        # Alerts
        alerts_html = ""
        if r["alerts"]:
            alerts_html = '<div class="alerts">'
            for level, msg in r["alerts"]:
                acolor = status_colors.get(level, "#6b7280")
                alerts_html += f'<div class="alert" style="border-left: 3px solid {acolor}; padding-left: 8px; margin: 4px 0; font-size: 0.85em;">[{level}] {msg}</div>'
            alerts_html += '</div>'

        bot_cards += f'''
        <div class="bot-card">
            <div class="bot-header" style="border-left: 4px solid {color};">
                <span class="bot-name">{r["name"]}</span>
                <span class="bot-status" style="color: {color};">{r["status"]}</span>
            </div>
            <div class="bot-metrics">{metrics_html}</div>
            {alerts_html}
        </div>'''

    # Account summary
    account_html = ""
    if account:
        equity = account.get("equity", 0)
        ret_pct = (equity - 100000) / 100000 * 100
        ret_class = "ok" if ret_pct >= 0 else "critical"
        account_html = f'''
        <div class="account-summary">
            <h2>Alpaca Account</h2>
            <div class="account-grid">
                <div class="metric"><span class="label">Equity</span><span class="value">${equity:,.2f}</span></div>
                <div class="metric"><span class="label">Cash</span><span class="value">${account.get("cash", 0):,.2f}</span></div>
                <div class="metric"><span class="label">Buying Power</span><span class="value">${account.get("buying_power", 0):,.2f}</span></div>
                <div class="metric"><span class="label">Return</span><span class="value {ret_class}">{ret_pct:+.1f}% (${equity - 100000:+,.2f})</span></div>
            </div>
        </div>'''

    # Overall verdict
    active_statuses = [r["status"] for r in reports if r["status"] != "DORMANT"]
    if "CRITICAL" in active_statuses:
        verdict = "CRITICAL"
        verdict_msg = "Immediate attention needed"
    elif "WARNING" in active_statuses:
        verdict = "WARNING"
        verdict_msg = "Review recommended"
    elif all(s == "HEALTHY" for s in active_statuses):
        verdict = "HEALTHY"
        verdict_msg = "All systems nominal"
    else:
        verdict = "UNKNOWN"
        verdict_msg = "Some bots have unknown status"
    verdict_color = status_colors.get(verdict, "#6b7280")

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bot Health Report — {now.strftime("%b %d, %Y %I:%M %p")}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ font-size: 1.5em; color: #f8fafc; margin-bottom: 4px; }}
    .subtitle {{ color: #94a3b8; font-size: 0.9em; margin-bottom: 20px; }}
    .verdict-banner {{ background: #1e293b; border: 1px solid {verdict_color}40; border-radius: 8px; padding: 16px; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; }}
    .verdict-dot {{ width: 14px; height: 14px; border-radius: 50%; background: {verdict_color}; box-shadow: 0 0 8px {verdict_color}80; }}
    .verdict-text {{ font-size: 1.1em; font-weight: 600; color: {verdict_color}; }}
    .verdict-msg {{ color: #94a3b8; font-size: 0.9em; margin-left: 8px; }}
    .account-summary {{ background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    .account-summary h2 {{ font-size: 1em; color: #94a3b8; margin-bottom: 10px; }}
    .account-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }}
    .bot-card {{ background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 12px; }}
    .bot-header {{ display: flex; justify-content: space-between; align-items: center; padding-left: 12px; margin-bottom: 12px; }}
    .bot-name {{ font-size: 1.1em; font-weight: 600; }}
    .bot-status {{ font-weight: 700; text-transform: uppercase; font-size: 0.85em; }}
    .bot-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 6px; }}
    .metric {{ display: flex; justify-content: space-between; padding: 4px 8px; background: #0f172a; border-radius: 4px; font-size: 0.85em; }}
    .label {{ color: #64748b; }}
    .value {{ font-weight: 600; }}
    .value.ok {{ color: #22c55e; }}
    .value.warning {{ color: #f59e0b; }}
    .value.critical {{ color: #ef4444; }}
    .alerts {{ margin-top: 10px; padding-top: 8px; border-top: 1px solid #334155; }}
    .footer {{ text-align: center; color: #475569; font-size: 0.8em; margin-top: 20px; padding-top: 12px; border-top: 1px solid #1e293b; }}
</style>
</head>
<body>
<div class="container">
    <h1>Bot Health Check & Performance Report</h1>
    <div class="subtitle">{now.strftime("%A, %B %d, %Y at %I:%M %p MT")}</div>

    <div class="verdict-banner">
        <div class="verdict-dot"></div>
        <span class="verdict-text">{verdict}</span>
        <span class="verdict-msg">{verdict_msg}</span>
    </div>

    {account_html}
    {bot_cards}

    <div class="footer">
        Generated by health_check.py — Auto-runs Mon-Fri 3:30 PM MT (options) + Daily 8:00 PM MT (crypto)
    </div>
</div>
</body>
</html>'''
    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bot Health Check & Performance Report")
    parser.add_argument("--options-only", action="store_true", help="Check options bots only (PutSeller, CallBuyer, AlpacaBot)")
    parser.add_argument("--crypto-only", action="store_true", help="Check CryptoBot only")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    # Select which bots to check
    if args.options_only:
        selected = {k: v for k, v in BOTS.items() if v["category"] == "options"}
    elif args.crypto_only:
        selected = {k: v for k, v in BOTS.items() if v["category"] == "crypto"}
    else:
        selected = BOTS

    # Get all pythonw processes once
    if not args.quiet:
        print("Scanning processes...")
    processes = get_pythonw_processes()

    # Run health checks
    reports = []
    for bot_name, bot_info in selected.items():
        if not args.quiet:
            print(f"  Checking {bot_name}...")
        report = check_bot_health(bot_name, bot_info, processes)
        reports.append(report)

    # Dashboard / Alpaca account
    account = None
    if not args.crypto_only:
        if not args.quiet:
            print("  Querying dashboard...")
        account = check_dashboard()

    # Generate reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_report = generate_text_report(reports, account)
    html_report = generate_html_report(reports, account)

    txt_path = REPORTS_DIR / f"health_{timestamp}.txt"
    html_path = REPORTS_DIR / f"health_{timestamp}.html"
    latest_html = REPORTS_DIR / "latest.html"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_report)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)

    # Update latest.html (copy, not symlink — avoids permission issues)
    with open(latest_html, "w", encoding="utf-8") as f:
        f.write(html_report)

    if not args.quiet:
        print()
        print(txt_report)
        print()
        print(f"Reports saved:")
        print(f"  HTML: {html_path}")
        print(f"  Text: {txt_path}")
        print(f"  Latest: {latest_html}")

    # Try Windows toast notification for critical/warning
    active_statuses = [r["status"] for r in reports if r["status"] != "DORMANT"]
    if "CRITICAL" in active_statuses or "WARNING" in active_statuses:
        worst = "CRITICAL" if "CRITICAL" in active_statuses else "WARNING"
        problems = [f"{r['name']}: {r['status']}" for r in reports if r["status"] not in ("HEALTHY", "DORMANT")]
        try_toast(f"Bot Health: {worst}", "\n".join(problems))

    # Cleanup old reports (keep last 30 days)
    cleanup_old_reports(REPORTS_DIR, days=30)

    return 0 if "CRITICAL" not in active_statuses else 1


def try_toast(title: str, message: str):
    """Try to show a Windows toast notification."""
    try:
        from win10toast import ToastNotifier
        toast = ToastNotifier()
        toast.show_toast(title, message, duration=10, threaded=True)
    except ImportError:
        pass
    except Exception:
        pass


def cleanup_old_reports(reports_dir: Path, days: int = 30):
    """Remove report files older than N days."""
    cutoff = datetime.now() - timedelta(days=days)
    for f in reports_dir.glob("health_*.*"):
        if f.name == "latest.html":
            continue
        try:
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                f.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
