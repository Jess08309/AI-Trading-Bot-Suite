#!/usr/bin/env python3
"""
Automated Bot Audit & Performance Analysis
============================================
Deep analysis engine that does what Copilot does during manual audits:
trade-by-trade analysis, pattern detection, root cause analysis,
strategy effectiveness scoring, and actionable recommendations.

Schedule: Every 3 days via Windows Task Scheduler

Usage:
  python bot_audit.py              # Full audit (all bots)
  python bot_audit.py --crypto     # CryptoBot only
  python bot_audit.py --options    # Options bots only

Output:
  C:/Bot/reports/audit_YYYYMMDD.html   (styled deep analysis report)
  C:/Bot/reports/audit_latest.html     (always the most recent)
  C:/Bot/reports/audit_history.json    (rolling performance snapshots)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, date
from io import FileIO, StringIO
from pathlib import Path
from urllib.request import Request, urlopen

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

REPORTS_DIR = Path("C:/Bot/reports")
HISTORY_FILE = REPORTS_DIR / "audit_history.json"
DASHBOARD_URL = "http://127.0.0.1:8088"

BOTS = {
    "CryptoBot": {
        "root": Path("C:/Bot"),
        "log_dir": Path("C:/Bot/logs"),
        "log_prefix": "trading_",
        "state_dir": Path("C:/Bot/data/state"),
        "trades_csv": Path("C:/Bot/data/trades.csv"),
        "state_file": "paper_balances.json",
        "positions_file": "positions.json",
        "category": "crypto",
        "initial_balance": 1000.0,
        "dormant": False,
    },
    "AlpacaBot": {
        "root": Path("C:/AlpacaBot"),
        "log_dir": Path("C:/AlpacaBot/logs"),
        "log_prefix": "alpacabot_",
        "state_dir": Path("C:/AlpacaBot/data/state"),
        "trades_csv": Path("C:/AlpacaBot/data/trades.csv"),
        "state_file": "bot_state.json",
        "positions_file": "positions.json",
        "category": "options",
        "initial_balance": 50000.0,
        "dormant": True,
    },
    "PutSeller": {
        "root": Path("C:/PutSeller"),
        "log_dir": Path("C:/PutSeller/logs"),
        "log_prefix": "putseller_",
        "state_dir": Path("C:/PutSeller/data/state"),
        "trades_csv": Path("C:/PutSeller/data/trades.csv"),
        "state_file": "bot_state.json",
        "positions_file": "positions.json",
        "category": "options",
        "initial_balance": 35000.0,
        "dormant": False,
    },
    "CallBuyer": {
        "root": Path("C:/CallBuyer"),
        "log_dir": Path("C:/CallBuyer/logs"),
        "log_prefix": "callbuyer_",
        "state_dir": Path("C:/CallBuyer/data/state"),
        "trades_csv": Path("C:/CallBuyer/data/trades.csv"),
        "state_file": "bot_state.json",
        "positions_file": "positions.json",
        "category": "options",
        "initial_balance": 15000.0,
        "dormant": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_log_tail(path: Path, n_bytes: int = 100_000) -> str:
    """Read tail of a potentially locked log file."""
    try:
        fio = FileIO(str(path), "r")
        size = fio.seek(0, 2)
        fio.seek(max(0, size - n_bytes))
        data = fio.read().decode("utf-8", errors="replace")
        fio.close()
        return data
    except Exception:
        return ""


def read_csv_trades(csv_path: Path) -> list[dict]:
    """Parse a trades CSV into list of dicts."""
    trades = []
    if not csv_path.exists():
        return trades
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content or content.count("\n") < 1:
            return trades
        reader = csv.DictReader(StringIO(content))
        for row in reader:
            # Normalize field names (strip whitespace from keys)
            clean = {k.strip(): v.strip() for k, v in row.items() if k}
            # Parse numeric fields
            for key in ["pnl", "pnl_usd", "pnl_pct", "pnl_dollar",
                        "entry_price", "exit_price", "size_usd",
                        "credit", "close_debit", "hold_days",
                        "confidence", "rule_score", "ml_proba",
                        "hold_time_hours", "qty"]:
                if key in clean and clean[key]:
                    try:
                        clean[key] = float(clean[key])
                    except (ValueError, TypeError):
                        pass
            trades.append(clean)
    except Exception:
        pass
    return trades


def extract_trades_from_logs(log_dir: Path, prefix: str) -> list[dict]:
    """Extract trade close events from log files (fallback when CSV is empty).

    Parses PutSeller format:
      CLOSED: {sym} ${short}/${long}{C/P} | PnL ${pnl} ({pct}%) | {days}d hold | {reason} ...
    """
    trades = []
    if not log_dir.exists():
        return trades

    # PutSeller/IronCondor close pattern
    close_re = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\S*\s+\[(?:INFO|WARNING)\].*"
        r"CLOSED:\s+(\w+)\s+\$[\d.]+/\$[\d.]+([CP])\s+\|\s+"
        r"PnL\s+\$([+-]?[\d,.]+)\s+\(([+-]?\d+)%\)\s+\|\s+"
        r"(\d+)d\s+hold\s+\|\s+(\w+)"
    )

    for log_path in sorted(log_dir.glob(f"{prefix}*.log")):
        try:
            text = read_log_tail(log_path, n_bytes=500_000)
            for m in close_re.finditer(text):
                ts_str, symbol, side_char, pnl_str, pct_str, days_str, reason = m.groups()
                pnl_val = float(pnl_str.replace(",", ""))
                trades.append({
                    "timestamp": ts_str,
                    "symbol": symbol,
                    "direction": "CALL" if side_char == "C" else "PUT",
                    "pnl_usd": pnl_val,
                    "pnl_pct": float(pct_str),
                    "hold_days": float(days_str),
                    "exit_reason": reason,
                    "won": pnl_val > 0,
                    "source": "log",
                })
        except Exception:
            continue

    # De-duplicate by (timestamp, symbol, pnl) since logs could overlap
    seen = set()
    deduped = []
    for t in trades:
        key = (t["timestamp"], t["symbol"], t["pnl_usd"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped


def find_latest_log(log_dir: Path, prefix: str) -> Path | None:
    if not log_dir.exists():
        return None
    candidates = sorted(
        log_dir.glob(f"{prefix}*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_recent_logs(log_dir: Path, prefix: str, days: int = 3) -> list[Path]:
    """Find all log files from the last N days."""
    if not log_dir.exists():
        return []
    cutoff = datetime.now() - timedelta(days=days)
    return sorted(
        [p for p in log_dir.glob(f"{prefix}*.log")
         if datetime.fromtimestamp(p.stat().st_mtime) > cutoff],
        key=lambda p: p.stat().st_mtime,
    )


def get_processes() -> list[dict]:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process pythonw -EA SilentlyContinue | "
             "Select-Object Id, @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB)}}, "
             "StartTime, @{N='Cmd';E={(Get-CimInstance Win32_Process -Filter "
             "\"ProcessId=$($_.Id)\").CommandLine}} | ConvertTo-Json"],
            capture_output=True, text=True, timeout=15,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        if r.stdout.strip():
            data = json.loads(r.stdout)
            return data if isinstance(data, list) else [data]
    except Exception:
        pass
    return []


def query_dashboard(endpoint: str = "/api/all") -> dict | None:
    try:
        req = Request(f"{DASHBOARD_URL}{endpoint}", method="GET")
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════

def parse_trade_timestamp(ts_str: str) -> datetime | None:
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None


def analyze_trades(trades: list[dict], bot_name: str) -> dict:
    """Deep trade-by-trade analysis."""
    if not trades:
        return {"has_trades": False, "total": 0}

    now = datetime.now()
    analysis = {"has_trades": True, "total": len(trades)}

    # Parse timestamps and P&L
    parsed = []
    for t in trades:
        ts_raw = t.get("timestamp", "")
        ts = parse_trade_timestamp(ts_raw)
        pnl = t.get("pnl_usd") or t.get("pnl") or t.get("pnl_dollar") or 0
        pnl_pct = t.get("pnl_pct") or 0
        if isinstance(pnl, str):
            try: pnl = float(pnl)
            except: pnl = 0
        if isinstance(pnl_pct, str):
            try: pnl_pct = float(pnl_pct)
            except: pnl_pct = 0
        symbol = t.get("symbol") or t.get("underlying") or "?"
        reason = t.get("exit_reason") or t.get("reason") or "?"
        direction = t.get("direction") or t.get("side") or t.get("spread_type") or "?"
        won_raw = t.get("won")
        if isinstance(won_raw, bool):
            won = won_raw
        elif isinstance(won_raw, str):
            won = won_raw.upper() == "WIN"
        else:
            won = pnl > 0
        parsed.append({
            "ts": ts, "pnl": float(pnl), "pnl_pct": float(pnl_pct),
            "symbol": symbol, "reason": reason, "direction": direction,
            "won": won, "raw": t,
        })

    # Sort by time
    parsed.sort(key=lambda x: x["ts"] or datetime.min)

    # ── Overall Stats ────────────────────────────────────────────────
    wins = [t for t in parsed if t["won"]]
    losses = [t for t in parsed if not t["won"]]
    analysis["wins"] = len(wins)
    analysis["losses"] = len(losses)
    analysis["win_rate"] = len(wins) / len(parsed) if parsed else 0

    pnls = [t["pnl"] for t in parsed]
    analysis["total_pnl"] = sum(pnls)
    analysis["avg_win"] = statistics.mean([t["pnl"] for t in wins]) if wins else 0
    analysis["avg_loss"] = statistics.mean([t["pnl"] for t in losses]) if losses else 0
    analysis["best_trade"] = max(pnls) if pnls else 0
    analysis["worst_trade"] = min(pnls) if pnls else 0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    analysis["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # ── Time-Based Analysis ──────────────────────────────────────────
    # Last 3 days vs previous period
    three_days_ago = now - timedelta(days=3)
    recent = [t for t in parsed if t["ts"] and t["ts"] > three_days_ago]
    older = [t for t in parsed if t["ts"] and t["ts"] <= three_days_ago]

    analysis["recent_trades"] = len(recent)
    analysis["recent_wins"] = len([t for t in recent if t["won"]])
    analysis["recent_wr"] = (analysis["recent_wins"] / len(recent)) if recent else None
    analysis["recent_pnl"] = sum(t["pnl"] for t in recent)

    analysis["older_trades"] = len(older)
    analysis["older_wr"] = (len([t for t in older if t["won"]]) / len(older)) if older else None

    # Trend detection
    if analysis["recent_wr"] is not None and analysis["older_wr"] is not None:
        delta = analysis["recent_wr"] - analysis["older_wr"]
        if delta > 0.1:
            analysis["trend"] = "IMPROVING"
        elif delta < -0.1:
            analysis["trend"] = "DECLINING"
        else:
            analysis["trend"] = "STABLE"
    else:
        analysis["trend"] = "INSUFFICIENT_DATA"

    # ── Per-Symbol Breakdown ─────────────────────────────────────────
    by_symbol = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0, "trades": 0})
    for t in parsed:
        s = t["symbol"]
        by_symbol[s]["trades"] += 1
        by_symbol[s]["pnl"] += t["pnl"]
        if t["won"]:
            by_symbol[s]["wins"] += 1
        else:
            by_symbol[s]["losses"] += 1

    # Best and worst symbols
    symbol_stats = []
    for sym, data in by_symbol.items():
        data["symbol"] = sym
        data["win_rate"] = data["wins"] / data["trades"] if data["trades"] > 0 else 0
        symbol_stats.append(data)

    symbol_stats.sort(key=lambda x: x["pnl"], reverse=True)
    analysis["best_symbols"] = symbol_stats[:5]
    analysis["worst_symbols"] = symbol_stats[-5:][::-1] if len(symbol_stats) > 5 else list(reversed(symbol_stats[-3:]))

    # Symbols that should be blacklisted (> 3 trades, < 25% WR, negative P&L)
    analysis["blacklist_candidates"] = [
        s for s in symbol_stats
        if s["trades"] >= 3 and s["win_rate"] < 0.25 and s["pnl"] < 0
    ]

    # ── Exit Reason Analysis ─────────────────────────────────────────
    by_reason = defaultdict(lambda: {"count": 0, "wins": 0, "total_pnl": 0})
    for t in parsed:
        r = t["reason"]
        by_reason[r]["count"] += 1
        by_reason[r]["total_pnl"] += t["pnl"]
        if t["won"]:
            by_reason[r]["wins"] += 1

    reason_stats = []
    for reason, data in by_reason.items():
        data["reason"] = reason
        data["win_rate"] = data["wins"] / data["count"] if data["count"] > 0 else 0
        data["avg_pnl"] = data["total_pnl"] / data["count"] if data["count"] > 0 else 0
        reason_stats.append(data)
    reason_stats.sort(key=lambda x: x["total_pnl"])
    analysis["exit_reasons"] = reason_stats

    # ── Streak Analysis ──────────────────────────────────────────────
    max_win_streak = max_loss_streak = current_streak = 0
    streak_type = None
    for t in parsed:
        if t["won"]:
            if streak_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if streak_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    analysis["max_win_streak"] = max_win_streak
    analysis["max_loss_streak"] = max_loss_streak
    analysis["current_streak_type"] = streak_type
    analysis["current_streak_len"] = current_streak

    # ── Direction Analysis (CryptoBot) ───────────────────────────────
    by_direction = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
    for t in parsed:
        d = t["direction"].upper()
        if d in ("LONG", "SHORT", "PUT", "CALL"):
            by_direction[d]["pnl"] += t["pnl"]
            if t["won"]:
                by_direction[d]["wins"] += 1
            else:
                by_direction[d]["losses"] += 1
    analysis["by_direction"] = dict(by_direction)

    # ── Equity Curve (cumulative P&L) ────────────────────────────────
    cumulative = []
    running = 0
    for t in parsed:
        running += t["pnl"]
        cumulative.append(running)
    analysis["equity_curve"] = cumulative

    # Max drawdown on equity curve
    peak = 0
    max_dd = 0
    for val in cumulative:
        peak = max(peak, val)
        dd = val - peak
        max_dd = min(max_dd, dd)
    analysis["max_drawdown_pnl"] = max_dd

    return analysis


def analyze_log_patterns(log_text: str, bot_name: str) -> dict:
    """Extract patterns and issues from log text."""
    patterns = {
        "errors": [],
        "warnings": [],
        "trade_events": [],
        "cycle_count": 0,
        "filter_stats": {},
        "issues": [],
    }

    lines = log_text.split("\n")

    error_counter = Counter()
    warning_counter = Counter()

    for line in lines:
        # Count cycles
        if re.search(r"Cycle \d+|cycle.*\d+.*complete", line, re.IGNORECASE):
            patterns["cycle_count"] += 1

        # Collect errors (deduplicate by pattern)
        if re.search(r"\bERROR\b|\bCRITICAL\b|Traceback", line, re.IGNORECASE):
            # Extract error category
            m = re.search(r"ERROR[:\s|]+\S+\.(\w+):\s*(.+)", line)
            if m:
                category = m.group(1) + ": " + m.group(2)[:80]
            else:
                category = line.strip()[:100]
            error_counter[category] += 1

        # Collect warnings
        if re.search(r"\bWARNING\b", line, re.IGNORECASE):
            m = re.search(r"WARNING[:\s|]+(.+)", line)
            if m:
                warning_counter[m.group(1).strip()[:80]] += 1

        # CryptoBot signal filters
        m = re.search(r"Signal filters:(.+)", line)
        if m:
            for kv in re.findall(r"(\w+)=(\d+)", m.group(1)):
                key, val = kv
                if key not in patterns["filter_stats"]:
                    patterns["filter_stats"][key] = []
                patterns["filter_stats"][key].append(int(val))

        # Trade events
        if re.search(r"\bOPEN\b.*\b(LONG|SHORT|PUT|CALL)\b|\bCLOSED\b|\bWIN\b|\bLOSS\b", line, re.IGNORECASE):
            patterns["trade_events"].append(line.strip()[:150])

    patterns["errors"] = [{"msg": k, "count": v} for k, v in error_counter.most_common(10)]
    patterns["warnings"] = [{"msg": k, "count": v} for k, v in warning_counter.most_common(10)]

    # Analyze filter stats for CryptoBot
    if patterns["filter_stats"]:
        filter_summary = {}
        for key, vals in patterns["filter_stats"].items():
            if vals:
                filter_summary[key] = {
                    "avg": round(statistics.mean(vals), 1),
                    "max": max(vals),
                    "total_blocked": sum(vals),
                }
        patterns["filter_summary"] = filter_summary

        # Flag problematic filters
        for key, summary in filter_summary.items():
            if key.startswith("skipped_") and summary["avg"] > 5:
                patterns["issues"].append(
                    f"Filter '{key}' blocking avg {summary['avg']:.0f} symbols/cycle "
                    f"(total {summary['total_blocked']} blocks)"
                )

    # Flag high error rates
    total_errors = sum(e["count"] for e in patterns["errors"])
    if total_errors > 20:
        patterns["issues"].append(f"{total_errors} errors in recent logs — needs investigation")

    return patterns


def build_recommendations(bot_name: str, trade_analysis: dict, log_patterns: dict,
                          state: dict, positions: list) -> list[dict]:
    """Generate actionable recommendations based on analysis."""
    recs = []

    def add(severity: str, title: str, detail: str):
        recs.append({"severity": severity, "title": title, "detail": detail})

    ta = trade_analysis
    lp = log_patterns

    if not ta.get("has_trades"):
        if not BOTS[bot_name].get("dormant"):
            add("WARNING", "No trades recorded",
                f"{bot_name} has 0 trade records in CSV. Either the bot hasn't entered any "
                f"trades or trade logging to CSV is broken. Check the bot's trade recording code.")
        return recs

    # ── Win Rate Issues ──────────────────────────────────────────────
    wr = ta.get("win_rate", 0)
    if wr < 0.30 and ta["total"] >= 10:
        add("CRITICAL", f"Win rate catastrophically low ({wr:.0%})",
            f"{ta['wins']}W / {ta['losses']}L across {ta['total']} trades. "
            f"Avg win: ${ta['avg_win']:+.2f}, Avg loss: ${ta['avg_loss']:+.2f}. "
            f"Profit factor: {ta['profit_factor']:.2f}. "
            f"Consider pausing this bot and reviewing entry criteria.")
    elif wr < 0.40 and ta["total"] >= 10:
        add("WARNING", f"Win rate below healthy ({wr:.0%})",
            f"Need ≥40% WR for viability. Currently {ta['wins']}W/{ta['losses']}L. "
            f"Review entry filters, consider tightening confidence thresholds.")

    # ── Profit Factor ────────────────────────────────────────────────
    pf = ta.get("profit_factor", 0)
    if pf < 1.0 and ta["total"] >= 5:
        add("CRITICAL", f"Negative expectancy (PF={pf:.2f})",
            f"Profit factor below 1.0 means the bot loses money on average. "
            f"Avg win ${ta['avg_win']:+.2f} vs avg loss ${ta['avg_loss']:+.2f}. "
            f"Either improve win rate or increase win/loss size ratio.")
    elif pf < 1.5 and ta["total"] >= 10:
        add("WARNING", f"Marginal profit factor ({pf:.2f})",
            f"PF should be ≥1.5 for comfortable margin. "
            f"Currently winning ${ta['avg_win']:.2f} avg vs losing ${abs(ta['avg_loss']):.2f} avg.")

    # ── Trend Detection ──────────────────────────────────────────────
    if ta.get("trend") == "DECLINING":
        add("WARNING", "Performance declining",
            f"Recent WR ({ta['recent_wr']:.0%} over {ta['recent_trades']} trades) "
            f"is worse than earlier ({ta['older_wr']:.0%} over {ta['older_trades']}). "
            f"May indicate market regime change or strategy decay.")
    elif ta.get("trend") == "IMPROVING":
        add("INFO", "Performance improving",
            f"Recent WR ({ta['recent_wr']:.0%}) is better than earlier ({ta['older_wr']:.0%}). "
            f"Recent fixes or market conditions are helping.")

    # ── Loss Streaks ─────────────────────────────────────────────────
    if ta.get("max_loss_streak", 0) >= 6:
        add("WARNING", f"Max loss streak: {ta['max_loss_streak']} in a row",
            f"Extended losing streaks suggest systematic issues, not just bad luck. "
            f"Current streak: {ta['current_streak_len']} {'wins' if ta['current_streak_type'] == 'win' else 'losses'}.")

    if ta.get("current_streak_type") == "loss" and ta.get("current_streak_len", 0) >= 4:
        add("WARNING", f"Currently on {ta['current_streak_len']}-loss streak",
            f"Active losing streak may trigger risk controls. "
            f"Meta-learner should be raising confidence thresholds.")

    # ── Symbol Concentration ─────────────────────────────────────────
    if ta.get("blacklist_candidates"):
        bad_syms = ", ".join(s["symbol"] for s in ta["blacklist_candidates"][:5])
        add("ACTION", f"Consider blacklisting: {bad_syms}",
            f"{len(ta['blacklist_candidates'])} symbols have ≥3 trades, <25% WR, and negative P&L. "
            f"These are consistently losing money.")

    worst = ta.get("worst_symbols", [])
    if worst and worst[0]["pnl"] < -50:
        sym = worst[0]
        add("WARNING", f"Worst symbol: {sym['symbol']} (${sym['pnl']:+.2f})",
            f"{sym['wins']}W/{sym['losses']}L ({sym['win_rate']:.0%} WR). "
            f"This symbol is a significant drag on performance.")

    # ── Exit Reason Analysis ─────────────────────────────────────────
    for er in ta.get("exit_reasons", []):
        if er["count"] >= 3 and er["win_rate"] < 0.2:
            add("WARNING", f"Exit reason '{er['reason']}' has {er['win_rate']:.0%} WR",
                f"{er['count']} trades exited via '{er['reason']}' with avg P&L ${er['avg_pnl']:+.2f}. "
                f"This exit path is almost always a loss — review the trigger logic.")

    # ── Direction Imbalance ──────────────────────────────────────────
    dirs = ta.get("by_direction", {})
    if "LONG" in dirs and "SHORT" in dirs:
        long_total = dirs["LONG"]["wins"] + dirs["LONG"]["losses"]
        short_total = dirs["SHORT"]["wins"] + dirs["SHORT"]["losses"]
        if long_total >= 3 and short_total >= 3:
            long_wr = dirs["LONG"]["wins"] / long_total
            short_wr = dirs["SHORT"]["wins"] / short_total
            if abs(long_wr - short_wr) > 0.20:
                worse = "SHORT" if short_wr < long_wr else "LONG"
                better = "LONG" if worse == "SHORT" else "SHORT"
                add("ACTION", f"{worse} trades underperform {better}",
                    f"{better}: {dirs[better]['wins']}W/{dirs[better]['losses']}L "
                    f"(${dirs[better]['pnl']:+.2f}). "
                    f"{worse}: {dirs[worse]['wins']}W/{dirs[worse]['losses']}L "
                    f"(${dirs[worse]['pnl']:+.2f}). "
                    f"Consider reducing or disabling {worse} entries.")

    # ── Log-Based Issues ─────────────────────────────────────────────
    for issue in lp.get("issues", []):
        add("WARNING", "Log pattern issue", issue)

    if lp.get("errors"):
        top_error = lp["errors"][0]
        if top_error["count"] >= 5:
            add("WARNING", f"Recurring error ({top_error['count']}x)",
                f"{top_error['msg']}")

    # ── Filter Effectiveness (CryptoBot) ─────────────────────────────
    fs = lp.get("filter_summary", {})
    if "skipped_sym_health" in fs and fs["skipped_sym_health"]["avg"] > 10:
        add("ACTION", f"sym_health filter blocking {fs['skipped_sym_health']['avg']:.0f} avg symbols",
            "Symbol health filter may be too aggressive. "
            "Check if profitable symbols are being excluded.")

    if "skipped_trend_mismatch" in fs and fs["skipped_trend_mismatch"]["avg"] > 8:
        add("INFO", f"Trend mismatch filter active ({fs['skipped_trend_mismatch']['avg']:.0f} avg)",
            "Trend filter is working as expected — blocking counter-trend entries.")

    # ── Position Alerts ──────────────────────────────────────────────
    if positions:
        now = datetime.now()
        for pos in positions:
            entry_str = pos.get("entry_time") or pos.get("open_date")
            if entry_str:
                try:
                    et = datetime.fromisoformat(entry_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    age_h = (now - et).total_seconds() / 3600
                    if age_h > 12:
                        add("WARNING", f"Position {pos.get('symbol', '?')} is {age_h:.0f}h old",
                            "Long-held position may be stuck. Check if exit conditions are being evaluated.")
                except (ValueError, TypeError):
                    pass

    # ── Inactivity ───────────────────────────────────────────────────
    if ta["recent_trades"] == 0 and not BOTS[bot_name].get("dormant"):
        add("WARNING", "No trades in last 3 days",
            f"Bot hasn't traded recently despite {lp.get('cycle_count', 0)} cycles. "
            f"Check if entry filters are too restrictive.")

    return recs


# ═══════════════════════════════════════════════════════════════════════
#  PERFORMANCE HISTORY TRACKING
# ═══════════════════════════════════════════════════════════════════════

def load_history() -> list[dict]:
    """Load rolling performance history."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_snapshot(reports: list[dict]):
    """Save today's metrics as a history snapshot."""
    history = load_history()

    snapshot = {
        "date": date.today().isoformat(),
        "bots": {},
    }
    for r in reports:
        snapshot["bots"][r["name"]] = {
            "balance": r.get("state", {}).get("balance", 0),
            "daily_pnl": r.get("state", {}).get("daily_pnl", 0),
            "total_pnl": r.get("state", {}).get("total_pnl", 0),
            "total_trades": r.get("trade_analysis", {}).get("total", 0),
            "win_rate": r.get("trade_analysis", {}).get("win_rate"),
            "open_positions": r.get("n_positions", 0),
            "status": r.get("status", "UNKNOWN"),
            "consecutive_losses": r.get("state", {}).get("consecutive_losses", 0),
        }

    # Deduplicate by date
    history = [h for h in history if h["date"] != snapshot["date"]]
    history.append(snapshot)

    # Keep last 90 days
    cutoff = (date.today() - timedelta(days=90)).isoformat()
    history = [h for h in history if h["date"] >= cutoff]

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def analyze_trends(history: list[dict], bot_name: str) -> dict:
    """Analyze multi-day trends from history."""
    points = []
    for snap in history:
        if bot_name in snap.get("bots", {}):
            points.append({
                "date": snap["date"],
                **snap["bots"][bot_name],
            })

    if len(points) < 2:
        return {"has_history": False}

    latest = points[-1]
    first = points[0]

    trend = {
        "has_history": True,
        "days_tracked": len(points),
        "first_date": first["date"],
        "balances": [p.get("balance", 0) for p in points],
        "dates": [p["date"] for p in points],
    }

    # Balance trend
    b_first = first.get("balance", 0)
    b_latest = latest.get("balance", 0)
    if b_first > 0:
        trend["balance_change_pct"] = ((b_latest - b_first) / b_first) * 100

    # Win rate trend
    wrs = [p.get("win_rate") for p in points if p.get("win_rate") is not None]
    if len(wrs) >= 2:
        trend["wr_first"] = wrs[0]
        trend["wr_latest"] = wrs[-1]
        trend["wr_change"] = wrs[-1] - wrs[0]

    return trend


# ═══════════════════════════════════════════════════════════════════════
#  PER-BOT AUDIT
# ═══════════════════════════════════════════════════════════════════════

def audit_bot(bot_name: str, bot_info: dict, processes: list[dict]) -> dict:
    """Run comprehensive audit for one bot."""
    report = {
        "name": bot_name,
        "category": bot_info["category"],
        "dormant": bot_info.get("dormant", False),
        "status": "UNKNOWN",
        "state": {},
        "trade_analysis": {},
        "log_patterns": {},
        "recommendations": [],
        "positions_raw": [],
        "n_positions": 0,
        "process_alive": False,
        "process_info": {},
    }

    # ── Process ──────────────────────────────────────────────────────
    root_lower = str(bot_info["root"]).lower().replace("\\", "/")
    for proc in processes:
        cmd = (proc.get("Cmd") or "").lower().replace("\\", "/")
        if root_lower in cmd and proc.get("WS_MB", 0) > 8:
            report["process_alive"] = True
            report["process_info"] = {
                "pid": proc.get("Id"),
                "memory_mb": proc.get("WS_MB"),
                "start_time": proc.get("StartTime"),
            }
            break

    # ── State ────────────────────────────────────────────────────────
    state_path = bot_info["state_dir"] / bot_info["state_file"]
    state = read_json(state_path)
    if state:
        if bot_name == "CryptoBot":
            report["state"] = {
                "balance": state.get("spot", 0) + state.get("futures", 0),
                "spot": state.get("spot", 0),
                "futures": state.get("futures", 0),
                "daily_pnl": state.get("daily_pnl", 0),
                "peak_balance": state.get("peak_balance", 0),
                "consecutive_losses": state.get("consecutive_losses", 0),
            }
        else:
            report["state"] = {
                "balance": state.get("current_balance", 0),
                "daily_pnl": state.get("daily_pnl", 0),
                "total_pnl": state.get("total_pnl", 0),
                "peak_balance": state.get("peak_balance", 0),
                "wins": state.get("wins", 0),
                "losses": state.get("losses", 0),
                "total_trades": state.get("total_trades", 0),
                "consecutive_losses": state.get("consecutive_losses", 0),
            }
            tt = report["state"]["total_trades"]
            if tt > 0:
                report["state"]["win_rate"] = report["state"]["wins"] / tt

    # ── Positions ────────────────────────────────────────────────────
    pos_path = bot_info["state_dir"] / bot_info["positions_file"]
    positions = read_json(pos_path)
    if positions and isinstance(positions, dict):
        report["positions_raw"] = list(positions.values())
        report["n_positions"] = len(positions)

    # ── Trade History ────────────────────────────────────────────────
    trades = read_csv_trades(bot_info["trades_csv"])
    # Fallback for CryptoBot: check history dir
    if not trades and bot_name == "CryptoBot":
        fallback = Path("C:/Bot/data/history/trade_history.csv")
        trades = read_csv_trades(fallback)
    # Fallback for options bots: extract trades from logs
    if not trades and bot_name in ("PutSeller", "CallBuyer"):
        trades = extract_trades_from_logs(bot_info["log_dir"], bot_info["log_prefix"])
        if trades:
            report["trade_source"] = "logs"
    report["trade_analysis"] = analyze_trades(trades, bot_name)

    # ── Log Analysis ─────────────────────────────────────────────────
    recent_logs = find_recent_logs(bot_info["log_dir"], bot_info["log_prefix"], days=3)
    combined_log = ""
    for lf in recent_logs:
        combined_log += read_log_tail(lf, n_bytes=50_000) + "\n"
    report["log_patterns"] = analyze_log_patterns(combined_log, bot_name)

    # ── Historical Trends ────────────────────────────────────────────
    history = load_history()
    report["trends"] = analyze_trends(history, bot_name)

    # ── Recommendations ──────────────────────────────────────────────
    report["recommendations"] = build_recommendations(
        bot_name, report["trade_analysis"], report["log_patterns"],
        report["state"], report["positions_raw"],
    )

    # ── Status ───────────────────────────────────────────────────────
    if report["dormant"]:
        report["status"] = "DORMANT"
    else:
        severities = [r["severity"] for r in report["recommendations"]]
        if "CRITICAL" in severities:
            report["status"] = "CRITICAL"
        elif "WARNING" in severities or "ACTION" in severities:
            report["status"] = "NEEDS_ATTENTION"
        elif report["process_alive"]:
            report["status"] = "HEALTHY"
        else:
            report["status"] = "DOWN"

    return report


# ═══════════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════

def severity_color(sev: str) -> str:
    return {
        "CRITICAL": "#ef4444", "WARNING": "#f59e0b", "ACTION": "#3b82f6",
        "INFO": "#22c55e",
    }.get(sev, "#94a3b8")


def status_color(status: str) -> str:
    return {
        "HEALTHY": "#22c55e", "NEEDS_ATTENTION": "#f59e0b",
        "CRITICAL": "#ef4444", "DOWN": "#ef4444",
        "DORMANT": "#64748b", "UNKNOWN": "#6b7280",
    }.get(status, "#6b7280")


def generate_html(reports: list[dict], account: dict | None) -> str:
    now = datetime.now()

    # Overall verdict
    active = [r for r in reports if not r["dormant"]]
    statuses = [r["status"] for r in active]
    if "CRITICAL" in statuses or "DOWN" in statuses:
        verdict = "CRITICAL"
        verdict_msg = "Issues found — action needed"
    elif "NEEDS_ATTENTION" in statuses:
        verdict = "NEEDS_ATTENTION"
        verdict_msg = "Some items need review"
    elif all(s == "HEALTHY" for s in statuses):
        verdict = "HEALTHY"
        verdict_msg = "All systems performing well"
    else:
        verdict = "UNKNOWN"
        verdict_msg = "Insufficient data for assessment"
    vc = status_color(verdict)

    # Account section
    acct_html = ""
    if account:
        eq = account.get("equity", 0)
        if isinstance(eq, str):
            try: eq = float(eq)
            except: eq = 0
        ret = ((eq - 100000) / 100000) * 100
        rc = "#22c55e" if ret >= 0 else "#ef4444"
        acct_html = f'''
        <div class="card">
            <h2>Alpaca Account</h2>
            <div class="metrics-grid">
                <div class="m"><span class="ml">Equity</span><span class="mv">${eq:,.2f}</span></div>
                <div class="m"><span class="ml">Return</span><span class="mv" style="color:{rc}">{ret:+.1f}%</span></div>
                <div class="m"><span class="ml">Cash</span><span class="mv">${float(account.get("cash",0)):,.2f}</span></div>
                <div class="m"><span class="ml">Buying Power</span><span class="mv">${float(account.get("buying_power",0)):,.2f}</span></div>
            </div>
        </div>'''

    # Per-bot sections
    bot_sections = ""
    for r in reports:
        sc = status_color(r["status"])
        ta = r["trade_analysis"]
        st = r["state"]
        lp = r["log_patterns"]
        tr = r.get("trends", {})

        # Header
        section = f'''
        <div class="card">
            <div class="bot-hdr" style="border-left: 4px solid {sc};">
                <div>
                    <span class="bot-nm">{r["name"]}</span>
                    <span class="bot-cat">{r["category"]}</span>
                </div>
                <span class="bot-st" style="color:{sc}">{r["status"]}</span>
            </div>'''

        # Process + Balance
        if r["process_alive"]:
            pi = r["process_info"]
            section += f'<div class="m"><span class="ml">Process</span><span class="mv ok">PID {pi.get("pid")} ({pi.get("memory_mb")}MB)</span></div>'
        elif not r["dormant"]:
            section += '<div class="m"><span class="ml">Process</span><span class="mv crit">NOT RUNNING</span></div>'

        if st.get("balance", 0) > 0 or st.get("total_trades", 0) > 0:
            bal = st.get("balance", 0)
            init = BOTS[r["name"]]["initial_balance"]
            ret_pct = ((bal - init) / init * 100) if init > 0 else 0
            rc = "#22c55e" if ret_pct >= 0 else "#ef4444"
            section += f'''
            <div class="metrics-grid">
                <div class="m"><span class="ml">Balance</span><span class="mv">${bal:,.2f}</span></div>
                <div class="m"><span class="ml">Return</span><span class="mv" style="color:{rc}">{ret_pct:+.1f}%</span></div>
                <div class="m"><span class="ml">Daily P&L</span><span class="mv">${st.get("daily_pnl",0):+,.2f}</span></div>
                <div class="m"><span class="ml">Positions</span><span class="mv">{r["n_positions"]}</span></div>
            </div>'''

        # Trade Analysis
        if ta.get("has_trades"):
            wr = ta["win_rate"]
            wrc = "#22c55e" if wr >= 0.5 else ("#f59e0b" if wr >= 0.4 else "#ef4444")
            pf = ta["profit_factor"]
            pfc = "#22c55e" if pf >= 1.5 else ("#f59e0b" if pf >= 1.0 else "#ef4444")
            trend = ta.get("trend", "?")
            trend_icon = {"IMPROVING": "&#x25B2;", "DECLINING": "&#x25BC;", "STABLE": "&#x25CF;"}.get(trend, "?")
            trend_color = {"IMPROVING": "#22c55e", "DECLINING": "#ef4444", "STABLE": "#94a3b8"}.get(trend, "#94a3b8")

            section += f'''
            <h3>Trade Performance</h3>
            <div class="metrics-grid">
                <div class="m"><span class="ml">Win Rate</span><span class="mv" style="color:{wrc}">{wr:.0%} ({ta["wins"]}W/{ta["losses"]}L)</span></div>
                <div class="m"><span class="ml">Profit Factor</span><span class="mv" style="color:{pfc}">{pf:.2f}</span></div>
                <div class="m"><span class="ml">Total P&L</span><span class="mv">${ta["total_pnl"]:+,.2f}</span></div>
                <div class="m"><span class="ml">Avg Win / Loss</span><span class="mv">${ta["avg_win"]:+.2f} / ${ta["avg_loss"]:+.2f}</span></div>
                <div class="m"><span class="ml">Best / Worst</span><span class="mv">${ta["best_trade"]:+.2f} / ${ta["worst_trade"]:+.2f}</span></div>
                <div class="m"><span class="ml">Max Loss Streak</span><span class="mv">{ta["max_loss_streak"]}</span></div>
                <div class="m"><span class="ml">Trend</span><span class="mv" style="color:{trend_color}">{trend_icon} {trend}</span></div>
                <div class="m"><span class="ml">Recent (3d)</span><span class="mv">{ta["recent_trades"]} trades, ${ta["recent_pnl"]:+.2f}</span></div>
            </div>'''

            # Best/worst symbols
            if ta.get("best_symbols"):
                section += '<h3>Top Symbols</h3><div class="sym-table"><table><tr><th>Symbol</th><th>Trades</th><th>WR</th><th>P&L</th></tr>'
                for s in ta["best_symbols"][:5]:
                    c = "#22c55e" if s["pnl"] >= 0 else "#ef4444"
                    section += f'<tr><td>{s["symbol"]}</td><td>{s["trades"]}</td><td>{s["win_rate"]:.0%}</td><td style="color:{c}">${s["pnl"]:+.2f}</td></tr>'
                section += '</table></div>'

            if ta.get("worst_symbols"):
                section += '<h3>Worst Symbols</h3><div class="sym-table"><table><tr><th>Symbol</th><th>Trades</th><th>WR</th><th>P&L</th></tr>'
                for s in ta["worst_symbols"][:5]:
                    c = "#22c55e" if s["pnl"] >= 0 else "#ef4444"
                    section += f'<tr><td>{s["symbol"]}</td><td>{s["trades"]}</td><td>{s["win_rate"]:.0%}</td><td style="color:{c}">${s["pnl"]:+.2f}</td></tr>'
                section += '</table></div>'

            # Exit reason breakdown
            if ta.get("exit_reasons"):
                section += '<h3>Exit Reasons</h3><div class="sym-table"><table><tr><th>Reason</th><th>Count</th><th>WR</th><th>Avg P&L</th></tr>'
                for er in ta["exit_reasons"]:
                    c = "#22c55e" if er["avg_pnl"] >= 0 else "#ef4444"
                    section += f'<tr><td>{er["reason"]}</td><td>{er["count"]}</td><td>{er["win_rate"]:.0%}</td><td style="color:{c}">${er["avg_pnl"]:+.2f}</td></tr>'
                section += '</table></div>'

            # Direction analysis
            dirs = ta.get("by_direction", {})
            if dirs:
                section += '<h3>Direction Analysis</h3><div class="sym-table"><table><tr><th>Direction</th><th>W/L</th><th>P&L</th></tr>'
                for d, data in dirs.items():
                    total = data["wins"] + data["losses"]
                    c = "#22c55e" if data["pnl"] >= 0 else "#ef4444"
                    wr_d = data["wins"] / total if total > 0 else 0
                    section += f'<tr><td>{d}</td><td>{data["wins"]}W/{data["losses"]}L ({wr_d:.0%})</td><td style="color:{c}">${data["pnl"]:+.2f}</td></tr>'
                section += '</table></div>'

        # Log patterns
        if lp.get("errors"):
            section += f'<h3>Recent Errors ({sum(e["count"] for e in lp["errors"])} total)</h3><div class="errors">'
            for e in lp["errors"][:5]:
                section += f'<div class="err-line">[x{e["count"]}] {e["msg"]}</div>'
            section += '</div>'

        # Recommendations
        if r["recommendations"]:
            section += '<h3>Recommendations</h3><div class="recs">'
            for rec in r["recommendations"]:
                rc = severity_color(rec["severity"])
                section += f'''<div class="rec" style="border-left:3px solid {rc}">
                    <div class="rec-hdr"><span class="rec-sev" style="color:{rc}">{rec["severity"]}</span> {rec["title"]}</div>
                    <div class="rec-detail">{rec["detail"]}</div>
                </div>'''
            section += '</div>'
        elif not r["dormant"]:
            section += '<div class="no-issues">No issues detected</div>'

        section += '</div>'
        bot_sections += section

    html = f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bot Audit — {now.strftime("%b %d, %Y")}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:20px;line-height:1.5}}
.container{{max-width:960px;margin:0 auto}}
h1{{font-size:1.6em;color:#f8fafc;margin-bottom:2px}}
h2{{font-size:1.1em;color:#94a3b8;margin-bottom:10px}}
h3{{font-size:0.95em;color:#94a3b8;margin:14px 0 6px;border-bottom:1px solid #1e293b;padding-bottom:4px}}
.subtitle{{color:#64748b;font-size:0.9em;margin-bottom:16px}}
.verdict{{background:#1e293b;border:1px solid {vc}40;border-radius:8px;padding:14px 18px;margin-bottom:16px;display:flex;align-items:center;gap:12px}}
.verdict-dot{{width:14px;height:14px;border-radius:50%;background:{vc};box-shadow:0 0 8px {vc}80}}
.verdict-text{{font-size:1.1em;font-weight:700;color:{vc}}}
.verdict-msg{{color:#94a3b8;font-size:0.9em}}
.card{{background:#1e293b;border-radius:8px;padding:16px;margin-bottom:12px}}
.bot-hdr{{display:flex;justify-content:space-between;align-items:center;padding-left:12px;margin-bottom:10px}}
.bot-nm{{font-size:1.15em;font-weight:700}}
.bot-cat{{color:#64748b;font-size:0.8em;margin-left:8px;text-transform:uppercase}}
.bot-st{{font-weight:700;font-size:0.85em;text-transform:uppercase}}
.metrics-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:5px}}
.m{{display:flex;justify-content:space-between;padding:4px 8px;background:#0f172a;border-radius:4px;font-size:0.85em}}
.ml{{color:#64748b}}.mv{{font-weight:600}}.mv.ok{{color:#22c55e}}.mv.crit{{color:#ef4444}}
.sym-table{{overflow-x:auto;margin:4px 0}}
table{{width:100%;border-collapse:collapse;font-size:0.85em}}
th{{text-align:left;color:#64748b;padding:4px 8px;border-bottom:1px solid #334155}}
td{{padding:4px 8px;border-bottom:1px solid #1e293b20}}
.errors{{margin:4px 0}}
.err-line{{font-size:0.8em;color:#f87171;padding:2px 8px;background:#0f172a;border-radius:3px;margin:2px 0;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.recs{{margin:6px 0}}
.rec{{padding:8px 12px;margin:4px 0;background:#0f172a;border-radius:6px}}
.rec-hdr{{font-weight:600;font-size:0.9em}}
.rec-sev{{font-size:0.75em;text-transform:uppercase;font-weight:700}}
.rec-detail{{font-size:0.82em;color:#94a3b8;margin-top:2px}}
.no-issues{{color:#22c55e;font-size:0.85em;padding:8px;text-align:center;background:#22c55e10;border-radius:6px}}
.footer{{text-align:center;color:#475569;font-size:0.75em;margin-top:20px;padding-top:12px;border-top:1px solid #1e293b}}
</style></head><body>
<div class="container">
<h1>Bot Audit & Performance Analysis</h1>
<div class="subtitle">{now.strftime("%A, %B %d, %Y at %I:%M %p MT")} — Auto-generated every 3 days</div>
<div class="verdict">
    <div class="verdict-dot"></div>
    <span class="verdict-text">{verdict}</span>
    <span class="verdict-msg">{verdict_msg}</span>
</div>
{acct_html}
{bot_sections}
<div class="footer">Generated by bot_audit.py — Next run in 3 days</div>
</div></body></html>'''
    return html


# ═══════════════════════════════════════════════════════════════════════
#  TEXT SUMMARY (for quick console view)
# ═══════════════════════════════════════════════════════════════════════

def generate_text(reports: list[dict], account: dict | None) -> str:
    now = datetime.now()
    lines = [
        "=" * 72,
        f"  BOT AUDIT & PERFORMANCE ANALYSIS",
        f"  {now.strftime('%A, %B %d, %Y at %I:%M %p MT')}",
        "=" * 72, "",
    ]

    if account:
        eq = float(account.get("equity", 0))
        lines.append(f"  ALPACA: ${eq:,.2f} ({(eq-100000)/1000:+.1f}K from $100K)")
        lines.append("")

    for r in reports:
        ta = r["trade_analysis"]
        st = r["state"]

        icon = {"HEALTHY": "OK", "NEEDS_ATTENTION": "!!", "CRITICAL": "XX",
                "DOWN": "XX", "DORMANT": "ZZ"}.get(r["status"], "??")
        lines.append(f"  [{icon}] {r['name']} — {r['status']}")
        lines.append(f"  {'─' * 52}")

        if st.get("balance", 0) > 0:
            lines.append(f"    Balance: ${st['balance']:,.2f} | Daily: ${st.get('daily_pnl',0):+,.2f}")

        if ta.get("has_trades"):
            lines.append(f"    Trades: {ta['total']} ({ta['wins']}W/{ta['losses']}L) | "
                         f"WR: {ta['win_rate']:.0%} | PF: {ta['profit_factor']:.2f}")
            lines.append(f"    Total P&L: ${ta['total_pnl']:+,.2f} | "
                         f"Trend: {ta.get('trend','?')} | "
                         f"Recent 3d: {ta['recent_trades']} trades ${ta['recent_pnl']:+,.2f}")

        if r["n_positions"]:
            lines.append(f"    Open Positions: {r['n_positions']}")

        if r["recommendations"]:
            lines.append(f"    RECOMMENDATIONS:")
            for rec in r["recommendations"]:
                lines.append(f"      [{rec['severity']}] {rec['title']}")
                lines.append(f"        {rec['detail'][:120]}")

        lines.append("")

    # Overall
    active = [r for r in reports if not r["dormant"]]
    total_recs = sum(len(r["recommendations"]) for r in active)
    critical = sum(1 for r in active for rec in r["recommendations"] if rec["severity"] == "CRITICAL")
    lines.append(f"  SUMMARY: {len(active)} active bots, {total_recs} recommendations ({critical} critical)")
    lines.append("=" * 72)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Automated Bot Audit")
    parser.add_argument("--crypto", action="store_true", help="CryptoBot only")
    parser.add_argument("--options", action="store_true", help="Options bots only")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    if args.crypto:
        selected = {k: v for k, v in BOTS.items() if v["category"] == "crypto"}
    elif args.options:
        selected = {k: v for k, v in BOTS.items() if v["category"] == "options"}
    else:
        selected = BOTS

    if not args.quiet:
        print("Running deep audit...")

    processes = get_processes()

    reports = []
    for name, info in selected.items():
        if not args.quiet:
            print(f"  Analyzing {name}...")
        reports.append(audit_bot(name, info, processes))

    account = query_dashboard("/api/account") if not args.crypto else None

    # Save history snapshot
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_snapshot(reports)

    # Generate reports
    ts = now_str = datetime.now().strftime("%Y%m%d")
    html = generate_html(reports, account)
    text = generate_text(reports, account)

    html_path = REPORTS_DIR / f"audit_{ts}.html"
    txt_path = REPORTS_DIR / f"audit_{ts}.txt"
    latest = REPORTS_DIR / "audit_latest.html"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(latest, "w", encoding="utf-8") as f:
        f.write(html)

    if not args.quiet:
        print()
        print(text)
        print()
        print(f"Reports saved:")
        print(f"  HTML: {html_path}")
        print(f"  Text: {txt_path}")
        print(f"  Latest: {latest}")

    # Cleanup old reports (keep 90 days)
    cutoff = datetime.now() - timedelta(days=90)
    for f in REPORTS_DIR.glob("audit_2*.*"):
        try:
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                f.unlink()
        except Exception:
            pass

    # Exit code
    active = [r for r in reports if not r["dormant"]]
    critical = any(r["status"] in ("CRITICAL", "DOWN") for r in active)
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
