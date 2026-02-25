"""
Real-time Trading Dashboard — WebSocket-powered live view
Reads state files from the bot engine and streams updates to the browser.
"""

import csv
import json
import os
import re
import asyncio
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, render_template_string, jsonify
from flask_sock import Sock

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
STATE_DIR = DATA_DIR / "state"
LOGS_DIR = ROOT_DIR / "logs"
TRADES_CSV = DATA_DIR / "trades.csv"

app = Flask(__name__)
sock = Sock(app)

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


def _read_trades_csv(limit: int = 500) -> List[Dict]:
    if not TRADES_CSV.exists():
        return []
    rows: deque = deque(maxlen=limit + 1)
    header = []
    try:
        with open(TRADES_CSV, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            for row in reader:
                rows.append(row)
    except Exception:
        return []
    if not header:
        return []
    result = []
    for row in rows:
        if len(row) >= len(header):
            result.append(dict(zip(header, row)))
    return result[-limit:]


def _parse_log_tail(n: int = 80) -> List[str]:
    if not LOGS_DIR.exists():
        return []
    logs = sorted(LOGS_DIR.glob("trading_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return []
    lines = deque(maxlen=n)
    try:
        with open(logs[0], "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                lines.append(line.rstrip())
    except Exception:
        return []
    return list(lines)


def _get_full_snapshot() -> Dict[str, Any]:
    """Build the complete dashboard state from all data files."""
    # Balances & circuit breaker state
    balances = _safe_json(STATE_DIR / "paper_balances.json")
    spot = float(balances.get("spot", 0))
    futures = float(balances.get("futures", 0))
    total = spot + futures
    peak = float(balances.get("peak_balance", total))
    daily_pnl = float(balances.get("daily_pnl", 0))
    consec_losses = int(balances.get("consecutive_losses", 0))
    initial = 5000.0  # INITIAL_SPOT + INITIAL_FUTURES
    total_pnl = total - initial
    drawdown_pct = ((total - peak) / peak * 100) if peak > 0 else 0

    # Positions
    raw_positions = _safe_json(STATE_DIR / "positions.json")
    positions = []
    price_history = _safe_json(STATE_DIR / "spot_price_history.json")
    futures_history = _safe_json(STATE_DIR / "futures_price_history.json")
    all_history = {**price_history, **futures_history}

    total_unrealized = 0
    for key, pos in raw_positions.items():
        symbol = pos.get("symbol", key)
        direction = pos.get("direction", "LONG")
        entry_price = float(pos.get("entry_price", 0))
        size = float(pos.get("size", 0))
        hist = all_history.get(symbol, [])
        current_price = float(hist[-1]) if hist else entry_price

        if direction == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price else 0
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100 if entry_price else 0
        pnl_usd = size * (pnl_pct / 100)
        total_unrealized += pnl_usd

        positions.append({
            "key": key,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "current_price": current_price,
            "size": size,
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usd": round(pnl_usd, 4),
            "peak_pnl_pct": float(pos.get("peak_pnl_pct", 0)),
            "ml_confidence": float(pos.get("ml_confidence", 0)),
            "entry_reason": pos.get("entry_reason", ""),
            "entry_time": pos.get("entry_time", ""),
            "is_futures": symbol.startswith("PI_"),
        })

    # Sort by PnL
    positions.sort(key=lambda p: p["pnl_usd"])

    # Trades history
    all_trades = _read_trades_csv(500)
    trades_display = []
    for t in reversed(all_trades[-100:]):
        trades_display.append({
            "timestamp": t.get("timestamp", ""),
            "symbol": t.get("symbol", ""),
            "direction": t.get("direction", ""),
            "entry_price": t.get("entry_price", ""),
            "exit_price": t.get("exit_price", ""),
            "size_usd": t.get("size_usd", ""),
            "pnl_usd": t.get("pnl_usd", ""),
            "pnl_pct": t.get("pnl_pct", ""),
            "exit_reason": t.get("exit_reason", ""),
            "entry_reason": t.get("entry_reason", ""),
        })

    # Stats from all trades
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if float(t.get("pnl_usd", 0)) > 0)
    losses = sum(1 for t in all_trades if float(t.get("pnl_usd", 0)) <= 0)
    win_rate = (wins / total_trades * 100) if total_trades else 0
    realized_pnl = sum(float(t.get("pnl_usd", 0)) for t in all_trades)
    avg_win = 0
    avg_loss = 0
    if wins:
        avg_win = sum(float(t["pnl_usd"]) for t in all_trades if float(t.get("pnl_usd", 0)) > 0) / wins
    if losses:
        avg_loss = sum(float(t["pnl_usd"]) for t in all_trades if float(t.get("pnl_usd", 0)) <= 0) / losses
    profit_factor = abs(avg_win * wins / (avg_loss * losses)) if (avg_loss * losses) != 0 else 0

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in all_trades:
        exit_reasons[t.get("exit_reason", "UNKNOWN")] += 1

    # Today's stats
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in all_trades if t.get("timestamp", "").startswith(today_str)]
    today_count = len(today_trades)
    today_wins = sum(1 for t in today_trades if float(t.get("pnl_usd", 0)) > 0)
    today_pnl = sum(float(t.get("pnl_usd", 0)) for t in today_trades)
    today_wr = (today_wins / today_count * 100) if today_count else 0

    # Per-symbol performance
    symbol_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in all_trades:
        sym = t.get("symbol", "UNKNOWN")
        pnl = float(t.get("pnl_usd", 0))
        symbol_stats[sym]["trades"] += 1
        symbol_stats[sym]["pnl"] += pnl
        if pnl > 0:
            symbol_stats[sym]["wins"] += 1
    symbol_perf = []
    for sym, s in symbol_stats.items():
        symbol_perf.append({
            "symbol": sym,
            "trades": s["trades"],
            "wins": s["wins"],
            "wr": round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0,
            "pnl": round(s["pnl"], 2),
        })
    symbol_perf.sort(key=lambda x: x["pnl"])

    # Direction stats
    long_trades = [t for t in all_trades if t.get("direction") == "LONG"]
    short_trades = [t for t in all_trades if t.get("direction") == "SHORT"]
    long_pnl = sum(float(t.get("pnl_usd", 0)) for t in long_trades)
    short_pnl = sum(float(t.get("pnl_usd", 0)) for t in short_trades)
    long_wr = (sum(1 for t in long_trades if float(t.get("pnl_usd", 0)) > 0) / len(long_trades) * 100) if long_trades else 0
    short_wr = (sum(1 for t in short_trades if float(t.get("pnl_usd", 0)) > 0) / len(short_trades) * 100) if short_trades else 0

    # Equity curve (from trades)
    equity_curve = []
    running = initial
    for t in all_trades:
        running += float(t.get("pnl_usd", 0))
        equity_curve.append({
            "ts": t.get("timestamp", "")[:19],
            "equity": round(running, 2),
        })

    # Circuit breaker / engine status from log
    log_lines = _parse_log_tail(40)
    engine_status = "UNKNOWN"
    last_cycle = ""
    cb_reason = ""
    for line in reversed(log_lines):
        if "Trading blocked:" in line and not cb_reason:
            cb_reason = line.split("Trading blocked:")[-1].strip()
            engine_status = "BLOCKED"
        if "TRADE" in line and "Cycle" in line and not last_cycle:
            last_cycle = line
        if "[RISK]" in line and not last_cycle:
            last_cycle = line
        if engine_status == "UNKNOWN" and ("OPEN " in line):
            engine_status = "TRADING"

    if engine_status == "UNKNOWN":
        engine_status = "RUNNING"

    # RL shadow report
    rl_report = _safe_json(STATE_DIR / "rl_shadow_report.json")
    rl_books = rl_report.get("books", {})

    # Config snapshot
    fingerprint = _safe_json(STATE_DIR / "runtime_fingerprint_latest.json")
    config_snap = fingerprint.get("config", {})

    return {
        "ts": datetime.now().isoformat(),
        "account": {
            "total_balance": round(total, 2),
            "spot_balance": round(spot, 2),
            "futures_balance": round(futures, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / initial * 100, 2) if initial else 0,
            "daily_pnl": round(daily_pnl, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "peak_balance": round(peak, 2),
            "drawdown_pct": round(drawdown_pct, 2),
            "consecutive_losses": consec_losses,
        },
        "engine": {
            "status": engine_status,
            "cb_reason": cb_reason,
            "last_cycle": last_cycle,
            "positions_count": len(positions),
        },
        "positions": positions,
        "trades": trades_display,
        "stats": {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "realized_pnl": round(realized_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "exit_reasons": dict(exit_reasons),
        },
        "today": {
            "trades": today_count,
            "wins": today_wins,
            "pnl": round(today_pnl, 2),
            "wr": round(today_wr, 1),
        },
        "direction": {
            "long_pnl": round(long_pnl, 2),
            "short_pnl": round(short_pnl, 2),
            "long_wr": round(long_wr, 1),
            "short_wr": round(short_wr, 1),
            "long_count": len(long_trades),
            "short_count": len(short_trades),
        },
        "symbol_perf": symbol_perf,
        "equity_curve": equity_curve[-200:],
        "rl_shadow": {
            "baseline": rl_books.get("baseline", {}),
            "rl": rl_books.get("rl", {}),
            "delta": rl_report.get("delta", {}),
        },
        "config": {
            k: v for k, v in config_snap.items()
            if k in (
                "MAX_POSITIONS_SPOT", "MAX_POSITIONS_FUTURES",
                "MAX_POSITIONS_PER_SYMBOL_SPOT", "MAX_POSITIONS_PER_SYMBOL_FUTURES",
                "STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "TRAILING_STOP_PCT",
                "MIN_ML_CONFIDENCE", "MIN_ENSEMBLE_SCORE",
                "COUNTER_TREND_ML_OVERRIDE", "CB_MAX_DRAWDOWN_PCT",
                "CB_MAX_CONSECUTIVE_LOSSES", "CB_DAILY_LOSS_LIMIT_PCT",
                "MAX_CORRELATION", "FUTURES_LEVERAGE", "PAPER_TRADING",
                "MODEL_RETRAIN_HOURS", "MIN_MODEL_TEST_ACCURACY",
                "TRADE_CYCLE_INTERVAL", "RISK_CHECK_INTERVAL",
            )
        },
        "log_tail": log_lines[-30:],
    }


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/snapshot")
def api_snapshot():
    return jsonify(_get_full_snapshot())


# ── WebSocket ──────────────────────────────────────────────────────────

@sock.route("/ws")
def ws_handler(ws):
    """Push full state every 5 seconds to connected clients."""
    try:
        while True:
            snapshot = _get_full_snapshot()
            ws.send(json.dumps(snapshot))
            time.sleep(5)
    except Exception:
        pass  # Client disconnected


# ── HTML Template ──────────────────────────────────────────────────────

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Bot — Live Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --green: #3fb950;
    --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
    --cyan: #39d2c0; --purple: #bc8cff;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, 'Segoe UI', Roboto, monospace; background:var(--bg); color:var(--text); font-size:13px; }
  a { color:var(--blue); text-decoration:none; }

  .header { background:var(--card); border-bottom:1px solid var(--border); padding:12px 24px; display:flex; align-items:center; justify-content:space-between; position:sticky; top:0; z-index:100; }
  .header h1 { font-size:16px; font-weight:600; }
  .header .status { display:flex; align-items:center; gap:8px; }
  .status-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .status-dot.connected { background:var(--green); box-shadow:0 0 6px var(--green); }
  .status-dot.disconnected { background:var(--red); box-shadow:0 0 6px var(--red); }
  .status-dot.blocked { background:var(--yellow); box-shadow:0 0 6px var(--yellow); }

  .grid { display:grid; gap:12px; padding:16px; }
  .grid-top { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
  .grid-main { grid-template-columns: 1fr 1fr; }
  .grid-full { grid-template-columns: 1fr; }
  @media(max-width:900px) { .grid-main { grid-template-columns:1fr; } }

  .card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; }
  .card h2 { font-size:12px; text-transform:uppercase; color:var(--muted); letter-spacing:1px; margin-bottom:10px; }
  .big-number { font-size:28px; font-weight:700; }
  .sub { font-size:11px; color:var(--muted); margin-top:2px; }

  .pos { color:var(--green); } .neg { color:var(--red); } .warn { color:var(--yellow); }

  .cb-banner { background:#2d1b00; border:1px solid var(--yellow); border-radius:8px; padding:14px 20px; margin:16px 16px 0; display:none; }
  .cb-banner.active { display:flex; align-items:center; gap:12px; }
  .cb-banner .icon { font-size:22px; }
  .cb-banner .text { font-size:13px; color:var(--yellow); }

  table { width:100%; border-collapse:collapse; font-size:12px; }
  thead th { text-align:left; padding:8px 6px; border-bottom:1px solid var(--border); color:var(--muted); font-weight:500; position:sticky; top:0; background:var(--card); }
  tbody td { padding:6px; border-bottom:1px solid #21262d; white-space:nowrap; }
  tbody tr:hover { background:#1c2333; }

  .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
  .badge-long { background:#0d2818; color:var(--green); }
  .badge-short { background:#2d0e0e; color:var(--red); }
  .badge-reason { background:#1c1d21; color:var(--muted); }

  .chart-container { position:relative; height:220px; }

  .stat-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #21262d; }
  .stat-row:last-child { border:none; }
  .stat-label { color:var(--muted); }
  .stat-value { font-weight:600; }

  .log-box { max-height:280px; overflow-y:auto; font-family:monospace; font-size:11px; line-height:1.6; background:#0d1117; border-radius:4px; padding:8px; }
  .log-box .warn-line { color:var(--yellow); }
  .log-box .err-line { color:var(--red); }
  .log-box .trade-line { color:var(--cyan); }

  .tab-bar { display:flex; gap:0; border-bottom:1px solid var(--border); margin-bottom:12px; }
  .tab-btn { padding:8px 16px; cursor:pointer; border:none; background:none; color:var(--muted); font-size:12px; border-bottom:2px solid transparent; }
  .tab-btn.active { color:var(--blue); border-bottom-color:var(--blue); }
  .tab-content { display:none; } .tab-content.active { display:block; }

  .scroll-table { max-height:400px; overflow-y:auto; }
</style>
</head>
<body>

<div class="header">
  <h1>Trading Bot — Live Dashboard</h1>
  <div class="status">
    <span class="status-dot disconnected" id="wsDot"></span>
    <span id="wsStatus" style="font-size:11px;color:var(--muted)">Connecting...</span>
    <span id="lastUpdate" style="font-size:11px;color:var(--muted);margin-left:12px"></span>
  </div>
</div>

<div class="cb-banner" id="cbBanner">
  <span class="icon">⚠</span>
  <div>
    <div class="text" style="font-weight:600;">Circuit Breaker Active</div>
    <div class="text" id="cbReason" style="font-size:12px;opacity:0.8"></div>
  </div>
</div>

<!-- KPI Cards -->
<div class="grid grid-top" id="kpiGrid">
  <div class="card">
    <h2>Total Balance</h2>
    <div class="big-number" id="totalBal">—</div>
    <div class="sub" id="totalPnl"></div>
  </div>
  <div class="card">
    <h2>Daily P&amp;L</h2>
    <div class="big-number" id="dailyPnl">—</div>
    <div class="sub" id="todayStats"></div>
  </div>
  <div class="card">
    <h2>Drawdown</h2>
    <div class="big-number" id="drawdown">—</div>
    <div class="sub" id="peakBal"></div>
  </div>
  <div class="card">
    <h2>Open Positions</h2>
    <div class="big-number" id="posCount">—</div>
    <div class="sub" id="unrealizedPnl"></div>
  </div>
  <div class="card">
    <h2>Win Rate</h2>
    <div class="big-number" id="winRate">—</div>
    <div class="sub" id="winLoss"></div>
  </div>
  <div class="card">
    <h2>Profit Factor</h2>
    <div class="big-number" id="profitFactor">—</div>
    <div class="sub" id="avgWinLoss"></div>
  </div>
</div>

<!-- Main panels -->
<div class="grid grid-main">
  <!-- Left: Positions + Trades -->
  <div class="card">
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('positions',this)">Open Positions</button>
      <button class="tab-btn" onclick="switchTab('trades',this)">Trade History</button>
    </div>
    <div class="tab-content active" id="tab-positions">
      <div class="scroll-table">
        <table><thead><tr>
          <th>Symbol</th><th>Side</th><th>Entry</th><th>Current</th><th>Size</th><th>P&L</th><th>Peak</th><th>ML Conf</th><th>Age</th>
        </tr></thead><tbody id="posTable"></tbody></table>
      </div>
    </div>
    <div class="tab-content" id="tab-trades">
      <div class="scroll-table">
        <table><thead><tr>
          <th>Time</th><th>Symbol</th><th>Side</th><th>P&L</th><th>%</th><th>Exit Reason</th><th>Entry Reason</th>
        </tr></thead><tbody id="tradeTable"></tbody></table>
      </div>
    </div>
  </div>

  <!-- Right: Equity curve + stats -->
  <div class="card">
    <h2>Equity Curve</h2>
    <div class="chart-container"><canvas id="equityChart"></canvas></div>
  </div>
</div>

<div class="grid grid-main">
  <!-- Symbol performance -->
  <div class="card">
    <h2>Performance by Symbol</h2>
    <div class="scroll-table">
      <table><thead><tr>
        <th>Symbol</th><th>Trades</th><th>Win Rate</th><th>P&L</th>
      </tr></thead><tbody id="symbolTable"></tbody></table>
    </div>
  </div>

  <!-- Stats panels -->
  <div class="card">
    <h2>Engine Stats</h2>
    <div class="stat-row"><span class="stat-label">Engine Status</span><span class="stat-value" id="engineStatus">—</span></div>
    <div class="stat-row"><span class="stat-label">Consecutive Losses</span><span class="stat-value" id="consecLosses">—</span></div>
    <div class="stat-row"><span class="stat-label">Spot Balance</span><span class="stat-value" id="spotBal">—</span></div>
    <div class="stat-row"><span class="stat-label">Futures Balance</span><span class="stat-value" id="futBal">—</span></div>
    <div class="stat-row"><span class="stat-label">Realized P&L</span><span class="stat-value" id="realizedPnl">—</span></div>
    <div class="stat-row"><span class="stat-label">Total Trades</span><span class="stat-value" id="totalTrades">—</span></div>

    <h2 style="margin-top:16px;">Direction Breakdown</h2>
    <div class="stat-row"><span class="stat-label">Long P&L</span><span class="stat-value" id="longPnl">—</span></div>
    <div class="stat-row"><span class="stat-label">Short P&L</span><span class="stat-value" id="shortPnl">—</span></div>
    <div class="stat-row"><span class="stat-label">Long WR / Count</span><span class="stat-value" id="longWr">—</span></div>
    <div class="stat-row"><span class="stat-label">Short WR / Count</span><span class="stat-value" id="shortWr">—</span></div>

    <h2 style="margin-top:16px;">Exit Reasons</h2>
    <div id="exitReasons"></div>
  </div>
</div>

<div class="grid grid-main">
  <!-- RL Shadow comparison -->
  <div class="card">
    <h2>RL Shadow Mode Comparison</h2>
    <div id="rlShadow"></div>
  </div>

  <!-- Config -->
  <div class="card">
    <h2>Active Config</h2>
    <div id="configPanel"></div>
  </div>
</div>

<!-- Log tail -->
<div class="grid grid-full">
  <div class="card">
    <h2>Live Log Feed</h2>
    <div class="log-box" id="logBox"></div>
  </div>
</div>

<script>
// ── Globals ──
let equityChart = null;
let ws = null;
let reconnectTimer = null;

// ── Helpers ──
function $(id) { return document.getElementById(id); }
function pnlClass(v) { return parseFloat(v) >= 0 ? 'pos' : 'neg'; }
function fmtUsd(v) { const n = parseFloat(v); return (n >= 0 ? '+' : '') + '$' + n.toFixed(2); }
function fmtPct(v) { const n = parseFloat(v); return (n >= 0 ? '+' : '') + n.toFixed(2) + '%'; }
function fmtPrice(v) { const n = parseFloat(v); return n >= 100 ? n.toFixed(2) : n >= 1 ? n.toFixed(4) : n.toFixed(6); }

function timeAgo(isoStr) {
  if (!isoStr) return '—';
  const d = new Date(isoStr.replace(' ', 'T'));
  const mins = Math.floor((Date.now() - d.getTime()) / 60000);
  if (mins < 60) return mins + 'm';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h ' + (mins % 60) + 'm';
  return Math.floor(hrs / 24) + 'd';
}

function shortTime(isoStr) {
  if (!isoStr) return '—';
  return isoStr.replace('T', ' ').substring(5, 16);
}

// ── Tabs ──
function switchTab(name, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  $('tab-' + name).classList.add('active');
}

// ── Chart ──
function initChart() {
  const ctx = $('equityChart').getContext('2d');
  equityChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{
      label: 'Equity',
      data: [],
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88,166,255,0.08)',
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 2,
    }]},
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: true, ticks: { maxTicksLimit: 8, color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e', font: { size: 10 }, callback: v => '$' + v.toFixed(0) }, grid: { color: '#21262d' } }
      }
    }
  });
}

function updateChart(curve) {
  if (!equityChart || !curve.length) return;
  equityChart.data.labels = curve.map(p => p.ts ? p.ts.substring(5, 16) : '');
  equityChart.data.datasets[0].data = curve.map(p => p.equity);
  // Color line based on current vs initial
  const last = curve[curve.length - 1].equity;
  const color = last >= 5000 ? '#3fb950' : '#f85149';
  equityChart.data.datasets[0].borderColor = color;
  equityChart.data.datasets[0].backgroundColor = color.replace(')', ',0.08)').replace('rgb', 'rgba');
  equityChart.update('none');
}

// ── Render ──
function render(d) {
  // KPIs
  $('totalBal').textContent = '$' + d.account.total_balance.toLocaleString(undefined, {minimumFractionDigits:2});
  $('totalBal').className = 'big-number ' + pnlClass(d.account.total_pnl);
  $('totalPnl').innerHTML = `<span class="${pnlClass(d.account.total_pnl)}">${fmtUsd(d.account.total_pnl)} (${fmtPct(d.account.total_pnl_pct)})</span> all-time`;

  $('dailyPnl').textContent = fmtUsd(d.account.daily_pnl);
  $('dailyPnl').className = 'big-number ' + pnlClass(d.account.daily_pnl);
  $('todayStats').textContent = d.today.trades + ' trades today | WR ' + d.today.wr.toFixed(1) + '%';

  $('drawdown').textContent = fmtPct(d.account.drawdown_pct);
  $('drawdown').className = 'big-number ' + (d.account.drawdown_pct < -5 ? 'neg' : d.account.drawdown_pct < -2 ? 'warn' : 'pos');
  $('peakBal').textContent = 'Peak: $' + d.account.peak_balance.toLocaleString(undefined, {minimumFractionDigits:2});

  $('posCount').textContent = d.engine.positions_count;
  $('unrealizedPnl').innerHTML = `Unrealized: <span class="${pnlClass(d.account.unrealized_pnl)}">${fmtUsd(d.account.unrealized_pnl)}</span>`;

  $('winRate').textContent = d.stats.win_rate.toFixed(1) + '%';
  $('winRate').className = 'big-number ' + (d.stats.win_rate >= 50 ? 'pos' : 'neg');
  $('winLoss').textContent = d.stats.wins + 'W / ' + d.stats.losses + 'L';

  $('profitFactor').textContent = d.stats.profit_factor.toFixed(2);
  $('profitFactor').className = 'big-number ' + (d.stats.profit_factor >= 1 ? 'pos' : 'neg');
  $('avgWinLoss').textContent = 'Avg: ' + fmtUsd(d.stats.avg_win) + ' / ' + fmtUsd(d.stats.avg_loss);

  // Circuit breaker banner
  if (d.engine.status === 'BLOCKED') {
    $('cbBanner').classList.add('active');
    $('cbReason').textContent = d.engine.cb_reason;
    $('wsDot').className = 'status-dot blocked';
  } else {
    $('cbBanner').classList.remove('active');
  }

  // Engine stats
  const statusColors = { TRADING: 'pos', BLOCKED: 'warn', RUNNING: 'pos', UNKNOWN: 'muted' };
  $('engineStatus').textContent = d.engine.status;
  $('engineStatus').className = 'stat-value ' + (statusColors[d.engine.status] || '');
  $('consecLosses').textContent = d.account.consecutive_losses;
  $('spotBal').textContent = '$' + d.account.spot_balance.toFixed(2);
  $('futBal').textContent = '$' + d.account.futures_balance.toFixed(2);
  $('realizedPnl').innerHTML = `<span class="${pnlClass(d.stats.realized_pnl)}">${fmtUsd(d.stats.realized_pnl)}</span>`;
  $('totalTrades').textContent = d.stats.total_trades;

  // Direction
  $('longPnl').innerHTML = `<span class="${pnlClass(d.direction.long_pnl)}">${fmtUsd(d.direction.long_pnl)}</span>`;
  $('shortPnl').innerHTML = `<span class="${pnlClass(d.direction.short_pnl)}">${fmtUsd(d.direction.short_pnl)}</span>`;
  $('longWr').textContent = d.direction.long_wr.toFixed(1) + '% / ' + d.direction.long_count;
  $('shortWr').textContent = d.direction.short_wr.toFixed(1) + '% / ' + d.direction.short_count;

  // Exit reasons
  let erHtml = '';
  for (const [reason, count] of Object.entries(d.stats.exit_reasons).sort((a,b) => b[1]-a[1])) {
    const pct = (count / d.stats.total_trades * 100).toFixed(1);
    erHtml += `<div class="stat-row"><span class="stat-label">${reason}</span><span class="stat-value">${count} (${pct}%)</span></div>`;
  }
  $('exitReasons').innerHTML = erHtml;

  // Positions table
  let posHtml = '';
  d.positions.forEach(p => {
    const sideClass = p.direction === 'LONG' ? 'badge-long' : 'badge-short';
    posHtml += `<tr>
      <td>${p.symbol}${p.is_futures ? ' ⚡' : ''}</td>
      <td><span class="badge ${sideClass}">${p.direction}</span></td>
      <td>${fmtPrice(p.entry_price)}</td>
      <td>${fmtPrice(p.current_price)}</td>
      <td>$${parseFloat(p.size).toFixed(0)}</td>
      <td class="${pnlClass(p.pnl_usd)}">${fmtUsd(p.pnl_usd)} (${fmtPct(p.pnl_pct)})</td>
      <td>${fmtPct(p.peak_pnl_pct)}</td>
      <td>${(p.ml_confidence * 100).toFixed(1)}%</td>
      <td>${timeAgo(p.entry_time)}</td>
    </tr>`;
  });
  $('posTable').innerHTML = posHtml || '<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:20px">No open positions</td></tr>';

  // Trades table
  let trHtml = '';
  d.trades.forEach(t => {
    const sideClass = t.direction === 'LONG' ? 'badge-long' : 'badge-short';
    trHtml += `<tr>
      <td>${shortTime(t.timestamp)}</td>
      <td>${t.symbol}</td>
      <td><span class="badge ${sideClass}">${t.direction}</span></td>
      <td class="${pnlClass(t.pnl_usd)}">${fmtUsd(t.pnl_usd)}</td>
      <td class="${pnlClass(t.pnl_pct)}">${fmtPct(t.pnl_pct)}</td>
      <td><span class="badge badge-reason">${t.exit_reason}</span></td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis">${t.entry_reason}</td>
    </tr>`;
  });
  $('tradeTable').innerHTML = trHtml;

  // Symbol performance
  let symHtml = '';
  d.symbol_perf.forEach(s => {
    symHtml += `<tr>
      <td>${s.symbol}</td>
      <td>${s.trades}</td>
      <td>${s.wr.toFixed(1)}%</td>
      <td class="${pnlClass(s.pnl)}">${fmtUsd(s.pnl)}</td>
    </tr>`;
  });
  $('symbolTable').innerHTML = symHtml;

  // Equity chart
  updateChart(d.equity_curve);

  // RL Shadow
  let rlHtml = '';
  if (d.rl_shadow.baseline && d.rl_shadow.baseline.trades > 0) {
    const b = d.rl_shadow.baseline, r = d.rl_shadow.rl, delta = d.rl_shadow.delta;
    rlHtml = `
      <table><thead><tr><th></th><th>Baseline</th><th>RL Agent</th><th>Delta</th></tr></thead><tbody>
      <tr><td class="stat-label">Equity</td><td>$${(b.equity||0).toFixed(2)}</td><td>$${(r.equity||0).toFixed(2)}</td><td class="${pnlClass(delta.equity||0)}">${fmtUsd(delta.equity||0)}</td></tr>
      <tr><td class="stat-label">Realized P&L</td><td class="${pnlClass(b.realized_pnl||0)}">${fmtUsd(b.realized_pnl||0)}</td><td class="${pnlClass(r.realized_pnl||0)}">${fmtUsd(r.realized_pnl||0)}</td><td class="${pnlClass(delta.realized_pnl||0)}">${fmtUsd(delta.realized_pnl||0)}</td></tr>
      <tr><td class="stat-label">Win Rate</td><td>${(b.win_rate||0).toFixed(1)}%</td><td>${(r.win_rate||0).toFixed(1)}%</td><td>—</td></tr>
      <tr><td class="stat-label">Trades</td><td>${b.trades||0}</td><td>${r.trades||0}</td><td>—</td></tr>
      <tr><td class="stat-label">Max Drawdown</td><td>${(b.max_drawdown_pct||0).toFixed(2)}%</td><td>${(r.max_drawdown_pct||0).toFixed(2)}%</td><td>${fmtPct(delta.max_drawdown_pct||0)}</td></tr>
      </tbody></table>`;
  } else {
    rlHtml = '<div style="color:var(--muted);padding:12px">No RL shadow data available yet</div>';
  }
  $('rlShadow').innerHTML = rlHtml;

  // Config
  let cfgHtml = '';
  for (const [k, v] of Object.entries(d.config)) {
    cfgHtml += `<div class="stat-row"><span class="stat-label">${k}</span><span class="stat-value">${v}</span></div>`;
  }
  $('configPanel').innerHTML = cfgHtml || '<div style="color:var(--muted)">No config loaded</div>';

  // Log tail
  let logHtml = '';
  d.log_tail.forEach(line => {
    let cls = '';
    if (line.includes('WARNING') || line.includes('blocked')) cls = 'warn-line';
    else if (line.includes('ERROR')) cls = 'err-line';
    else if (line.includes('OPEN ') || line.includes('CLOSE ')) cls = 'trade-line';
    logHtml += `<div class="${cls}">${line.replace(/</g,'&lt;')}</div>`;
  });
  $('logBox').innerHTML = logHtml;
  $('logBox').scrollTop = $('logBox').scrollHeight;

  // Timestamp
  $('lastUpdate').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

// ── WebSocket ──
function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    $('wsDot').className = 'status-dot connected';
    $('wsStatus').textContent = 'Live';
    if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      render(data);
      // Update dot based on engine status
      if (data.engine.status === 'BLOCKED') {
        $('wsDot').className = 'status-dot blocked';
        $('wsStatus').textContent = 'CB Active';
      } else {
        $('wsDot').className = 'status-dot connected';
        $('wsStatus').textContent = 'Live';
      }
    } catch(err) { console.error('WS parse error:', err); }
  };

  ws.onclose = () => {
    $('wsDot').className = 'status-dot disconnected';
    $('wsStatus').textContent = 'Disconnected — reconnecting...';
    if (!reconnectTimer) {
      reconnectTimer = setInterval(connectWs, 5000);
    }
  };

  ws.onerror = () => { ws.close(); };
}

// ── Init ──
window.addEventListener('load', () => {
  initChart();
  connectWs();
  // Fallback: fetch snapshot via HTTP if WS doesn't connect in 3s
  setTimeout(() => {
    if (!ws || ws.readyState !== 1) {
      fetch('/api/snapshot').then(r => r.json()).then(render).catch(() => {});
    }
  }, 3000);
});
</script>
</body>
</html>
"""


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8089
    print(f"Starting Live Dashboard on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
