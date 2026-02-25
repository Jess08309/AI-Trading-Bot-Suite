import csv
import json
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_STATE_DIR = ROOT_DIR / "data" / "state"
DATA_HISTORY_DIR = ROOT_DIR / "data" / "history"
LOGS_DIR = ROOT_DIR / "logs"

app = Flask(__name__)


def _safe_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _latest_trading_log() -> Path | None:
    if not LOGS_DIR.exists():
        return None
    logs = sorted(LOGS_DIR.glob("trading_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _parse_recent_trade_rows(limit: int = 200) -> List[Dict[str, Any]]:
    path = DATA_HISTORY_DIR / "trade_history.csv"
    if not path.exists():
        return []

    rows: deque[str] = deque(maxlen=max(400, limit * 4))
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(line)
    except Exception:
        return []

    parsed: List[Dict[str, Any]] = []
    for line in rows:
        cols = next(csv.reader([line]))
        if len(cols) < 5:
            continue
        if cols[0].lower() in {"timestamp", "symbol"}:
            continue
        if "T" not in cols[0]:
            continue

        try:
            ts = datetime.fromisoformat(cols[0].replace("Z", "+00:00"))
        except Exception:
            continue

        symbol = cols[1]
        side = cols[2]
        try:
            price = float(cols[3])
        except Exception:
            price = 0.0
        try:
            amount = float(cols[4])
        except Exception:
            amount = 0.0
        aux = None
        if len(cols) > 5:
            try:
                aux = float(cols[5])
            except Exception:
                aux = cols[5]

        parsed.append(
            {
                "timestamp": ts.isoformat(),
                "symbol": symbol,
                "side": side,
                "price": price,
                "amount": amount,
                "aux": aux,
            }
        )

    parsed.sort(key=lambda x: x["timestamp"], reverse=True)
    return parsed[:limit]


def _status_payload() -> Dict[str, Any]:
    balances = _safe_json(DATA_STATE_DIR / "paper_balances.json", {"spot": 0.0, "futures": 0.0})
    positions = _safe_json(DATA_STATE_DIR / "positions.json", {})
    perf = _safe_json(DATA_STATE_DIR / "performance_report.json", {})
    fingerprint = _safe_json(DATA_STATE_DIR / "runtime_fingerprint_latest.json", {})
    recent_trades = _parse_recent_trade_rows(limit=100)

    spot = float(balances.get("spot", 0.0) or 0.0)
    futures = float(balances.get("futures", 0.0) or 0.0)
    total_balance = spot + futures

    cfg = fingerprint.get("config", {}) if isinstance(fingerprint, dict) else {}
    initial_spot = float(cfg.get("INITIAL_SPOT_BALANCE", 2500.0) or 2500.0)
    initial_futures = float(cfg.get("INITIAL_FUTURES_BALANCE", 2500.0) or 2500.0)
    initial_total = initial_spot + initial_futures

    total_pnl = total_balance - initial_total
    total_return_pct = (total_pnl / initial_total * 100.0) if initial_total else 0.0

    open_positions: List[Dict[str, Any]] = []
    if isinstance(positions, dict):
        for key, value in positions.items():
            if not isinstance(value, dict):
                continue
            open_positions.append(
                {
                    "id": key,
                    "symbol": value.get("symbol", key),
                    "direction": value.get("direction", "UNKNOWN"),
                    "size": float(value.get("size", 0.0) or 0.0),
                    "entry_price": float(value.get("entry_price", 0.0) or 0.0),
                    "entry_time": value.get("entry_time"),
                    "ml_confidence": float(value.get("ml_confidence", 0.0) or 0.0),
                    "entry_reason": value.get("entry_reason", ""),
                    "stop_loss": float(value.get("stop_loss", 0.0) or 0.0),
                    "take_profit": float(value.get("take_profit", 0.0) or 0.0),
                }
            )
    open_positions.sort(key=lambda x: (x["symbol"], x["id"]))

    return {
        "timestamp": datetime.now().isoformat(),
        "balances": {
            "spot": spot,
            "futures": futures,
            "total": total_balance,
            "initial_total": initial_total,
            "pnl": total_pnl,
            "return_pct": total_return_pct,
        },
        "stats": {
            "open_positions": len(open_positions),
            "win_rate_pct": float(perf.get("win_rate_pct", 0.0) or 0.0),
            "total_trades": int(perf.get("total_trades", 0) or 0),
            "expectancy": float(perf.get("expectancy", 0.0) or 0.0),
            "profit_factor": float(perf.get("profit_factor", 0.0) or 0.0),
            "max_drawdown_pct": float(perf.get("max_drawdown_pct", 0.0) or 0.0),
        },
        "open_positions": open_positions,
        "recent_trades": recent_trades[:40],
        "symbol_stats": (perf.get("symbols", {}) if isinstance(perf, dict) else {}),
    }


def _transparency_payload(limit: int = 200) -> Dict[str, Any]:
    transparency = _safe_json(DATA_STATE_DIR / "trade_transparency.json", [])
    if not isinstance(transparency, list):
        transparency = []

    items = []
    for record in transparency[-limit:]:
        if not isinstance(record, dict):
            continue
        signal = record.get("signal", {}) if isinstance(record.get("signal"), dict) else {}
        items.append(
            {
                "symbol": record.get("symbol"),
                "direction": record.get("direction"),
                "status": record.get("status"),
                "entry_time": record.get("entry_time"),
                "exit_time": record.get("exit_time"),
                "entry_price": record.get("entry_price"),
                "exit_price": record.get("exit_price"),
                "pnl_usd": record.get("pnl_usd"),
                "pnl_pct": record.get("pnl_pct"),
                "hold_minutes": record.get("hold_minutes"),
                "exit_reason": record.get("exit_reason"),
                "signal": signal,
            }
        )

    items.reverse()
    return {"timestamp": datetime.now().isoformat(), "count": len(items), "items": items}


def _thought_payload(limit: int = 180) -> Dict[str, Any]:
    runtime = _safe_json(DATA_STATE_DIR / "runtime_fingerprint_latest.json", {})
    rl_shadow = _safe_json(DATA_STATE_DIR / "rl_shadow_report.json", {})

    log_file = _latest_trading_log()
    entries: List[Dict[str, Any]] = []
    if log_file and log_file.exists():
        lines: deque[str] = deque(maxlen=2500)
        try:
            with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    clean = line.strip()
                    if clean:
                        lines.append(clean)
        except Exception:
            lines = deque()

        pattern = re.compile(r"^([0-9\-: ]+) \| ([A-Z]+)\s+\| (.*)$")
        keywords = (
            "Signal filters",
            "Counter-trend",
            "Sentiment status",
            "OPEN ",
            "CLOSE ",
            "ALERT[",
            "Cycle",
            "Scheduled model retrain",
            "Model",
            "Skip ",
            "Direction",
        )
        for line in reversed(lines):
            if not any(k in line for k in keywords):
                continue
            match = pattern.match(line)
            if match:
                ts, level, message = match.groups()
            else:
                ts, level, message = "", "INFO", line

            kind = "info"
            if "OPEN " in message or "CLOSE " in message:
                kind = "action"
            elif "ALERT[" in message or level == "WARNING":
                kind = "warning"
            elif "Signal filters" in message or "Counter-trend" in message or "Sentiment status" in message:
                kind = "reasoning"

            entries.append({"timestamp": ts, "level": level, "kind": kind, "message": message})
            if len(entries) >= limit:
                break

    summary = {}
    if isinstance(rl_shadow, dict):
        delta = rl_shadow.get("delta", {}) if isinstance(rl_shadow.get("delta"), dict) else {}
        baseline = (rl_shadow.get("books", {}) or {}).get("baseline", {}) if isinstance(rl_shadow.get("books", {}), dict) else {}
        rl_book = (rl_shadow.get("books", {}) or {}).get("rl", {}) if isinstance(rl_shadow.get("books", {}), dict) else {}
        summary = {
            "shadow_cycle": rl_shadow.get("cycle"),
            "delta_equity": delta.get("equity"),
            "delta_realized_pnl": delta.get("realized_pnl"),
            "delta_drawdown_pct": delta.get("max_drawdown_pct"),
            "baseline_win_rate": baseline.get("win_rate"),
            "rl_win_rate": rl_book.get("win_rate"),
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "runtime_flags": runtime.get("flags", {}) if isinstance(runtime, dict) else {},
        "runtime_config": runtime.get("config", {}) if isinstance(runtime, dict) else {},
        "shadow_summary": summary,
        "log_entries": entries,
    }


def _chart_payload() -> Dict[str, Any]:
    perf = _safe_json(DATA_STATE_DIR / "performance_report.json", {})
    recent_trades = _parse_recent_trade_rows(limit=1200)

    daily_raw = perf.get("daily_pnl", {}) if isinstance(perf, dict) else {}
    daily_pnl = []
    if isinstance(daily_raw, dict):
        for day, value in sorted(daily_raw.items(), key=lambda x: x[0]):
            try:
                daily_pnl.append({"day": day, "pnl": float(value)})
            except Exception:
                continue

    symbols_raw = perf.get("symbols", {}) if isinstance(perf, dict) else {}
    symbol_stats = []
    if isinstance(symbols_raw, dict):
        for symbol, stats in symbols_raw.items():
            if not isinstance(stats, dict):
                continue
            symbol_stats.append(
                {
                    "symbol": symbol,
                    "trades": int(stats.get("trades", 0) or 0),
                    "pnl": float(stats.get("pnl", 0.0) or 0.0),
                    "win_rate_pct": float(stats.get("win_rate_pct", 0.0) or 0.0),
                }
            )
    symbol_stats.sort(key=lambda x: x["trades"], reverse=True)

    equity_points: List[Dict[str, Any]] = []
    cycle_points: List[Dict[str, Any]] = []
    latest_cycle = None

    log_file = _latest_trading_log()
    if log_file and log_file.exists():
        total_pattern = re.compile(r"^([0-9\-: ]+) \| [A-Z]+\s+\| Total: \$([0-9,]+(?:\.[0-9]+)?)")
        cycle_pattern = re.compile(r"^([0-9\-: ]+) \| [A-Z]+\s+\| --- Cycle ([0-9]+)")
        lines: deque[str] = deque(maxlen=4000)
        try:
            with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    clean = line.strip()
                    if clean:
                        lines.append(clean)
        except Exception:
            lines = deque()

        for line in lines:
            m_total = total_pattern.match(line)
            if m_total:
                ts, value = m_total.groups()
                try:
                    equity_points.append({"timestamp": ts, "total": float(value.replace(",", ""))})
                except Exception:
                    pass
            m_cycle = cycle_pattern.match(line)
            if m_cycle:
                ts, cycle_value = m_cycle.groups()
                try:
                    c = int(cycle_value)
                    latest_cycle = c
                    cycle_points.append({"timestamp": ts, "cycle": c})
                except Exception:
                    pass

    return {
        "timestamp": datetime.now().isoformat(),
        "latest_cycle": latest_cycle,
        "equity_curve": equity_points[-240:],
        "cycle_curve": cycle_points[-240:],
        "daily_pnl": daily_pnl[-120:],
        "symbol_stats": symbol_stats,
        "recent_trades": recent_trades[:500],
    }


@app.get("/api/status")
def api_status():
    return jsonify(_status_payload())


@app.get("/api/transparency")
def api_transparency():
    limit = int(request.args.get("limit", "200"))
    limit = max(20, min(limit, 1000))
    return jsonify(_transparency_payload(limit=limit))


@app.get("/api/thoughts")
def api_thoughts():
    limit = int(request.args.get("limit", "180"))
    limit = max(20, min(limit, 800))
    return jsonify(_thought_payload(limit=limit))


@app.get("/api/all")
def api_all():
    return jsonify(
        {
            "status": _status_payload(),
            "transparency": _transparency_payload(limit=200),
            "thoughts": _thought_payload(limit=180),
        "charts": _chart_payload(),
        }
    )


@app.get("/api/charts")
def api_charts():
    return jsonify(_chart_payload())


@app.get("/")
def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Bot Control Center</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #0f1218; color: #e8ecf1; }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 16px; }
    h1 { margin: 0 0 14px 0; color: #73f0a8; font-size: 26px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 12px; }
    .tab { border: 1px solid #2a3342; background: #18202b; color: #dbe4ef; padding: 10px 14px; cursor: pointer; border-radius: 8px; }
    .tab.active { border-color: #73f0a8; color: #73f0a8; }
    .panel { display: none; }
    .panel.active { display: block; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 12px; }
    .card { background: #17202a; border: 1px solid #263141; border-radius: 10px; padding: 12px; cursor: pointer; }
    .card .label { color: #92a2b6; font-size: 12px; }
    .card .val { font-size: 22px; font-weight: 700; margin-top: 4px; }
    .good { color: #73f0a8; }
    .bad { color: #ff7676; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .box { background: #17202a; border: 1px solid #263141; border-radius: 10px; padding: 10px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #243041; }
    th { color: #73f0a8; position: sticky; top: 0; background: #17202a; }
    .small { font-size: 12px; color: #8ca1b9; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
    .pill.long { background: #143827; color: #73f0a8; }
    .pill.short { background: #3d1b1b; color: #ff9e9e; }
    .pill.open { background: #1e3348; color: #96c5ff; }
    .pill.closed { background: #2e2f34; color: #c7ced8; }
    .log { font-family: Consolas, monospace; font-size: 12px; padding: 6px; border-bottom: 1px solid #243041; }
    .log.warning { color: #ffb07c; }
    .log.action { color: #96c5ff; }
    .log.reasoning { color: #c7b2ff; }
    .toolbar { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; flex-wrap: wrap; }
    button { background: #1e2b3a; border: 1px solid #2f4258; color: #dbe4ef; border-radius: 8px; padding: 8px 10px; cursor: pointer; }
    button:hover { border-color: #73f0a8; }
    .muted { color: #8ca1b9; }
    .chartWrap { background: #111923; border: 1px solid #263141; border-radius: 8px; padding: 8px; }
    .chart { width: 100%; height: 220px; display: block; }
    .chartTall { width: 100%; height: 300px; display: block; }
    .axisText { fill: #8ca1b9; font-size: 11px; }
  </style>
</head>
<body>
  <div class='wrap'>
    <h1>Bot Control Center</h1>
    <div class='tabs'>
      <button class='tab active' data-tab='status'>Status</button>
      <button class='tab' data-tab='transparency'>Transparency</button>
      <button class='tab' data-tab='thoughts'>Thought Process</button>
      <button class='tab' data-tab='charts'>Charts</button>
      <label class='small muted' style='margin-left:auto;display:flex;align-items:center;gap:6px;'>
        <input type='checkbox' id='liveToggle' checked>
        Live
      </label>
      <select id='refreshRate' class='small' style='background:#1e2b3a;border:1px solid #2f4258;color:#dbe4ef;border-radius:8px;padding:7px;'>
        <option value='3000'>3s</option>
        <option value='5000' selected>5s</option>
        <option value='10000'>10s</option>
      </select>
      <button id='refreshBtn' style='margin-left:auto;'>Refresh Now</button>
    </div>

    <div id='status' class='panel active'>
      <div id='statusCards' class='cards'></div>
      <div class='grid2'>
        <div class='box'>
          <div class='toolbar'><strong>Open Positions</strong><span class='small muted' id='openCount'></span></div>
          <table id='positionsTable'><thead><tr><th>Symbol</th><th>Dir</th><th>Size</th><th>Entry</th><th>ML Conf</th></tr></thead><tbody></tbody></table>
        </div>
        <div class='box'>
          <div class='toolbar'><strong>Recent Trades</strong><span class='small muted' id='tradeCount'></span></div>
          <table id='tradesTable'><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Price</th><th>Amount</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
      <div class='box' style='margin-top:12px;'>
        <div class='toolbar'><strong>Symbol Statistics</strong></div>
        <table id='symbolsTable'><thead><tr><th>Symbol</th><th>Trades</th><th>Win Rate</th><th>PnL</th></tr></thead><tbody></tbody></table>
      </div>
    </div>

    <div id='transparency' class='panel'>
      <div class='grid2'>
        <div class='box'>
          <div class='toolbar'><strong>Decision Records</strong><span id='transCount' class='small muted'></span></div>
          <table id='transTable'><thead><tr><th>Time</th><th>Symbol</th><th>Dir</th><th>Status</th><th>PnL%</th></tr></thead><tbody></tbody></table>
        </div>
        <div class='box'>
          <div class='toolbar'><strong>Selected Decision Details</strong></div>
          <pre id='transDetail' style='white-space:pre-wrap;font-size:12px;'></pre>
        </div>
      </div>
    </div>

    <div id='thoughts' class='panel'>
      <div class='cards' id='thoughtCards'></div>
      <div class='box'>
        <div class='toolbar'>
          <strong>Reasoning / Actions Log</strong>
          <button data-filter='all' class='logFilter'>All</button>
          <button data-filter='action' class='logFilter'>Actions</button>
          <button data-filter='reasoning' class='logFilter'>Reasoning</button>
          <button data-filter='warning' class='logFilter'>Warnings</button>
        </div>
        <div id='logList'></div>
      </div>
    </div>

    <div id='charts' class='panel'>
      <div class='grid2'>
        <div class='box'>
          <div class='toolbar'><strong>Live Equity Curve</strong><span id='eqMeta' class='small muted'></span></div>
          <div class='chartWrap'><svg id='equitySvg' class='chart'></svg></div>
        </div>
        <div class='box'>
          <div class='toolbar'><strong>Daily PnL</strong></div>
          <div class='chartWrap'><svg id='dailySvg' class='chart'></svg></div>
        </div>
      </div>
      <div class='box' style='margin-top:12px;'>
        <div class='toolbar'>
          <strong>Per-Symbol Drill Down</strong>
          <select id='symbolSelect' style='background:#1e2b3a;border:1px solid #2f4258;color:#dbe4ef;border-radius:8px;padding:6px;'></select>
          <span id='symbolMeta' class='small muted'></span>
        </div>
        <div class='grid2'>
          <div class='chartWrap'><svg id='symbolBarSvg' class='chartTall'></svg></div>
          <div>
            <table id='symbolTradesTable'><thead><tr><th>Time</th><th>Side</th><th>Price</th><th>Amount</th></tr></thead><tbody></tbody></table>
          </div>
        </div>
      </div>
    </div>

    <div class='small muted' id='lastRefresh' style='margin-top:10px;'></div>
  </div>

<script>
let allData = null;
let currentLogFilter = 'all';
let refreshTimer = null;
let liveMode = true;
let refreshMs = 5000;

function fmt(n, d=2) {
  const v = Number(n || 0);
  return v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
}

function pnlClass(v) { return Number(v) >= 0 ? 'good' : 'bad'; }

function escapeHtml(v) {
  return String(v ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function drawLineSvg(svgEl, points, color, yKey) {
  const width = 900;
  const height = 220;
  svgEl.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svgEl.innerHTML = '';
  if (!points || points.length < 2) {
    svgEl.innerHTML = `<text x='10' y='20' class='axisText'>Not enough points</text>`;
    return;
  }
  const vals = points.map(p => Number(p[yKey] || 0));
  let minY = Math.min(...vals);
  let maxY = Math.max(...vals);
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  const pad = 22;
  const xStep = (width - pad * 2) / (points.length - 1);
  const yScale = (height - pad * 2) / (maxY - minY);
  const coords = points.map((p, i) => {
    const x = pad + i * xStep;
    const y = height - pad - ((Number(p[yKey] || 0) - minY) * yScale);
    return `${x},${y}`;
  }).join(' ');

  svgEl.innerHTML = `
    <line x1='${pad}' y1='${pad}' x2='${pad}' y2='${height - pad}' stroke='#2f4258'/>
    <line x1='${pad}' y1='${height - pad}' x2='${width - pad}' y2='${height - pad}' stroke='#2f4258'/>
    <polyline fill='none' stroke='${color}' stroke-width='2' points='${coords}' />
    <text x='${pad}' y='14' class='axisText'>${fmt(maxY)}</text>
    <text x='${pad}' y='${height - 6}' class='axisText'>${fmt(minY)}</text>
  `;
}

function drawBarSvg(svgEl, rows, labelKey, valueKey) {
  const width = 900;
  const height = 220;
  svgEl.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svgEl.innerHTML = '';
  if (!rows || !rows.length) {
    svgEl.innerHTML = `<text x='10' y='20' class='axisText'>No data</text>`;
    return;
  }

  const pad = 24;
  const maxAbs = Math.max(...rows.map(r => Math.abs(Number(r[valueKey] || 0))), 1);
  const zeroY = (height - pad) / 2;
  const usable = (height - pad * 2) / 2;
  const barW = Math.max(3, Math.floor((width - pad * 2) / rows.length) - 2);

  let bars = '';
  let labels = '';
  rows.forEach((r, idx) => {
    const value = Number(r[valueKey] || 0);
    const h = Math.max(1, Math.abs(value) / maxAbs * usable);
    const x = pad + idx * (barW + 2);
    const y = value >= 0 ? (zeroY - h) : zeroY;
    const color = value >= 0 ? '#73f0a8' : '#ff7676';
    bars += `<rect x='${x}' y='${y}' width='${barW}' height='${h}' fill='${color}'><title>${escapeHtml(r[labelKey])}: ${fmt(value)}</title></rect>`;
    if (idx % Math.ceil(rows.length / 12) === 0) {
      labels += `<text x='${x}' y='${height - 4}' class='axisText'>${escapeHtml(String(r[labelKey]).slice(-5))}</text>`;
    }
  });

  svgEl.innerHTML = `
    <line x1='${pad}' y1='${zeroY}' x2='${width - pad}' y2='${zeroY}' stroke='#2f4258'/>
    ${bars}
    ${labels}
    <text x='${pad}' y='14' class='axisText'>+${fmt(maxAbs)}</text>
    <text x='${pad}' y='${height - 26}' class='axisText'>-${fmt(maxAbs)}</text>
  `;
}

function setTabs() {
  document.querySelectorAll('.tab').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    };
  });
}

function renderStatus(status) {
  const cards = [
    ['Total Balance', `$${fmt(status.balances.total)}`, pnlClass(status.balances.pnl)],
    ['Spot Balance', `$${fmt(status.balances.spot)}`, ''],
    ['Futures Balance', `$${fmt(status.balances.futures)}`, ''],
    ['Total PnL', `$${fmt(status.balances.pnl)}`, pnlClass(status.balances.pnl)],
    ['Return %', `${fmt(status.balances.return_pct)}%`, pnlClass(status.balances.return_pct)],
    ['Win Rate', `${fmt(status.stats.win_rate_pct)}%`, ''],
    ['Open Positions', `${status.stats.open_positions}`, ''],
    ['Total Trades', `${status.stats.total_trades}`, '']
  ];

  document.getElementById('statusCards').innerHTML = cards.map(([k,v,c]) =>
    `<div class='card'><div class='label'>${k}</div><div class='val ${c}'>${v}</div></div>`).join('');

  const pbody = document.querySelector('#positionsTable tbody');
  document.getElementById('openCount').textContent = `(${status.open_positions.length})`;
  pbody.innerHTML = status.open_positions.slice(0, 200).map(p =>
    `<tr><td>${p.symbol}</td><td><span class='pill ${String(p.direction).toLowerCase()==='long'?'long':'short'}'>${p.direction}</span></td><td>${fmt(p.size,4)}</td><td>${fmt(p.entry_price,6)}</td><td>${fmt((p.ml_confidence||0)*100)}%</td></tr>`
  ).join('') || `<tr><td colspan='5' class='muted'>No open positions</td></tr>`;

  const tbody = document.querySelector('#tradesTable tbody');
  document.getElementById('tradeCount').textContent = `(${status.recent_trades.length})`;
  tbody.innerHTML = status.recent_trades.map(t => {
    const ts = t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : '-';
    return `<tr><td>${ts}</td><td>${t.symbol}</td><td>${t.side}</td><td>${fmt(t.price,6)}</td><td>${fmt(t.amount,6)}</td></tr>`;
  }).join('') || `<tr><td colspan='5' class='muted'>No recent trades</td></tr>`;

  const sb = document.querySelector('#symbolsTable tbody');
  const symbols = Object.entries(status.symbol_stats || {}).sort((a,b)=> (b[1].trades||0)-(a[1].trades||0));
  sb.innerHTML = symbols.map(([sym, s]) =>
    `<tr><td>${sym}</td><td>${s.trades||0}</td><td>${fmt(s.win_rate_pct||0)}%</td><td class='${pnlClass(s.pnl||0)}'>${fmt(s.pnl||0)}</td></tr>`
  ).join('') || `<tr><td colspan='4' class='muted'>No symbol statistics</td></tr>`;
}

function renderTransparency(trans) {
  document.getElementById('transCount').textContent = `(${trans.count})`;
  const tbody = document.querySelector('#transTable tbody');
  const items = trans.items || [];
  tbody.innerHTML = items.slice(0, 250).map((r, idx) => {
    const ts = r.entry_time ? new Date(r.entry_time).toLocaleString() : '-';
    const dirClass = String(r.direction).toLowerCase()==='long' ? 'long' : 'short';
    const stClass = String(r.status).toLowerCase()==='open' ? 'open' : 'closed';
    return `<tr data-idx='${idx}'><td>${ts}</td><td>${r.symbol||''}</td><td><span class='pill ${dirClass}'>${r.direction||''}</span></td><td><span class='pill ${stClass}'>${r.status||''}</span></td><td class='${pnlClass(r.pnl_pct||0)}'>${r.pnl_pct==null?'-':fmt(r.pnl_pct)}%</td></tr>`;
  }).join('') || `<tr><td colspan='5' class='muted'>No transparency records</td></tr>`;

  Array.from(tbody.querySelectorAll('tr[data-idx]')).forEach(row => {
    row.onclick = () => {
      const rec = items[Number(row.dataset.idx)];
      document.getElementById('transDetail').textContent = JSON.stringify(rec, null, 2);
    };
  });

  if (items.length) {
    document.getElementById('transDetail').textContent = JSON.stringify(items[0], null, 2);
  } else {
    document.getElementById('transDetail').textContent = 'No data';
  }
}

function renderThoughts(th) {
  const s = th.shadow_summary || {};
  const cards = [
    ['Shadow Cycle', s.shadow_cycle ?? '-'],
    ['Δ Equity', s.delta_equity ?? 0],
    ['Δ Realized PnL', s.delta_realized_pnl ?? 0],
    ['Δ Drawdown %', s.delta_drawdown_pct ?? 0],
    ['Baseline Win Rate', s.baseline_win_rate ?? 0],
    ['RL Win Rate', s.rl_win_rate ?? 0]
  ];
  document.getElementById('thoughtCards').innerHTML = cards.map(([k,v]) =>
    `<div class='card'><div class='label'>${k}</div><div class='val ${typeof v==='number'?pnlClass(v):''}'>${typeof v==='number'?fmt(v):v}</div></div>`
  ).join('');

  const logs = th.log_entries || [];
  const filtered = logs.filter(x => currentLogFilter === 'all' ? true : x.kind === currentLogFilter);
  document.getElementById('logList').innerHTML = filtered.map(x =>
    `<div class='log ${x.kind}'><span class='muted'>${x.timestamp} [${x.level}]</span> ${x.message}</div>`
  ).join('') || `<div class='log muted'>No matching thought-process lines.</div>`;
}

function renderCharts(charts) {
  const eq = charts.equity_curve || [];
  const daily = charts.daily_pnl || [];
  const symbols = charts.symbol_stats || [];
  const recentTrades = charts.recent_trades || [];

  drawLineSvg(document.getElementById('equitySvg'), eq, '#96c5ff', 'total');
  drawBarSvg(document.getElementById('dailySvg'), daily, 'day', 'pnl');
  document.getElementById('eqMeta').textContent = `latest cycle: ${charts.latest_cycle ?? '-'} | points: ${eq.length}`;

  const sel = document.getElementById('symbolSelect');
  const prev = sel.value;
  sel.innerHTML = symbols.map(s => `<option value='${escapeHtml(s.symbol)}'>${escapeHtml(s.symbol)}</option>`).join('');
  if (prev && symbols.find(s => s.symbol === prev)) sel.value = prev;

  const chosen = sel.value || (symbols[0] ? symbols[0].symbol : '');
  const ranked = symbols.slice(0, 30);
  drawBarSvg(document.getElementById('symbolBarSvg'), ranked, 'symbol', 'pnl');

  const meta = symbols.find(s => s.symbol === chosen);
  document.getElementById('symbolMeta').textContent = meta
    ? `Trades: ${meta.trades} | Win Rate: ${fmt(meta.win_rate_pct)}% | PnL: ${fmt(meta.pnl)}`
    : 'No symbol selected';

  const tbody = document.querySelector('#symbolTradesTable tbody');
  const filtered = recentTrades.filter(t => t.symbol === chosen).slice(0, 80);
  tbody.innerHTML = filtered.map(t => {
    const ts = t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : '-';
    return `<tr><td>${ts}</td><td>${escapeHtml(t.side)}</td><td>${fmt(t.price,6)}</td><td>${fmt(t.amount,6)}</td></tr>`;
  }).join('') || `<tr><td colspan='4' class='muted'>No recent trades for symbol</td></tr>`;

  sel.onchange = () => renderCharts({ ...charts, recent_trades: recentTrades, symbol_stats: symbols });
}

function scheduleRefresh() {
  if (refreshTimer) clearTimeout(refreshTimer);
  if (!liveMode) return;
  refreshTimer = setTimeout(fetchAll, refreshMs);
}

async function fetchAll() {
  const res = await fetch('/api/all');
  allData = await res.json();
  renderStatus(allData.status);
  renderTransparency(allData.transparency);
  renderThoughts(allData.thoughts);
  renderCharts(allData.charts);
  document.getElementById('lastRefresh').textContent = `Last refresh: ${new Date().toLocaleString()}`;
  scheduleRefresh();
}

document.getElementById('refreshBtn').onclick = () => fetchAll();
document.getElementById('liveToggle').onchange = (e) => {
  liveMode = !!e.target.checked;
  scheduleRefresh();
};
document.getElementById('refreshRate').onchange = (e) => {
  refreshMs = Number(e.target.value || 5000);
  scheduleRefresh();
};
document.querySelectorAll('.logFilter').forEach(btn => {
  btn.onclick = () => {
    currentLogFilter = btn.dataset.filter;
    if (allData) renderThoughts(allData.thoughts);
  };
});

setTabs();
fetchAll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("DASH_PORT", "8088"))
    debug = os.getenv("DASH_DEBUG", "false").strip().lower() == "true"
    app.run(host=host, port=port, debug=debug)
