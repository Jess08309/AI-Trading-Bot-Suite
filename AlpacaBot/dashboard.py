"""
AlpacaBot Web Dashboard
=======================
Real-time web dashboard for the scalp trading bot.
Shows account status, positions, trades, signals, and activity log.
Auto-refreshes every 5 seconds via AJAX.
"""
import json
import logging
import threading
import traceback
from datetime import datetime

from flask import Flask, jsonify, render_template_string

log = logging.getLogger("alpacabot.dashboard")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlpacaBot Scalp Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0e17;
            color: #e0e6ed;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1f35 0%, #0d1326 100%);
            border-bottom: 2px solid #2a3a5c;
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 22px;
            background: linear-gradient(90deg, #00d4ff, #7b68ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .status {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 13px;
        }
        .header .dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        .dot.green { background: #00ff88; box-shadow: 0 0 8px #00ff8855; }
        .dot.yellow { background: #ffbb00; box-shadow: 0 0 8px #ffbb0055; }
        .dot.red { background: #ff4444; box-shadow: 0 0 8px #ff444455; }

        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            padding: 16px 24px;
        }
        .card {
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 16px;
        }
        .card h3 {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #64748b;
            margin-bottom: 8px;
        }
        .card .value {
            font-size: 28px;
            font-weight: 700;
        }
        .card .sub {
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
        }
        .positive { color: #00ff88; }
        .negative { color: #ff4444; }
        .neutral { color: #94a3b8; }

        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 12px;
            padding: 0 24px 16px;
        }

        .panel {
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 16px;
            max-height: 420px;
            overflow-y: auto;
        }
        .panel h2 {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #7b68ee;
            margin-bottom: 12px;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 8px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th {
            text-align: left;
            color: #64748b;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 6px 8px;
            border-bottom: 1px solid #1e293b;
        }
        td {
            padding: 8px;
            border-bottom: 1px solid #0f172a;
        }
        tr:hover { background: #1e293b33; }

        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-call { background: #00ff8822; color: #00ff88; }
        .badge-put { background: #ff444422; color: #ff4444; }
        .badge-win { background: #00ff8822; color: #00ff88; }
        .badge-loss { background: #ff444422; color: #ff4444; }

        .activity-item {
            padding: 6px 0;
            border-bottom: 1px solid #0f172a;
            font-size: 12px;
            line-height: 1.4;
        }
        .activity-item .time {
            color: #64748b;
            font-family: monospace;
            margin-right: 8px;
        }
        .activity-item.warning { color: #ffbb00; }
        .activity-item.error { color: #ff4444; }

        .bottom-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
            padding: 0 24px 24px;
        }

        .config-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px 16px;
            font-size: 13px;
        }
        .config-grid .label { color: #64748b; }
        .config-grid .val { color: #e0e6ed; text-align: right; }

        .risk-bar {
            height: 6px;
            background: #1e293b;
            border-radius: 3px;
            margin-top: 8px;
            overflow: hidden;
        }
        .risk-bar .fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s;
        }

        #lastUpdate {
            font-size: 11px;
            color: #475569;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .pulsing { animation: pulse 2s infinite; }

        .paper-badge {
            background: #ffbb0033;
            color: #ffbb00;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AlpacaBot SCALP Dashboard</h1>
        <div class="status">
            <span id="paperBadge" class="paper-badge">PAPER</span>
            <span><span id="statusDot" class="dot green"></span><span id="statusText">Loading...</span></span>
            <span id="lastUpdate"></span>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>Portfolio Value</h3>
            <div class="value" id="equity">$0.00</div>
            <div class="sub" id="buyingPower">Buying power: $0</div>
        </div>
        <div class="card">
            <h3>Session P&L</h3>
            <div class="value" id="totalPnl">$0.00</div>
            <div class="sub" id="pnlBreakdown">Closed: $0 | Open: $0</div>
        </div>
        <div class="card">
            <h3>Positions</h3>
            <div class="value" id="posCount">0 / 3</div>
            <div class="sub" id="signalCount">Signals today: 0</div>
        </div>
        <div class="card">
            <h3>Uptime</h3>
            <div class="value" id="uptime">00:00:00</div>
            <div class="sub" id="cycleCount">Cycle: 0</div>
        </div>
    </div>

    <div class="main-grid">
        <div class="panel">
            <h2>Open Positions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Strike</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>Qty</th>
                        <th>P&L</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody id="positionsTable"></tbody>
            </table>
            <div id="noPositions" style="text-align:center;color:#475569;padding:24px;font-size:13px;">
                No open positions
            </div>
        </div>
        <div class="panel">
            <h2>Activity Log</h2>
            <div id="activityLog"></div>
        </div>
    </div>

    <div class="bottom-grid">
        <div class="panel">
            <div class="scanner-header">
                <h2>Scanner Signals</h2>
                <div class="scan-stats">
                    <span class="badge-scan" id="scanBadge">OFF</span>
                    &nbsp; <span id="scanStats"></span>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Score</th>
                        <th>Price</th>
                        <th>RSI</th>
                        <th>Trend</th>
                    </tr>
                </thead>
                <tbody id="scannerTable"></tbody>
            </table>
            <div id="noScanner" style="text-align:center;color:#475569;padding:24px;font-size:13px;">
                Scanner idle — waiting for first scan
            </div>
        </div>
        <div class="panel">
            <h2>Recent Trades</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>P&L</th>
                        <th>P&L%</th>
                        <th>Exit Reason</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody id="tradesTable"></tbody>
            </table>
            <div id="noTrades" style="text-align:center;color:#475569;padding:24px;font-size:13px;">
                No trades yet
            </div>
        </div>
        <div class="panel">
            <h2>Risk & Config</h2>
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
                    <span>Daily P&L</span>
                    <span id="dailyPnl">$0.00</span>
                </div>
                <div class="risk-bar"><div class="fill" id="dailyBar" style="width:50%;background:#00ff88;"></div></div>
            </div>
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
                    <span>Drawdown</span>
                    <span id="drawdown">0.0%</span>
                </div>
                <div class="risk-bar"><div class="fill" id="ddBar" style="width:0%;background:#ffbb00;"></div></div>
            </div>
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
                    <span>Consecutive Losses</span>
                    <span id="consecLosses">0 / 5</span>
                </div>
                <div class="risk-bar"><div class="fill" id="lossBar" style="width:0%;background:#ff4444;"></div></div>
            </div>
            <div id="breakerAlert" style="display:none;background:#ff444422;border:1px solid #ff4444;border-radius:6px;padding:8px;margin:8px 0;font-size:12px;color:#ff4444;">
                CIRCUIT BREAKER ACTIVE
            </div>
            <hr style="border-color:#1e293b;margin:12px 0;">
            <div class="config-grid">
                <span class="label">Watchlist</span><span class="val" id="cfgWatch">-</span>
                <span class="label">Target DTE</span><span class="val" id="cfgDte">-</span>
                <span class="label">Stop Loss</span><span class="val" id="cfgSl">-</span>
                <span class="label">Take Profit</span><span class="val" id="cfgTp">-</span>
                <span class="label">Trailing Stop</span><span class="val" id="cfgTs">-</span>
                <span class="label">Max Positions</span><span class="val" id="cfgMaxPos">-</span>
            </div>
            <hr style="border-color:#1e293b;margin:12px 0;">
            <div style="font-size:12px;">
                <div style="display:flex;justify-content:space-between;margin:4px 0;">
                    <span class="label">MSFT</span><span id="priceMSFT">-</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin:4px 0;">
                    <span class="label">NVDA</span><span id="priceNVDA">-</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        function fmt(n, dec=2) {
            if (n === null || n === undefined) return '-';
            return '$' + Number(n).toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec});
        }
        function pct(n) {
            if (n === null || n === undefined) return '-';
            return (n * 100).toFixed(1) + '%';
        }
        function pnlClass(n) {
            return n > 0 ? 'positive' : n < 0 ? 'negative' : 'neutral';
        }

        async function refresh() {
            try {
                const resp = await fetch('/api/state');
                const d = await resp.json();

                // Status
                const dot = document.getElementById('statusDot');
                const st = document.getElementById('statusText');
                st.textContent = d.status || 'Unknown';
                if (d.running && d.status.includes('Active')) {
                    dot.className = 'dot green';
                } else if (d.running) {
                    dot.className = 'dot yellow';
                } else {
                    dot.className = 'dot red';
                }

                // Paper badge
                document.getElementById('paperBadge').style.display = d.config.paper ? '' : 'none';

                // Cards
                const acct = d.account || {};
                document.getElementById('equity').textContent = fmt(acct.portfolio_value || acct.equity);
                document.getElementById('buyingPower').textContent = 'Buying power: ' + fmt(acct.buying_power);

                const tp = d.total_pnl || 0;
                document.getElementById('totalPnl').textContent = fmt(tp);
                document.getElementById('totalPnl').className = 'value ' + pnlClass(tp);
                document.getElementById('pnlBreakdown').textContent =
                    'Closed: ' + fmt(d.closed_pnl) + ' | Open: ' + fmt(d.open_pnl);

                document.getElementById('posCount').textContent = d.num_positions + ' / ' + d.max_positions;
                document.getElementById('signalCount').textContent = 'Signals today: ' + d.signals_today + ' | Trades: ' + d.trades_today;

                document.getElementById('uptime').textContent = d.uptime || '00:00:00';
                document.getElementById('cycleCount').textContent = 'Cycle: ' + d.cycle;

                // Positions
                const ptb = document.getElementById('positionsTable');
                const np = document.getElementById('noPositions');
                if (d.positions && d.positions.length > 0) {
                    np.style.display = 'none';
                    ptb.innerHTML = d.positions.map(p => `
                        <tr>
                            <td><strong>${p.underlying}</strong></td>
                            <td><span class="badge badge-${p.direction}">${p.direction.toUpperCase()}</span></td>
                            <td>$${p.strike}</td>
                            <td>${fmt(p.entry_price)}</td>
                            <td>${fmt(p.current_price)}</td>
                            <td>${p.qty}</td>
                            <td class="${pnlClass(p.pnl)}">${fmt(p.pnl)} (${pct(p.pnl_pct)})</td>
                            <td>${p.score}</td>
                        </tr>
                    `).join('');
                } else {
                    np.style.display = '';
                    ptb.innerHTML = '';
                }

                // Activity log
                const al = document.getElementById('activityLog');
                if (d.activity_log && d.activity_log.length > 0) {
                    al.innerHTML = d.activity_log.map(a => `
                        <div class="activity-item ${a.level || ''}">
                            <span class="time">${a.time}</span>${a.msg}
                        </div>
                    `).join('');
                }

                // Trades
                const ttb = document.getElementById('tradesTable');
                const nt = document.getElementById('noTrades');
                if (d.trade_history && d.trade_history.length > 0) {
                    nt.style.display = 'none';
                    ttb.innerHTML = d.trade_history.slice(0, 20).map(t => {
                        const pnl = typeof t.pnl === 'string' ? parseFloat(t.pnl) : t.pnl;
                        const pnlp = typeof t.pnl_pct === 'string' ? parseFloat(t.pnl_pct) : t.pnl_pct;
                        const dir = t.direction || '';
                        const exitTime = t.exit_time ? t.exit_time.split('T')[1]?.split('.')[0] || '' : '';
                        return `
                            <tr>
                                <td>${t.underlying || ''}</td>
                                <td><span class="badge badge-${dir}">${dir.toUpperCase()}</span></td>
                                <td class="${pnlClass(pnl)}">${fmt(pnl)}</td>
                                <td class="${pnlClass(pnlp)}">${pct(pnlp)}</td>
                                <td>${t.exit_reason || ''}</td>
                                <td>${exitTime}</td>
                            </tr>
                        `;
                    }).join('');
                } else {
                    nt.style.display = '';
                    ttb.innerHTML = '';
                }

                // Risk
                const risk = d.risk || {};
                const dpnl = risk.daily_pnl || 0;
                document.getElementById('dailyPnl').textContent = fmt(dpnl);
                document.getElementById('dailyPnl').className = pnlClass(dpnl);

                const dd = risk.drawdown || 0;
                document.getElementById('drawdown').textContent = (dd * 100).toFixed(1) + '%';
                document.getElementById('ddBar').style.width = Math.min(100, Math.abs(dd) / 0.15 * 100) + '%';
                document.getElementById('ddBar').style.background = Math.abs(dd) > 0.10 ? '#ff4444' : '#ffbb00';

                const cl = risk.consecutive_losses || 0;
                document.getElementById('consecLosses').textContent = cl + ' / 5';
                document.getElementById('lossBar').style.width = (cl / 5 * 100) + '%';

                const ba = document.getElementById('breakerAlert');
                if (risk.breaker_active) {
                    ba.style.display = '';
                    ba.textContent = 'CIRCUIT BREAKER: ' + (risk.breaker_reason || 'Active');
                } else {
                    ba.style.display = 'none';
                }

                // Config
                const cfg = d.config || {};
                document.getElementById('cfgWatch').textContent = (cfg.watchlist || []).join(', ');
                document.getElementById('cfgDte').textContent = cfg.dte + 'd';
                document.getElementById('cfgSl').textContent = (cfg.stop_loss * 100).toFixed(0) + '%';
                document.getElementById('cfgTp').textContent = '+' + (cfg.take_profit * 100).toFixed(0) + '%';
                document.getElementById('cfgTs').textContent = (cfg.trailing_stop * 100).toFixed(0) + '%';
                document.getElementById('cfgMaxPos').textContent = cfg.max_positions;

                // Scanner
                const sc = d.scanner || {};
                const scanBadge = document.getElementById('scanBadge');
                if (sc.enabled) {
                    scanBadge.textContent = 'ON';
                    scanBadge.style.background = '#00ff8822';
                    scanBadge.style.color = '#00ff88';
                    scanBadge.style.borderColor = '#00ff8855';
                } else {
                    scanBadge.textContent = 'OFF';
                }
                const scanStatsEl = document.getElementById('scanStats');
                if (sc.total_scans > 0) {
                    scanStatsEl.textContent = `Scan #${sc.total_scans} | ${sc.symbols_scanned} symbols | ${sc.signals_found} signals`;
                }

                const stb = document.getElementById('scannerTable');
                const ns = document.getElementById('noScanner');
                const topSigs = sc.latest_signals || [];
                if (topSigs.length > 0) {
                    ns.style.display = 'none';
                    stb.innerHTML = topSigs.map(s => {
                        const dirCls = s.direction === 'call' ? 'call' : 'put';
                        const scoreW = Math.min(100, s.score / 15 * 100);
                        const scoreColor = s.score >= 8 ? '#00ff88' : s.score >= 6 ? '#ffbb00' : '#94a3b8';
                        const rsiColor = s.rsi < 30 ? '#00ff88' : s.rsi > 70 ? '#ff4444' : '#94a3b8';
                        return `
                            <tr class="scanner-row">
                                <td class="rank">${s.rank}</td>
                                <td><strong>${s.symbol}</strong></td>
                                <td><span class="badge badge-${dirCls}">${s.direction.toUpperCase()}</span></td>
                                <td>${s.score} <span class="score-bar" style="width:${scoreW}%;background:${scoreColor}"></span></td>
                                <td>$${Number(s.price).toFixed(2)}</td>
                                <td style="color:${rsiColor}">${s.rsi}</td>
                                <td>${s.trend}</td>
                            </tr>
                        `;
                    }).join('');
                } else {
                    ns.style.display = '';
                    stb.innerHTML = '';
                }

                // Prices
                const prices = d.prices || {};
                document.getElementById('priceMSFT').textContent = prices.MSFT ? '$' + Number(prices.MSFT).toFixed(2) : '-';
                document.getElementById('priceNVDA').textContent = prices.NVDA ? '$' + Number(prices.NVDA).toFixed(2) : '-';

                // Update timestamp
                document.getElementById('lastUpdate').textContent = 'Updated: ' + new Date().toLocaleTimeString();

            } catch (e) {
                document.getElementById('statusDot').className = 'dot red pulsing';
                document.getElementById('statusText').textContent = 'Connection lost';
            }
        }

        // Refresh every 5 seconds
        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>
"""


class Dashboard:
    """Flask-based web dashboard for AlpacaBot."""

    def __init__(self, engine, host, port):
        self.engine = engine
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML)

        @self.app.route("/api/state")
        def api_state():
            try:
                state = self.engine.get_dashboard_state()
                return jsonify(state)
            except Exception as e:
                log.error("Dashboard API error:\n" + traceback.format_exc())
                return jsonify({"error": str(e), "running": False}), 500

    def start(self):
        """Start dashboard in background thread."""
        thread = threading.Thread(target=self._run_server, daemon=True)
        thread.start()
        log.info("Dashboard started at http://localhost:" + str(self.port))

    def _run_server(self):
        """Run Flask server (called in background thread)."""
        wlog = logging.getLogger("werkzeug")
        wlog.setLevel(logging.WARNING)
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
