# Master Chess - Bot Control & Monitoring Workspace

Control panel for the AI crypto trading bot. Launcher scripts, dashboards,
performance reports, and backtesting harness.

The live bot engine lives at D:\042021\CryptoBot (see its own README.md).

---

## Quick Start

    # Launch (Quality Aggressive v5 profile)
    .\START_BOT_AGGRESSIVE.bat

    # Watch live output
    .\WATCH_CONSOLE.ps1

    # Quick P/L report
    python quick_report.py

---

## Launcher Profiles

| Script | Description |
|--------|-------------|
| START_BOT_AGGRESSIVE.bat | Active - v5 Quality Aggressive: trades every cycle, ML>=0.62 |
| START_BOT_CONSERVATIVE.bat | Higher thresholds, longer cycles, safer |
| START_BOT.bat | Base launcher (called by profiles) |
| RESTART_BOT_STRICT.ps1 | Graceful stop + restart |

## Dashboards and Monitors

| Script | Purpose |
|--------|---------|
| WATCH_CONSOLE.ps1 | Tail the live bot log |
| WATCH_TRADES.ps1 | Watch trade entries/exits |
| UNIFIED_DASHBOARD.ps1 | All-in-one status view |
| TRANSPARENCY_DASHBOARD.ps1 | Signal reasoning breakdown |
| SHADOW_RL_DASHBOARD.ps1 | RL shadow-mode performance |
| THINKING_DASHBOARD.ps1 | ML decision introspection |
| STATUS_MONITOR.ps1 | Bot health/uptime check |
| VIEW_EXIT_REASONS.ps1 | Trade exit reason summary |

## Tools

| File | Purpose |
|------|---------|
| quick_report.py | P/L and win-rate report |
| backtest_harness.py | Historical backtests |
| main.py | CLI entrypoint: report, backtest, exporter |
| LOCK_CURRENT_PROFILE.ps1 | Lock config profile to disk |
| UNLOCK_PROFILE.ps1 | Unlock config changes |

## Project Layout

    START_BOT*.bat                  Launchers
    *.ps1                           Dashboards and monitors
    main.py                         CLI entrypoint
    quick_report.py                 Performance report
    backtest_harness.py             Backtesting
    requirements.txt                Python deps (reports only)
    data/                           Cached history and state
    docs/                           Documentation
    logs/                           Console output logs
    models/                         Model snapshots
    monitoring/                     Prometheus exporter
    tests/                          Unit tests
    tools/                          Extra utilities
    _ARCHIVE/                       Pre-v5 snapshots
    _BACKUPS/                       Zip backups

## Data Paths

| Path | Content |
|------|---------|
| data/history/trade_history.csv | All historical trades |
| data/state/coin_performance.json | Per-symbol metrics |
| data/state/paper_balances.json | Current balances |

## Git Rules

- .env, logs, backup ZIPs, and __pycache__ are gitignored
- Commit only code, docs, and intentionally tracked state

Last Updated: February 16, 2026
