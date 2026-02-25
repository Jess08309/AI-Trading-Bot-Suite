# Monitoring Stack (Prometheus + Grafana)

## What this provides
- `state_exporter.py`: exposes bot state metrics from `data/state/*.json`
- Prometheus scrape endpoint: `http://localhost:9090`
- Grafana UI: `http://localhost:3000` (admin/admin)
- Exporter metrics endpoint: `http://localhost:9108`

## Start stack
From `C:\Master Chess\monitoring`:

```powershell
docker compose up -d
```

## Key metrics exposed
- `trading_total_pnl_usd`
- `trading_win_rate_pct`
- `trading_sharpe_ratio`
- `trading_max_drawdown_pct`
- `trading_backtest_total_pnl_usd`
- `trading_backtest_trades_total`
- `trading_balance_spot_usd`
- `trading_balance_futures_usd`
- `trading_open_positions`

## Grafana setup
1. Add data source: Prometheus URL `http://prometheus:9090`
2. Create panels for the metrics above
3. Save dashboard as "Trading Bot Performance"

## Stop stack
```powershell
docker compose down
```
