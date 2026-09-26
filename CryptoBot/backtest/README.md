# CryptoBot Backtest Lab

Consolidated home for CryptoBot backtesting: data download, model training,
and the short-vs-long baseline driver. This folder does not change any live
trading logic or config -- it only measures the existing strategy against
real historical data.

## Why this consolidation happened

CryptoBot previously had **9 backtest-related scripts scattered across
`tools/`**, most of them hand-duplicating (and slowly drifting from) the
live entry/exit logic in `cryptotrades/core/trading_engine.py`, plus one
harness (`backtest_harness.py`, not touched by this consolidation) with a
missing data dependency. An audit found:

| Tool | Verdict | Reason |
|---|---|---|
| `tools/legacy/backtest_6mo.py` | Archived | Hand-duplicated v5-v11 signal/exit logic, used a `FUTURES_TO_SPOT` proxy mapping instead of real futures data |
| `tools/legacy/optimize_backtest.py` | Archived | Parameter sweep on the standalone rules-only strategy (see below), not the live engine |
| `tools/legacy/optimize_backtest_regime.py` | Archived | Round-2 sweep, same standalone strategy |
| `tools/legacy/optimize_backtest_v3.py` | Archived | Round-3 sweep, same standalone strategy |
| `tools/legacy/optimize_backtest_v4.py` | Archived | Round-4/final sweep, same standalone strategy |
| `tools/legacy/local_backtest_qc_port.py` | Archived | Rules-only spot-long-only mirror of the QuantConnect port; base engine the 4 sweep scripts above import from; closed investigation (best tuned result still -15.5% net, i.e. a rules-only strategy without the live ML gate has no edge here) |
| `tools/legacy/train_from_historical.py` | Archived | Superseded by `retrain_on_6mo.py` (same indicator stack, 6-month 1-min data instead of a non-existent cache CSV) |

All 7 are preserved unmodified under [`tools/legacy/`](../tools/legacy/),
each with an archival banner explaining why, for history/reference. They are
not run in CI and not maintained going forward.

### Tools that survived (still in `tools/`, unchanged)
- `download_6mo_candles.py` -- downloads 6 months of 1-min spot candles from Coinbase (free, no API key) to `data/historical/1min/`.
- `download_kraken_futures_ohlc.py` -- downloads 6 months of 1-min **real** Kraken Futures OHLC (free, public endpoint, no API key) to `data/historical/1min_futures/`. Added as part of this consolidation to replace the old `FUTURES_TO_SPOT` proxy-mapping approach.
- `retrain_on_6mo.py` -- trains the live `MarketPredictor` model (`cryptotrades/utils/market_predictor.py`) on the 6-month spot dataset.
- `walk_forward.py` -- rolling train/test ML validation + historical-trade Monte Carlo, both driven by the live `FeatureEngine`/`MarketPredictor`/`config`.

### Why no new `futures_lab.py` was built
`cryptotrades/utils/backtester.py` already contains `FuturesBacktester`,
which drives the **live** `MarketPredictor` + live `config` thresholds
(long/short confidence, TP/SL, leverage, funding costs, slippage, partial
fills) for both long and short futures trades -- it is not a hand-duplicated
copy. `run_baseline.py` (this folder) simply feeds it real historical data.
Building a separate `futures_lab.py` would have re-created what already
exists. **Caveat:** `FuturesBacktester` only recently gained real futures
OHLC input (via `download_kraken_futures_ohlc.py`); before this
consolidation it had only ever been fed spot-proxy or synthetic data in
practice, so this is the first time it's being validated against genuine
Kraken Futures price action end-to-end.

### Open item: ML model path discrepancy (flagged, not fixed)
- Live config's `ML_MODEL_PATH` and `retrain_on_6mo.py`'s output both point
  to `data/models/market_model.joblib` (gitignored, machine-local).
- The now-archived `train_from_historical.py` instead wrote to a **different**
  path, `models/trading_model.joblib` (no `data/` prefix) -- and that
  mismatched path is what's actually present on the droplet
  (`models/trading_model_*.joblib` files exist there).
- It's unclear from this audit alone which path the live `TradingBot` loads
  at runtime in production. **This was intentionally left unresolved** --
  no production config changes were made as part of this consolidation.
  Verify which model file the running bot actually loads before trusting
  `retrain_on_6mo.py`'s output to affect live trading.

## Running the baseline

```bash
cd CryptoBot

# 1. Download 6 months of real 1-min spot candles (Coinbase, free)
python3 tools/download_6mo_candles.py

# 2. Download 6 months of real 1-min Kraken Futures candles (free, public)
python3 tools/download_kraken_futures_ohlc.py

# 3. Train the live MarketPredictor model on the 6-month spot dataset
python3 tools/retrain_on_6mo.py

# 4. Run the short-vs-long baseline with the strictest execution-cost
#    assumptions (higher slippage, partial fills, funding costs, fees)
SIM_REALISM_PROFILE=strict python3 backtest/run_baseline.py
```

`run_baseline.py` reports, per spot/futures symbol and in aggregate:
trades, win rate, P&L, max drawdown, Sharpe ratio, profit factor, and exit
reason breakdown -- plus a dedicated **futures long-vs-short** breakdown
(trades, win rate, net P&L, profit factor, exit reasons per direction) to
answer whether the live short-side futures logic is adding value. Full
results are written to `data/state/baseline_report.json` (gitignored).

For walk-forward ML validation and historical-trade Monte Carlo simulation
(a complementary, longer-running analysis), use `tools/walk_forward.py`
separately -- it is not part of the `run_baseline.py` flow.

## Data layout (all gitignored, re-fetchable)
```
data/historical/1min/          <- spot 1-min candles (download_6mo_candles.py)
data/historical/1min_futures/  <- futures 1-min candles (download_kraken_futures_ohlc.py)
data/models/market_model.joblib <- trained model (retrain_on_6mo.py)
data/state/baseline_report.json <- run_baseline.py output
```
