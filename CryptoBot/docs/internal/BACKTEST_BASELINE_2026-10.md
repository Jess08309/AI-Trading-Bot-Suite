# October 2026 baseline: blocked, not validated

Audit date: 2026-10-03 UTC. **No production strategy change is shipped. No
profitability claim is supported.** Paper trading and order-enable flags,
credentials, retired bots, and PR #39 are untouched.

## Executed work and data coverage

This fresh workspace had no historical CSVs, trained models, live trade
records, or production state. Public probes to Coinbase candles, Kraken
instruments/charts, and Alpaca crypto market data all failed DNS resolution
(`NameResolutionError`, no address associated with hostname). No account
endpoint was called and no credentials were used.

| Requested execution | Observed result |
|---|---|
| `python3 tools/download_6mo_candles.py` | Repeated Coinbase DNS failures; stopped with `timeout 25s` (124), no candles |
| `python3 tools/download_kraken_futures_ohlc.py` | Instruments and chart DNS failures; stopped with `timeout 25s` (124), no candles |
| `python3 tools/retrain_on_6mo.py --meta-output data/models/market_model_training_meta.json` | “No data files”; no model or metadata produced |
| Strict `backtest/run_baseline.py --model data/models/market_model.joblib --model-meta data/models/market_model_training_meta.json` (before edits) | Exit 1: could not load model |
| `python3 tools/walk_forward.py` | “trades.csv not found”; returns before ML validation |
| `python3 tools/walk_forward.py --skip-mc` | “historical_prices_2yr.csv not found”; writes config-only JSON, **not a validation result** |

The commands used the existing requirements in a temporary virtualenv;
dependency declarations were not changed. Download retries were deliberately
bounded instead of spending hours retrying unreachable hosts. Raw console
outputs remain under `/tmp/cryptobot-*.txt` in this session, not committed.

Requested window: approximately 2026-04-06 through 2026-10-03 (the downloaders
use 180 days, not calendar months). Acquired spot/futures rows: **0**;
acquisition coverage: **0%**, across all 17 requested spot and 15 futures
symbols. There is no measured market-series gap coverage because there is
no series. Futures source substitutions (PI to PF, MATIC to POL) must be
audited in `_symbol_sources.json` when data is available; a substituted
contract is not evidence about the original contract's fills.

## Baseline versus improved

N/A means **not measured**, not zero trades or break-even. This table is also
the PR summary; it must not be replaced with synthetic-market numbers.

| Run / evidence label | Trades | WR | PF | MaxDD | Net P&L |
|---|---:|---:|---:|---:|---:|
| Full-window baseline, would be **in-sample** after full-window training; blocked | N/A | N/A | N/A | N/A | N/A |
| Held-out baseline, **out-of-sample pending**; blocked | N/A | N/A | N/A | N/A | N/A |
| Improved held-out strategy, **not run / none shipped** | N/A | N/A | N/A | N/A | N/A |

Per-symbol spot/futures returns, Sharpe, exit-reason distribution, and
futures LONG/SHORT trades, win rates, PF and P&L are all **unavailable**.
**LONG-versus-SHORT verdict: inconclusive; no directional gate is justified.**
Monte Carlo profit probability, loss tails, and walk-forward train/test
accuracy are also unavailable. Historical loss claims are reviewed separately
in [LIVE_DATA_REVIEW.md](LIVE_DATA_REVIEW.md), not counted as current results.

## Critical correction: the lab is not the production engine

The task and older lab README assumed shared strategy/model/config. Source
inspection disproves that assumption:

| Concern | Actual implementation (paths relative to `CryptoBot/`) |
|---|---|
| Production startup | `cryptotrades/core/trading_engine.py:1439-1456,1639-1655`: constructs `MLModel()`, loads state, then `load_or_train`; optional forced startup training |
| Production model path | `core/trading_engine.py:623-636`: `models/trading_model.joblib`; challenger uses `_challenger.joblib` and is shadow-only |
| Startup file selection | `core/trading_engine.py:638-661`: loads exactly `self.model_path`, checks feature count, trains on missing/unloadable/mismatched model if history exists; does **not** select the newest versioned file |
| Production writes | `core/trading_engine.py:943-954`: accepted `_train` model written to the same `self.model_path` plus timestamp/accuracy backup |
| Production schedule | `core/trading_engine.py:84,288,3495-3530`: `cfg.MODEL_RETRAIN_HOURS=8`, env `MODEL_RETRAIN_HOURS_OVERRIDE`; at least two series with 60 points and 240 total points |
| Lab model | `cryptotrades/utils/market_predictor.py:32-44,72-80`: `MarketPredictor.load_model()` loads only its configured path, default `data/models/market_model.joblib`; no fallback to production |
| Lab config | `cryptotrades/utils/config.py:110,210`: `MODEL_RETRAIN_INTERVAL=21600` and `ML_MODEL_PATH=data/models/market_model.joblib`; these do **not** drive production startup/scheduling |
| Offline trainer | `tools/retrain_on_6mo.py:64-65,206-255,270-299`: GradientBoosting training, saves selected fold model and training-window metadata |

References prefixed `core/` above are under `cryptotrades/`. Production
`MLModel` and lab `FeatureEngine` currently both use seven named indicators
(`core/trading_engine.py:614-617`, `utils/feature_engine.py:24-32`), but that
does **not** make their training, calibration, probability semantics,
timeframe, or entry/exit policies equivalent. Do not copy one artifact over
the other merely to resolve the names.

All relative model paths resolve against the **running process cwd**.
`cryptotrades/main.py` adjusts import paths and loads its adjacent `.env`,
but does not `chdir`. If cwd is `CryptoBot/`, production loads
`CryptoBot/models/trading_model.joblib`; if cwd is `CryptoBot/cryptotrades/`,
it loads `CryptoBot/cryptotrades/models/trading_model.joblib`. Neither the
droplet's current cwd nor loaded model identity is established here. The
live-data report supplies read-only discovery commands.

The lab calls `run_full_backtest` (`utils/backtester.py:1052-1107`), not
`TradingBot`. That runner does not propagate `utils.config` strategy
thresholds to its constructors: spot TP defaults are 2%/1.5%
(`backtester.py:145-165`), not the displayed config TP 5%; futures thresholds
are constructor defaults (`backtester.py:676-697`). Strict execution-cost
defaults are read from utils config, but production uses internal `cfg`,
locked-profile overrides, limit-order/spread guards, ATR exits,
symbol/direction pauses, and portfolio risk rules absent from this lab.
Thus changing a displayed config value may not change a simulated entry.

The simulator consumes **closes only** and indexes them as one-minute
observations: gaps are compressed, OHLC extremes and real limit-order
latency/rejections are not replayed. Each symbol gets an independent
starting balance, not a shared portfolio. Per-symbol Sharpe is
mean trade return / standard deviation × sqrt(trade count)
(`backtester.py:601-616`), not annualized minute Sharpe.

**Resolution:** document the two model paths, not silently redirect a live
model. Promoting any lab candidate requires a separately reviewed replay of
the actual production decision path and forward paper confirmation.

## Leakage-safe tooling shipped (lab only)

`backtest/run_baseline.py` now accepts `--start-date` (inclusive) and
`--end-date` (exclusive), interpreted as UTC if timezone is omitted, and
filters rows **before** candle aggregation. Default full-window execution
is unchanged. No pre-test prices warm up indicators or open positions.
Reports include actual selected timestamps/coverage, temporal leakage label,
per-symbol exit counts and pooled trade statistics. Worst-symbol drawdown
and mean-symbol Sharpe are explicitly **not portfolio** statistics.

`--require-out-of-sample` refuses unknown/overlapping training windows and
non-strict profiles before simulation. Matching training metadata is required:
metadata is a provenance assertion, not a cryptographic artifact check.
Missing/malformed explicit metadata fails rather than quietly claiming OOS.
These tests verify infrastructure correctness, **not a profitable strategy**.

For real data obtained on an allowed host, run from an isolated checkout
with cwd at `CryptoBot/`, **never the running bot's working directory**:

```bash
python3 tools/download_6mo_candles.py
python3 tools/download_kraken_futures_ohlc.py
python3 tools/retrain_on_6mo.py --end-date 2026-08-14T00:00:00Z \
  --output data/models/market_model_frozen.joblib \
  --meta-output data/models/market_model_frozen_training_meta.json
SIM_REALISM_PROFILE=strict python3 backtest/run_baseline.py \
  --model data/models/market_model_frozen.joblib \
  --model-meta data/models/market_model_frozen_training_meta.json \
  --start-date 2026-08-14T00:00:00Z --end-date 2026-10-03T00:00:00Z \
  --require-out-of-sample --output data/state/baseline_oos.json
python3 tools/walk_forward.py
```

This reserves about 28% (50 of 180 days) for a chronologically later test;
the trainer drops all rows at/after the cutoff before forward labels are
built. Keep the metadata and model frozen together. Select candidates using
earlier development folds, not repeated tuning on this final test.
The trainer concatenates symbol samples before `TimeSeriesSplit`; its CV
accuracy is **not** a global chronological OOS trading gate.

Existing `walk_forward.py` additionally requires `data/trades.csv` for
Monte Carlo and `data/historical/historical_prices_2yr.csv` for ML. Despite
its imports, its rolling validator fits its own RandomForest on row-count
windows (`tools/walk_forward.py:339-386`), not production `MLModel`; its
forward-label boundary also needs purging before accuracy is trustworthy.
Do not equate its ML accuracy or bootstrap of existing trades with a strict
walk-forward production P&L test. These pre-existing limitations are documented,
not rewritten without real-data validation.

## Candidates rejected for shipping

| Hypothesis | Evidence / required comparison | Decision |
|---|---|---|
| Prefer futures SHORT or LONG | Compare both directions on multiple strict OOS windows | Rejected: no direction results; futures enable flags untouched |
| Add ONDO/thin-pair blacklist | PRs #17/#29/#37 show unconfirmed fills and exit blocking; need current symbol expectancy and fill counts | Rejected: no current live series or OOS gain; existing `SYMBOL_BLACKLIST` mechanism preserved |
| Tighten stops / breach confirmation | Compare exit losses and actual fill slippage with configured/ATR stops | Rejected: no exit data; close-only lab cannot reproduce broker exit blocking |
| Change score/confidence gates | Compare PF, net P&L and drawdown under matching production rules | Rejected: unavailable data and lab/live threshold mismatch |
| Redirect production to offline model | Distinct loaders/training procedures verified in source | Rejected: naming discrepancy is architectural, not proof of a wrong live file |

None was simulated and found inferior; they are **rejected for deployment
because validation is unavailable**, not falsely claimed to be backtested.
No new live config knob or revert knob is needed: there is no live change.
Future acceptance requires higher strict OOS net P&L and PF without worse
drawdown, sufficient trades across regimes, and matching production replay.

## Test baseline and verification

Initial system Python lacked pytest. After installing existing dependencies
in `/tmp/cryptobot-venv`, repo-root pytest required the existing CryptoBot
import paths (otherwise `test_backtest_harness.py` cannot import its module).

```bash
cd /home/runner/work/AI-Trading-Bot-Suite/AI-Trading-Bot-Suite
PYTHONPATH="$PWD/CryptoBot:$PWD/CryptoBot/cryptotrades" \
  /tmp/cryptobot-venv/bin/python -m pytest \
  "$PWD/CryptoBot/tests" "$PWD/CryptoBot/cryptotrades/tests" "$PWD/tools/tests" -q
```

Before changes: **169 passed, 11 failed**. The failures are all in
`CryptoBot/tests/test_critical_paths.py`:

- `TestFillConfirmation`: `test_buy_confirmed_fill_returns_qty`,
  `test_buy_zero_fill_returns_zero_and_cancels`,
  `test_repeated_broker_zero_qty_available_reconciles_explicitly`,
  `test_sell_full_fill_closes_position`,
  `test_sell_qty_available_none_falls_back_to_position_qty`,
  `test_sell_qty_available_positive_clamps_requested_qty`.
- `TestUnconfirmedFillCooldown`:
  `test_confirmed_fill_resets_unconfirmed_fill_strikes`,
  `test_three_consecutive_unconfirmed_fills_block_symbol`.
- `TestSellSpotQtyAvailableTruthiness`:
  `test_qty_available_none_falls_back_to_position_qty`,
  `test_qty_available_partial_clamps_request`.
- `TestDustReconciliation`: `test_confirmed_sell_resets_retry_streak`.

Observed errors include missing `order_retry_manager`/`expectancy_tracker`
on test-created bots and sell mocks expecting the older order flow. These
unrelated tests and production paths were not altered to hide failures.
New focused tests: **18 passed** (UTC/date boundaries, gaps, empty/short
windows, unknown/overlap labels, mixed futures/spot leakage, strict gate,
aggregation and CLI rejection). After changes: **187 passed, the identical
11 failures**, no new failures; separate `cryptotrades/tests` invocation:
**101 passed**. CLI help, missing-model exit 1 and invalid-window exit 2 were
manually verified. `git diff --check` passed. Code/security review is performed
before completing the PR; any unavailable scan will be disclosed.
