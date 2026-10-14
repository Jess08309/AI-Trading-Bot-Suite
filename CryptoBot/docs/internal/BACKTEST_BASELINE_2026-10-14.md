# October 2026 baseline retry: blocked, not validated

Audit date: 2026-10-14 UTC. **The production strategy was not changed and no
profitability claim is supported.** This was a research-only attempt in a fresh
checkout; no production host, broker account, or credentials were accessed.

## Execution and data coverage

The existing dependencies from `requirements.txt` and
`cryptotrades/requirements.txt` were installed in
`/tmp/cryptobot-oct2026-venv`. Dependency declarations were not changed.
Commands ran with cwd at `CryptoBot/`.

Both public endpoint probes failed DNS resolution:

- Coinbase `api.exchange.coinbase.com`: `NameResolutionError` —
  `Failed to resolve 'api.exchange.coinbase.com' ([Errno -5] No address
  associated with hostname)`.
- Kraken Futures `futures.kraken.com`: `NameResolutionError` —
  `Failed to resolve 'futures.kraken.com' ([Errno -5] No address associated
  with hostname)`.

The required downloaders were each attempted with a 25-second bound and then
with an 18-second unbuffered diagnostic bound; each stopped with timeout exit
124. The Coinbase run logged repeated DNS failures fetching `BTC-USD`. Kraken's
instruments-list request failed with the same DNS error; the downloader warned
that it would use configured symbols as-is, then logged repeated DNS failures
fetching `PI_XBTUSD`. The bounds stopped both runs before their complete symbol
loops, so no claim is made that every symbol received an individual HTTP
request.

The downloaders created no candle CSVs. The local inventory therefore has zero
rows for every requested symbol (not a claim about rows acquired remotely):

| Spot symbol | Rows | Earliest / latest UTC | Largest gap |
|---|---:|---|---:|
| BTC-USD | 0 | N/A | N/A |
| ETH-USD | 0 | N/A | N/A |
| SOL-USD | 0 | N/A | N/A |
| ADA-USD | 0 | N/A | N/A |
| AVAX-USD | 0 | N/A | N/A |
| DOGE-USD | 0 | N/A | N/A |
| LINK-USD | 0 | N/A | N/A |
| XRP-USD | 0 | N/A | N/A |
| LTC-USD | 0 | N/A | N/A |
| UNI-USD | 0 | N/A | N/A |
| XLM-USD | 0 | N/A | N/A |
| BCH-USD | 0 | N/A | N/A |
| DOT-USD | 0 | N/A | N/A |
| MATIC-USD | 0 | N/A | N/A |
| ATOM-USD | 0 | N/A | N/A |
| NEAR-USD | 0 | N/A | N/A |
| AAVE-USD | 0 | N/A | N/A |

| Futures symbol | Rows | Earliest / latest UTC | Largest gap |
|---|---:|---|---:|
| PI_XBTUSD | 0 | N/A | N/A |
| PI_ETHUSD | 0 | N/A | N/A |
| PI_SOLUSD | 0 | N/A | N/A |
| PI_ADAUSD | 0 | N/A | N/A |
| PI_AVAXUSD | 0 | N/A | N/A |
| PI_DOGEUSD | 0 | N/A | N/A |
| PI_MATICUSD | 0 | N/A | N/A |
| PI_LTCUSD | 0 | N/A | N/A |
| PI_LINKUSD | 0 | N/A | N/A |
| PI_SHIBUSD | 0 | N/A | N/A |
| PI_XRPUSD | 0 | N/A | N/A |
| PI_DOTUSD | 0 | N/A | N/A |
| PI_UNIUSD | 0 | N/A | N/A |
| PI_ATOMUSD | 0 | N/A | N/A |
| PI_XLMUSD | 0 | N/A | N/A |

The requested downloader window was approximately 2026-04-17 through
2026-10-14 (180 days). Acquisition coverage on disk is **0%**: 0 of 4,406,400
expected spot minute rows and 0 of 3,888,000 expected futures minute rows.
There is no market series from which to measure actual first/last timestamps or
gaps. Kraken did not produce `_symbol_sources.json`; the failed instrument
lookup means PI→PF or MATIC→POL substitutions cannot be verified. No
substituted contract is treated as evidence about its original contract.

## Training and OOS attempt

The frozen-model trainer was invoked:

```bash
python3 tools/retrain_on_6mo.py \
  --output data/models/market_model_frozen.joblib \
  --meta-output data/models/market_model_frozen_training_meta.json
```

It exited 0 with `No data files in .../data/historical/1min`; no model or
training metadata was produced. Since no actual data window exists, no
data-derived split cutoff can be selected and no `--end-date` was supplied.
Training and the strict OOS evaluation are **blocked**, not skipped in favor of
synthetic data.

The strict-profile backtest preflight was also invoked:

```bash
SIM_REALISM_PROFILE=strict python3 backtest/run_baseline.py \
  --model data/models/market_model_frozen.joblib \
  --output data/state/baseline_oos_2026-10-14.json
```

It exited 1 with `ERROR: could not load model from
data/models/market_model_frozen.joblib`; no backtest JSON was written. Because
there is neither a frozen model nor training-window metadata, the
`--require-out-of-sample` gate could not be reached. No result JSON or model
binary is committed.

## Results

`N/A` means **not measured**, not zero trades or break-even.

| Run / evidence label | Trades | WR | PF | MaxDD | Net P&L |
|---|---:|---:|---:|---:|---:|
| Full-window baseline, would be **in-sample** after full-window training; blocked | N/A | N/A | N/A | N/A | N/A |
| Held-out baseline, **out-of-sample pending**; blocked | N/A | N/A | N/A | N/A | N/A |
| Improved held-out strategy, **not run / none shipped** | N/A | N/A | N/A | N/A | N/A |

Per-symbol returns, Sharpe, exit-reason distribution, futures LONG/SHORT
trades, win rates, profit factors and P&L are unavailable. LONG-versus-SHORT is
**inconclusive; no directional gate is justified**. Monte Carlo and walk-forward
results are also unavailable. No result cell is replaced with a synthetic
market-data estimate.

## Scope and interpretation

The backtest lab uses `MarketPredictor` and `utils.backtester`, not the
production `TradingBot` decision path. Its outcomes, if later measured, would
not by themselves validate production profitability. See
[the prior baseline audit](BACKTEST_BASELINE_2026-10.md) for the lab/live
differences and known simulator limitations.

No files under `cryptotrades/` or live strategy, risk, order-enable, or
credential settings were changed. The only repository change for this attempt
is this report; generated data, models, state, and temporary logs were not
committed.
