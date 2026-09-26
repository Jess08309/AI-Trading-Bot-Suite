# Indicator Correlation Analysis (Workstream E pre-staging)

Status: **BLOCKED ON DATA ACCESS — script ready, real run pending.**

This is read-only pre-staging for Workstream E: computing all 14
indicators produced by `AlpacaBot/core/indicators.py::compute_all_indicators`
(no changes made to that file) over 6 months of SPY 10-minute bars, and
reporting the pairwise Pearson correlation matrix + a cluster analysis.

## How to generate this report

Use `tools/compute_indicator_correlation.py`, added in this same PR:

```bash
# Option A: live pull from Alpaca (needs ALPACA_API_KEY / ALPACA_API_SECRET
# in the environment and network access to the Alpaca market data API):
python3 tools/compute_indicator_correlation.py --months 6

# Option B: from a local bar cache (e.g. a CSV with a "close" column,
# such as one produced by AlpacaBot/tools/download_5min.py-style tooling
# adapted to a 10-minute timeframe):
python3 tools/compute_indicator_correlation.py --input path/to/spy_10min.csv
```

Either invocation writes this file (`docs/indicator_correlation.md`) with
the full 14x14 correlation matrix and a cluster summary (indicators
grouped by `|Pearson r| >= --corr-threshold`, default `0.8`).

## Why this file does not yet contain the real SPY analysis

This document was authored in a sandboxed environment with **no network
route to Alpaca's market data API** (`data.alpaca.markets` does not
resolve) and **no Alpaca credentials**. Concretely:

```
$ curl -sS -o /dev/null -w "%{http_code}\n" https://data.alpaca.markets/v2/stocks/SPY/bars
curl: (6) Could not resolve host: data.alpaca.markets
```

Rather than fabricate a correlation matrix and present it as if it were
computed from real SPY price data (which would be actively misleading
for a document that feeds into live trading/ML decisions), this PR ships
the ready-to-run script and asks that it be executed once from an
environment with real Alpaca market-data access (e.g. the droplet, or
any machine with `ALPACA_API_KEY`/`ALPACA_API_SECRET` and network
access), after which this file should be regenerated and committed.

## Functional validation performed in this sandbox

To prove the script itself is correct end-to-end (data loading → sliding
`compute_all_indicators` window, matching the same
`chunk = prices[i - lookback : i + 1]` pattern already used by
`AlpacaBot/tools/backtest.py` and friends → Pearson correlation →
connected-components clustering → markdown rendering), it was run
locally against a **synthetic, randomly-generated price series** (NOT
real SPY data, ~2,000 samples) via `--input`. That run completed without
error and produced a well-formed 14x14 matrix and a plausible cluster
grouping (e.g. the mean-reversion/oscillator family — RSI, Bollinger
Band %B, CCI, Stochastic, Williams %R, and z-score — showed high mutual
correlation, which is the expected/known relationship between those
indicator families and is a useful sanity check that the pipeline is
wired correctly). That synthetic output is intentionally **not**
included here to avoid it being mistaken for the real SPY analysis.

## Next step

Run one of the two commands above from an environment with real Alpaca
market data access, then replace this file's content with the generated
report (the script will overwrite this same path).
