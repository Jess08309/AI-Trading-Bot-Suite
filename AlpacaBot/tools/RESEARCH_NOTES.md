# AlpacaBot Strategy-Improvement Research: Comparative Backtest Results

RESEARCH ONLY. This PR adds two new files under `AlpacaBot/tools/` and does
not modify any existing strategy, backtest, or live code. AlpacaBot was
stopped for the duration of this work (owner-authorized) and has not been
restarted -- that decision is left to the owner.

## Files added
- `AlpacaBot/tools/bar_cache.py` -- shared 1-minute bar downloader/cache
  (mirrors `download_5min.py`'s chunked-download pattern; all coarser
  timeframes are derived locally by resampling this one dataset).
- `AlpacaBot/tools/backtest_tuned.py` -- the 3-config comparative backtest.

## STEP 1: Indicator redundancy (already computed in a prior session, verified here)
See `docs/indicator_correlation.md` for the full matrix (5,476 samples,
trailing 6mo SPY 10-min bars). STOP condition does **not** trigger -- RSI /
MACD histogram / BB position are not mutually redundant (max pairwise
|r| = 0.78, under the 0.8 threshold). One real 5-way cluster was found
instead: `bb_position / zscore / williams_r / cci / stochastic`. Trimmed
4-indicator set used by Configs B/C, chosen from the data: **rsi, macd_hist,
bb_position, volatility_ratio** (volatility_ratio is the most orthogonal 4th
indicator available -- max |r| = 0.03 against the other three).

## STEP 2/3: Comparative backtest (SPY, QQQ, AAPL, MSFT, NVDA; 145 trading days, $50k)

| Config | Final | P&L | P&L% | Trades | Win Rate | Profit Factor | Avg Hold | Max DD |
|---|---|---|---|---|---|---|---|---|
| A: BASELINE (10-min, 14-ind, +ML gate) | $51,132 | +$1,132 | +2.3% | 53 | 47.2% | 1.04 | 0.80d | -19.6% |
| **B: SCALP-TUNED (2-min, 4-ind, 15m trend filter)** | **$86,538** | **+$36,538** | **+73.1%** | 74 | 55.4% | 1.72 | 0.80d | -20.5% |
| C: SWING-TUNED (1-hour, 4-ind, daily trend filter) | $50,065 | +$65 | +0.1% | 23 | 47.8% | 1.00 | 1.85d | -27.4% |

**Winner: Config B (SCALP-TUNED)**, +$35,406 vs baseline. Config C roughly
breaks even (PF 1.00) with the worst drawdown of the three.

Config A's ML gate checked 354 rule-based signals, blocked 283 for low
confidence and 18 for direction disagreement -- only 53 (15%) passed both
gates and became trades.

## Scope decisions / caveats (see backtest_tuned.py's module docstring for full detail)
- Config A reproduces the rule-based indicator/threshold/exit/sizing layer
  from current `core/config.py`, PLUS the real ML confidence/agreement gate
  (actual saved `options_model.joblib`, replayed exactly as
  `trading_engine.py` calls it). Sentiment / SPY-regime-filter /
  meta-learner ensemble / put-win-rate auto-disable are **not** modeled --
  no existing backtest tool in this repo models them either.
- Universe (SPY/QQQ/AAPL/MSFT/NVDA) is fixed by the task spec; note
  `core/scanner.py`'s `SCANNER_UNIVERSE` comments document SPY/QQQ/MSFT as
  historically-eliminated (unprofitable) symbols excluded from the live
  scanner's real universe.
- Single ~7-month historical window (2026-03 to 2026-09), not walk-forward
  or out-of-sample validated -- meaningful overfitting risk, especially for
  Config B's outperformance. Treat as a first-pass comparative signal, not
  a production-ready result.
- Strike selection is a single-strike ITM-target simplification (no real
  options chain data available in a closes-only backtest) -- same class of
  simplification already used by `backtest_mtf.py`.
