# Config B Validation: Replication + Out-of-Sample Test

RESEARCH ONLY. Adds `--start-date`/`--end-date`/`--cache-label` to
`backtest_tuned.py` (+ `fetch_1min_window_cached` in `bar_cache.py`) so an
explicit non-overlapping historical window can be backtested without
disturbing the original trailing-days cache. No existing strategy/live code
touched. AlpacaBot remains stopped throughout.

## TEST 1 — Replication (same window: 2026-03-02 to 2026-09-25)
Re-ran the merged `tools/backtest_tuned.py` with no changes, fully cached
data. Result is **bit-for-bit identical** to the original PR #20 run:

| Config | Final | P&L | Trades | WR | PF | MaxDD |
|---|---|---|---|---|---|---|
| A: BASELINE | $51,132 | +$1,132 (+2.3%) | 53 | 47.2% | 1.04 | -19.6% |
| B: SCALP-TUNED | $86,538 | +$36,538 (+73.1%) | 74 | 55.4% | 1.72 | -20.5% |
| C: SWING-TUNED | $50,065 | +$65 (+0.1%) | 23 | 47.8% | 1.00 | -27.4% |

Confirms the backtest is deterministic and the original +73.1% was not a
coding error or random-seed artifact.

## TEST 2 — Out-of-sample (non-overlapping window: 2025-08-04 to 2026-03-02, immediately preceding the original)
Freshly downloaded 1-min data for this period (cached under a distinct
`1min_oos` label, original cache untouched). Same symbols, same rules, same
parameters — no tweaking.

| Config | Final | P&L | Trades | WR | PF | MaxDD |
|---|---|---|---|---|---|---|
| A: BASELINE | $44,481 | -$5,519 (-11.0%) | 42 | 42.9% | 0.74 | -17.7% |
| **B: SCALP-TUNED** | **$39,509** | **-$10,491 (-21.0%)** | 14 | 28.6% | **0.11** | -21.9% |
| C: SWING-TUNED | $59,429 | +$9,429 (+18.9%) | 21 | 57.1% | 1.48 | -20.6% |

## Side-by-side

| Config | In-sample P&L | Out-of-sample P&L | Beats baseline OOS? |
|---|---|---|---|
| A: BASELINE | +2.3% | -11.0% | -- |
| B: SCALP-TUNED | **+73.1%** | **-21.0%** | **No -- underperforms A by $4,972** |
| C: SWING-TUNED | +0.1% | +18.9% | Yes -- beats A by $14,948 |

## Verdict (per the owner's stated decision rule)
**Config B does NOT beat Config A out-of-sample -- it does worse.** Profit
factor collapsed from 1.72 (in-sample) to 0.11 (out-of-sample); trade count
collapsed from 74 to 14 (the trimmed 4-indicator set + 15-min trend filter
barely qualified any signals in the different regime). This is textbook
overfitting to the original March-September 2026 window.

**Conclusion: the +73% was overfitting. Do not deploy Config B. AlpacaBot's
live strategy is not touched.**

Side note (not a recommendation, just an observation worth flagging):
Config C (roughly breakeven in-sample) outperformed on this particular OOS
window (+18.9%, PF 1.48). One favorable out-of-sample period is not
validation either -- would need further non-overlapping windows before
drawing any conclusion about C.
