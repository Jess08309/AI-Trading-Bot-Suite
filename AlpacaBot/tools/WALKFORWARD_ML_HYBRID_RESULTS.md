# Walk-Forward Validation of Config C + ML Hybrid (Config D)

RESEARCH ONLY. AlpacaBot remained stopped throughout. No existing strategy or
live code modified.

## CRITICAL BUG FOUND AND FIXED MID-TASK: resampling was positional, not calendar-time

`resample_intraday()` (used by Configs A/B/C/D to build 2-min/10-min/15-min/
1-hour bars from 1-min data) took every Nth *row*, not every Nth real time
interval. Two nominally-similar fetches of almost the same period started at
different exact minutes and had irregular 1-min gaps, so "bar #17" could land
on a different wall-clock hour in each fetch. Confirmed via a stark result
discrepancy between two overlapping-but-not-identical windows. **Fixed**:
`resample_intraday()` now uses real pandas `.resample()` on a DatetimeIndex
(calendar-time bars, left-labeled/closed to match Alpaca's own bar
convention). Per owner instruction, **everything was re-run from scratch**
with the fix (no re-download needed -- the bug was in the in-memory
resampling step, not the cached 1-min data itself).

### Original 200-day comparison (2026-03-02 to 2026-09-25): numbers changed, same overall story
| Config | Before fix | After fix |
|---|---|---|
| A: BASELINE | +2.3% | **+10.2%** |
| B: SCALP-TUNED | +73.1% | **+80.5%** |
| C: SWING-TUNED | +0.1% | **+51.4%** |

### TEST 2 out-of-sample (2025-08-04 to 2026-03-02): conclusion UNCHANGED -- Config B still overfits
| Config | Before fix | After fix |
|---|---|---|
| A: BASELINE | -11.0% | **-15.3%** |
| B: SCALP-TUNED | -21.0% | **-17.3%** (in-sample was +80.5% -- still a collapse; still underperforms A) |
| C: SWING-TUNED | +18.9% | **+10.9%** |

**PR #21's conclusion holds**: Config B is still overfit, still not recommended for deployment.

## PHASE 1 -- Walk-forward validation of Config C (rules), 4 non-overlapping windows

| Window | Period | P&L (buggy) | P&L (fixed) | WR | PF | Trades |
|---|---|---|---|---|---|---|
| A | 2024-03-01 to 2024-09-01 | -12.3% | **+18.7%** | 60.8% | 1.22 | 51 |
| B | 2024-09-01 to 2025-03-01 | -15.7% | **-12.1%** | 30.0% | 0.54 | 10 |
| C | 2025-03-01 to 2025-08-01 | +10.9% | **+8.6%** | 45.2% | 1.13 | 31 |
| D | 2025-08-01 to 2026-03-01 | -6.6% | **+2.5%** | 44.8% | 1.05 | 29 |

**The bug fix reverses the Phase 1 gate decision.** Buggy data showed only
1/4 profitable windows (fail). Fixed data shows **3/4 profitable windows
(A, C, D)** -- passes the owner's "3+ profitable windows" gate. Proceeded to
Phase 2 per the decision rule.

## PHASE 2 -- ML Hybrid (Config D): GradientBoostingClassifier vs pure rules

Config D = Config C's exact timeframe/trend-filter/DTE/holds, but signal
direction/confidence come from a GradientBoostingClassifier (mirrors
CryptoBot's `market_predictor.py::MarketPredictor` pattern) trained on the
same 4 indicators to predict 3-bars-ahead direction, confidence threshold
0.55. Model trained ONCE on a pretrain window strictly before all 4 test
windows (2022-09-01 to 2024-03-01, 18 months) and frozen -- same no-look-
ahead methodology as this repo's CryptoBot pipeline v4 frozen gate model.

| Window | Config C (rules) | Config D (ML hybrid) |
|---|---|---|
| A | +18.7% (PF 1.22, WR 60.8%, 51 trades) | -5.9% (PF 0.96, WR 42.3%, 78 trades) |
| B | -12.1% (PF 0.54, WR 30.0%, 10 trades) | **+355.3%** (PF 1.71, WR 50.7%, 152 trades) |
| C | +8.6% (PF 1.13, WR 45.2%, 31 trades) | -23.5% (PF 0.45, WR 34.6%, 26 trades) |
| D | +2.5% (PF 1.05, WR 44.8%, 29 trades) | +10.1% (PF 1.19, WR 56.7%, 30 trades) |
| **Profitable windows** | **3/4** | **2/4** |

### Window B's +355% needs a caveat, not a headline
The frozen classifier's own TimeSeriesSplit CV accuracy on its pretrain data
was **52.2%** -- barely above a coin flip. The arithmetic behind the +355%
checks out (WR 50.7% x avg win $5,541 vs avg loss $3,320, compounded across
152 trades at 15%-of-equity sizing each -- percentage-of-equity sizing
genuinely can produce this kind of blowup once a winning stretch begins), but
a near-random classifier producing a 7x account in one window and losing
money in the other three is the signature of **variance/luck in trade
sequencing, not a robust edge**. It should not be trusted to generalize.

## Verdict: pure rules (Config C) beat the ML hybrid (Config D)
- Config C is profitable in 3/4 windows, consistently (WR 45-61%, PF
  1.05-1.22, no extreme outliers).
- Config D is profitable in only 2/4 windows, and is **worse than Config C**
  in 3 of the 4 individual windows (A, C, and arguably B on a risk-adjusted
  basis given the 52.2% CV accuracy) -- its only "win" is a single
  high-variance outlier built on a barely-better-than-random classifier.
- **Answer to the Phase 2 question: pure rules are better. The ML hybrid
  does not improve Config C** -- it trades the rule system's modest,
  consistent edge for high variance and a fragile, low-confidence classifier.

**Recommendation: do not deploy either Config C or Config D to the live bot
without further validation** (Config C passed this walk-forward test but
still needs, at minimum, a review of the SPY/QQQ/MSFT-historically-
unprofitable caveat noted in earlier PRs, and ideally a 5th+ non-overlapping
window before any live consideration). Config D should be set aside --the
52.2% CV accuracy indicates the underlying 4-indicator feature set does not
carry a strong 3-bar-ahead directional signal for a GBM classifier to
exploit reliably.

AlpacaBot was not restarted. All decisions are the owner's to make.
