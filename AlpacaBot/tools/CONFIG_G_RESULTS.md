# Config G (Gamma-Scalp, 1-7 DTE) Validation

RESEARCH ONLY. AlpacaBot remained stopped throughout; not restarted.

## Design notes / deviations from the literal spec (both approved by owner mid-task)
1. **Strike selection switched from the shared 5%-ITM convention to ATM (0% offset).**
   As originally implemented (reusing A/B/C/D's 5%-ITM `select_strike`), 2%
   position sizing on a $50k account ($1,000 max spend) could not afford even
   1 contract for SPY/QQQ/AAPL/MSFT -- a 5%-ITM strike bakes in 5% of the
   underlying's price as pure intrinsic value alone (e.g. SPY ~$650-770 =>
   $32.50-38.50/share minimum, i.e. $3,250-3,850/contract, before any time
   value). Verified via single-symbol smoke tests: SPY/AAPL produced **zero**
   trades, NVDA (cheapest symbol) alone produced 77. Switching to ATM (a) is
   the more correct convention for gamma scalping anyway (max gamma sits ATM,
   not deep ITM) and (b) fixed affordability across the whole universe.
2. Everything else implemented exactly as specified: 5-min bars, same 4
   indicators (rsi/macd_hist/bb_position/volatility_ratio), 1-hour trend
   filter (calls only in uptrend, puts only in downtrend, flat = no
   constraint), DTE fixed at 7 (top of the stated 1-7 range, leaving room
   before the 3-DTE hard exit), stop-loss -40%, take-profit +75%, theta stop
   (exits at cumulative P&L% <= -4% x days_held), hard exit at 3 DTE
   remaining, no trailing-stop/no MAX_HOLD (not part of the stated rules),
   IV-percentile < 40 entry filter (reuses `core/indicators.py::iv_percentile`
   unmodified), 2% position sizing.

## Results (same 4 walk-forward windows as Config C)

| Window | Period | Config C (1-hour) | Config G (5-min) |
|---|---|---|---|
| A | 2024-03 to 2024-09 | +18.7% (PF 1.22, WR 60.8%, 51 trades) | **-8.5%** (PF 0.84, WR 4.2%, 690 trades) |
| B | 2024-09 to 2025-03 | -12.1% (PF 0.54, WR 30.0%, 10 trades) | **+22.8%** (PF 1.22, WR 5.7%, 995 trades) |
| C | 2025-03 to 2025-08 | +8.6% (PF 1.13, WR 45.2%, 31 trades) | **+11.5%** (PF 1.13, WR 5.7%, 1019 trades) |
| D | 2025-08 to 2026-03 | +2.5% (PF 1.05, WR 44.8%, 29 trades) | **+23.3%** (PF 1.21, WR 5.9%, 1162 trades) |
| **Profitable windows** | | **3/4** | **3/4** |

Exit reasons for Config G are dominated by THETA_STOP (94-96% of all exits in
every window) -- consistent with the intended payoff shape: frequent small
controlled losses (avg loss $20-55), rare large wins from TAKE_PROFIT
(avg win $700-1,100), net positive in 3 of 4 windows.

## Verdict: the faster timeframe PRESERVES the edge -- noise does not kill it
Config G is profitable in the same 3/4 windows as Config C, and by raw
magnitude its winning windows are *larger* (+22.8%/+11.5%/+23.3% vs C's
+18.7%/+8.6%/+2.5%). The 5-min timeframe did not introduce a noise-driven
failure mode.

## Caveat that must be resolved before calling this "proven" for real deployment
**Trade frequency is extreme**: 690-1,162 trades per 6-month window (roughly
5-10 trades/day across 5 symbols), average hold 0.08-0.17 days (2-4 hours).
This backtest prices options via a pure theoretical Black-Scholes mid-price
with **no bid-ask spread or slippage modeled** -- every other config in this
research effort trades at most ~30-90 times per window, where a few cents of
slippage per trade is a rounding error; at 700-1,200 trades, even $0.05-0.10
of unmodeled slippage per contract could materially erode or reverse the
theoretical edge shown here. **This has NOT been tested and should be treated
as an open risk, not a settled conclusion**, before proceeding further.

## Recommendation
Per the task's own framing ("report if the faster timeframe preserves the
edge or if noise kills it"): **the edge is preserved, not killed by noise** --
Config G is not killed. However, given the trade-frequency/slippage caveat
above, I'd call this "provisionally proven, pending a slippage-sensitivity
check" rather than unconditionally proven. IronCondor validation is a
decision for the owner; not proceeding to it myself. Not deploying anything;
AlpacaBot remains stopped.
