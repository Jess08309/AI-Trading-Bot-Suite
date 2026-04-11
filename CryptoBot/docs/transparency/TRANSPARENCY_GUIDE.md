# TRANSPARENCY FEATURES - Complete Guide

## Overview

The bot now has **full transparency** into every decision it makes. You can see:
- **Why trades were opened** (signal strength, confidence, indicators)
- **Why trades were closed** (stop loss, take profit, ML signal, etc.)
- **Per-symbol performance** (win rate, avg profit/loss, exit patterns)
- **Daily summaries** (best/worst trades, total P/L, exit reason breakdown)

## New Features

### 1. Exit Reason Tracking ✅
Every trade is now logged with **why it was closed**:
- **TAKE_PROFIT** - Hit profit target
- **STOP_LOSS** - Hit stop loss
- **TRAILING_STOP** - Trailing stop triggered
- **ML_SIGNAL** - ML model signaled reversal (RSI overbought, ensemble bearish, signal flip)

### 2. Signal Strength Logging ✅
Entry details include:
- **ML Confidence** - How confident the model was (0-100%)
- **Volatility** - Market volatility at entry
- **RSI** - Relative Strength Index
- **Regime** - Trending vs ranging market
- **Correlation** - Position correlation
- **Sentiment** - News sentiment
- **Quality Score** - 0-6 points for scaled sizing
- **Position Multiplier** - 1x, 2x, or 3x based on setup quality

### 3. Per-Symbol Performance ✅
Track how each coin performs:
- Total trades, wins, losses
- Win rate percentage
- Average win/loss
- Best/worst trades
- Average hold time
- Exit reason breakdown per symbol

### 4. Daily Summaries ✅
End-of-day performance snapshots:
- Total trades and P/L
- Win rate
- Best and worst trades
- Exit reason distribution
- Average win/loss

## How to Use

### Quick Status Check
```powershell
# View current positions with live P/L (updates every 5 seconds)
.\STATUS_MONITOR.ps1
```

### View Recent Exits
```powershell
# See last 15 closed trades with exit reasons and signal details
.\VIEW_EXIT_REASONS.ps1
```

### Full Transparency Dashboard
```powershell
# Interactive menu with all transparency data
.\TRANSPARENCY_DASHBOARD.ps1
```

**Dashboard Options:**
1. **Daily Summary** - Today's performance overview
2. **Per-Symbol Stats** - Performance by coin
3. **Recent Trades** - Last 10 closed trades with signal details
4. **Open Positions** - Current positions with entry signals
5. **All (Full Report)** - Everything at once

### Data Files

All transparency data is stored in `C:\Master Chess\data\state\`:

- **trade_transparency.json** - Complete trade history with entry/exit details
- **symbol_performance.json** - Per-symbol statistics
- **daily_summary.json** - Today's performance summary

## Example Output

### Exit Reason Breakdown
```
TAKE_PROFIT
  Trades: 12 | Wins: 12 (100.0% win rate)
  Total P/L: $245.32

STOP_LOSS
  Trades: 8 | Wins: 0 (0.0% win rate)
  Total P/L: -$156.78

ML_SIGNAL
  Trades: 5 | Wins: 3 (60.0% win rate)
  Total P/L: $23.45
```

### Trade Details
```
PI_ETHUSD SHORT | P/L: $45.23 (+2.34%)
Exit Reason: TAKE_PROFIT
Entry: $1,961.90 @ 14:23
 Exit: $1,916.15 @ 15:47
  Hold: 84 minutes

Entry Signal:
  Confidence: 32.1% | Volatility: 0.012 | RSI: 68
  Regime: trending | Correlation: 0.15
  ► SCALED POSITION: 2x size (Quality Score: 4/6)
```

### Symbol Performance
```
PI_ETHUSD
  Trades: 15 (W:9 L:6) | Win Rate: 60.0%
  Total P/L: $123.45 (4.23%)
  Avg Win: 3.21% | Avg Loss: -1.87%
  Best: 5.67% | Worst: -3.21%
  Avg Hold: 92 minutes
  Exit Reasons: TAKE_PROFIT(7), STOP_LOSS(4), ML_SIGNAL(4)
```

## Understanding Signal Quality

The bot scores each entry opportunity on **6 factors** (0-6 points):

| Factor | Points | Conditions |
|--------|--------|------------|
| High Confidence | +2 | ML confidence > 75% |
| Low Volatility | +2 | Volatility < 1.5% |
| Trending Regime | +1 | Clear trend detected |
| Low Correlation | +1 | Correlation < 0.3 |
| RSI Zone | +0 to +3 | Spot only (oversold bonus) |
| Sentiment | +0 to +2 | Spot only (positive news) |

**Position Sizing by Quality:**
- **0-2 points** = 1x size (standard)
- **3-4 points** = 2x size (strong setup)
- **5-6 points** = 3x size (perfect setup)

## Transparency Goals

This system ensures you always know:
1. **Why it entered** - Full signal breakdown with confidence, indicators, regime
2. **Why it exited** - Exact reason (profit target, stop loss, reversal signal)
3. **How it's performing** - Win rate, P/L, and patterns by symbol
4. **What's working** - Which exit strategies and signal patterns are profitable

## Configuration

Transparency tracking is **always on** - it doesn't affect strategy or learning.

To enable/disable scaled sizing (1-3x multipliers):
```python
# In config.py
ENABLE_SCALED_SIZING = True  # Set to False to always use 1x sizing
```

## Next Steps

As the bot trades, you'll accumulate transparency data showing:
- Which coins perform best
- Which exit reasons are most profitable
- How signal quality correlates with success
- Optimal hold times per symbol

Use this data to understand the bot's behavior and validate that it's trading intelligently!

---

**Remember:** Transparency doesn't change how the bot trades - it just lets you see inside every decision. The more trades that accumulate, the more insights you'll gain into what's working and what's not.
