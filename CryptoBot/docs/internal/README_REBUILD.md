# Crypto Trading Bot - Restored

## ✅ Status: OPERATIONAL

The bot has been rebuilt from scratch with the rotation system fully integrated.

## What's Working

### Core Features
- ✅ Bot starts and runs continuously
- ✅ Logging to console and `bot_output.log`
- ✅ Signal handlers (Ctrl+C graceful shutdown)
- ✅ 3-minute check interval
- ✅ 9 active trading pairs
- ✅ 15-coin watchlist for rotation

### Rotation System
- ✅ CoinPerformanceTracker integrated
- ✅ Weekly rotation checks (every 168 hours)
- ✅ Inactive coin monitoring (samples 3 coins every 10th cycle)
- ✅ Composite scoring (0-100 points)
- ✅ Automatic recommendations to drop bottom 2 performers

### Performance Tracking
- ✅ Trade recording (`record_trade()`)
- ✅ Volatility tracking (`record_volatility()`)
- ✅ Score calculation with 6 metrics
- ✅ Rotation candidate selection
- ✅ Data persisted to `coin_performance.json`

## How to Start

### Option 1: Batch Script
```cmd
cd D:\042021\CryptoBot
start_bot.bat
```

### Option 2: Direct Python
```cmd
cd D:\042021\CryptoBot
.venv\Scripts\activate
python cryptotrades\main.py
```

## Configuration

### Trading Pairs
Currently active (9 coins):
- BTC-USD, ETH-USD, SOL-USD
- ADA-USD, AVAX-USD, DOGE-USD
- MATIC-USD, LTC-USD, LINK-USD

### Watchlist  
Full rotation pool (15 coins):
- Active 9 + XRP-USD, DOT-USD, UNI-USD, ATOM-USD, XLM-USD, AAVE-USD

### Timings
- **Check interval**: 180 seconds (3 minutes)
- **Rotation check**: 168 hours (7 days)
- **Inactive sampling**: Every 10 cycles (~30 minutes)
- **Sample size**: 3 inactive coins per check

### Rotation Thresholds
- **Bottom performers evaluated**: 2 coins
- **Score improvement required**: 20% (1.20x multiplier)

## Next Steps to Complete

The bot is running but needs actual trading logic added:

1. **Add Coinbase Client Initialization**
   - Load API keys from `.env`
   - Initialize `Client(api_key, api_secret)`

2. **Add Price Fetching**
   - Use Coinbase API to get current prices
   - Calculate volatility from price movements

3. **Add Trading Logic**
   - Implement buy/sell decision rules
   - Execute orders via Coinbase API
   - Call `perf_tracker.record_trade()` after each trade

4. **Add Sentiment Analysis**
   - RSS feed is already fixed (no more hanging)
   - Integrate `get_news_sentiment()` calls
   - Pass sentiment to `calculate_coin_score()`

5. **Add ML Model**
   - Load `trade_model.joblib` if exists
   - Use predictions for trading decisions
   - Pass confidence to scoring

## File Structure

```
D:\042021\CryptoBot\
├── cryptotrades\
│   ├── main.py                      # Entry point
│   ├── core\
│   │   ├── trading_engine.py        # ✅ REBUILT - Main bot logic
│   │   └── __init__.py
│   └── utils\
│       ├── coin_performance.py      # ✅ Rotation tracker
│       └── news_sentiment.py        # ✅ RSS timeout fixed
├── start_bot.bat                    # ✅ NEW - Quick start script
├── bot_output.log                   # Log file (created on first run)
├── coin_performance.json            # Performance data (auto-created)
└── .env                             # Your API keys
```

## Logs

All activity is logged to:
- **Console**: Real-time output
- **File**: `D:\042021\CryptoBot\bot_output.log`

## Rotation System Details

### Scoring Metrics (0-100 points)
1. **Win Rate** (30 pts): % of profitable trades
2. **Profit** (25 pts): Average profit per trade
3. **Volatility** (20 pts): Price movement (higher = more opportunity)
4. **Trade Frequency** (15 pts): Activity level
5. **Sentiment** (±10 pts): News sentiment bonus/penalty
6. **ML Confidence** (10 pts): Model prediction confidence

### How Rotation Works
1. Every 7 days, bot calculates scores for all 15 watchlist coins
2. Identifies bottom 2 performers from active 9
3. Finds best 2 performers from inactive 6
4. If inactive score > active score × 1.20, recommends swap
5. Logs recommendation (manual approval for now)

### Sample Output
```
========================================
ROTATION CHECK TRIGGERED
========================================

🔄 ROTATION RECOMMENDED:
  DROP: ['ETH-USD', 'DOGE-USD']
  ADD:  ['XRP-USD', 'AAVE-USD']

Scores:
  ETH-USD: 45.2/100
  DOGE-USD: 38.7/100
  XRP-USD: 72.4/100
  AAVE-USD: 68.9/100
```

## Testing

Test the import:
```cmd
cd D:\042021\CryptoBot
.venv\Scripts\python.exe -c "from cryptotrades.core.trading_engine import main; print('OK')"
```

Test rotation tracker:
```cmd
.venv\Scripts\python.exe -c "from utils.coin_performance import CoinPerformanceTracker; t=CoinPerformanceTracker(); print(t.calculate_coin_score('BTC-USD', 0, 0.5))"
```

## Known Issues

- ❌ Trading logic is placeholder only (doesn't actually trade yet)
- ❌ Volatility calculation is mocked (needs real price data)
- ❌ Rotation recommendations logged but not auto-executed
- ⚠️  Original `trading_engine.py` was corrupted - this is a fresh rebuild

## Recovery Notes

**Feb 2, 2026**: Original file corrupted by debugging attempts. Rebuilt from scratch using:
- Extracted imports from backup
- Rotation features from documentation  
- Clean minimal architecture
- All rotation modules verified working

The bot now runs successfully with rotation system integrated!
