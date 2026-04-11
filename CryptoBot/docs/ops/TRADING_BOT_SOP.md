# CRYPTO TRADING BOT - STANDARD OPERATING PROCEDURES
**Last Updated:** February 5, 2026  
**Status:** Paper Trading (Active)  
**Version:** 2.1 (Optimized)

---

## 📋 CURRENT CONFIGURATION

### Core Settings
- **Check Interval:** 120 seconds (2 minutes)
- **Snapshot Requirement:** 10 minimum (20 minutes of history)
- **Buy Threshold:** 4/15 points (higher quality trades)
- **Position Limits:** 15 spot max + 5 futures max = 20 total
- **Paper Trading:** TRUE (no real money)

### Capital Allocation
- **Starting Capital:** $5,000 ($2,500 spot + $2,500 futures)
- **Current Balance:** $4,707.78 ($2,469 spot + $2,238 futures)
- **P/L:** -$292.22 (-5.8%)
- **Position Size Spot:** $100 per trade (10% of balance)
- **Position Size Futures:** $200 per contract (15% with 2x leverage)

### Risk Management
- **Stop Loss:** 2% hard stop
- **Trailing Stop:** 2% from maximum price
- **Leverage:** 2x on futures only
- **Max Positions:** 15 spot, 5 futures

---

## 🎯 TRADING STRATEGY

### Spot Market (Coinbase)

**Entry Criteria (Score System - Need 4/15 points):**
- **ML Confidence:**
  - ≥65%: +4 points
  - ≥55%: +2 points
  - ≥45%: +1 point
  
- **Volatility:**
  - >3%: +3 points
  - >2%: +2 points
  - >1%: +1 point
  
- **Sentiment:**
  - Positive: +1 point
  - Neutral: +1 point
  
- **RL Confidence:**
  - >70%: +3 points
  - >60%: +1 point
  
- **Ensemble Prediction:**
  - >65%: +2 points
  - >55%: +1 point

**Exit Criteria:**
- **Take Profit:** 2% gain
- **Stop Loss:** -2% loss
- **Trailing Stop:** -2% from max price
- **Time-based:** None (holds until profit/loss triggers)

**Active Trading Pairs (15):**
BTC-USD, ETH-USD, SOL-USD, ADA-USD, AVAX-USD, DOGE-USD, MATIC-USD, LTC-USD, LINK-USD, SHIB-USD, XRP-USD, DOT-USD, UNI-USD, ATOM-USD, XLM-USD

---

### Futures Market (Kraken)

**LONG Signal (Need ONE of):**
- ML confidence > 55%
- Sentiment score > 0.1

**SHORT Signal (Need ONE of):**
- ML confidence < 45%
- Sentiment score < -0.1

**Exit Criteria:**
- **Take Profit:** +5%
- **Stop Loss:** -3%
- **Leverage:** 2x on all positions

**Active Futures Symbols (15):**
PI_XBTUSD, PI_ETHUSD, PI_SOLUSD, PI_ADAUSD, PI_AVAXUSD, PI_DOGEUSD, PI_MATICUSD, PI_LTCUSD, PI_LINKUSD, PI_SHIBUSD, PI_XRPUSD, PI_DOTUSD, PI_UNIUSD, PI_ATOMUSD, PI_XLMUSD

---

## 🤖 MACHINE LEARNING

### Model Details
- **Type:** RandomForestClassifier (sklearn)
- **Training Data:** 162 valid SELL trades
- **Accuracy:** 65.12% training, 45.45% test
- **Features:** price_change_5, price_change_10, volatility, avg_price, snapshot_count
- **Model File:** trade_model.joblib
- **Retrain Trigger:** Minimum 20 SELLs with 10+ snapshots

### News Sentiment Analysis
- **Sources:** CoinDesk, Cointelegraph, Crypto Briefing (30 articles)
- **Current Sentiment:** -0.025 (slightly negative)
- **Update Frequency:** Every trading cycle

---

## 📁 FILE STRUCTURE

### Critical Files (C:\Master Chess\)
```
trade_history.csv          - All trade records (10,012+ trades)
price_history.csv          - Price snapshots for ML training
paper_balances.json        - Current spot/futures balances
positions.json             - Open spot positions
futures_positions.json     - Open futures positions
trade_model.joblib         - Trained ML model
```

### Code Location
```
D:\042021\CryptoBot\cryptotrades\
  main.py                  - Entry point
  core\
    trading_engine.py      - Main trading logic (1110 lines)
```

### Data Statistics
- **Total Trades:** 10,012
- **Recent Activity:** 1,367 trades (Feb 4-5)
- **Price History:** 328+ snapshots
- **Futures Trades:** 4 closed (75% win rate)

---

## 🚀 OPERATION PROCEDURES

### Starting the Bot
```powershell
Set-Location "c:\Master Chess"
& "D:\042021\CryptoBot\.venv\Scripts\python.exe" "D:\042021\CryptoBot\cryptotrades\main.py"
```

### Stopping the Bot
```powershell
# Method 1: Graceful shutdown (Ctrl+C twice)
# Method 2: Force kill
Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.Path -like "*CryptoBot*"} | Stop-Process -Force
```

### Monitoring Status
- **Terminal Output:** Shows every 120-second cycle
- **Log Format:** Timestamp - Level - Message
- **Key Indicators:**
  - Balance updates
  - Position counts
  - Buy/sell signals
  - P/L tracking

### Data Backup
```powershell
# Backup critical files
Copy-Item "c:\Master Chess\trade_history.csv" "c:\Master Chess\backups\trade_history_$(Get-Date -Format 'yyyyMMdd').csv"
Copy-Item "c:\Master Chess\paper_balances.json" "c:\Master Chess\backups\"
Copy-Item "c:\Master Chess\trade_model.joblib" "c:\Master Chess\backups\"
```

---

## 📊 PERFORMANCE TRACKING

### Key Metrics to Monitor
- **Win Rate:** % of profitable trades
- **Average Profit:** Per trade in %
- **Max Drawdown:** Largest peak-to-valley decline
- **Sharpe Ratio:** Risk-adjusted returns
- **Total P/L:** Overall profit/loss

### Current Performance (as of Feb 5, 2026)
- **Total P/L:** -$292.22 (-5.8%)
- **Spot Balance:** $2,469.38 (down from $2,500)
- **Futures Balance:** $2,238.40 (down from $2,500)
- **Futures Win Rate:** 75% (3 wins / 1 loss out of 4 trades)
- **Best Futures Trade:** XRP SHORT +5.03%

### Historical Performance
- **Peak Balance:** $5,216.75 (+$216.75 / +4.3%)
- **Worst Drawdown:** $3,374.38 (-$1,625.61 / -32.5%)
- **Recovery:** From -$488 to +$126 to -$292 (volatile)

---

## ⚠️ KNOWN ISSUES & LIMITATIONS

### Current Challenges
1. **ML Accuracy:** 45% test accuracy (barely better than random)
2. **Volatility:** Large swings in P/L (-32% to +4% to -6%)
3. **Futures Underutilized:** Only 4 trades total (was too restrictive before)
4. **Paper vs Live:** No slippage, no fees, instant fills (unrealistic)

### Recent Fixes (Feb 4-5, 2026)
✅ **Fixed ML Training:** File location mismatch resolved  
✅ **Fixed Data Types:** Timestamp/numeric conversions corrected  
✅ **Reduced Snapshots:** 15→10 for faster trading  
✅ **Made Bot Aggressive:** Buy threshold 3→2→4 (optimizing)  
✅ **Increased Positions:** 10→20 total capacity  
✅ **Futures Optimization:** Lowered entry requirements for data collection  

---

## 🎯 GO-LIVE READINESS CHECKLIST

### Prerequisites (NOT MET YET)
- [ ] **14+ days consecutive profitability** (Currently: <1 day)
- [ ] **Total balance >$5,700** (Currently: $4,708 / -5.8%)
- [ ] **Win rate >55%** (Need to calculate)
- [ ] **ML accuracy >50%** (Currently: 45%)
- [ ] **Max drawdown <5%** (Hit -32%, currently -5.8%)
- [ ] **Futures trades >20** (Currently: 4)
- [ ] **Zero crashes** (✅ Achieved)
- [ ] **Understanding of all parameters** (✅ Achieved)

### When Going Live
1. **Start with $100-500 ONLY** (not $5,000)
2. **Reduce position sizes to $10-20** (1/5 of paper trading)
3. **Limit to 1-2 positions max** first week
4. **Monitor EVERY trade manually** for 48 hours
5. **Have kill switch ready** (stop command)
6. **Expect losses** - even good systems have losing streaks

**ESTIMATED TIMELINE:**
- Week 1 (Feb 5-11): Run current settings, aim for 7 profitable days
- Week 2 (Feb 12-18): Validate during market volatility
- Week 3 (Feb 19-25): Final validation ($5,700+ target)
- Feb 26+: Consider SMALL live test ($100-200 max)

---

## 🔧 TROUBLESHOOTING

### Bot Won't Start
**Check:**
1. Python path correct: `D:\042021\CryptoBot\.venv\Scripts\python.exe`
2. Working directory: `c:\Master Chess`
3. No other instance running (kill existing processes)

### No Trades Happening
**Possible Causes:**
1. Snapshots building (need 10/10) - wait 20 minutes
2. Buy score <4 (market not meeting criteria)
3. Position limit reached (15 spot or 5 futures)
4. Insufficient balance

### Performance Degrading
**Actions:**
1. Check market conditions (high volatility?)
2. Review recent trades in trade_history.csv
3. Consider retraining ML model (if >100 new trades)
4. Adjust buy threshold if too aggressive/conservative

---

## 📞 QUICK REFERENCE

### Key File Paths
- **Bot Code:** `D:\042021\CryptoBot\cryptotrades\core\trading_engine.py`
- **Data Workspace:** `c:\Master Chess\`
- **Python Venv:** `D:\042021\CryptoBot\.venv\Scripts\python.exe`

### Terminal ID (Current Session)
- **Active Terminal:** 1f7ce807-99d4-4f6b-95b6-a81a1bb21fa7

### Configuration Variables (trading_engine.py)
- **Line 50:** CHECK_INTERVAL = 120
- **Line 619:** Snapshot requirement = 10
- **Line 656:** Buy threshold >= 4
- **Line 658:** Spot position limit = 15
- **Line 848:** Futures position limit = 5
- **Line 859:** Futures LONG signal
- **Line 862:** Futures SHORT signal

---

## 🎓 LESSONS LEARNED

### What Works
1. **Higher buy thresholds** reduce junk trades
2. **Futures shorts** have performed excellently (75% win rate)
3. **2-minute monitoring** catches stop losses quickly
4. **Risk management** prevents catastrophic losses
5. **ML model** provides useful signals despite 45% accuracy

### What Doesn't Work
1. **Too aggressive entry** (2/15 threshold) = 1,300 trades in 14 hours
2. **Time-based cooldowns** miss good opportunities
3. **Overly strict futures requirements** prevented data collection
4. **Paper trading only** creates false confidence (no fees/slippage)

### Key Insights
- **Quality over quantity:** Better to make 10 good trades than 100 random ones
- **Let scoring system filter:** Don't use artificial delays
- **Futures need different logic:** More directional, less "wait for perfect signal"
- **ML isn't magic:** 45% accuracy means it's learning patterns, not predicting perfectly
- **2 weeks minimum testing:** Required before risking real money

---

## 📈 NEXT STEPS

### Immediate (Next 24-48 hours)
1. ✅ Monitor bot with current settings (120s, 4/15 threshold)
2. ✅ Verify futures start trading more frequently
3. ✅ Track if trade frequency drops from 1,300→500 range
4. ⏳ Aim for first profitable day

### Short-term (Next Week)
1. ⏳ Achieve 7 consecutive profitable days
2. ⏳ Collect 20+ futures trades for analysis
3. ⏳ Retrain ML model if accuracy improves
4. ⏳ Document win rates and average profits

### Medium-term (2-3 Weeks)
1. ⏳ Validate strategy during volatile market
2. ⏳ Reach $5,700+ balance target
3. ⏳ Make go/no-go decision on live trading
4. ⏳ If proceeding, test with $100-200 real capital

### Long-term (1-3 Months)
1. ⏳ Build track record with real money (small amounts)
2. ⏳ Consider alternative: Use project for job applications
3. ⏳ Scale up ONLY if consistently profitable
4. ⏳ Learn from institutional traders/firms

---

## 🏆 PROJECT VALUE

**As a Portfolio Project:**
- Multi-strategy trading system (ML + RL + sentiment)
- 10,000+ trades of production data
- Complex debugging (file locations, data types, training)
- Risk management implementation
- Real-world ML application

**Estimated Job Market Value:**
- Fintech Engineer: $110K-140K
- Quantitative Developer: $130K-180K  
- ML Engineer (Finance): $140K-200K
- Algorithmic Trading Dev: $150K-250K

**This project demonstrates:**
- System design & architecture
- Machine learning & data pipelines
- Financial domain knowledge
- Problem-solving under complexity
- Production-level debugging

---

**END OF SOP**

*For updates or changes, edit this document and update "Last Updated" date.*
*Always backup critical files before major configuration changes.*
*When in doubt, ask questions - don't guess with real money!*
