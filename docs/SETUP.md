# Setup Guide - Crypto Trading Bot

> **Step-by-step installation and configuration instructions**

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Bot](#running-the-bot)
5. [Monitoring & Management](#monitoring--management)
6. [Troubleshooting](#troubleshooting)
7. [Safety & Best Practices](#safety--best-practices)

---

## Prerequisites

### System Requirements

**Operating System:**
- Windows 10 or later (64-bit)
- Linux/Mac support possible but untested

**Hardware:**
- CPU: Any modern processor (bot uses <5% CPU)
- RAM: 500 MB minimum, 1 GB recommended
- Disk Space: 500 MB for code + data
- Internet: Stable connection required (bot makes API calls every 60s)

**Software:**
- Python 3.11 or higher ([Download here](https://www.python.org/downloads/))
- PowerShell (included with Windows)
- Text editor (VS Code recommended, Notepad works)
- Git (optional, for version control)

### API Access (Optional for Paper Trading)

**Coinbase Advanced Trade:**
- Account: [coinbase.com/advanced-trade](https://coinbase.com/advanced-trade)
- API Keys: Not required for paper trading
- Live Trading: Generate API keys in Account Settings → API

**Kraken Futures:**
- Account: [futures.kraken.com](https://futures.kraken.com)
- API Keys: Not required for paper trading
- Live Trading: Generate API keys in Futures Settings

**Note:** Bot works in paper trading mode WITHOUT API keys. All trades are simulated.

---

## Installation

### Step 1: Install Python

1. Download Python 3.11+ from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ **IMPORTANT:** Check "Add Python to PATH"
4. Click "Install Now"
5. Verify installation:
   ```powershell
   python --version
   # Should show: Python 3.11.x or higher
   ```

### Step 2: Get the Bot Code

**Option A: Download ZIP (Easiest)**
1. Download project ZIP file
2. Extract to `D:\042021\CryptoBot\`
3. You should see: `cryptotrades/`, `data/`, `README.md`, etc.

**Option B: Git Clone (Recommended)**
```powershell
cd D:\042021\
git clone <repository-url> CryptoBot
cd CryptoBot
```

### Step 3: Create Virtual Environment

**Why?** Isolates bot dependencies from system Python

```powershell
# Navigate to project folder
cd D:\042021\CryptoBot

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1

# You should see (.venv) in your prompt
```

**Troubleshooting Activation:**
If you get "execution policy" error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Install Dependencies

```powershell
# Make sure virtual environment is activated (see (.venv) in prompt)
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected packages:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `feedparser` - RSS feed parsing
- `requests` - API calls
- `joblib` - Model serialization

**Verification:**
```powershell
pip list
# Should show all packages installed
```

### Step 5: Verify File Structure

Your directory should look like this:
```
D:\042021\CryptoBot\
├── cryptotrades/
│   ├── main.py
│   ├── core/
│   │   └── trading_engine.py
│   ├── utils/
│   │   └── news_sentiment.py
│   └── ...
├── data/
│   ├── models/
│   │   └── trade_model.joblib
│   ├── history/
│   │   ├── trade_history.csv
│   │   └── price_history.csv
│   └── state/
│       ├── positions.json
│       ├── futures_positions.json
│       └── paper_balances.json
├── .venv/
├── requirements.txt
├── README.md
└── ...
```

If `data/` folders don't exist, create them:
```powershell
mkdir data\models
mkdir data\history
mkdir data\state
```

---

## Configuration

### Default Settings (No Changes Needed)

The bot comes pre-configured with safe defaults:

**trading_engine.py (Line 42-60):**
```python
# Trading parameters
PAPER_TRADING = True              # Safe mode (no real money)
CHECK_INTERVAL = 60               # Price checks every 60 seconds
BUY_THRESHOLD = 4.0               # Min score to buy (4 out of 15)
SPOT_POSITION_LIMIT = 15          # Max spot positions
FUTURES_POSITION_LIMIT = 10       # Max futures positions
STOP_LOSS_PERCENT = 2.0           # Spot: exit at -2%
FUTURES_STOP_LOSS = 3.0           # Futures: exit at -3%
TAKE_PROFIT_PERCENT = 2.0         # Spot: exit at +2%
FUTURES_TAKE_PROFIT = 5.0         # Futures: exit at +5%
STARTING_BALANCE = 5000           # Paper trading capital
TRADE_AMOUNT_USD = 100            # Spot: $100 per trade
FUTURES_TRADE_AMOUNT_USD = 200    # Futures: $200 per contract
```

**These settings are SAFE and well-tested.** Only change if you understand the implications.

---

### Optional: Customize Trading Parameters

**To Modify Settings:**

1. Open `D:\042021\CryptoBot\cryptotrades\core\trading_engine.py`
2. Find the section around line 42-60
3. Change values carefully
4. Save file

**Common Customizations:**

**More Conservative (Safer):**
```python
BUY_THRESHOLD = 6.0               # Fewer, higher-quality trades
SPOT_POSITION_LIMIT = 10          # Smaller portfolio
STOP_LOSS_PERCENT = 1.5           # Tighter stop-loss
TRADE_AMOUNT_USD = 50             # Smaller position sizes
```

**More Aggressive (Riskier):**
```python
BUY_THRESHOLD = 3.0               # More trades
SPOT_POSITION_LIMIT = 20          # Larger portfolio
STOP_LOSS_PERCENT = 3.0           # Wider stop-loss
TRADE_AMOUNT_USD = 150            # Bigger position sizes
```

**⚠️ WARNING:** Changing these can significantly impact performance. Test in paper trading first.

---

### Optional: Add API Keys (Live Trading)

**DON'T DO THIS unless you've tested in paper mode for 30+ days.**

1. Create `.env` file in `D:\042021\CryptoBot\`:
   ```plaintext
   COINBASE_API_KEY=your_key_here
   COINBASE_API_SECRET=your_secret_here
   KRAKEN_API_KEY=your_key_here
   KRAKEN_API_SECRET=your_secret_here
   ```

2. In `trading_engine.py`, change:
   ```python
   PAPER_TRADING = False  # DANGER: Real money mode
   ```

3. **Start with small amounts** (e.g., $100 total, not $5,000)

---

## Running the Bot

### Method 1: One-Click Startup (Recommended)

1. Navigate to workspace folder:
   ```powershell
   cd "c:\Master Chess"
   ```

2. Double-click `START_BOT.bat`

   OR run in PowerShell:
   ```powershell
   .\START_BOT.bat
   ```

3. You should see:
   ```
   ═══════════════════════════════════════════
   🤖 CRYPTO TRADING BOT STARTING...
   ═══════════════════════════════════════════
   
   Mode: PAPER TRADING (Safe Mode)
   Check Interval: 60 seconds
   Buy Threshold: 4.0 / 15
   Position Limits: 15 spot, 10 futures
   Starting Balance: $5000.00
   
   [CYCLE 1] Checking prices...
   ```

### Method 2: Manual Startup

```powershell
# Activate virtual environment
cd D:\042021\CryptoBot
.venv\Scripts\Activate.ps1

# Run bot
python cryptotrades\main.py
```

### What to Expect

**First 10 Minutes:**
```
[CYCLE 1-10] Building price snapshots...
- Fetching prices for 30 symbols (15 spot + 15 futures)
- Need 10 snapshots minimum before trading
- No trades will execute yet
```

**After 10 Minutes:**
```
[CYCLE 11] Ready to trade!

BTC-USD: 
  Price: $45,234.50
  ML Prediction: 0.62 (62% buy confidence)
  RL Action: BUY
  Sentiment: +0.4 (bullish news)
  Volatility: 2.3% (low)
  Buy Score: 7.2 / 15 ✅
  
>> BUY SIGNAL: BTC-USD at $45,234.50 (0.00221 BTC = $100)
>> Position opened. Stop-loss: $44,329 | Take-profit: $46,139

Active Positions: 1/15 spot, 0/10 futures
Balance: $4,900.00 (-$100 in positions)
```

**Ongoing Operation:**
```
[CYCLE 25] 
  - 3 spot positions open (BTC, ETH, SOL)
  - 1 futures position (XRP SHORT)
  - BTC hit +2% → TAKE_PROFIT → Sold for +$2.00 profit
  - ETH at +1.3% → Holding
  - SOL at -0.8% → Holding
  - XRP SHORT at +3.2% → Holding (target: +5%)
```

---

## Monitoring & Management

### Check Bot Status

**While Running:**
The terminal shows live updates every 60 seconds.

**View Active Positions:**
```powershell
# Spot positions
Get-Content "D:\042021\CryptoBot\data\state\positions.json" | ConvertFrom-Json | Format-List

# Futures positions
Get-Content "D:\042021\CryptoBot\data\state\futures_positions.json" | ConvertFrom-Json | Format-List
```

**Check Current Balance:**
```powershell
Get-Content "D:\042021\CryptoBot\data\state\paper_balances.json" | ConvertFrom-Json
```

**View Trade History:**
```powershell
# Last 20 trades
Get-Content "D:\042021\CryptoBot\data\history\trade_history.csv" | Select-Object -Last 20
```

### Performance Analysis

**Run Analysis Script:**
```powershell
cd "c:\Master Chess"
python analyze_performance.py
```

**Output:**
```
=== PERFORMANCE ANALYSIS ===
Total Trades: 1,231
Closed Trades: 1,231

Win Rate: 58.2% (717 wins, 488 losses)
Average P/L: -0.087% per trade
Total P/L: -$1,116.30 (-22.3%)

Best Trade: +19.8% (PI_XRPUSD FUTURES SHORT)
Worst Trade: -15.3% (PI_XRPUSD FUTURES LONG)

By Market Type:
  Spot:    1,096 trades | 59.4% win rate | -0.092% avg
  Futures:   135 trades | 48.9% win rate | -0.046% avg
```

### Stopping the Bot

**Graceful Shutdown:**
1. Press `Ctrl+C` in the terminal
2. Bot will finish current cycle
3. Save all state to disk
4. Exit cleanly

**Force Stop (Emergency):**
1. Close the terminal window
2. OR run: `Stop-Process -Name python -Force`

**⚠️ WARNING:** Force stop may lose unsaved data. Use graceful shutdown when possible.

---

## Troubleshooting

### Problem: Bot Won't Start

**Error: "Python not found"**
```
Solution:
1. Reinstall Python with "Add to PATH" checked
2. OR manually add Python to PATH
3. Restart PowerShell
```

**Error: "No module named 'pandas'"**
```
Solution:
1. Activate virtual environment: .venv\Scripts\Activate.ps1
2. Install dependencies: pip install -r requirements.txt
```

**Error: "Permission denied"**
```
Solution:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Problem: No Trades Happening

**Symptom:** Bot runs but never buys anything

**Check 1: Waiting for Snapshots?**
```
Need 10 price snapshots (10 minutes) before trading starts.
Wait 10 minutes and check again.
```

**Check 2: Buy Scores Too Low?**
```
Look for "Buy Score: X / 15" in logs.
If all scores are < 4.0, no trades will execute.
This is normal during low-volatility periods.
```

**Check 3: Position Limits Reached?**
```
If you see "Max positions reached (15/15)", 
bot won't buy until something sells.
Wait for take-profit or stop-loss to trigger.
```

**Check 4: API Issues?**
```
If you see "Failed to fetch prices" errors,
check internet connection.
Try restarting bot.
```

---

### Problem: Bot Losing Money

**Expected Behavior:**
- Short-term losses are normal (learning phase)
- Win rate around 50-60% is good
- Small losses per trade (-0.1% to -0.5%) are acceptable during testing

**Red Flags:**
- ❌ Win rate < 40%
- ❌ Average loss per trade > -1%
- ❌ Worst trade < -5%
- ❌ Balance drops > 20% in one day

**Actions:**
1. **STOP THE BOT** (Ctrl+C)
2. Run performance analysis: `python analyze_performance.py`
3. Check if stop-losses are triggering correctly
4. Consider increasing BUY_THRESHOLD (4.0 → 5.0 or 6.0)
5. Review [PERFORMANCE.md](PERFORMANCE.md) for detailed analysis

---

### Problem: High CPU/Memory Usage

**Normal Usage:**
- CPU: 2-5%
- RAM: 200-500 MB
- Disk: Minimal (logs written every trade)

**If Higher:**
```
1. Check for multiple bot instances:
   Get-Process python

2. Kill duplicates:
   Stop-Process -Name python

3. Restart bot fresh
```

---

### Problem: Missing Trade History

**Symptom:** `trade_history.csv` is empty or missing

**Solutions:**
```powershell
# Check if file exists
Test-Path "D:\042021\CryptoBot\data\history\trade_history.csv"

# If missing, create empty file
New-Item "D:\042021\CryptoBot\data\history\trade_history.csv" -ItemType File

# Bot will recreate on next trade
```

---

## Safety & Best Practices

### DO's ✅

1. ✅ **Start with paper trading**
   - Run for 30+ days minimum
   - 1,000+ trades to validate strategy
   - Understand why trades happen

2. ✅ **Monitor regularly**
   - Check performance daily
   - Review trade history weekly
   - Analyze win rate and P/L metrics

3. ✅ **Keep backups**
   ```powershell
   # Backup state files
   Copy-Item "data\state\*" "backups\state_$(Get-Date -Format 'yyyyMMdd')\"
   
   # Backup trade history
   Copy-Item "data\history\*" "backups\history_$(Get-Date -Format 'yyyyMMdd')\"
   ```

4. ✅ **Document changes**
   - Keep notes on parameter changes
   - Track what works and what doesn't
   - Git commit messages for code changes

5. ✅ **Test modifications**
   - Change one parameter at a time
   - Run for 100+ trades before evaluating
   - Compare before/after performance

---

### DON'Ts ❌

1. ❌ **Never skip paper trading**
   - Don't go live without extensive testing
   - Don't assume it will work with real money
   - Don't trust backtest results alone

2. ❌ **Don't over-optimize**
   - Resist urge to change parameters daily
   - Don't chase "perfect" settings
   - Overfitting to past data won't predict future

3. ❌ **Don't risk more than you can afford to lose**
   - If using real money, start with $100-$500
   - Never invest rent/bill money
   - Crypto is high risk, bot adds algorithmic risk

4. ❌ **Don't leave it unattended for weeks**
   - Check performance at least weekly
   - Markets change, bots need adjustments
   - Technical issues can accumulate

5. ❌ **Don't trust blindly**
   - Understand why bot makes decisions
   - Question unexpected behavior
   - Kill switch if something seems wrong

---

### Emergency Procedures

**If Things Go Very Wrong:**

1. **STOP THE BOT IMMEDIATELY**
   ```powershell
   # Ctrl+C in terminal
   # OR force kill
   Stop-Process -Name python -Force
   ```

2. **Assess Damage**
   ```powershell
   # Check balance
   Get-Content "data\state\paper_balances.json"
   
   # Review recent trades
   Get-Content "data\history\trade_history.csv" | Select-Object -Last 50
   ```

3. **Identify Root Cause**
   - Stop-loss not triggering? (60s interval issue)
   - Too many trades? (lower threshold)
   - Market crash? (normal, wait it out)
   - Bug in code? (check error logs)

4. **Fix and Restart**
   - Address root cause
   - Test fix in paper mode first
   - Monitor closely for 100 trades

---

## Next Steps

**After Setup:**
1. ✅ Read [README.md](README.md) - Overview and features
2. ✅ Review [ARCHITECTURE.md](ARCHITECTURE.md) - How it works
3. ✅ Study [PERFORMANCE.md](PERFORMANCE.md) - What we learned
4. ✅ Run bot for 24 hours
5. ✅ Analyze first 100 trades
6. ✅ Decide if adjustments needed

**For Live Trading (30+ days from now):**
1. ⏳ Validate 1,000+ paper trades
2. ⏳ Confirm positive P/L over 7-day period
3. ⏳ Test stop-loss protection thoroughly
4. ⏳ Set up API keys (Coinbase, Kraken)
5. ⏳ Start with $100-$500 ONLY
6. ⏳ Scale up slowly if profitable

**Good luck, and trade safely! 🚀**

---

## Support

**Documentation:**
- README.md - Project overview
- ARCHITECTURE.md - Technical details
- PERFORMANCE.md - Results analysis
- SETUP.md - This file

**Common Issues:**
- Check Troubleshooting section above
- Review error messages in terminal
- Inspect log files in `logs/bot_output.log`
- Analyze trade history in `data/history/trade_history.csv`

---

*Last Updated: February 6, 2026*  
*Version: 1.0*
