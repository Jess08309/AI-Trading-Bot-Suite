# Quick Start Guide - Trading Bot

> **One-page reference for starting, monitoring, and stopping the bot**

---

## 🚀 Starting the Bot

### Method 1: One-Click (Recommended)
```powershell
cd "c:\Master Chess"
.\START_BOT.bat
```

### Method 2: Manual
```powershell
cd "D:\042021\CryptoBot"
.venv\Scripts\Activate.ps1
python cryptotrades\main.py
```

**What to Expect:**
- First 10 minutes: Building price snapshots
- After 10 minutes: Trading begins
- Updates every 60 seconds

---

## 👀 Monitoring

### Check Balance
```powershell
Get-Content "D:\042021\CryptoBot\data\state\paper_balances.json"
```

### View Active Positions
```powershell
# Spot
Get-Content "D:\042021\CryptoBot\data\state\positions.json" | ConvertFrom-Json

# Futures
Get-Content "D:\042021\CryptoBot\data\state\futures_positions.json" | ConvertFrom-Json
```

### Run Performance Analysis
```powershell
cd "c:\Master Chess"
python analyze_performance.py
```

### Check Logs
```powershell
Get-Content "D:\042021\CryptoBot\logs\bot_output.log" -Tail 50
```

---

## 🛑 Stopping the Bot

### Graceful Stop (Recommended)
1. Press `Ctrl+C` in the terminal
2. Wait for "Shutdown complete" message
3. Close terminal

### Force Stop (Emergency Only)
```powershell
Stop-Process -Name python -Force
```

---

## ⚙️ Current Configuration

| Setting | Value |
|---------|-------|
| Mode | PAPER TRADING (safe) |
| Check Interval | 60 seconds |
| Buy Threshold | 4/15 points |
| Spot Positions | Max 15 |
| Futures Positions | Max 10 |
| Stop Loss | 2% spot, 3% futures |
| Take Profit | 2% spot, 5% futures |
| Starting Balance | $5,000 |
| Current Balance | $3,883.70 |

---

## 🔍 Troubleshooting

### Bot Won't Start
```powershell
# Check Python
python --version  # Should be 3.11+

# Activate environment
cd "D:\042021\CryptoBot"
.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### No Trades Happening
- Wait 10 minutes (building snapshots)
- Check buy scores in logs (need ≥4/15)
- Verify internet connection

### High Losses
1. Stop bot immediately (`Ctrl+C`)
2. Run analysis: `python analyze_performance.py`
3. Review worst trades
4. Check if stop-losses triggering correctly

---

## 📊 Key Metrics to Watch

**Good Signs:**
- ✅ Win rate 55-65%
- ✅ Average P/L > 0%
- ✅ Worst loss < -4%
- ✅ Balance trending up

**Warning Signs:**
- ⚠️ Win rate < 50%
- ⚠️ Average P/L < -0.5%
- ⚠️ Worst loss > -5%
- ⚠️ Balance drops > 10% in one day

**STOP Immediately If:**
- ❌ Balance drops > 20%
- ❌ Worst trade < -10%
- ❌ Win rate < 40%
- ❌ Repeated errors in logs

---

## 📁 Important Files

| File | Location | Purpose |
|------|----------|---------|
| Bot Launcher | `c:\Master Chess\START_BOT.bat` | One-click start |
| Main Code | `D:\042021\CryptoBot\cryptotrades\main.py` | Entry point |
| Trading Engine | `D:\042021\CryptoBot\cryptotrades\core\trading_engine.py` | Core logic |
| Balance | `D:\042021\CryptoBot\data\state\paper_balances.json` | Current money |
| Trade History | `D:\042021\CryptoBot\data\history\trade_history.csv` | All trades |
| Logs | `D:\042021\CryptoBot\logs\bot_output.log` | Runtime logs |

---

## 🎯 Daily Checklist

### Morning (Start Bot)
1. ✅ Check yesterday's performance
2. ✅ Review overnight trades
3. ✅ Start bot: `.\START_BOT.bat`
4. ✅ Verify successful startup

### During Day (Monitor)
1. ✅ Check balance 2-3 times
2. ✅ Review active positions
3. ✅ Watch for error messages
4. ✅ Note any unusual behavior

### Evening (Review)
1. ✅ Run performance analysis
2. ✅ Check win rate
3. ✅ Review worst trades
4. ✅ Stop bot if needed (or leave running 24/7)

---

## 🔐 Safety Reminders

**DO:**
- ✅ Run in paper trading mode
- ✅ Monitor daily
- ✅ Stop if losses exceed 20%
- ✅ Keep backups of data files

**DON'T:**
- ❌ Change PAPER_TRADING to False (yet)
- ❌ Modify parameters without testing
- ❌ Leave unmonitored for weeks
- ❌ Panic over short-term losses

---

## 📞 Need Help?

**Documentation:**
- Overview: `D:\042021\CryptoBot\README.md`
- Technical: `D:\042021\CryptoBot\ARCHITECTURE.md`
- Performance: `D:\042021\CryptoBot\PERFORMANCE.md`
- Setup: `D:\042021\CryptoBot\SETUP.md`
- Completion: `D:\042021\CryptoBot\CLEANUP_COMPLETE.md`

**Common Issues:**
- Check SETUP.md troubleshooting section
- Review error messages in logs
- Run performance analysis
- Check git commits for recent changes

---

**Quick Commands Cheat Sheet:**
```powershell
# Start bot
cd "c:\Master Chess"; .\START_BOT.bat

# Check balance
Get-Content "D:\042021\CryptoBot\data\state\paper_balances.json"

# Analyze performance
cd "c:\Master Chess"; python analyze_performance.py

# Check logs
Get-Content "D:\042021\CryptoBot\logs\bot_output.log" -Tail 50

# Stop bot
Ctrl+C (in bot terminal)
```

---

*Keep this guide handy for quick reference!*

*Last Updated: February 6, 2026*
