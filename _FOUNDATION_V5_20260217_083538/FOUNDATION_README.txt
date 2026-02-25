===============================================================
  MASTER CHESS TRADING BOT - FOUNDATION BASELINE v5.0
  Saved: 2026-02-17 08:36:47
===============================================================

VERSION: v5.0 QUALITY-FIRST
STATUS:  PROVEN PROFITABLE (12h paper trading test)

PERFORMANCE AT SAVE TIME:
  - 104 trades since v5 deploy (8pm Feb 16)
  - 71.8% win rate (74W / 29L / 1F)
  - +$31.48 P&L on $5,000 paper balance
  - Only 1 stop-loss hit in 104 trades
  - ML model: 76% train / 74% test accuracy
  - Balance: Spot=$2523.5978 Futures=$2504.6011

KEY CONFIGURATION:
  - GradientBoosting ML classifier
  - MIN_ML_CONFIDENCE: 0.62
  - MIN_ENSEMBLE_SCORE: 0.58
  - SIDE market filter: ON
  - Kelly criterion position sizing
  - Trailing stops + max hold time exits
  - Per-symbol auto-pause after 4 consecutive losses
  - Circuit breakers: 5 consecutive losses / -5% daily / -10% drawdown

FILES INCLUDED:
  engine/       - Core trading engine + config + main entry point
  config/       - .env, locked_profile, runtime fingerprint
  state/        - Positions, balances, RL agent, meta learner
  launchers/    - All .bat/.ps1 scripts from Master Chess
  monitoring/   - Dashboard and monitoring tools
  models/       - Trained ML model + RL agent weights

TO RESTORE:
  1. Copy engine/*.py -> D:\042021\CryptoBot\cryptotrades\core\
  2. Copy engine/main.py -> D:\042021\CryptoBot\cryptotrades\main.py
  3. Copy config/.env -> D:\042021\CryptoBot\.env
  4. Copy config/locked_profile.json -> D:\042021\CryptoBot\data\state\
  5. Copy state/* -> D:\042021\CryptoBot\data\state\
  6. Copy models/* -> D:\042021\CryptoBot\data\models\
  7. Copy launchers/* -> C:\Master Chess\
  8. Run: C:\Master Chess\START_BOT_AGGRESSIVE.bat

DO NOT MODIFY THESE FILES. This is the proven baseline.
===============================================================
