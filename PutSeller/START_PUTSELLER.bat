@echo off
title PutSeller - Credit Put Spreads (Right Leg)
cd /d "%~dp0"
echo ============================================
echo   PutSeller - Credit Put Spread Bot
echo   Strategy: Sell OTM Bull Put Spreads
echo   Allocation: 35%% of Alpaca Account
echo ============================================
echo.

if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
)

.venv\Scripts\python.exe main.py
pause
