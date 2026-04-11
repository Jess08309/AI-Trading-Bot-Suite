@echo off
echo ================================================
echo   AlpacaBot v2.0 SCALP - Options Trading Bot
echo   Strategy: MSFT + NVDA  1DTE  5-min bars
echo   Paper Trading Mode
echo ================================================
echo.

cd /d "%~dp0"

REM Activate venv if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

python main.py

pause
