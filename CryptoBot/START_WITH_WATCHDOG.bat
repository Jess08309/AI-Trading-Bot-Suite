@echo off
setlocal

echo ================================================
echo   Master Chess - Bot + Watchdog Launcher
echo ================================================
echo.

rem --- Start the bot in its own window ---
echo [1/2] Starting trading bot...
start "Trading Bot" cmd /c "cd /d C:\Bot && START_BOT_AGGRESSIVE.bat"

rem --- Wait a moment for the bot to initialize ---
timeout /t 10 /nobreak >nul

rem --- Start the watchdog in this window ---
echo [2/2] Starting watchdog monitor...
echo.
echo Watchdog will check bot health every 60 seconds.
echo Auto-restart is ON. Press Ctrl+C to stop watchdog.
echo.

cd /d C:\Bot
"C:\Bot\.venv\Scripts\python.exe" BOT_WATCHDOG.py --interval 60

echo.
echo Watchdog exited.
pause
