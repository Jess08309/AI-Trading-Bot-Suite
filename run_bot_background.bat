@echo off
set BOT_LOCK_SKIP=1
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
	echo ERROR: venv missing at .venv\Scripts\pythonw.exe
	exit /b 1
)

if not exist "logs" mkdir "logs"

start /b "" ".venv\Scripts\pythonw.exe" "cryptotrades\main.py" > "logs\bot_output.log" 2>&1
echo Bot started in background. Check logs\bot_output.log for status.
timeout /t 2
