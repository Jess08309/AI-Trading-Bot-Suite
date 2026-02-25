@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: venv missing at .venv\Scripts\python.exe
  exit /b 1
)

echo Starting unified web dashboard on http://127.0.0.1:8088
start "Bot Dashboard" cmd /c ".venv\Scripts\python.exe tools\dashboard\web_dashboard.py"
start "" http://127.0.0.1:8088

endlocal