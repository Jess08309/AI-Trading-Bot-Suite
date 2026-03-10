@echo off
title Trading Command Center
cd /d "%~dp0"

echo.
echo   Trading Command Center
echo   ========================
echo   Starting on http://127.0.0.1:8088
echo.

.venv\Scripts\python.exe tools\dashboard\unified_dashboard.py
pause
