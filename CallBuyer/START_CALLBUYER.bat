@echo off
title CallBuyer - Left Leg
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
echo.
echo  ============================================
echo   CallBuyer v1.0 - Momentum Call Buying
echo   Appendage: Left Leg
echo  ============================================
echo.
python main.py
pause
