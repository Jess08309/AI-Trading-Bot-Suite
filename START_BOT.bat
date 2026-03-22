@echo off
setlocal

echo ========================================
echo   Master Chess Bot Launcher
echo ========================================

set "BOT_DIR=C:\Bot"
set "PYTHON_EXE=%BOT_DIR%\.venv\Scripts\python.exe"
set "BOT_MAIN=%BOT_DIR%\cryptotrades\main.py"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found: %PYTHON_EXE%
    pause
    exit /b 1
)

if not exist "%BOT_MAIN%" (
    echo ERROR: Bot entrypoint not found: %BOT_MAIN%
    pause
    exit /b 1
)

cd /d "%BOT_DIR%"
if not defined ENABLE_FUTURES set "ENABLE_FUTURES=true"
if not defined ENABLE_COINBASE_FUTURES_DATA set "ENABLE_COINBASE_FUTURES_DATA=true"
if not defined ENABLE_KRAKEN_FUTURES_FALLBACK set "ENABLE_KRAKEN_FUTURES_FALLBACK=true"
if not defined DIRECTION_BIAS set "DIRECTION_BIAS=short_lean"
if not defined DIRECTION_BIAS_STRENGTH set "DIRECTION_BIAS_STRENGTH=0.01"
if not defined RL_SHADOW_MODE set "RL_SHADOW_MODE=true"
if not defined RL_LIVE_SIZE_CONTROL set "RL_LIVE_SIZE_CONTROL=false"
if not defined RL_LIVE_SIZE_MIN_MULT set "RL_LIVE_SIZE_MIN_MULT=0.5"
if not defined RL_LIVE_SIZE_MAX_MULT set "RL_LIVE_SIZE_MAX_MULT=1.5"
if not defined FUTURES_LEVERAGE_OVERRIDE set "FUTURES_LEVERAGE_OVERRIDE=2"
if not defined USE_LOCKED_PROFILE set "USE_LOCKED_PROFILE=true"
if not defined LOCKED_PROFILE_PATH set "LOCKED_PROFILE_PATH=%BOT_DIR%\data\state\locked_profile.json"

if /I "%USE_LOCKED_PROFILE%"=="true" (
    if not exist "%LOCKED_PROFILE_PATH%" (
        echo WARNING: Locked profile requested but file is missing:
        echo          %LOCKED_PROFILE_PATH%
        echo          Run LOCK_CURRENT_PROFILE.ps1 to create it.
        echo          Continuing without locked profile to avoid startup failure.
        set "USE_LOCKED_PROFILE=false"
    )
)

echo Running from: %BOT_DIR%
echo Python: %PYTHON_EXE%
echo Entrypoint: %BOT_MAIN%
echo Direction bias: %DIRECTION_BIAS% (strength=%DIRECTION_BIAS_STRENGTH%)
echo RL shadow mode: %RL_SHADOW_MODE%
echo RL live size control: %RL_LIVE_SIZE_CONTROL% (%RL_LIVE_SIZE_MIN_MULT%x-%RL_LIVE_SIZE_MAX_MULT%x)
echo Futures leverage override: %FUTURES_LEVERAGE_OVERRIDE%x
echo Locked profile: %USE_LOCKED_PROFILE% (%LOCKED_PROFILE_PATH%)
echo.
echo [REMINDER] Restart safety: keep locked profile ON to prevent config drift.
echo [REMINDER] If strategy settings are intentionally changed, run LOCK_CURRENT_PROFILE.ps1 again.
echo.
echo Starting bot... (live output below)
echo Press Ctrl+C to stop.
echo.

"%PYTHON_EXE%" -u "%BOT_MAIN%"

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Bot exited with code %EXIT_CODE%
if not "%EXIT_CODE%"=="0" pause

endlocal
exit /b %ERRORLEVEL%