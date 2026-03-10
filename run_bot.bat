@echo off
set BOT_LOCK_SKIP=1

rem Run from this folder (portable)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
	echo ERROR: venv missing at .venv\Scripts\python.exe
	exit /b 1
)

if "%ENABLE_FUTURES%"=="" set "ENABLE_FUTURES=true"
if "%ENABLE_COINBASE_FUTURES_DATA%"=="" set "ENABLE_COINBASE_FUTURES_DATA=false"
if "%ENABLE_KRAKEN_FUTURES_FALLBACK%"=="" set "ENABLE_KRAKEN_FUTURES_FALLBACK=true"
if "%DIRECTION_BIAS%"=="" set "DIRECTION_BIAS=neutral"
if "%DIRECTION_BIAS_STRENGTH%"=="" set "DIRECTION_BIAS_STRENGTH=0.04"
if "%RL_SHADOW_MODE%"=="" set "RL_SHADOW_MODE=true"
if "%RL_LIVE_SIZE_CONTROL%"=="" set "RL_LIVE_SIZE_CONTROL=false"
if "%RL_LIVE_SIZE_MIN_MULT%"=="" set "RL_LIVE_SIZE_MIN_MULT=0.5"
if "%RL_LIVE_SIZE_MAX_MULT%"=="" set "RL_LIVE_SIZE_MAX_MULT=1.5"
if "%USE_LOCKED_PROFILE%"=="" set "USE_LOCKED_PROFILE=true"
if "%LOCKED_PROFILE_PATH%"=="" set "LOCKED_PROFILE_PATH=data\state\locked_profile.json"

".venv\Scripts\python.exe" "cryptotrades\main.py"
