@echo off
setlocal

echo ========================================
echo   Master Chess - QUALITY AGGRESSIVE v5
echo ========================================
echo.

set "ENABLE_FUTURES=true"
set "ENABLE_COINBASE_FUTURES_DATA=true"
set "ENABLE_KRAKEN_FUTURES_FALLBACK=true"
set "DIRECTION_BIAS=neutral"
set "DIRECTION_BIAS_STRENGTH=0.02"
set "RL_SHADOW_MODE=true"
set "RL_LIVE_SIZE_CONTROL=false"
set "RL_LIVE_SIZE_MIN_MULT=0.5"
set "RL_LIVE_SIZE_MAX_MULT=2.0"
set "USE_LOCKED_PROFILE=true"
set "LOCKED_PROFILE_PATH=D:\042021\CryptoBot\data\state\locked_profile.json"
set "LOCKED_PROFILE_ALLOW_ENV_OVERRIDES=true"

rem === Speed: trade every cycle (1 min) but quality gates are enforced ===
set "AGGRESSIVE_FUTURES_BURST=true"
set "FORCE_MODEL_RETRAIN_ON_START=true"
set "TRADE_CYCLE_INTERVAL_OVERRIDE=1"
set "MODEL_RETRAIN_HOURS_OVERRIDE=2"

rem === Quality gates: HIGHER bars than before ===
set "MIN_ML_CONFIDENCE_OVERRIDE=0.62"
set "MIN_ENSEMBLE_SCORE_OVERRIDE=0.58"
set "SIDE_MARKET_FILTER_OVERRIDE=true"

rem === Circuit breakers: keep them ACTIVE ===
set "CB_MAX_CONSECUTIVE_LOSSES_OVERRIDE=5"
set "CB_DAILY_LOSS_LIMIT_PCT_OVERRIDE=-5"
set "CB_MAX_DRAWDOWN_PCT_OVERRIDE=-10"

rem === Position limits ===
set "MAX_POSITIONS_SPOT_OVERRIDE=50"
set "MAX_POSITIONS_FUTURES_OVERRIDE=30"
set "MAX_POSITIONS_PER_SYMBOL_SPOT_OVERRIDE=2"
set "MAX_POSITIONS_PER_SYMBOL_FUTURES_OVERRIDE=2"
set "MAX_CORRELATION_OVERRIDE=0.85"
set "FUTURES_LEVERAGE_OVERRIDE=2"

rem === Risk / reward: balanced stops ===
set "STOP_LOSS_PCT_OVERRIDE=-3.0"
set "TAKE_PROFIT_PCT_OVERRIDE=3.0"
set "TRAILING_STOP_PCT_OVERRIDE=2.0"
set "FUTURES_STOP_LOSS_OVERRIDE=-4.0"

rem === Symbol auto-pause ===
set "SYMBOL_PAUSE_CONSECUTIVE_LOSSES_OVERRIDE=4"

echo Quality gates: ML_prob^>=%MIN_ML_CONFIDENCE_OVERRIDE% Ensemble^>=%MIN_ENSEMBLE_SCORE_OVERRIDE% SIDE_filter=ON
echo Risk: SL=%STOP_LOSS_PCT_OVERRIDE%%% TP=%TAKE_PROFIT_PCT_OVERRIDE%%% Trail=%TRAILING_STOP_PCT_OVERRIDE%%%
echo Circuit breakers: losses=%CB_MAX_CONSECUTIVE_LOSSES_OVERRIDE% daily=%CB_DAILY_LOSS_LIMIT_PCT_OVERRIDE%%% drawdown=%CB_MAX_DRAWDOWN_PCT_OVERRIDE%%%
echo Per-symbol caps: spot=%MAX_POSITIONS_PER_SYMBOL_SPOT_OVERRIDE% futures=%MAX_POSITIONS_PER_SYMBOL_FUTURES_OVERRIDE% ^| Max corr=%MAX_CORRELATION_OVERRIDE%
echo Futures leverage: %FUTURES_LEVERAGE_OVERRIDE%x ^| Symbol pause after: %SYMBOL_PAUSE_CONSECUTIVE_LOSSES_OVERRIDE% consecutive losses
echo Force retrain: %FORCE_MODEL_RETRAIN_ON_START% ^| Locked profile: %USE_LOCKED_PROFILE%
echo.

call "C:\Master Chess\START_BOT.bat"

endlocal
