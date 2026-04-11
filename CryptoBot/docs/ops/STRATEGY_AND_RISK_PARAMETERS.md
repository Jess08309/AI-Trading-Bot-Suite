# Strategy and Risk Parameters

This document captures the current live-engine strategy/risk configuration used by `trading_engine.py`.

## Core Strategy
- Universe:
  - Spot: `BTC-USD, ETH-USD, SOL-USD, ADA-USD, AVAX-USD, DOGE-USD, LINK-USD, XRP-USD`
  - Futures: `PI_XBTUSD, PI_ETHUSD, PI_SOLUSD`
- Directional logic:
  - Long bias when ML score is strong-up, trend is supportive, RSI is not overbought
  - Short bias when ML score is strong-down, trend is supportive, RSI is not oversold
- Sentiment source:
  - Fear & Greed index (Alternative.me), cached with fallback behavior

## Execution and Risk Defaults
- Initial balances:
  - Spot: `$2,500`
  - Futures: `$2,500`
- Position sizing:
  - Max position size: `10%` of balance
  - Min position size: `2%` of balance
  - Kelly-like confidence scaling with volatility adjustment
- Position limits:
  - Spot max open: `5`
  - Futures max open: `3`
- Stops/targets:
  - Spot stop loss: `-2.0%`
  - Futures stop loss: `-3.0%`
  - Take profit: `+4.0%`
  - Trailing stop: `1.5%`
- Correlation control:
  - Max pairwise correlation: `0.80`
  - Correlation lookback: `120` points (increased from `60`)
- Circuit breaker:
  - Max consecutive losses: `3`
  - Daily loss limit: `-5%`
  - Max drawdown: `-10%`

## Market Data and Fallbacks
- Spot data: Coinbase public endpoints
- Futures data routing:
  1. Coinbase product endpoints (preferred)
  2. Kraken futures fallback when Coinbase data is unavailable
- Liquidity gate:
  - Spot minimum depth: `$50,000`
  - Futures minimum depth: `$100,000`
  - Required depth ratio to order size: `5x`

## Recommended Validation Workflow
1. Compile check:
   - `python -m py_compile cryptotrades/core/trading_engine.py`
2. Backtest (historical):
   - `python backtest_harness.py --instrument spot --months 3`
   - `python backtest_harness.py --instrument futures --months 6`
3. Performance report refresh:
   - `python quick_report.py`
4. Monitoring review:
   - Prometheus/Grafana stack in `monitoring/`

## Nice-to-Have Roadmap
- Portfolio rebalancing logic
- Dynamic position sizing by portfolio heat
- Multi-exchange support for execution
- ML model versioning and model registry metadata
