# AlpacaBot - Options Trading Bot

Python-based options trading bot targeting SPY, QQQ, AAPL, MSFT using Alpaca's API.

## Status: Scaffolding Phase

## Planned Features
- Options chain analysis (Greeks, IV, spread strategies)
- ML signal generation (ported from CryptoBot)
- Risk management (defined-risk trades, position limits)
- Paper trading → live migration path

## Structure
```
core/           # Trading engine, API client, options logic
data/           # Historical data, models, state
logs/           # Trade logs
tools/          # Analysis & reporting scripts
tests/          # Unit tests
```

## Getting Started
1. Get Alpaca API keys (paper trading)
2. `pip install -r requirements.txt`
3. Configure `config.py` with API keys
4. `python main.py`
