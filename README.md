# Autonomous Multi-Asset Trading Platform

**Solo Developer & Architect** | ~56,000 lines of Python across 200+ files  
**Stack:** Python · PyTorch · scikit-learn · NumPy · Pandas · Flask · Prometheus · Grafana · Docker · REST APIs · WebSockets

---

## What Is This?

A fully autonomous multi-bot trading platform built from scratch — four independent trading engines covering crypto spot/futures, stock options scalping, credit spreads (iron condors), and momentum call buying. Runs 24/7 on a single machine, managing its own positions, risk, and model retraining without human intervention.

Every component — ML pipelines, reinforcement learning agents, circuit breakers, dashboards — was designed, coded, and tested by me. No wrappers, no tutorial code, no off-the-shelf strategy libraries.

**Status:** Actively running in paper trading mode. Continuously iterating.

---

## The Four Bots

### 1. CryptoBot — AI-Driven Crypto Spot & Futures
- **Markets:** 100+ cryptocurrencies (18 spot via Alpaca, 8-12 futures via Kraken)
- **Runtime:** 24/7, 60-second risk checks, trade decisions every 5 minutes
- **ML:** Gradient Boosting + GPU Neural Net ensemble (50%) + 11-source sentiment composite (35%) + Deep Q-Network RL agent in shadow mode (15%)
- **NLP:** DistilBERT transformer scoring RSS feeds (CoinDesk, Cointelegraph, The Block, Bitcoin Magazine) + Reddit posts from 5 crypto subreddits
- **Universe Scanner:** Auto-discovers top coins by market cap every 30 minutes with 72-hour probation on new entries
- **Regime Detection:** 4 market regimes (Trending Up/Down, Ranging, High Volatility) using ADX, SMA alignment, ATR ratios, Bollinger squeeze

### 2. AlpacaBot — Stock Options Scalper
- **Markets:** 61 stocks (AAPL, NVDA, MSFT, TSLA, META, etc.)
- **Runtime:** Market hours, 15-second risk checks, signal scan every 20 minutes
- **ML:** VotingClassifier ensemble (GBM + Random Forest + ExtraTrees) with 20 features
- **Per-Symbol DTE Map:** Backtested 150+ configurations across 61 symbols for optimal days-to-expiration
- **RL Shadow Agent:** Tabular Q-learning (243 states), needs 50-trade outperformance to get promoted

### 3. IronCondor — Credit Spread Income Engine
- **Markets:** 13 mega-cap symbols (SPY, QQQ, AAPL, MSFT, NVDA, GOOGL, AMZN, META, etc.)
- **Runtime:** Market hours, 15-minute scan interval
- **Strategy:** Sells OTM credit spreads at 16 delta (~84% probability of profit), 30-60 DTE, 50% profit take
- **Safety:** SPY crash filter, regime-based position caps, IV/HV ratio filter, earnings guard

### 4. CallBuyer — Momentum Breakout Calls
- **Markets:** High-beta stocks during breakouts
- **Runtime:** Market hours, 10-minute scan interval
- **Strategy:** Buys ITM calls on momentum breakouts with volume surge confirmation (≥1.3× average)
- **Safety:** Earnings guard (7-day block), regime detection blocks entries in downtrends

---

## Key Features

### Machine Learning & AI
- 4 independent ML pipelines with domain-specific features
- Walk-forward validation with `TimeSeriesSplit` — no look-ahead bias
- Auto-retraining every 8 hours (crypto) or every 15 trade outcomes (options)
- GPU-accelerated PyTorch neural net on CUDA (RTX 4080)
- Deep Q-Network with experience replay and shadow mode architecture

### Risk Management
- Circuit breakers: consecutive loss limits, daily loss caps, max drawdown thresholds
- Per-symbol auto-pause after consecutive losses
- Position correlation gating (max 0.60 Pearson correlation)
- Portfolio-level exposure cap across all bots (25% of equity)
- SPY crash filter and regime-based position caps

### Sentiment & NLP
- 11 free sentiment sources for crypto (Fear & Greed, whale tracking, funding rates, etc.)
- 5 sentiment sources for stocks (news scoring, VIX, SPY trend, market breadth, options flow)
- DistilBERT transformer for NLP scoring of RSS feeds and Reddit posts

### Infrastructure
- **Unified Dashboard:** Flask web app showing all 4 bots — real-time P&L, positions, trades, health
- **Bot Watchdog:** 7 automated health checks with auto-restart and exponential backoff
- **Monitoring:** Prometheus + Grafana stack via Docker Compose
- **Backtesting Suite:** 27 scripts (~7,000 lines) — walk-forward, per-symbol DTE sweep, strategy comparison

---

## Project Structure

```
├── cryptotrades/              # CryptoBot engine
│   ├── core/                  # Trading engine, universe scanner, perf engine
│   ├── utils/                 # ML, sentiment, regime detection, RL agent, etc.
│   ├── strategies/            # Strategy modules
│   └── tests/                 # Unit tests (28 tests)
├── tools/
│   ├── dashboard/             # Unified web dashboard (all 4 bots)
│   └── backtests/             # Backtesting scripts and analysis
├── monitoring/                # Prometheus + Grafana Docker stack
├── tests/                     # Integration tests (37 tests)
├── docs/                      # Architecture, setup, operations guides
├── BOT_WATCHDOG.py            # Health monitor with auto-restart
├── main.py                    # CLI entrypoint
├── quick_report.py            # Performance report generator
├── backtest_harness.py        # Historical backtesting
├── START_BOT*.bat             # Launch profiles (Default, Aggressive, Conservative)
├── STATUS_MONITOR.ps1         # Real-time balance and position display
└── PERFORMANCE_TUNER.ps1      # System optimization (power plan, process priority)
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Jess08309/CryptoBot-Updated-20260321.git
cd CryptoBot-Updated-20260321

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r cryptotrades/requirements.txt

# Set up your API keys (create a .env file — NOT tracked by git)
# ALPACA_API_KEY=your_key_here
# ALPACA_API_SECRET=your_secret_here

# Run tests
python -m pytest tests/ cryptotrades/tests/ -v

# Launch the bot
.\START_BOT.bat
```

---

## Tests

All **65 tests passing** as of March 21, 2026:
- 37 integration tests (config validation, ML predictor, regime detection, backtesting)
- 28 component tests (circuit breaker, position sizer, performance tracker, retry logic)

---

## Recent Updates (March 21, 2026)

- **Full code audit** — all 4 bots audited for bugs, security, and code quality
- **CallBuyer regime adjustments** — added regime-specific position sizing and confidence offsets for all 4 market regimes
- **Alpaca migration** — CryptoBot migrated from Coinbase to Alpaca (7 files updated)
- **Defensive guardrails** — SPY crash filter, regime-based position caps, earnings guard, portfolio exposure cap
- **Unified dashboard** — all 4 bots + guardrails panel in a single web UI
- **Security cleanup** — removed all state/data files from git tracking, comprehensive .gitignore

---

## Feedback Welcome!

If you have any questions, comments, suggestions, or advice about the code, architecture, or strategies — **please open an Issue or leave a comment!** I'm always looking to learn and improve. Whether it's a bug you spotted, a better approach to something, or just general feedback — I'd love to hear it.

---

**~56,000 lines of Python · 200+ files · 4 independent trading engines · 65 tests passing · 24/7 autonomous operation**

Last Updated: March 21, 2026
