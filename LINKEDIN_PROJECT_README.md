# Autonomous Crypto Trading Engine

**Role:** Solo Developer & Architect
**Stack:** Python · scikit-learn · NumPy · Pandas · Flask · REST APIs · JSON state management

---

## Overview

Designed and built a fully autonomous cryptocurrency trading engine from scratch — a production-grade system that monitors 30+ markets across spot and futures, generates ML-driven signals, manages risk in real time, and executes trades 24/7 without human intervention.

This isn't a tutorial bot or a wrapper around someone else's library. Every component — from the machine learning pipeline to the circuit breaker logic to the position management system — was architected, implemented, and battle-tested by me.

---

## Technical Highlights

**Machine Learning Pipeline**
- Built a custom ML signal engine using Gradient Boosting (scikit-learn) that trains on live-accumulated market data — not static backtests
- Engineered features from multi-timeframe price history: returns, volatility, RSI, moving average crossovers, trend slope, and volume proxies
- Automated model retraining on a scheduled cadence with out-of-sample accuracy gating to reject underperforming models before deployment

**Reinforcement Learning (Shadow Mode)**
- Implemented a DQN-based RL agent that runs in shadow mode alongside the primary strategy
- Shadow agent independently tracks its own P&L, drawdown, and win rate against the baseline — providing a live A/B test without risking capital
- Online learning from shadow trade outcomes feeds back into the agent's Q-values each cycle

**Risk Management System**
- Multi-layered circuit breaker: consecutive loss limits, daily loss caps, and max drawdown thresholds — all persisted across restarts
- Per-symbol auto-pause after consecutive losses with configurable cooldown windows
- Per-direction (LONG/SHORT) rolling win-rate tracker that automatically suspends a failing direction
- Position correlation gating to prevent over-concentration in correlated assets
- Trailing stops, stale profit-decay exits, and max-hold time enforcement

**Signal Quality Architecture**
- Trend-following core with strict counter-trend suppression (requires 92%+ ML confidence to trade against the trend)
- Sideways market filter that blocks low-conviction entries in ranging conditions
- RSI guardrails, sentiment integration (Fear & Greed Index with staleness decay), and ensemble scoring
- Kelly Criterion-based position sizing adjusted for volatility and portfolio exposure

**Infrastructure & Operations**
- Multi-exchange support: Coinbase (spot) and Kraken (futures) with automatic failover and spot-proxy fallback
- Flask-based live monitoring dashboard with real-time P&L, position tracking, and strategy comparison
- Full state persistence (positions, balances, price history, model artifacts) enabling seamless recovery from restarts
- Configurable locked-profile system for reproducible deployments with environment variable overrides
- Runtime fingerprinting for audit trails — every startup captures engine hash, config snapshot, and model metadata

**Data & Observability**
- CSV trade journal with full entry/exit detail for accounting and strategy analysis
- Structured logging with daily rotation and alert system (webhook + email support)
- Drawdown and daily-loss alerting with configurable cooldowns

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Main Loop (60s)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Fetch    │→│  Risk     │→│  Signal Gen   │  │
│  │  Prices   │  │  Check    │  │  (every 5m)  │  │
│  └──────────┘  └──────────┘  └──────┬───────┘  │
│                                      │          │
│  ┌──────────┐  ┌──────────┐  ┌──────▼───────┐  │
│  │  State    │←│  Execute  │←│  Quality     │  │
│  │  Persist  │  │  Trades   │  │  Gates       │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │  RL Shadow Agent (parallel evaluation)   │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## What I Learned

- How to build production ML systems that retrain on live data without overfitting
- The critical difference between backtested performance and live execution
- Why risk management isn't a feature — it's the architecture
- Designing for failure: state persistence, circuit breakers, and graceful degradation
- The discipline of treating a side project with production-grade engineering standards

---

**Status:** Actively running · Paper trading mode · Continuously iterating
