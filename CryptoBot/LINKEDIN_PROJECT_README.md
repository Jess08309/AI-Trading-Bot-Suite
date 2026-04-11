# Autonomous Multi-Asset Trading Platform

**Role:** Solo Developer & Architect
**Codebase:** ~56,000 lines of Python across 200+ files
**Stack:** Python · PyTorch · scikit-learn · NumPy · Pandas · Flask · Prometheus · Grafana · Docker · REST APIs · WebSockets

---

## Overview

Designed and built a fully autonomous multi-bot trading platform from scratch — four independent trading engines covering crypto spot/futures, stock options scalping, credit spreads (iron condors), and momentum call buying. The platform runs 24/7 on a single machine, managing its own positions, risk, and model retraining without human intervention.

Every component — from the machine learning pipelines to the reinforcement learning agents to the circuit breaker logic — was architected, implemented, and battle-tested by me. No wrappers, no tutorial code, no off-the-shelf strategy libraries.

---

## The Four Bots

### 1. CryptoBot — AI-Driven Crypto Spot & Futures Trading
**Markets:** 100+ cryptocurrencies (18 spot via Alpaca, 8-12 futures via Kraken)
**Runtime:** 24/7, 60-second risk checks, trade decisions every 5 minutes

Trades both long and short across spot and leveraged futures (2×). Uses a three-model ensemble:

| Model | Weight | How It Works |
|-------|--------|-------------|
| Gradient Boosting + GPU Neural Net | 50% | Sklearn GBM (200 estimators) ensembled with a 4-layer PyTorch neural net (15→128→64→32→2) trained on 15 technical features. Runs on CUDA GPU for sub-millisecond batch inference. |
| 11-Source Sentiment Composite | 35% | Aggregates Fear & Greed Index, whale transactions, Bitcoin funding rates, liquidation proxies, open interest, DXY strength, mempool congestion, stablecoin ratios, and more — all from free public APIs. |
| Deep Q-Network (RL Agent) | 15% | DQN with experience replay (10K buffer), 5 position-sizing actions. Runs in shadow mode with parallel virtual portfolios to prove itself before getting real control. |

**NLP Sentiment Pipeline:** Ingests RSS feeds from CoinDesk, Cointelegraph, The Block, and Bitcoin Magazine plus Reddit posts from 5 crypto subreddits. Scores articles using DistilBERT transformer (with keyword-based fallback). Coin-specific routing maps headlines to the right asset.

**Universe Scanner:** Every 30 minutes, pulls the top 250 coins by market cap from CoinGecko, cross-references with available Alpaca pairs, filters by volume and liquidity, scores by 5 weighted factors, and enforces a 72-hour probation period on new discoveries to prevent chasing pumps.

**Regime Detection:** Classifies the market into 4 regimes (Trending Up, Trending Down, Ranging, High Volatility) using ADX, SMA alignment, ATR ratios, and Bollinger squeeze. Regime flips trigger cooldown periods that block new entries during transitions.

---

### 2. AlpacaBot — Stock Options Scalper
**Markets:** 61 stocks (AAPL, NVDA, MSFT, TSLA, META, etc.)
**Runtime:** Market hours, 15-second risk checks, signal scan every 20 minutes

Scalps short-dated call options (1-3 DTE) with per-symbol optimized DTE derived from 150+ backtests. Puts are disabled entirely after backtesting showed an 18% win rate.

| Component | Details |
|-----------|---------|
| ML Model | VotingClassifier ensemble (GBM + Random Forest + ExtraTrees) with 20 features including 14 technical indicators + 6 options-context features (hour of day, intraday range, overnight gap, vol regime) |
| Sentiment | 5 sources — Alpaca news (keyword scoring), VIX via VIXY ETF, SPY trend, market breadth, options flow anomalies |
| Meta-Learner | Blends ML (35%), Sentiment (15%), and Rule Score (50%) with dynamic thresholds that tighten on losing streaks |
| RL Shadow Agent | Tabular Q-learning (243 discrete states), 4 actions — observes every trade, needs 50-trade outperformance to get promoted |
| Risk Manager | Graduated response (not binary) — scales position size from 100% down to 10% as losses accumulate, with per-symbol and per-direction pause logic |

**Per-Symbol DTE Map:** Backtested 150 configurations across 61 symbols to find the optimal DTE for each stock. Top performers: AVGO (2 DTE, profit factor 1.34), PYPL (2 DTE, PF 2.13), NFLX (2 DTE, PF 1.57).

**Morning Window:** Lower confidence thresholds 9:30-11:00 ET to capture momentum at the open. Higher bars in the afternoon when momentum fades.

---

### 3. IronCondor — Credit Spread Income Engine
**Markets:** 13 mega-cap symbols (SPY, QQQ, AAPL, MSFT, NVDA, GOOGL, AMZN, META, etc.)
**Runtime:** Market hours, 15-minute scan interval

Sells OTM credit spreads at 16 delta (~84% probability of profit) with 30-60 DTE, targeting 50% profit take. Both bull put spreads and bear call spreads — the full iron condor profile. Inspired by tastytrade methodology.

| Component | Details |
|-----------|---------|
| ML Model | GBM classifier trained on 12 credit-spread-quality features (IV/HV ratio, support distance, credit quality, OI liquidity, sector relative strength, OTM buffer, etc.) |
| IV Filter | Only enters when implied volatility ≥ 1.2× historical volatility (fat premiums) |
| Earnings Guard | Checks yfinance for upcoming earnings — blocks trades within the earnings window. Fail-closed: if yfinance is unavailable, the trade is blocked |
| Spread Sizing | Width scales with price: $5 spreads (<$200 stocks), $10 ($200-500), $25 (>$500) |
| Exit Rules | 50% profit take, 1.5× credit stop-loss, 21 DTE time exit (avoids gamma risk), emergency exit at 2% from short strike |
| Crash Filter | Blocks all new spreads when SPY drops >1.5% intraday or VIXY spikes >20% — prevents selling into a market crash |
| Regime Caps | In TRENDING_DOWN or HIGH_VOL regimes, tightens max positions from 37P/20C to 5P/3C — reduces exposure when conditions deteriorate |

---

### 4. CallBuyer — Momentum Breakout Calls
**Markets:** High-beta stocks during breakouts
**Runtime:** Market hours, 10-minute scan interval

Buys ITM calls (~2% in-the-money, delta ~0.65) on momentum breakouts with volume surge confirmation (≥1.3× average). 45 DTE target for time cushion. Small allocation (15%), high-risk/high-reward.

| Component | Details |
|-----------|--------|
| ML Model | GBM classifier trained on breakout-quality features (volume surge, ATR expansion, relative strength, support bounce distance) |
| Earnings Guard | Checks yfinance for upcoming earnings — blocks call buys within 7 days of earnings. Fail-closed: if yfinance is unavailable, the trade is blocked. ETFs bypass. |
| Regime Detection | Classifies market regime; blocks entries in TRENDING_DOWN. Adjusts confidence thresholds per regime. |
| Exit Rules | 50% take-profit, -25% stop-loss, trailing stop after 20% gain, 7 DTE time exit |

---

## Shared Infrastructure

### Unified Dashboard
A Flask web application that shows all four bots in a single UI — real-time P&L, open positions, recent trades, and process health for every bot. Grand total P&L aggregation across the entire platform.

### Monitoring Stack (Prometheus + Grafana)
Docker Compose deployment with three services:
- **State Exporter** — reads JSON state files from all bots, exposes Prometheus metrics (P&L, win rate, Sharpe ratio, drawdown, positions)
- **Prometheus** — scrapes metrics every 15 seconds
- **Grafana** — visualization dashboards for historical performance

### Bot Watchdog
Continuous health monitor with 7 automated checks: process alive, log freshness, crash detection (regex for ERROR/Traceback), balance drawdown alerts, win-rate degradation, stale position detection, and consecutive loss tracking. Auto-restarts crashed bots with exponential backoff. Windows toast notifications for critical alerts.

### Backtesting Suite
27 backtest scripts (~7,000 lines) covering:
- Multi-timeframe composite scoring (3-month, 6-month, 12-month weighted)
- Walk-forward optimization
- Per-symbol DTE sweep (source of the optimized DTE map)
- Options-specific metrics (ATR%, Volume CV, Vol Regime scoring)
- Strategy variant comparison

### Operational Tooling
PowerShell and batch scripts for production operations:
- **Performance Tuner** — sets High Performance power plan, AboveNormal process priority, disables Nagle's algorithm for lower latency
- **Status Monitor** — real-time balance and position display with live price lookups
- **Profile Locking** — snapshots runtime config into `locked_profile.json` to prevent config drift between restarts
- **Launch Profiles** — Default, Aggressive (trade every cycle, higher confidence bars), and Conservative (wider trend lean, lower RL multiplier)
- **Trade Analytics** — exit reason distribution analyzer, live trade feed, console log tail

---

## Portfolio-Level Safety

All three stock/options bots enforce a shared aggregate risk cap and bot-specific guardrails:

| Guardrail | Scope | How It Works |
|-----------|-------|--------------|
| Portfolio Exposure Cap | All 3 stock bots | Each bot reads position files from all others. If aggregate worst-case risk exceeds 25% of account equity, new entries are blocked across the board. |
| SPY Crash Filter | IronCondor | Before scanning for new spreads, checks if SPY is down >1.5% intraday or VIXY is up >20%. If either triggers, all new spread entries are blocked for that cycle. |
| Regime Position Caps | IronCondor | In TRENDING_DOWN or HIGH_VOL regimes, max positions tighten from 37 puts / 20 calls to 5 puts / 3 calls. Prevents overcommitting during stress. |
| Earnings Guard | CallBuyer | Checks yfinance earnings calendar before every call purchase. Blocks entries within 7 days of earnings. Fail-closed design. ETFs whitelisted. |
| Cross-Bot State Files | All | Each bot writes `positions.json` with standardized risk fields (`max_loss_total`, `entry_total`, `cost`). Portfolio cap reads all three files with BOM-safe encoding. |

---

## Technical Highlights

### Machine Learning
- **4 independent ML pipelines** — each bot has its own model trained on domain-specific features
- **Ensemble methods** — GBM, Random Forest, ExtraTrees via VotingClassifier (AlpacaBot); GBM + GPU Neural Net (CryptoBot)
- **Walk-forward validation** with `TimeSeriesSplit` — no look-ahead bias
- **Quality gates** — models must pass out-of-sample accuracy thresholds (53-55%) before deployment
- **Auto-retraining** — every 8 hours (crypto) or every 15 new trade outcomes (options)
- **GPU acceleration** — PyTorch neural net on CUDA (RTX 4080) for sub-millisecond batch prediction

### Reinforcement Learning
- **Deep Q-Network** with experience replay, target network soft updates, and ε-greedy exploration
- **Shadow mode architecture** — RL agent runs parallel virtual portfolios against a baseline, learning from thousands of shadow trades before influencing real decisions
- **Reward shaping** — amplifies large wins, dampens noise from tiny wins, exploration bonus for under-traded symbols

### Risk Management (Every Bot)
- **Circuit breakers** — consecutive loss limits, daily loss caps, max drawdown thresholds (all persisted across restarts)
- **Per-symbol auto-pause** after consecutive losses with configurable cooldowns
- **Per-direction tracking** — automatically suspends LONG or SHORT if that direction's win rate drops below threshold
- **Position correlation gating** — Pearson correlation on rolling returns prevents over-concentration (max 0.60)
- **Execution realism** (paper trading) — slippage simulation (5-8 bps), partial fill probability (30%), trading fees, funding costs, and orderbook liquidity gates

### Sentiment & NLP
- **11 free sentiment sources** for crypto — Fear & Greed, whale tracking, funding rates, liquidation proxies, DXY, mempool, stablecoin ratios, and more
- **5 sentiment sources** for stocks — news keyword scoring, VIX, SPY trend, market breadth, options flow
- **DistilBERT transformer** for NLP scoring of RSS feeds and Reddit posts
- **Coin/stock-specific routing** — headlines are mapped to the correct asset via keyword matching

### Position Sizing
- **Kelly Criterion** base with quarter-Kelly fraction for safety
- **Multi-factor scaling** — volatility (0.5×-1.6×), time-of-day (0.5× during quiet hours), win/loss streak (0.5×-1.3×), regime multiplier, ML confidence scaling
- **Quiet hours detection** — reduces sizes during low-liquidity windows

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Multi-Bot Trading Platform                          │
│                                                                          │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌─────────────┐  │
│  │   CryptoBot   │ │   AlpacaBot   │ │  IronCondor   │ │  CallBuyer  │  │
│  │  Spot+Futures  │ │ Options Scalp │ │Credit Spreads │ │  Momentum   │  │
│  │   24/7, 60s   │ │ Mkt Hrs, 15s  │ │ Mkt Hrs, 30s  │ │ Mkt Hrs,10m │  │
│  │               │ │               │ │               │ │             │  │
│  │  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │ │ ┌─────────┐│  │
│  │  │ ML+GPU  │  │ │  │ Voting  │  │ │  │   GBM   │  │ │ │   GBM   ││  │
│  │  │Ensemble │  │ │  │Ensemble │  │ │  │ Spreads │  │ │ │Breakout ││  │
│  │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │ │ └────┬────┘│  │
│  │       │       │ │       │       │ │       │       │ │      │     │  │
│  │  ┌────▼────┐  │ │  ┌────▼────┐  │ │  ┌────▼────┐  │ │ ┌────▼────┐│  │
│  │  │Sentiment│  │ │  │Sentiment│  │ │  │ IV/HV   │  │ │ │Meta-    ││  │
│  │  │11 src   │  │ │  │ 5 src   │  │ │  │ Filter  │  │ │ │Learner  ││  │
│  │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │ │ └────┬────┘│  │
│  │       │       │ │       │       │ │       │       │ │      │     │  │
│  │  ┌────▼────┐  │ │  ┌────▼────┐  │ │  ┌────▼────┐  │ │ ┌────▼────┐│  │
│  │  │RL Agent │  │ │  │RL Agent │  │ │  │Earnings │  │ │ │  Risk   ││  │
│  │  │(Shadow) │  │ │  │(Shadow) │  │ │  │ Guard   │  │ │ │ Manager ││  │
│  │  └─────────┘  │ │  └─────────┘  │ │  └─────────┘  │ │ └─────────┘│  │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └─────┬─────┘  │
│          │                 │                 │               │         │
│          ▼                 ▼                 ▼               ▼         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │               Shared Infrastructure Layer                      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────────────┐  │    │
│  │  │ Unified  │ │ Watchdog │ │ Regime    │ │ Prometheus +   │  │    │
│  │  │Dashboard │ │ (7 health│ │ Detector  │ │ Grafana        │  │    │
│  │  │(all bots)│ │  checks) │ │ (4 modes) │ │ (Docker)       │  │    │
│  │  └──────────┘ └──────────┘ └───────────┘ └────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  APIs: Alpaca (stocks+options+crypto) · Kraken (futures+WS) · CoinGecko │
│        · alternative.me · blockchain.info · RSS · Reddit · DistilBERT    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## What I Learned

- How to build **production ML systems** that retrain on live data without overfitting — walk-forward validation, quality gates, model versioning
- The critical difference between **backtested performance and live execution** — slippage, partial fills, and liquidity matter more than signal accuracy
- Why **risk management isn't a feature — it's the architecture** — circuit breakers, per-symbol pauses, and correlation gates prevent catastrophic losses
- Building **reinforcement learning agents** that prove themselves in shadow mode before getting real responsibility
- The discipline of **multi-system orchestration** — 4 bots, 4 strategies, shared monitoring, independent failure domains
- **Designing for failure**: state persistence, crash recovery, watchdogs, and graceful degradation across every component
- **NLP at scale**: transformer-based sentiment scoring on live RSS/Reddit feeds mapped to specific trading assets
- Treating a side project with **production-grade engineering standards** — observability, deployment profiles, config locking, and audit trails

---

**~56,000 lines of Python · 200+ files · 4 independent trading engines · 24/7 autonomous operation**
**Status:** Actively running · Paper trading mode · Continuously iterating
