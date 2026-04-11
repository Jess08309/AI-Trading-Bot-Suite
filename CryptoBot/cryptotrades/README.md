# CryptoBot — AI-Powered Cryptocurrency Trading Engine

An autonomous cryptocurrency trading system that combines machine learning, deep reinforcement learning, real-time sentiment analysis, and regime detection to trade 100+ crypto assets across spot and futures markets. Built in Python, designed for 24/7 unattended operation.

> **Status:** Paper trading (simulated) on Alpaca + Kraken Futures  
> **Runtime:** Continuous — cycles every 60 seconds, trades every 5 minutes  
> **Hardware:** Optimized for CUDA GPUs (RTX 4080) but runs on CPU

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
  - [Trading Engine](#1-trading-engine)
  - [ML Model (GradientBoosting + GPU Neural Net)](#2-ml-model)
  - [Sentiment Analysis (11 Sources)](#3-sentiment-analysis-11-free-sources)
  - [News & Social Sentiment](#4-news--social-sentiment)
  - [Reinforcement Learning Agent](#5-reinforcement-learning-agent-dqn)
  - [Meta-Learner Ensemble](#6-meta-learner-ensemble)
  - [Universe Scanner](#7-universe-scanner)
  - [Regime Detection](#8-regime-detection)
  - [Risk Management](#9-risk-management)
- [Signal Generation Pipeline](#signal-generation-pipeline)
- [Trade Execution](#trade-execution)
- [Performance Engine](#performance-engine)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Data & State Files](#data--state-files)
- [Technology Stack](#technology-stack)

---

## How It Works

CryptoBot runs a continuous loop that:

1. **Scans** 100+ cryptocurrencies every 30 minutes, ranking them by volume, momentum, and market cap
2. **Fetches** real-time prices from Alpaca (spot) and Kraken (futures) every 60 seconds
3. **Monitors** open positions for stop-loss, take-profit, trailing stops, and time-based exits every cycle
4. **Generates** trade signals every 5 minutes by running each symbol through 11 quality filters
5. **Executes** the best signals with position sizes tuned by Kelly Criterion, volatility, regime, and time-of-day
6. **Learns** continuously — retraining the ML model every 8 hours and updating the RL agent after every trade

Every trade decision passes through a **multi-model ensemble** that blends:

| Model | Weight | Purpose |
|-------|--------|---------|
| Gradient Boosting + GPU Neural Net | 50% | Technical indicator patterns (15 features) |
| 11-Source Sentiment Composite | 35% | Market fear/greed, whale activity, funding rates |
| Deep Q-Network (RL) | 15% | Adaptive position sizing from experience |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CryptoBot Engine                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │  Universe     │   │  Price Feed  │   │  Sentiment       │   │
│  │  Scanner      │   │  (Alpaca +   │   │  (11 sources +   │   │
│  │  (250 coins)  │   │   Kraken WS) │   │   RSS + Reddit)  │   │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘   │
│         │                  │                     │              │
│         ▼                  ▼                     ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Signal Generation Pipeline                  │   │
│  │  ML Model → Regime Filter → RSI → Trend → Correlation  │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │  Meta-Learner │   │  Risk Mgr    │   │  RL Agent        │   │
│  │  Ensemble     │◄─►│  (Circuit    │   │  (DQN Shadow     │   │
│  │  (ML+Sent+RL) │   │   Breaker)   │   │   Mode)          │   │
│  └──────┬───────┘   └──────────────┘   └──────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Execution Layer                             │   │
│  │  Position Sizing → Liquidity Gate → Slippage Model      │   │
│  │  → Paper Trade Simulation (or Live via Alpaca/Kraken)    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Trading Engine

**File:** `core/trading_engine.py` (~3,700 lines)

The central orchestrator. Runs the main loop, coordinates all subsystems, manages positions, and executes trades.

**Key parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Risk/Reward Ratio | 2.3:1 | Asymmetric — small losses, bigger wins |
| Stop Loss (Spot) | -1.5% | ATR-adaptive with 0.8% floor |
| Take Profit (Spot) | +3.5% | Fixed target |
| Stop Loss (Futures) | -2.0% | Tighter for leveraged positions |
| Take Profit (Futures) | +4.0% | Higher target for futures |
| Trailing Stop | 0.8% | Activates after +1.2% profit |
| Max Positions (Spot) | 4 | Per-asset-class limit |
| Max Positions (Futures) | 3 | Per-asset-class limit |
| Max Position Size | 8% | Of total portfolio per trade |
| Futures Leverage | 2× | Conservative leverage |
| Max Hold (Spot) | 2.5h flat / 5h forced | Time-based exit |
| Max Hold (Futures) | 1.5h flat / 3.5h forced | Shorter for leveraged |
| Trade Interval | 5 minutes | Between trade decision cycles |
| Risk Check Interval | 60 seconds | Continuous position monitoring |

**Exit strategies** (checked every 60 seconds):
- **Stop Loss** — ATR-scaled, tightened 30% in high-volatility regimes
- **Take Profit** — Fixed percentage target
- **Trailing Stop** — Locks in profit, fires at 0.8% drawdown from peak
- **Stale Profit Decay** — Exits if a winning position starts giving back gains (arms at +1.5%, exits if decays by 0.8%)
- **Max Hold Flat** — Closes positions going nowhere after 2.5h (spot) or 1.5h (futures)
- **Max Hold Forced** — Absolute time limit: 5h spot, 3.5h futures
- **Hold Decay** — Early close if held >60% of max time and P/L within ±0.3%

---

### 2. ML Model

**Algorithm:** Scikit-learn GradientBoostingClassifier + PyTorch GPU Neural Network

**15 Technical Features** computed from price history:
- SMA (5, 10, 20-period), EMA (12, 26-period)
- RSI (14-period), MACD signal line
- Bollinger Band width and %B
- ATR (14-period), Stochastic %K/%D
- Volume ratio, price momentum, trend slope

**Training:**
- Walk-forward validation with `TimeSeriesSplit(n_splits=3)` and 2% gap
- Minimum 55% out-of-sample accuracy required to deploy
- Retrains automatically every 8 hours with fresh market data
- Class balancing via sample weights

**GPU Neural Network** (ensemble partner — 40% weight):
- Architecture: `Linear(15→128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128→64) → BatchNorm → ReLU → Dropout(0.2) → Linear(64→32) → BatchNorm → ReLU → Linear(32→2)`
- Trains on same data as sklearn model
- Adam optimizer, ReduceLROnPlateau scheduler, early stopping (patience=10)
- Batch prediction: ~0.5ms for 100 symbols on GPU
- **Final prediction: 60% sklearn + 40% GPU neural net**

---

### 3. Sentiment Analysis (11 Free Sources)

**File:** `utils/enhanced_sentiment.py`

All sources are free and require no API keys. Each has its own in-memory cache with disk fallback for crash recovery.

| # | Source | Weight | What It Measures | Cache |
|---|--------|--------|------------------|-------|
| 1 | Fear & Greed Index | 20% | Market emotion (0-100 scale) | 10 min |
| 2 | Top Coin 24h Momentum | 15% | BTC/ETH/SOL/ADA/etc. price changes | 5 min |
| 3 | Whale Transaction Tracker | 12% | BTC transactions >10 BTC, flow direction | 10 min |
| 4 | Bitcoin Funding Rates | 12% | Kraken futures funding (contrarian signal) | 5 min |
| 5 | Liquidation Proxy | 10% | Basis spread + funding squeeze detection | 5 min |
| 6 | Long/Short Ratio | 8% | Mark vs index price spread | 5 min |
| 7 | Dollar Strength (DXY) | 8% | USD strength vs 6 currencies (EUR, JPY, GBP, CAD, CHF, SEK) | 30 min |
| 8 | Stablecoin Market Ratio | 8% | USDT/USDC/DAI share of total market | 30 min |
| 9 | Mempool Congestion | 4% | Bitcoin unconfirmed transactions + hashrate | 10 min |
| 10 | Global Market Data | 3% | BTC/ETH dominance, total market cap trend | 5 min |
| 11 | Open Interest | — | BTC/ETH/SOL/ADA/DOGE/LINK aggregate OI | 5 min |

**Output:** Composite sentiment score from -1.0 (extremely bearish) to +1.0 (extremely bullish), plus per-coin 24h momentum scores.

**Data Sources:** alternative.me, CoinGecko, Kraken Futures API, blockchain.info, floatrates.com — all public, all free.

---

### 4. News & Social Sentiment

**File:** `utils/news_sentiment.py`

| Source | Details |
|--------|---------|
| **RSS Feeds** | CoinDesk, Cointelegraph, The Block, Bitcoin Magazine, Crypto Briefing (top 10 articles each) |
| **Reddit** | r/cryptocurrency, r/bitcoin, r/ethereum, r/CryptoMarkets, r/defi (top 15 hot posts, public JSON API) |
| **CoinGecko** | Trending coins + global dominance data |
| **Twitter** | Optional (requires API key) — searches #crypto, #bitcoin, #ethereum |

**NLP Scoring:**
- **Primary:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) — transformer-based, returns [-1, 1]
- **Fallback:** Keyword-based scoring with 60 crypto-specific positive/negative terms
- **Coin-specific routing:** Keyword mapping (e.g., "bitcoin"/"btc"/"satoshi" → BTC)
- **Recency-weighted:** More recent articles have higher impact

---

### 5. Reinforcement Learning Agent (DQN)

**File:** `utils/rl_agent.py`

| Parameter | Value |
|-----------|-------|
| Algorithm | Double DQN with experience replay |
| Network | 2 hidden layers × 64 units, LayerNorm + ReLU |
| State Space | 5-dim: [sentiment, volatility, trend, RSI, ML confidence] |
| Action Space | 5 position multipliers: [0.25×, 0.5×, 1.0×, 1.5×, 2.0×] |
| Replay Buffer | 10,000 experiences, batch size 32 |
| Target Network | Synced every 50 updates |
| Exploration | ε-greedy: ε=0.15 → decays → min 0.05 |
| Optimizer | Adam (lr=0.001), Huber loss, gradient clipping |

**Shadow Mode (Phase 1):** The RL agent runs two virtual portfolios in parallel with every trade — a *baseline* (fixed 1× sizing) and an *RL-sized* clone. It learns from shadow outcomes without affecting live (paper) trades. This lets the RL agent accumulate thousands of training examples before being given real control.

**Reward Shaping:** Amplifies large wins (>50% → 1.5×), dampens tiny wins (<10% → 0.5×). Exploration bonus for coins with <10 historical trades.

---

### 6. Meta-Learner Ensemble

**File:** `utils/meta_learner.py`

Blends the three model channels into one buy/sell decision:

```
Final Signal = (ML × 50%) + (Sentiment × 35%) + (RL × 15%)
```

- Weights are **dynamic** — recalculated from rolling accuracy on the last 50 predictions
- RL channel only included if it expresses a strong opinion (|prediction - 0.5| > 0.1)
- Buy threshold: adaptive ~0.55, modulated ±5% by sentiment
- Sell threshold: adaptive ~0.45, modulated ±5% by sentiment
- Thresholds self-adjust ±2% based on recent win rate
- State persisted to `data/state/meta_learner.json`

---

### 7. Universe Scanner

**File:** `core/universe_scanner.py`

Dynamically discovers and ranks tradeable coins every 30 minutes:

1. Pulls **top 250 coins** by market cap from CoinGecko
2. Gets **trending coins** (momentum factor)
3. Cross-references with **Alpaca** available crypto pairs (currently 36 symbols)
4. Filters: minimum $50M market cap, $5M 24h volume, excludes 57+ stablecoins/wrapped tokens
5. **Scores** each coin:

| Factor | Weight | Description |
|--------|--------|-------------|
| Volume | 30% | 24h trading volume (log-scaled, $1B = max) |
| Momentum | 25% | 24h price change magnitude (sweet spot: 2-15%) |
| Market Cap | 20% | Larger = safer / more liquid |
| Trending | 15% | CoinGecko trending bonus |
| Volatility | 10% | 7-day change as trading opportunity proxy |

**Probation System:** New discoveries enter a mandatory **72-hour observation period** before the bot can trade them. This prevents chasing pumps. The 18 core symbols (BTC, ETH, SOL, etc.) are always included regardless.

**Futures Discovery:** Cross-references spot watchlist with Kraken's available perpetual futures contracts (up to 30 futures).

---

### 8. Regime Detection

**File:** `utils/regime_detector.py`

Classifies the market into one of four regimes every 15 minutes using BTC as the reference:

| Regime | Indicators | Bot Behavior |
|--------|-----------|--------------|
| **TRENDING_UP** | ADX > 25, positive SMA slope | Normal entries, LONG bias |
| **TRENDING_DOWN** | ADX > 25, negative SMA slope | Blocks LONG unless ML ≥ 0.65, SHORT bias |
| **RANGING** | ADX < 25, narrow Bollinger bands | Shortens max hold times |
| **HIGH_VOLATILITY** | ATR ratio > 1.5× | Tightens stops 30%, reduces position sizes |

**Regime Flip Detection:** When the regime changes (e.g., TRENDING_UP → TRENDING_DOWN), the bot enters a **cooldown period** where all new entries are blocked. This prevents whipsaw losses during transitions. Flip severity is scored, and whipsaw detection tracks rapid flip frequency.

---

### 9. Risk Management

**Circuit Breaker** — Auto-pauses all trading when:
- 5 consecutive losing trades
- Daily P/L drops below -4%
- Drawdown from equity peak exceeds -8%
- Fast streak guard: 3 losses in 5 trades OR -2% in recent window → 30-minute pause

**Position-Level Controls:**
- Max 8% of portfolio per position
- Max correlation 0.60 between open positions (Pearson on rolling returns)
- Per-symbol auto-pause after 4 consecutive losses (2-hour cooldown)
- Per-direction pause if LONG or SHORT win rate drops below 30% over last 20 trades (1-hour cooldown)
- Symbol health gate: excludes symbols with profit factor < 0.50 over last 20 trades

**Execution Realism** (paper trading):
- Slippage simulation: 5 bps (spot), 8 bps (futures)
- Partial fill probability: 30% chance of partial fill (60-95% fill ratio)
- Trading fees: 0.07% spot, 0.06% futures
- Futures funding cost: 0.02% per 8 hours
- Orderbook liquidity gate: checks depth before entering (min $5K spot, $15K futures)

**Quiet Hours:** Reduces position sizes by 50% during low-liquidity windows (UTC 2-6 AM, 10 PM-midnight).

---

## Signal Generation Pipeline

Every 5 minutes, each symbol passes through **11 quality filters** before a trade signal is produced:

```
Symbol → Blacklist Check
       → Symbol Health Gate (PF ≥ 0.50)
       → Minimum 50 Data Points
       → Per-Symbol Loss Pause Check
       → ML Prediction (must be ≥ 52% confident)
       → Regime-Adjusted Confidence Threshold
       → RSI Filter (LONG: RSI < 68, SHORT: RSI > 32)
       → Low-Volatility Filter
       → Momentum Quality Gate
       → SIDE Market Filter (skip sideways unless ML ≥ 0.52)
       → Counter-Trend Override (ML ≥ 0.56, with 20% penalty)
       → ✅ Signal Produced
```

Signals that pass all filters include: symbol, direction (LONG/SHORT), confidence score, ML score, RSI, trend, sentiment, correlation, volatility, and a human-readable reason string.

---

## Trade Execution

Signals are sorted by confidence (highest first), then executed:

1. **Circuit breaker** pre-check
2. **Correlation gate** — skip if too correlated with existing positions (>0.60)
3. **Position sizing** — Kelly Criterion base, adjusted by:
   - Volatility scaling (0.5× to 1.6×)
   - Time-of-day (0.5× during quiet hours)
   - Win/loss streak (0.5× to 1.3×)
   - Regime multiplier
   - Floor: never below 50% of base size
4. **Liquidity gate** — orderbook depth must support the order
5. **Slippage** applied to simulated entry price
6. **Fees** deducted from paper balance (0.07% spot)
7. **Position created** with calculated stop-loss, take-profit, and trailing stop levels
8. **Shadow trade** opened in both baseline and RL virtual books for comparison

---

## Performance Engine

**File:** `core/perf_engine.py`

Hardware-optimized components:

| Component | Technology | Speedup |
|-----------|-----------|---------|
| **Concurrent Price Fetcher** | ThreadPoolExecutor (10 workers) + HTTP connection pooling | 10× vs sequential (2s vs 20s for 100 symbols) |
| **GPU Neural Network** | PyTorch on CUDA (RTX 4080, 12GB VRAM) | ~0.5ms batch prediction for 100 symbols |

The concurrent fetcher uses `requests.Session` with keep-alive connection pooling, reducing TCP handshake overhead to near-zero for repeated calls to the same Alpaca and Kraken endpoints.

---

## Project Structure

```
cryptotrades/
├── main.py                          # Entry point (single-instance lock, .env loading)
├── requirements.txt                 # Dependencies
├── .env                             # API keys and configuration overrides
│
├── core/
│   ├── trading_engine.py            # Main engine (~3,700 lines) — loop, signals, execution
│   ├── perf_engine.py               # Concurrent fetcher + GPU neural network
│   └── universe_scanner.py          # Dynamic coin discovery and ranking
│
├── utils/
│   ├── config.py                    # Centralized configuration (all params + env overrides)
│   ├── enhanced_sentiment.py        # 11-source sentiment composite
│   ├── news_sentiment.py            # RSS + Reddit + NLP sentiment
│   ├── meta_learner.py              # ML/Sentiment/RL ensemble
│   ├── rl_agent.py                  # Deep Q-Network reinforcement learning
│   ├── technical_indicators.py      # 25+ indicators (SMA, EMA, RSI, MACD, Bollinger, ATR...)
│   ├── regime_detector.py           # 4-regime market classification
│   ├── circuit_breaker.py           # Trading pause on loss streaks / drawdown
│   ├── correlation_tracker.py       # Pearson correlation to prevent over-concentration
│   ├── position_sizer.py            # Kelly Criterion position sizing
│   ├── execution_model.py           # Slippage, partial fills, funding costs
│   ├── kraken_futures_ws.py         # Kraken WebSocket for real-time futures prices
│   ├── transparency.py              # Trade decision audit logging
│   ├── alerting.py                  # Discord/webhook alerts
│   └── ...                          # Additional utilities
│
├── data/
│   ├── trades.csv                   # All closed trades (timestamps, P/L, reasons)
│   ├── state/                       # Persisted state (positions, balances, models, caches)
│   └── models/                      # Trained ML model files (.joblib)
│
├── models/                          # GPU model weights (.pt)
├── logs/                            # Daily log files (trading_YYYYMMDD.log)
└── train_model.py                   # Offline model training script
```

---

## Configuration

All parameters are set in `core/trading_engine.py` (`TradingConfig`) and `utils/config.py`, with every value overridable via environment variables in `.env`.

**Key environment variables:**

```bash
# API Keys
ALPACA_API_KEY=...                    # Alpaca crypto + equities
ALPACA_API_SECRET=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
KRAKEN_API_KEY=...                    # Kraken futures
KRAKEN_API_SECRET=...

# Mode
PAPER_TRADING=true                    # true = simulated, false = real money
SIM_REALISM_PROFILE=strict            # off | normal | strict
DIRECTION_MODE=both                   # both | long_only | short_only

# Tuning (examples)
MIN_ML_CONFIDENCE=0.52
MAX_CORRELATION=0.60
CB_MAX_CONSECUTIVE_LOSSES=5
CB_DAILY_LOSS_LIMIT_PCT=-4.0
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional — falls back to CPU)

### Installation

```bash
cd cryptotrades
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Optional: GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Optional: NLP sentiment
pip install transformers feedparser
```

### Configuration

1. Copy or edit `.env` with your API keys (Alpaca for crypto spot, Kraken for futures)
2. All trading parameters have sensible defaults — see `utils/config.py`

### Running

```bash
python main.py
```

The bot acquires a single-instance lock (port 49632), loads configuration, and enters the main loop. It runs continuously until interrupted with `Ctrl+C`.

### Offline Model Training

```bash
python train_model.py
```

Fetches 365 days of historical data from CoinGecko for BTC, ETH, SOL, and ADA, trains a GradientBoosting model with walk-forward validation, and saves it to `models/`.

---

## Data & State Files

The bot persists all state to disk for crash recovery:

| File | Purpose |
|------|---------|
| `data/trades.csv` | Complete trade log (entry/exit prices, P/L, reasons) |
| `data/state/positions.json` | Current open positions |
| `data/state/paper_balances.json` | Spot/futures balances, peak equity, daily P/L |
| `data/state/price_history.json` | Last 500 prices per symbol (~8h at 1-min intervals) |
| `data/state/rl_agent.json` | DQN weights, replay buffer, per-coin stats |
| `data/state/rl_shadow_report.json` | Baseline vs RL portfolio comparison |
| `data/state/meta_learner.json` | Ensemble weights and model accuracy history |
| `data/state/scanner_probation.json` | Universe scanner new-symbol observation tracking |
| `logs/trading_YYYYMMDD.log` | Daily log files |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **ML** | scikit-learn (GradientBoosting), PyTorch (DQN + GPU neural net) |
| **NLP** | HuggingFace Transformers (DistilBERT) with keyword fallback |
| **Spot Data** | Alpaca Crypto API (`data.alpaca.markets/v1beta3/crypto/us`) |
| **Futures Data** | Kraken Futures REST + WebSocket (`futures.kraken.com`) |
| **Sentiment** | CoinGecko, alternative.me, blockchain.info, Kraken, floatrates.com |
| **News** | RSS (CoinDesk, Cointelegraph, The Block), Reddit JSON API |
| **Concurrency** | ThreadPoolExecutor (10 workers), HTTP connection pooling |
| **GPU** | CUDA (RTX 4080) via PyTorch — lazy-loaded to avoid OpenMP deadlock |
| **State** | JSON persistence with crash recovery + daily CSV trade log |

---

*Built for autonomous 24/7 operation. All trading is simulated unless explicitly configured for live execution.*
