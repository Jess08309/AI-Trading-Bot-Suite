<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Alpaca-Paper%20Trading-FFD700?logo=alpaca&logoColor=black" />
  <img src="https://img.shields.io/badge/Oracle%20Cloud-24%2F7-F80000?logo=oracle&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-ML%20%2B%20LangGraph-blueviolet" />
  <img src="https://img.shields.io/badge/Status-Live%20(Paper)-brightgreen" />
</p>

# 🤖 AI Trading Bot Suite

**Four autonomous trading bots running 24/7 on Oracle Cloud — crypto, options spreads, and momentum calls — powered by machine learning, multi-agent AI, and adaptive risk management.**

> *One codebase. Four strategies. Zero manual intervention.*

---

## 📊 The Bots at a Glance

| Bot | Strategy | Markets | Trade Frequency | AI Stack |
|-----|----------|---------|-----------------|----------|
| **CryptoBot** | ML momentum + trend | 18 spot + 8 futures | 24/7, ~30-60 trades/day | GBM ensemble, LangGraph multi-agent, NLP sentiment |
| **PutSeller** | Credit spreads (iron condors) | 225+ stocks/ETFs | Market hours, ~2-5 spreads/day | ML qualification, meta-learner |
| **CallBuyer** | Momentum call buying | 149 stocks | Market hours, ~1-3 calls/day | 14-indicator scoring + ML ensemble |
| **AlpacaBot** | Options scalping (disabled) | 301 stocks | — | 4-layer AI ensemble (on pause) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Oracle Cloud (Ubuntu 24.04)                  │
│                    2 CPU · 15 GB RAM · 24/7                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  CryptoBot   │  │  PutSeller   │  │  CallBuyer   │          │
│  │  (15K lines) │  │  (5.4K lines)│  │  (4.4K lines)│          │
│  │              │  │              │  │              │          │
│  │ Spot+Futures │  │ Bull Puts +  │  │  ITM Calls   │          │
│  │ Long & Short │  │ Bear Calls   │  │  Breakouts   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌──────────────────────────────────────────────────┐          │
│  │              Alpaca Brokerage API                 │          │
│  │         Paper Trading · REST + Websocket          │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │          Shared Risk Management Layer             │          │
│  │  Capital allocation · Circuit breakers · Limits   │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Watchdog    │  │  Dashboards  │  │   Systemd    │          │
│  │  Auto-restart │  │  Flask UIs   │  │   Services   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔥 CryptoBot — The Flagship (15,000+ lines)

The most sophisticated bot in the suite. Trades both spot and perpetual futures on crypto markets around the clock.

### How It Works
1. **Every 60 seconds**: Fetches live prices for 26 symbols (18 spot + 8 futures)
2. **ML Signal Generation**: A Gradient Boosting model trained on 15 technical indicators generates directional predictions with confidence scores
3. **Multi-Agent Validation** *(optional)*: A LangGraph-orchestrated team of AI agents (Technical Analyst, Sentiment Analyst, Risk Manager) reviews each signal
4. **Regime Detection**: Classifies the market as TRENDING, RANGING, or HIGH_VOLATILITY and adapts stop-losses, position sizing, and hold times accordingly
5. **Execution**: Opens long or short positions with adaptive stops (ATR-based + regime-adjusted)
6. **Exit Management**: Six exit strategies compete — stop loss, take profit, trailing stop, hold decay, time-based, and regime shift

### Key Features
- **Walk-Forward ML Training**: 3-fold time-series cross-validation prevents overfitting. Models auto-rejected below 55% accuracy
- **NLP Sentiment**: Scans crypto news/social feeds for directional bias
- **Symbol Health Filter**: Auto-excludes coins with poor recent profit factor
- **Correlation Guard**: Limits exposure to highly-correlated positions
- **Atomic State Saves**: All position and balance data written via temp file + rename — crash-safe

### Tech Stack
`Python 3.11` · `scikit-learn` · `LangGraph` · `GPT-4o-mini` · `NumPy` · `Alpaca Crypto API`

---

## 💰 PutSeller — The Income Machine (5,400+ lines)

Sells credit spreads (bull put spreads + bear call spreads) to collect premium from time decay. The most consistent earner when markets cooperate.

### How It Works
1. **Every 15 minutes**: Scans 225+ qualified stocks/ETFs for spread opportunities
2. **BEST Entry Logic**: Checks HV20 (historical volatility), IV/HV ratio, and VWAP delta to find statistically favorable entries
3. **Chain Analysis**: Walks the options chain to find strikes with optimal delta (0.15-0.30), adequate spread width, and target credit
4. **Iron Condor Construction**: Opens both a bull put spread AND a bear call spread on qualifying symbols — profit from range-bound movement
5. **Risk Controls**: Max 12 put spreads + 8 call spreads, per-underlying limits, leveraged ETF guardrails, earnings avoidance

### Key Features
- **Earnings Guard**: Automatically skips any symbol with earnings within the DTE window
- **Leveraged ETF Protection**: 30+ leveraged ETFs get reduced quantity (1 contract max) and wider OTM strikes
- **MLEG + Fallback Close**: Tries multi-leg close first, falls back to individual legs with proper `position_intent` (buy_to_close / sell_to_close)
- **Meta-Learner**: Adapts confidence thresholds based on recent win/loss streaks

### Tech Stack
`Python 3.11` · `Alpaca Trading API` · `Options Chain API` · `NumPy` · `scikit-learn`

---

## 📈 CallBuyer — The Momentum Hunter (4,400+ lines)

Buys in-the-money call options on stocks showing strong momentum breakouts. High risk, high reward — the aggressive leg of the portfolio.

### How It Works
1. **Every 10 minutes**: Scans watchlist for momentum setups using a 14-indicator scoring system
2. **Feature Engine**: Computes RSI, MACD, Bollinger %B, ATR, volume surge, sector momentum, VWAP distance, and more
3. **ML Scoring**: Gradient Boosting model assigns a probability score; meta-learner adjusts the threshold
4. **Contract Selection**: Finds the best ITM call option (delta > 0.60) with adequate volume and tight spread
5. **Exit Management**: Take profit, stop loss, trailing stop from high water mark, and DTE-based forced exit

### Key Features
- **Timezone-Aware**: All market timing uses Eastern Time (works on UTC cloud servers)
- **Cross-Platform Portfolio Check**: Monitors aggregate exposure across all three options bots
- **Pre-Market Warmup**: Loads and trains ML model 30 minutes before market open

### Tech Stack
`Python 3.11` · `Alpaca Trading API` · `scikit-learn` · `NumPy`

---

## 🧠 AlpacaBot — The Veteran (Currently Paused)

Originally the most ambitious bot with a 4-layer AI ensemble for options scalping. Currently paused at 0% allocation after a drawdown during testing, but the architecture remains for future use.

### Architecture
- **Layer 1**: 14 technical indicators → feature vector
- **Layer 2**: Gradient Boosting ML model
- **Layer 3**: Rule-based meta-learner with adaptive thresholds
- **Layer 4**: Walk-forward backtester qualification gate

### Why It's Paused
During paper testing, put-side trades showed a catastrophic 18% win rate (12W/54L). Rather than continue losing paper money, the allocation was zeroed out. The code is fully functional and ready to re-enable with tuned parameters.

---

## 🛡️ Risk Management

Every bot shares a layered risk management system:

| Layer | What It Does |
|-------|-------------|
| **Circuit Breaker** | Halts trading after daily loss limit (-3%) or consecutive losses (5+) |
| **Position Limits** | Per-symbol, per-strategy, and portfolio-wide caps |
| **Capital Allocation** | PutSeller 35% · CallBuyer 15% · AlpacaBot 0% · CryptoBot separate balance |
| **Risk Utilization Cap** | 85% max portfolio risk — blocks new trades when exceeded |
| **Leveraged ETF Guard** | Special limits for 3x ETFs (TQQQ, SOXL, etc.) |
| **Earnings Blackout** | No options trades on symbols with upcoming earnings |
| **Correlation Guard** | CryptoBot limits exposure to correlated coins |

---

## 🚀 Deployment

All bots run as `systemd` services on Oracle Cloud with automatic restart:

```bash
# Service management (on server)
sudo systemctl status cryptobot putseller callbuyer
sudo systemctl restart cryptobot

# Logs
sudo journalctl -u cryptobot -f --no-pager
sudo journalctl -u putseller --since "1 hour ago"
```

### Local Development (Windows)
```bash
# Each bot has its own venv
cd CryptoBot && python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env  # Add your Alpaca API keys
python main.py
```

---

## 📁 Project Structure

```
TradingBots/
├── CryptoBot/              # 24/7 crypto spot + futures
│   ├── cryptotrades/       # Core engine
│   │   ├── core/          # Trading engine, ML model, indicators
│   │   ├── utils/         # 21 utility modules
│   │   └── tests/         # Unit tests
│   ├── agents/            # LangGraph multi-agent system
│   ├── tools/             # Analysis & audit scripts
│   ├── deploy/            # Oracle Cloud deployment configs
│   ├── monitoring/        # Prometheus + Docker monitoring
│   └── docs/              # Architecture documentation
│
├── PutSeller/              # Credit spread income strategy
│   ├── core/              # Put engine, API client, risk manager
│   ├── tools/             # Position recovery utilities
│   └── tests/             # Critical path tests
│
├── CallBuyer/              # Momentum call buying
│   ├── core/              # Call engine, feature engine, ML
│   └── tests/             # Critical path tests
│
├── AlpacaBot/              # Options scalping (paused)
│   ├── core/              # Trading engine, ML, meta-learner
│   ├── tools/             # 17 analysis & backtest tools
│   └── tests/             # Critical path tests
│
└── README.md               # You are here
```

---

## 🔧 Configuration

Each bot uses a `.env` file for configuration. See `.env.example` in each bot's directory for the template.

**Required API Keys:**
- [Alpaca](https://alpaca.markets/) — Brokerage API for stocks, options, and crypto
- [OpenAI](https://openai.com/) — Only needed if enabling CryptoBot's multi-agent system

---

## 📊 Performance Tracking

Each bot maintains:
- **Trade CSV**: Every closed trade logged with entry/exit prices, PnL, hold time, and exit reason
- **Meta-Learner State**: Adaptive thresholds that evolve with win/loss streaks
- **ML Model Versions**: Timestamped model snapshots with accuracy scores
- **Audit Reports**: Automated weekly deep audits (CryptoBot) with HTML reports

---

## ⚠️ Disclaimer

This is a **paper trading** system built for educational and research purposes. It is not financial advice. The bots trade with simulated money on Alpaca's paper trading environment. Past simulated performance does not guarantee future results.

---

## 👤 Author

Built by [Jess08309](https://github.com/Jess08309) — a solo developer exploring the intersection of AI, machine learning, and algorithmic trading.

---

<p align="center">
  <i>Four bots. Three markets. One goal: let the machines trade while you sleep.</i>
</p>
