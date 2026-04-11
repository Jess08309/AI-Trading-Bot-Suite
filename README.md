<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Alpaca-Paper%20Trading-FFD700?logo=alpaca&logoColor=black" />
  <img src="https://img.shields.io/badge/Oracle%20Cloud-24%2F7-F80000?logo=oracle&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-ML%20%2B%20LangGraph-blueviolet" />
  <img src="https://img.shields.io/badge/Status-Live%20(Paper)-brightgreen" />
  <img src="https://img.shields.io/badge/Lines%20of%20Code-53%2C900+-informational" />
</p>

# 🤖 AI Trading Bot Suite

**Four autonomous trading bots running 24/7 on Oracle Cloud — crypto, options spreads, and momentum calls — powered by machine learning, multi-agent AI, and adaptive risk management.**

**53,900+ lines of code · 234 files · 169 Python modules · 4 months of development**

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

### 🧠 The Multi-Agent AI System (LangGraph)

CryptoBot includes a full **multi-agent AI system** built on [LangGraph](https://github.com/langchain-ai/langgraph) — a team of 4 specialized GPT-4o-mini agents that validate every trade signal before execution:

```
┌─────────────────────────────────────────────────────┐
│                  ML Signal Generated                 │
│              "BUY BTC/USD, confidence 0.72"          │
└──────────────────────┬──────────────────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │     LangGraph StateGraph     │
        │      (Parallel → Sequential) │
        └──────┬───────────────┬───────┘
               ▼               ▼
   ┌──────────────────┐ ┌──────────────────┐
   │ Technical Analyst │ │ Sentiment Analyst│
   │                   │ │                  │
   │ Reviews indicators│ │ Scans crypto news│
   │ Confirms signal   │ │ & social feeds   │
   │ direction matches │ │ for directional  │
   │ chart structure   │ │ bias / red flags  │
   └────────┬─────────┘ └────────┬─────────┘
            └────────┬───────────┘
                     ▼
          ┌──────────────────┐
          │   Risk Manager   │
          │                  │
          │ Checks portfolio │
          │ exposure, sizing │
          │ & correlation    │
          └────────┬─────────┘
                   ▼
          ┌──────────────────┐
          │   Orchestrator   │
          │                  │
          │ Combines all 3   │
          │ opinions → final │
          │ GO / NO-GO       │
          └──────────────────┘
```

| Agent | Model | Purpose |
|-------|-------|---------|
| **Technical Analyst** | GPT-4o-mini | Validates that indicators (RSI, MACD, Bollinger, ATR) support the ML signal direction |
| **Sentiment Analyst** | GPT-4o-mini | Scans crypto news and social media for sentiment that could override technical signals |
| **Risk Manager** | GPT-4o-mini | Reviews portfolio exposure, position correlation, and sizing before approving |
| **Orchestrator** | GPT-4o-mini | Weighs all three opinions and makes the final trade/no-trade decision with a confidence score |

**Why only CryptoBot has this (not the options bots):**
- CryptoBot trades **24/7 with 30-60 trades per day** — enough volume to justify AI validation on every signal
- Options bots trade **2-5 times per day** during market hours. Their edge comes from math (time decay, fixed-risk spreads), not directional prediction — an AI committee adds cost without proportional benefit
- At low trade frequency, the ML model + meta-learner + rule-based scoring already filters well enough

**Current status: DISABLED** — The agent system was originally costing **~$37/day (~$1,100/month)** in OpenAI API calls. Every 60-second trade cycle hit the API with 4 agent calls, 24/7 — that's 1,440 cycles × 4 agents × ~500-1000 tokens each = millions of tokens per day. On a paper trading account making ~$18/week, spending $259/week on API calls didn't make sense.

**Cost optimizations we built before disabling:**

| Optimization | What It Does | Savings |
|---|---|---|
| **Rate Limiting** | Hard 60-second minimum between API calls (`advisor.py`). Instead of calling OpenAI on every signal (10-100/min), it calls once per minute max | ~98% fewer API calls |
| **GPT-4o-mini** | All 4 agents use `gpt-4o-mini` ($0.15/1M tokens) instead of GPT-4o ($2.50/1M tokens) | 15x cheaper per call |
| **External Data Caching** | All free API data (CoinGecko, Binance funding rates, Reddit sentiment, Fear & Greed index) cached in memory with 5-30 min TTLs — no repeated calls for the same data | $0 for market data |
| **Token Caps** | Each agent limited to 512 max tokens per response (orchestrator gets 2048 for multi-symbol handling). Keeps responses short and focused | ~50% token reduction |
| **Stale Cache Fallback** | If an external API fails, returns the last cached response instead of erroring out. Data quality report flags feeds as GOOD/DEGRADED/POOR so agents weight accordingly | Zero downtime |

**With all optimizations: ~$0.27/day (~$100/year)** — down from the original $37/day. The system is fully production-ready for minimal-cost operation.

The code is fully intact in `CryptoBot/agents/` (12 files), and `memory.json` continues recording trade outcomes for future training. To re-enable: set `AGENT_ENABLED=true` in `.env` — the cost optimizations are already baked in.

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

## 🔨 How This Got Built — The 4-Month Journey

This project started in January 2026 as a single crypto trading script and evolved into a 53,900-line, four-bot system over four months of continuous development. It was not a smooth ride.

### The Team

This entire codebase was built through a collaboration between a human developer ([Jess08309](https://github.com/Jess08309)), **GitHub Copilot** (primary coding agent — architecture, implementation, debugging, deployment), and **ChatGPT** (strategic advisor — code review, root cause analysis, second opinions on fixes). The workflow looked like this:

1. **Copilot** writes the code, runs the audits, and deploys to production
2. **ChatGPT** reviews the findings, classifies severity, and sanity-checks the fix approach
3. **Human** makes the final call, tests in paper trading, and decides what ships

This three-way verification process caught bugs that no single agent would have found alone. When Copilot audited the codebase and found 27 issues, ChatGPT independently classified 16 as "fix now" and 11 as "wait for data" — preventing both under-fixing and over-engineering.

### The Rocky Start

The early months were rough. Some highlights from the failure log:

- **CryptoBot launched with a 0% win rate.** Nine consecutive iterations of filter tuning had made the entry criteria so strict that the bot literally could not open a trade. When it finally did trade, HOLD_DECAY exits were killing 77% of positions before they had time to profit.
- **AlpacaBot lost 89% of its capital** ($44,700 from a $100,000 paper balance) before puts were disabled. The 18% win rate on put-side trades was catastrophic — 12 wins against 54 losses.
- **PutSeller consumed 100% of buying power** because an environment variable wasn't loading (Python read the config at class definition time, before `.env` was parsed). It hit 106.7% risk utilization and exhausted all buying power at $318.
- **CallBuyer went 552 cycles without opening a single trade** because a max RSI filter of 70 silently killed every momentum candidate. The one thing a momentum bot looks for — high RSI — was the thing blocking it.
- **CryptoBot spawned 36+ zombie processes** because the watchdog was checking for `python.exe` while the bot ran under `pythonw.exe`, and a broken port lock with `SO_REUSEADDR` let zombies rebind the same socket.
- **PutSeller's MLEG orders created reversed positions** — when a multi-leg close timed out but filled later, the individual-leg fallback also executed, flipping credit spreads into debit spreads. 14 stop-losses fired in 8 minutes, losing $2,488 in a single cascade.
- **A PowerShell BOM character** (3 invisible bytes: `EF BB BF`) corrupted a JSON state file, causing Python's `json.load()` to fail silently and lose all tracked positions.

### The Debugging Process

Every fix followed the same pattern:

1. **Audit** — Copilot scans the full codebase, cross-referencing logs, state files, and trade CSVs
2. **Triage** — ChatGPT reviews the findings and classifies: *fix now (code bug)* vs. *wait for data (parameter tuning)*
3. **Implement** — Copilot writes the fix with minimal scope — no refactoring, no feature creep
4. **Deploy** — SCP the fixed files to Oracle Cloud, restart the systemd service, verify clean startup in `journalctl`
5. **Monitor** — Wait for live cycles to confirm the fix works under real market conditions

The discipline of separating "broken code" from "needs tuning" was critical. After a paper account reset on March 29, the rule became: only fix bugs that prevent the code from functioning. Parameter tuning requires 1-2 weeks of clean data first.

### By the Numbers

| Bot | Files | Lines of Code |
|-----|-------|---------------|
| CryptoBot | 145 | 29,149 |
| AlpacaBot | 50 | 14,697 |
| PutSeller | 21 | 5,507 |
| CallBuyer | 16 | 4,239 |
| **Total** | **234** | **53,903** |

- **169 Python modules** across ML models, trading engines, risk managers, meta-learners, API clients, and utility libraries
- **60+ bugs found and fixed** across multiple audit cycles
- **12-file multi-agent AI system** built on LangGraph with cost optimizations reducing API spend from $37/day to $0.27/day
- **Deployed on Oracle Cloud** with systemd services, automatic restart, and watchdog monitoring

Four months of building, breaking, debugging, and rebuilding — and the bots are still running.

---

## ⚠️ Disclaimer

This is a **paper trading** system built for educational and research purposes. It is not financial advice. The bots trade with simulated money on Alpaca's paper trading environment. Past simulated performance does not guarantee future results.

---

## 👤 Author

Built by [Jess08309](https://github.com/Jess08309) with **GitHub Copilot** and **ChatGPT** — a human-AI collaboration exploring the intersection of machine learning, autonomous agents, and algorithmic trading.

---

<p align="center">
  <i>53,900 lines of code. Four bots. Three markets. Four months. One goal: let the machines trade while you sleep.</i>
</p>
