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

### The Early Days (January–February)

Nothing worked at first. That's not an exaggeration:

- **CryptoBot launched with a 0% win rate.** Nine consecutive iterations of filter tuning had made the entry criteria so strict that the bot literally could not open a trade.
- **AlpacaBot lost 89% of its capital** ($44,700 from a $100,000 paper balance). The 18% win rate on put-side trades was catastrophic — 12 wins against 54 losses.
- **CallBuyer went 552 cycles without opening a single trade** because a max RSI filter of 70 silently killed every momentum candidate — the one thing a momentum bot looks for.
- **CryptoBot spawned 36+ zombie processes** because the watchdog checked for `python.exe` while the bot ran under `pythonw.exe`.

But every failure taught us something, and every fix made the system measurably better.

### Hundreds of Fix Cycles

This wasn't a "five-round" process. It was **hundreds of individual debug-fix-deploy-test cycles** over four months — sometimes a dozen in a single day. Tweak a threshold, deploy, watch the logs, see what breaks next, fix that, deploy again. The 60+ bug fixes in the codebase don't count the parameter adjustments, the config changes, the log format tweaks, the dozens of times a "fix" introduced a new problem that needed its own fix.

Some changes took 5 minutes. Some took an entire weekend of tracing through log files, state files, API responses, and trade CSVs to find the root cause. The MLEG double-execution bug alone required building two recovery tools, a position rebuilder, and rewriting the close logic — then testing it across multiple market sessions to confirm it actually worked.

Here are a few landmark moments from hundreds of iterations, grouped roughly by phase:

**January–February: Getting off the ground**

The first two months were mostly building — writing trading engines, ML models, API clients, risk managers, meta-learners, and the indicator suites. But every component needed its own rounds of debugging before it could even talk to the others. CryptoBot went through 9 consecutive iterations just on entry filter tuning before it could open its first trade. CallBuyer's buy/sell functions crashed on launch because of a wrong parameter name (`contract_symbol` vs `option_symbol`). AlpacaBot's 4-layer ensemble was the most complex architecture but also had the most integration bugs — state files overwriting each other, warmup functions calling methods that didn't exist.

**March: The breaking-and-fixing month**

March was where the real debugging happened. Every day was a new discovery:

- PutSeller's allocation wasn't loading from `.env` → fixed dotenv load order → then discovered the meta-learner was loading stale state that overrode streak counts → rewrote `_load_state()` → then found MLEG orders were double-executing → built cancel-before-fallback logic → then realized `position_intent` was missing from the SDK calls → rewrote close logic with raw HTTP → then 14 stop-losses cascaded in 8 minutes → built position recovery tools → then a PowerShell BOM character corrupted the recovered state file → switched to BOM-free UTF-8 writes. That was one bot over about two weeks.
- CryptoBot's watchdog was spawning zombie processes → fixed PID detection → then found the port lock was letting zombies rebind → removed `SO_REUSEADDR` → then HOLD_DECAY was exiting 40% of trades prematurely → tuned threshold → then tuned again → then tuned a third time → then discovered it was still the #1 exit reason after 627 trades → tuned the PnL floor → then extended max hold times → then found the ML model had a feature mismatch after pruning → added retrain trigger.
- CallBuyer sat at zero trades for 552 cycles. Found the RSI cap. Raised it. Still no trades. Found `get_bars(days=60)` was returning below the 50-bar minimum. Fixed it. Immediately found 33 candidates. Opened 2 trades. Then the meta-learner blocked all subsequent trades because the confidence floor was too high during warmup. Lowered it. Then the bid fallback referenced a key that didn't exist. Fixed it. Every fix unlocked the next bottleneck.

**Late March–Early April: Stabilization**

This was the phase where things started actually working — but "working" still meant daily monitoring and adjustments:

- CryptoBot's post-overhaul run: 62 trades, +$18.40 — first sustained profitability after 1,300+ trades in the red
- Capital allocation across bots: went from 115% (over-allocated) down to properly coordinated 35%+15%+0% with PORTFOLIO_MAX_PCT bumped to 50% so they'd stop blocking each other
- Direction mode, symbol health filters, regime detection thresholds, correlation limits — all tuned through repeated cycles of "deploy → watch 50 trades → analyze → adjust"
- Built the full multi-agent AI system (12 files, LangGraph DAG, 4 GPT agents), got it working, then optimized cost from $37/day down to $0.27/day, then disabled it to let the base system prove itself first

**April 10–11: The deep audit**

Three parallel code audits found 27 remaining issues. ChatGPT classified 16 as "fix now." All 16 were patched and deployed in a single session — futures data deduplication, atomic state writes, NaN feature leaks, earnings check gaps, timezone bugs, position_intent on API calls. The cleanest the codebase has ever been.

### Where It Stands Now

The bots are running on Oracle Cloud 24/7 with zero crashes and zero errors in the logs:

- **CryptoBot**: Cycle 62+ on the latest deployment, scanning 33 spot symbols and 4 futures contracts every 60 seconds, sentiment analysis live (Fear & Greed at 15/Extreme Fear), ML model retraining every 2 hours, RANGING regime detection active
- **PutSeller**: 14 tracked positions loaded cleanly, waiting for Monday's market open with proper earnings guards, leveraged ETF protection, and the double-execution fix in place
- **CallBuyer**: Running clean cycles with timezone-aware market detection, cross-platform paths, and the fixed bid fallback logic ready for Monday

The system went from "nothing works" to "everything trades" to "trades profitably" over four months of continuous iteration. Each bug found made the system more robust. Each audit cycle caught things the previous one missed. The codebase today has 60+ specific bug fixes baked in — every one of them earned the hard way.

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
- **60+ bugs found and fixed** across hundreds of debug-fix-deploy-test cycles
- **Hundreds of iterations** — parameter tuning, config changes, log analysis, threshold adjustments, deploy-and-watch cycles over 4 months
- **12-file multi-agent AI system** built on LangGraph with cost optimizations reducing API spend from $37/day to $0.27/day
- **Deployed on Oracle Cloud** with systemd services, automatic restart, and watchdog monitoring

Four months. Hundreds of cycles. Every iteration better than the last.

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
