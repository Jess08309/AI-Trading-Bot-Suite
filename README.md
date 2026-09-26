<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/Alpaca-Paper%20Trading-FFD700?logo=alpaca&logoColor=black" alt="Alpaca Paper Trading" />
  <img src="https://img.shields.io/badge/Status-Paper%20Trading%20%26%20Research-blue" alt="Status: Paper Trading & Research" />
  <img src="https://img.shields.io/badge/QuantConnect-LEAN%20Research-F38B00?logo=quantconnect&logoColor=white" alt="QuantConnect LEAN Research" />
  <img src="https://img.shields.io/badge/Live%20Orders-Disabled%20(Safety%20Default)-red" alt="Live Orders Disabled" />
</p>

# 🤖 AI Trading Bot Suite

**A multi-strategy automated algorithmic trading and quantitative research platform covering options credit spreads, momentum calls, options scalping, and cryptocurrency trend/momentum strategies.**

> **Current Status (as of September 6, 2026):** All bots in this repository operate strictly in **paper/demo trading** and **QuantConnect research/backtesting** modes. **Live trading is NOT enabled.** Backtests, simulations, and paper trading results are research tools and do not guarantee future profitability.

---

## 📌 Table of Contents

1. [Project Purpose & Architecture](#project-purpose)
2. [Current Status & Research Context](#current-status)
3. [Safety Posture & Operational Guardrails](#safety-posture)
4. [The Bots at a Glance](#bots-at-a-glance)
5. [PutSeller & QuantConnect Research (Primary Focus)](#putseller-research)
6. [Strategy Deep Dives](#strategy-deep-dives)
   - [PutSeller (Credit Spreads & Iron Condors)](#putseller-details)
   - [CallBuyer (Momentum ITM Calls)](#callbuyer-details)
   - [CryptoBot (Spot & Futures Momentum)](#cryptobot-details)
   - [AlpacaBot (Options Scalping - Paused)](#alpacabot-details)
7. [Operational Safety & Execution Engineering](#safety-engineering)
8. [Backtesting Methodology & Limitations](#backtesting-limitations)
9. [Development Roadmap](#development-roadmap)
10. [Repository Navigation](#repository-navigation)
11. [Local Setup & Deployment](#local-setup)
12. [How This Got Built — The Development Journey](#development-journey)
13. [⚠️ Disclaimer](#disclaimer)
14. [👤 Author & Credits](#author-credits)

---

<a id="project-purpose"></a>
## 🎯 Project Purpose & Architecture

The **AI Trading Bot Suite** is an open-source quantitative trading and algorithmic research platform designed to evaluate, backtest, and paper-trade multiple algorithmic strategies across US equities, equity options, and digital assets.

Rather than relying on a single monolithic strategy, the suite employs a modular **four-bot architecture** where each bot targets a distinct market dynamic:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           System & Execution Architecture                       │
│                     Cloud VM (Ubuntu 24.04) / Local Environment                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐               │
│   │   CryptoBot    │    │   PutSeller    │    │   CallBuyer    │               │
│   │ 24/7 Momentum  │    │ Credit Spreads │    │ Momentum Calls │               │
│   │ Spot & Futures │    │  Iron Condors  │    │  ITM Breakouts │               │
│   └───────┬────────┘    └───────┬────────┘    └───────┬────────┘               │
│           │                     │                     │                         │
│           ▼                     ▼                     ▼                         │
│   ┌────────────────────────────────────────────────────────────┐                │
│   │                 Cross-Bot Risk & Safety Layer              │                │
│   │  • Coordinated capital allocation (35% / 15% / 0%)         │                │
│   │  • Sibling-bot position ownership verification             │                │
│   │  • Daily loss circuit breakers & max utilization caps      │                │
│   │  • One-sided quote filters & pre-order safety gates        │                │
│   └─────────────────────────────┬──────────────────────────────┘                │
│                                 │                                               │
│                                 ▼                                               │
│   ┌────────────────────────────────────────────────────────────┐                │
│   │                   Alpaca Brokerage API                     │                │
│   │              Paper Trading (REST + WebSockets)             │                │
│   └────────────────────────────────────────────────────────────┘                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                         Research & Simulation Engine                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────────────────────┐     ┌────────────────────────────────────┐   │
│   │     QuantConnect / LEAN      │     │            Backtest Lab            │   │
│   │ • Rules-only mechanical port │     │ • Local event-driven simulations   │   │
│   │ • Minute-resolution options  │     │ • Black-Scholes pricing models     │   │
│   │ • Separated tuning & holdout │     │ • Historical option chain replay   │   │
│   └──────────────────────────────┘     └────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

The core architectural tenet of this project is that **operational safety, execution realism, state reconciliation, and risk controls precede capital allocation**.

---

<a id="current-status"></a>
## 🚦 Current Status & Research Context

*Status current as of September 6, 2026:*

- **Paper / Demo Trading Phase:** All trading bots are configured exclusively for paper and demo environments. No real capital is at risk.
- **QuantConnect LEAN Research:** Core strategy logic is being evaluated in QuantConnect using high-resolution historical option chain data to evaluate underlying statistical properties without reliance on heuristic market filters.
- **Ongoing Validation:** The system is continuously tested against paper execution data to diagnose discrepancies between simulated backtest fills and real-world order book dynamics.
- **No Profitability Representation:** Neither historical backtests nor paper trading returns are presented as proof of profitability. The platform is an evolving research project.

---

<a id="safety-posture"></a>
## 🛡️ Safety Posture & Operational Guardrails

Automated options and multi-asset trading presents severe operational risks if execution and risk boundaries are not strictly enforced. The suite maintains a rigorous safety posture:

1. **Live Orders Disabled:** The system is explicitly configured with live trading disabled across all modules and scripts.
2. **Credential Confidentiality:** API keys, secret keys, and webhook URLs must remain strictly private in local `.env` files and must never be committed to source control.
3. **Controlled Live Gate:** Live deployment is not a casual toggle. Transitioning any component to live trading requires:
   - Statistically meaningful sample sizes of completed trades.
   - Out-of-sample validation across varying market volatility regimes.
   - Rigorous operational error tracking with zero unhandled exceptions.
   - Explicit manual review and authorization.
4. **Defensive Defaults:** If market quotes become stale, API connections disconnect, or unexpected order states occur, the execution engine enters a defensive hold/reconcile state rather than attempting unhedged market orders.

---

<a id="bots-at-a-glance"></a>
## 📊 The Bots at a Glance

| Bot | Strategy | Target Markets | Trade Frequency | Technology Stack | Capital Allocation |
|-----|----------|----------------|-----------------|------------------|--------------------|
| **PutSeller** | Defined-risk credit spreads (bull puts, bear calls, iron condors) | 225+ US Equities & ETFs | 15-min scan (market hours) | Python 3.11, QuantConnect LEAN, Alpaca API, NumPy, scikit-learn | 35% (Paper) |
| **CallBuyer** | Momentum breakouts on ITM calls | 149 High-Volume Stocks | 10-min scan (market hours) | Python 3.11, 14-indicator feature engine, Gradient Boosting, Alpaca API | 15% (Paper) |
| **CryptoBot** | ML momentum + trend following | 18 Spot + 8 Perpetual Futures | 60-second loop (24/7) | Python 3.11, scikit-learn, LangGraph (optional multi-agent), Alpaca Crypto API | Separate Balance |
| **AlpacaBot** | Options scalping (4-layer ML ensemble) | 301 US Equities | High frequency (market hours) | Python 3.11, 4-layer ML ensemble, walk-forward validator | 0% (Paused) |

---

<a id="putseller-research"></a>
## 🔬 PutSeller & QuantConnect Research (Primary Focus)

PutSeller is the primary research and validation focus of the suite. To evaluate whether credit spread mechanics offer a repeatable statistical edge independent of external ML models or complex sentiment indicators, PutSeller was ported to a standalone **rules-only implementation** on **QuantConnect / LEAN**.

### QuantConnect Port Architecture
- **Rules-Only Mechanical Skeleton:** The QuantConnect port (`PutSeller/quantconnect/main.py` and `lean_workspace/PutSeller_IronCondor/main.py`) focuses purely on systematic mechanical rules: watchlist selection, DTE targets, strike delta boundaries, credit thresholds, stop loss multipliers, profit targets, and holding duration rules.
- **Historical Minute-Resolution Data:** Runs on QuantConnect historical **minute-resolution equity and options data**, incorporating realistic bid/ask quotes and intraday volatility spikes.
- **Separated Data Windows:** To guard against overfitting and data snooping, research adheres to separated testing windows:
  - **Tuning Period (2021-01-01 to 2023-12-31):** Used for parameter sweeps, strike selection tuning, and exit rule calibration.
  - **Validation Period (2024-01-01 to 2026-08-01):** Evaluated strictly as out-of-sample data to measure parameter decay and robustness.
  - **Holdout Period:** Preserved for final out-of-sample verification.

### Current Intended Research Configuration
The current baseline configuration under evaluation in QuantConnect is:

```python
SHORT_DELTA_MAX       = 0.18    # Max delta for short strike selection (balances OTM buffer and premium)
TAKE_PROFIT_PCT       = 0.60    # Close position upon capturing 60% of initial theoretical credit
STOP_LOSS_MULT        = 1.40    # Trigger exit if current debit to close exceeds 1.4x initial credit
LIMIT_CREDIT_FRACTION = 0.85    # Entry limit order requires >= 85% of theoretical mid-price credit
# Limit Order Policy: Fixed credit threshold with stale cancellation; no decaying limit-order retry
```

> **Note on Limit-Order Retries:** Historical backtests demonstrated that relaxing limit orders via a decaying credit fraction ("walking the order down") significantly eroded net returns by accepting low-credit fills that failed to compensate for stop-loss tail risk. The algorithm strictly requires fixed credit quality or cancels the entry.

---

<a id="strategy-deep-dives"></a>
## 🔍 Strategy Deep Dives

<a id="putseller-details"></a>
### 💰 PutSeller — The Income Machine
- **Core Concept:** Premium collection through systematic sale of out-of-the-money (OTM) credit spreads (bull put spreads and bear call spreads) on highly liquid tickers (SPY, QQQ, IWM, AAPL, MSFT, NVDA, etc.).
- **DTE Window:** Targets 30–60 DTE (nominal target 45 DTE), exiting positions before expiration (e.g., at 21 DTE or upon hitting profit targets).
- **Position Limits:** Maximum 12 put spreads and 8 call spreads portfolio-wide, with a cap of 2 spreads per underlying.
- **Risk Protections:** Earnings avoidance (skips underlyings with earnings inside the DTE window) and leveraged ETF position caps (1 contract max on 3x instruments like TQQQ/SOXL).

<a id="callbuyer-details"></a>
### 📈 CallBuyer — The Momentum Hunter
- **Core Concept:** Directional breakout trading via long in-the-money (ITM) call options (delta > 0.60) on stocks exhibiting strong momentum surges.
- **Feature Pipeline:** 14 technical features (RSI, MACD histogram, Bollinger %B, ATR expansion, sector relative strength, VWAP distance, volume breakout) evaluated through a Gradient Boosting classifier.
- **Execution & Exits:** Uses dynamic take-profit thresholds, trailing stops from high-water marks, and mandatory DTE-based liquidations.

<a id="cryptobot-details"></a>
### 🔥 CryptoBot — The Flagship (Spot & Futures)
- **Core Concept:** 24/7 momentum and trend following across 18 spot pairs and 8 perpetual futures contracts.
- **Market Regime Detection:** Classifies market state into `TRENDING`, `RANGING`, or `HIGH_VOLATILITY`, adjusting ATR stops and hold times accordingly.
- **Multi-Agent AI Architecture (LangGraph):** Features a 4-agent validation team (Technical Analyst, Sentiment Analyst, Risk Manager, Orchestrator) using GPT-4o-mini.
- **Agent Status:** The multi-agent system has been fully cost-optimized (rate limiting, data caching, token caps reducing cost from ~$37/day to ~$0.27/day) and is currently paused (`AGENT_ENABLED=false`) while baseline ML models are evaluated independently.

<a id="alpacabot-details"></a>
### 🧠 AlpacaBot — The Veteran (Currently Paused)
- **Core Concept:** Multi-layer AI options scalper combining technical indicators, Gradient Boosting, a rule-based meta-learner, and walk-forward backtest qualification.
- **Status:** **Paused at 0% allocation.** During initial paper testing, put-side scalps suffered severe drawdown (18% win rate). The architecture and tooling are preserved in the codebase, but the bot is held inactive pending complete strategy redesign.

---

<a id="safety-engineering"></a>
## 🛡️ Operational Safety & Execution Engineering

Automated trading suites face complex edge cases when managing multi-leg derivative positions and multi-bot accounts. The codebase implements extensive defensive safeguards:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Operational Safety Framework                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Pre-Order Safety Gate      2. Multi-Leg Order Tracking                  │
│     • Real-time equity check      • Atomic state transitions                │
│     • Broker leg conflict check   • Deterministic client_order_id           │
│     • Two-sided quote filter      • Cancel-race fill detection              │
│                                                                             │
│  3. Orphan-Leg Protection      4. Broker Ground-Truth Reconciliation        │
│     • Exposed leg detection       • Periodic local vs broker state sync     │
│     • Safe close fallback         • Ghost position elimination              │
│     • Capped retries & alerts     • Persistent incident logging & audit     │
│                                                                             │
│  5. Cross-Bot Coordination     6. Circuit Breakers                          │
│     • Sibling-bot state checks    • Daily loss limit (-3%)                  │
│     • Non-overlapping ownership   • Consecutive loss halt (5 losses)        │
│     • Isolated allocation caps    • Exponential backoff on failed closes    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Multi-Leg Order Tracking:** Multi-leg entries (short and long legs) are tracked atomically. If one leg fills while the other fails or cancels, the system flags the position rather than silently dropping it.
- **Orphan-Leg Detection & Recovery:** If an unhedged leg is detected due to partial fills or execution disconnects, the engine triggers automated closing orders. If closing fails after 3 attempts, it halts retries and raises an escalation alert to prevent uncontrolled order loops.
- **State Reconciliation:** Periodic background reconciliation compares local position state files with broker ground truth (`get_option_positions`), cleaning up stale "ghost" positions and adopting untracked fills.
- **Persistent Incidents & Side Blocking:** Anomalies and one-leg incidents are recorded in persistent incident logs (`eng.incidents`). While an incident is open, new entries on that underlying and side are blocked.
- **Cross-Bot Ownership Coordination:** Sibling bots sharing an Alpaca account inspect one another’s state files (e.g., PutSeller checking `CallBuyer/data/state/positions.json`) to prevent accidentally modifying or adopting contracts owned by another strategy.
- **Quote-Quality Gates:** Quotes with zero bid or zero ask (`one_sided`) are rejected from midpoint calculations to prevent erroneous stop-loss or take-profit triggers caused by illiquid off-market ticks.

> **Risk Reality:** While these safety systems significantly reduce software, execution, and state errors, they **cannot eliminate financial risk**. Market gap openings, exchange halts, extreme illiquidity, and broker platform outages can still result in unexpected losses.

---

<a id="backtesting-limitations"></a>
## 📉 Backtesting Methodology & Limitations

Rigorous quantitative research requires acknowledging the inherent limitations of simulated backtests:

1. **Fill & Liquidity Realism:** Backtesting platforms typically assume limit orders fill whenever price touches the limit. In live markets, queue priority, quote fading, and thin market depth frequently result in partial fills, missed entries, or adverse slippage.
2. **Transaction Costs & Drag:** Options multi-leg strategies incur contract fees, exchange fees, and regulatory costs on every leg. High turnover can rapidly turn a gross-profitable backtest into a net-negative strategy after fees.
3. **QuantConnect Date Limits:** On QuantConnect Community-tier accounts, certain datasets enforce a rolling 90-day out-of-sample cutoff from the current real date, limiting recent historical backtest ranges.
4. **Raw Orders vs. Completed Round Trips:** Quantitative analysis must distinguish between raw order count (individual leg submissions, cancellations, adjustments) and completed round trips (the matched open and close of an entire position). Headline order counts do not equal trade count.

---

<a id="development-roadmap"></a>
## 🗺️ Development Roadmap

- [x] **Four-Bot Architecture:** Unified repository structure with modular bot components and shared risk conventions.
- [x] **Execution Safety Hardening:** Multi-leg order tracking, orphan-leg handling, broker reconciliation, and cross-bot position guards.
- [x] **PutSeller QuantConnect Port:** Rules-only LEAN algorithm with minute-resolution options data.
- [ ] **Custom Metrics & Reporting Validation:** Finalize robust LEAN reporting for completed round-trip trade statistics, fee attribution, and quote diagnostics.
- [ ] **QuantConnect Parameter Sweeps (2021–2023):** Conduct systematic parameter optimization exclusively within the historical tuning period.
- [ ] **Validation Period Testing (2024–2026):** Execute validation runs on unseen data to test parameter persistence and robustness.
- [ ] **Paper Trading Reality Parity:** Continue live paper trading across active bots to benchmark paper fill rates against LEAN simulation models.
- [ ] **Allocation Validation:** Reassess capital allocation across all bots, including evaluating a potential strategy overhaul for AlpacaBot (currently at 0%).
- [ ] **Gated Live Deployment Review:** Define strict statistical readiness gates before any live capital execution is evaluated.

---

<a id="repository-navigation"></a>
## 📂 Repository Navigation

```
AI-Trading-Bot-Suite/
├── PutSeller/                  # Credit spread & iron condor strategy (Primary Focus)
│   ├── core/                   # Trading engine, API client, risk manager, feature engine
│   ├── quantconnect/           # Standalone LEAN cloud algorithm (main.py)
│   ├── tools/                  # Position recovery, orphan audit, and diagnostic scripts
│   ├── utils/                  # Logging and helper utilities
│   └── tests/                  # Unit tests (order safety, critical paths, state transitions)
│
├── CallBuyer/                  # Momentum ITM call buying strategy
│   ├── core/                   # Call engine, feature engine, ML model, risk manager
│   ├── quantconnect/           # QuantConnect LEAN implementation
│   ├── tools/                  # Position auditor and diagnostics
│   └── tests/                  # Unit tests for feature engine and trade logic
│
├── CryptoBot/                  # 24/7 Spot & Futures crypto momentum bot
│   ├── cryptotrades/           # Core trading engine, order execution, indicators
│   ├── agents/                 # LangGraph multi-agent AI validation system (4 agents)
│   ├── quantconnect/           # LEAN crypto algorithm port
│   ├── monitoring/             # Prometheus / Docker monitoring configuration
│   ├── tools/                  # Analysis, tuning, and orphan cleanup utilities
│   └── tests/                  # Unit tests and backtest harness tests
│
├── AlpacaBot/                  # 4-layer ML options scalping bot (Paused at 0% allocation)
│   ├── core/                   # Scalping engine, ML models, meta-learner
│   ├── quantconnect/           # LEAN scalping implementation
│   ├── tools/                  # Backtest and analysis tools
│   └── tests/                  # Unit and backtest bias tests
│
├── backtest_lab/               # Local backtesting harness
│   ├── engine.py / engine_v2.py# Event-driven local simulation engines
│   ├── pricing.py              # Black-Scholes and options Greeks calculations
│   └── alpaca_data.py          # Historical data ingestion and caching
│
├── lean_workspace/             # QuantConnect LEAN CLI local workspace
│   ├── PutSeller_IronCondor/   # LEAN workspace project for PutSeller
│   ├── CallBuyer_Momentum/     # LEAN workspace project for CallBuyer
│   ├── AlpacaBot_Scalp/        # LEAN workspace project for AlpacaBot
│   └── lean.json               # LEAN engine configuration
│
└── README.md                   # Repository documentation (You are here)
```

---

<a id="local-setup"></a>
## 💻 Local Setup & Deployment

### Prerequisites
- Python 3.11+
- [Alpaca Markets](https://alpaca.markets/) Paper Trading account (API Key + Secret)
- (Optional) [QuantConnect](https://www.quantconnect.com/) account for cloud LEAN backtesting

### Local Setup
Each bot is self-contained with its own dependencies and configuration:

```bash
# 1. Clone the repository
git clone https://github.com/Jess08309/AI-Trading-Bot-Suite.git
cd AI-Trading-Bot-Suite

# 2. Set up a virtual environment for a specific bot (e.g., PutSeller)
cd PutSeller
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and enter your Alpaca Paper Trading API keys

# 5. Run unit tests
python3 -m unittest discover -s tests
```

### Production Service Management (Cloud VM)
On a cloud server (e.g., Ubuntu on Oracle Cloud), the bots run as managed `systemd` background services:

```bash
# Check service status
sudo systemctl status putseller callbuyer cryptobot

# Stream live service logs
sudo journalctl -u putseller -f --no-pager
```

---

<a id="development-journey"></a>
## 🔨 How This Got Built — The Development Journey

This codebase began in January 2026 as a standalone crypto trading script and expanded over months into an integrated multi-strategy suite spanning over 53,000 lines of code across 230+ files.

### Collaborative Engineering Model
The system was engineered through a structured human-AI collaboration:
1. **Developer ([Jess08309](https://github.com/Jess08309)):** Product direction, strategy requirements, risk bounds, paper trading validation, and deployment oversight.
2. **Claude Opus:** Primary software engineering agent — architecting modules, writing trading logic, porting strategies to LEAN, and building regression test suites.
3. **ChatGPT:** Strategic quantitative advisor and peer reviewer — auditing logic, triaging defect severity, and evaluating algorithmic tradeoffs.

### Hardening Through Iteration
Real-world paper market data uncovered dozens of subtle edge cases that guided iterative hardening:
- **Order Race Conditions:** Discovered and fixed cases where canceled multi-leg orders filled during the cancellation window, building handlers to capture and track the resulting positions.
- **Quote Sanitization:** Identified that one-sided options quotes (zero bid/ask) produced distorted midpoints that caused erroneous stop-loss triggers, leading to strict two-sided quote gates.
- **Limit Order Realism:** Replaced decaying limit-order retries with fixed-credit requirements after backtesting proved that walking limit orders downward severely degraded net returns.
- **Cross-Bot Coordination:** Added sibling-bot state inspection to prevent collisions when multiple bots trade options within the same brokerage account.

---

<a id="disclaimer"></a>
## ⚠️ Disclaimer

**This software is experimental and intended strictly for educational, academic, and research purposes.**

- **Not Financial Advice:** Nothing in this repository constitutes financial, investment, legal, or tax advice.
- **Substantial Risk of Loss:** Algorithmic trading across equities, options, and cryptocurrencies involves significant risk of monetary loss. 
- **Simulations vs. Reality:** Past performance, historical backtests, and simulated paper trading results do not guarantee future profitability or performance.
- **No Warranty:** The software is provided "as is", without warranty of any kind. The authors and contributors assume no liability for any direct or indirect financial losses resulting from the use of this software.

---

<a id="author-credits"></a>
## 👤 Author & Credits

- **Author & Maintainer:** [Jess08309](https://github.com/Jess08309)
- **Collaborators:** Claude Opus & ChatGPT (Human-AI Quantitative Research Initiative)
