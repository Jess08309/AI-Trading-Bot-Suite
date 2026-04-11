# Automated Trading Bot System

> **Built from scratch by a solo developer** — ~56,000 lines of code across 200+ files, running 24/7 on a cloud server.

---

## What Is This?

This is a collection of automated trading bots that buy and sell investments around the clock without any human involvement. Think of them like robot traders that watch the markets, spot opportunities, make trades, and manage risk — all on their own.

The whole system runs on a free cloud server (a rented computer on the internet), so it works 24/7 even when my laptop is off.

Everything here — the trading logic, the AI models, the safety systems, the dashboard — was designed and coded by me from the ground up. No copy-paste from tutorials, no off-the-shelf trading libraries.

**Current status:** Running in paper trading mode (simulated money, real market data). Building a track record before committing real capital.

---

## The Four Bots

### 1. CryptoBot — The Crypto Trader
**What it does:** Buys and sells cryptocurrencies like Bitcoin, Ethereum, and Solana.

**How it works in plain English:**
- It watches about 33 different crypto coins at the same time
- Every few minutes, it runs each coin through a scoring system that looks at things like: Is the price trending up or down? How fast is it moving? Is trading volume high or low? Are other similar coins moving the same way?
- A machine learning model (a type of AI that learns from past data) gives each coin a confidence score — basically "how sure am I this trade will make money?"
- If the score is high enough, it places a trade
- Once in a trade, it constantly monitors for the right time to sell — either when it hits a profit target, a stop-loss (a safety limit to prevent big losses), or a trailing stop (a moving floor that locks in profits as the price goes up)
- It also reads crypto news and social media to gauge market sentiment — if everyone is panicking, it adjusts accordingly

**When it runs:** 24/7, because crypto markets never close.

---

### 2. PutSeller (aka IronCondor) — The Options Spread Trader
**What it does:** Sells options contracts to collect small, steady premiums — like being the house at a casino.

**How it works in plain English:**
- Options are contracts that give someone the right to buy or sell a stock at a certain price by a certain date. When you *sell* an option, someone pays you money upfront (called the "premium"), and you're betting the stock *won't* move past a specific price
- PutSeller uses a strategy called "credit spreads" — it simultaneously sells one option and buys another cheaper one as insurance, which limits how much it can lose on any single trade
- It scans through hundreds of stocks looking for ones that are calm enough (not about to have an earnings report or major event) and whose options are priced richly enough to be worth the trade
- It checks things like: How volatile has this stock been? Are option prices inflated right now? Is there enough cushion before we'd start losing money?
- Each trade has a built-in expiration date (usually about 35 days out), so positions naturally close themselves over time
- It runs both "bull put spreads" (betting a stock won't drop too far) and "bear call spreads" (betting a stock won't rise too far)

**When it runs:** During US stock market hours only (9:30 AM – 4:00 PM Eastern).

**Allocation:** 35% of the account — the largest single chunk, because this strategy is designed to be the steadiest earner.

---

### 3. CallBuyer — The Momentum Chaser
**What it does:** Buys call options on stocks that are surging upward — riding the wave of momentum.

**How it works in plain English:**
- A "call option" is a contract that lets you profit when a stock goes up, with limited downside (you can only lose what you paid for the contract)
- CallBuyer scans for stocks showing strong upward momentum — things like price acceleration, positive trend signals, and unusually high trading volume
- When it finds a strong candidate, it buys a call option with about 14 days until it expires
- A machine learning model and a "meta-learner" (a system that adapts its strategy based on recent wins and losses) both weigh in on the decision
- This is the most aggressive of the three stock-market bots — higher risk, higher potential reward

**When it runs:** During US stock market hours.

**Allocation:** 15% of the account — smallest chunk because it's the riskiest.

---

### 4. AlpacaBot — The Stock Trader (Currently Retired)
**What it did:** Traded stocks directly (not options) using a 14-indicator scoring system with a multi-layer AI ensemble.

**Current status:** Turned off after losing 89% of its allocated capital. The AI models weren't reliable enough for direct stock trading, and the position sizing was too aggressive. Rather than patching it, the capital was redirected to the options bots, which have more built-in risk protection. It still exists in the code and could be reactivated with a different approach.

---

## How the AI Works

Each bot uses **machine learning** — software that studies thousands of historical trades and price patterns to learn what setups tend to make money and which ones tend to lose.

### The "Wisdom of Crowds" Approach
Instead of one single AI making all decisions, the bots use an **ensemble approach** — multiple different models and rule-based checks that all "vote" on whether to take a trade. Think of it like asking 10 different experts for their opinion and only pulling the trigger when most of them agree.

The scoring considers things like:
- **Price patterns** — mathematical formulas applied to price and volume data (moving averages, momentum indicators, volatility measures, etc.)
- **Market mood** — is the overall market trending up, down, or sideways? The bots adjust their behavior based on this
- **Volatility** — how wild are price swings? Wilder markets mean wider safety nets and smaller bet sizes
- **Diversification** — are a bunch of similar investments all moving the same way? If so, don't pile into all of them (avoid putting all eggs in one basket)

### The Self-Adjusting System
Each bot has a "meta-learner" — a layer that watches the bot's recent performance and automatically adjusts how aggressive or conservative it should be:
- **On a winning streak?** It gets slightly more confident and takes more trades
- **On a losing streak?** It tightens up, becomes pickier, and might even pause new trades until conditions improve

### Quality Control: The Backtester
Before any stock is allowed into the trading universe, it has to pass a historical simulation — we replay how the bot's strategy *would have* performed on that stock over the last 3, 6, and 12 months. Only stocks that show consistent profitability across all three time windows make the cut. This eliminates stocks that just happened to look good recently but have a poor long-term track record.

---

## The Cloud Server

### Why Not Just Run on a Laptop?
Originally, all four bots ran on my laptop. The problems were obvious:
- Close the laptop? Bots stop
- Internet hiccup? Bots can't trade
- Windows update forces a restart? Any open positions are unmanaged
- Crypto trades 24/7, meaning the laptop could *literally never be turned off*

### The Solution: A Free Cloud Server
- **Provider:** Oracle Cloud (free tier — costs $0/month)
- **Location:** Phoenix, Arizona data center — professional-grade internet, backup power, climate control
- **Specs:** 2 processor cores, 15 GB memory, 42 GB storage, running Ubuntu Linux
- **Uptime:** Designed for 24/7 operation with automatic recovery from crashes

### The Safety Net: Auto-Restart Watchdog
A **watchdog service** checks every 2 minutes that all bots are alive and healthy. If one crashes, the watchdog automatically restarts it within seconds. Each bot also has a built-in lock that prevents two copies from accidentally running at the same time — because running two copies of the same bot leads to chaos (see the "Zombie Army" story below).

---

## The Journey: Problems Solved and Lessons Learned

Building this system has been a months-long process of writing code, finding problems, fixing them, and then finding new problems. Here are some of the biggest challenges:

### The Zero Win Rate Crisis
After CryptoBot's first 46 trades, it had **zero wins**. Every single trade lost money. The investigation revealed that a safety feature called "HOLD_DECAY" was programmed to close positions that weren't moving quickly enough. The problem? It was closing trades *right before* they would have hit the profit target. After relaxing this setting and making profit targets smaller (but actually achievable), the bot started winning within the first day of the fix.

### The $44,700 Meltdown
AlpacaBot (the stock trader) lost 89% of its capital — about $44,700 in simulated money. The AI models weren't reliable enough for direct stock trading, and the bet sizes were too large. Rather than throwing good money after bad trying to fix it, the bot was shut down entirely. Sometimes the right move is knowing when to walk away from an approach that isn't working.

### The Zombie Army
At one point, **36 copies of CryptoBot were running simultaneously**. The watchdog (the program that checks if bots are alive) had a bug where it couldn't detect that CryptoBot was already running, so it kept starting new copies. Each zombie was fighting over the same trades and writing to the same log files. Imagine 36 chefs all trying to cook in the same kitchen with the same ingredients at the same time. Fixed by improving the watchdog's detection and adding a lock that prevents more than one copy from starting.

### The Double-Execution Disaster
When PutSeller tried to close an options position, it would send the close order and wait 30 seconds for confirmation. When the confirmation didn't come in time, it assumed the order failed and sent a backup close through a different method. But the original order *eventually did go through* — meaning both orders executed, and the bot accidentally **opened new positions in the opposite direction**. Nine positions got flipped before anyone noticed. Fixed by making the bot cancel the original order before attempting the backup, and building a recovery tool to detect and undo reversed positions.

### The Lucky $13,310 Accident
A bug caused CallBuyer to keep buying more of the same Google (GOOGL) call option — 10 contracts instead of the intended limit. The position was huge and dangerous. But when it was finally closed, it locked in **+$13,310 in profit**. A happy accident, but not good risk management. The bug was fixed to prevent it from happening again.

### The Invisible 50-Day Problem
CallBuyer needs at least 50 days of price history to run its analysis. It was asking for 60 days of data, but it was asking for 60 *calendar* days. Weekends and holidays mean 60 calendar days only gives about 41 *trading* days — not enough. Every stock silently came back with "not enough data," and the bot made **zero trades for weeks** without any error message. Changing the request from 60 to 90 calendar days immediately found candidates and opened trades. A tiny one-number fix that took hours to diagnose.

### The 115% Budget Problem
The three active bots were configured to use 60% + 40% + 15% = 115% of the account. You can't spend more than 100% of your money. The broker kept rejecting orders. Rebalanced to 50% + 35% + 15% = 100%.

### The $37/Day AI Experiment
We built a sophisticated multi-agent AI system using OpenAI's GPT-4 (the same AI behind ChatGPT) to help CryptoBot make smarter decisions. It worked — but cost $37/day in API fees just for the computing time. For a paper trading account that isn't making real money yet, that's not sustainable. The system was disabled, and the bots fell back to their built-in models which cost nothing to run.

---

## Risk Management: The Safety Systems

Every bot has multiple layers of protection to limit losses:

1. **Position sizing** — No single trade risks more than a small percentage of the account. Bigger account = bigger trades, but always proportional
2. **Stop-losses** — Every trade has a maximum loss limit. If the price moves against us past a set threshold, sell immediately to cut the loss
3. **Daily loss caps** — If a bot loses too much in one day, it stops trading until tomorrow. No revenge trading
4. **Diversification checks** — Don't hold too many similar positions (e.g., no loading up on 5 different tech stocks that all move together)
5. **Emergency exits** — If a stock suddenly crashes 5%+ against us, close immediately regardless of other rules
6. **Budget guards** — Never try to spend more money than the account actually has
7. **Losing streak protection** — After several losses in a row, the bot automatically becomes much pickier about new trades and waits for conditions to improve
8. **Earnings avoidance** — Don't make bets right before a company reports earnings (stock prices jump unpredictably around earnings)

---

## Account Overview

| Item | Details |
|------|---------|
| **Broker** | Alpaca (paper trading account) |
| **Starting Capital** | ~$100,000 (simulated) |
| **Current Equity** | ~$91,000 |
| **PutSeller Allocation** | 35% |
| **CallBuyer Allocation** | 15% |
| **CryptoBot** | Separate crypto balance |
| **AlpacaBot** | 0% (disabled) |

Everything is running on **paper money** — simulated dollars with real market data. The goal is to prove the system works consistently before putting real money on the line.

---

## The Business Vision

This isn't just a personal trading project — it's the foundation for a business:

1. **Prove it works** (current phase) — Accumulate 100+ closed trades with a strong track record
2. **Go public with results** — Build a public website showing live performance stats
3. **Sell trade alerts** — Offer a paid subscription ($29–99/month) where people get real-time notifications of every trade: what to buy, when, at what price, and when to sell
4. **Scale up** — Multiple strategies, tiered pricing, and potentially a copy-trading feature where subscribers' accounts automatically mirror the bot's trades

The business model is selling *information* (trade signals and strategy access), not managing other people's money — which keeps it legal and accessible without needing a hedge fund license.

### Revenue Potential
- 50 subscribers × $49/month = ~$2,500/month
- 200 subscribers × $49/month = ~$10,000/month
- 500 subscribers × $49/month = ~$24,500/month

---

## The Dashboard

A web-based dashboard shows everything happening across all bots in real time:
- Current profit/loss for each bot
- Every open position and its status
- Trade history
- Account health and system status

---

## Project Structure

```
C:\Bot\                          — CryptoBot (main hub)
├── cryptotrades\               — Core trading code
│   ├── core\                   — Trading engine, risk manager, AI models
│   ├── utils\                  — Helpers (correlation tracking, logging, etc.)
│   └── main.py                 — Entry point
├── tools\                      — Utilities
│   ├── dashboard\              — Web dashboard (port 8088)
│   ├── bot_audit.py            — Automated performance audit
│   └── health_check.py         — System health check
├── BOT_WATCHDOG.py             — Auto-restart service for all bots
├── data\state\                 — Saved state (positions, AI models, config)
└── logs\                       — Trade logs

C:\PutSeller\                    — Options spread bot (35% of account)
├── core\                       — Put engine, risk manager, meta-learner
├── data\state\                 — Positions, trade history
└── logs\                       — Daily log files

C:\CallBuyer\                    — Momentum call buying bot (15% of account)
├── core\                       — Call engine, risk manager, feature scoring
├── data\state\                 — Positions, trade history
└── logs\                       — Daily log files

C:\AlpacaBot\                    — Stock trader (currently disabled)
├── tools\backtest_qualified.py — Multi-timeframe backtester
└── ...                         — Same structure as above
```

---

## Cloud Server Details

The system runs on a free Oracle Cloud server in Phoenix, Arizona.

```
Server IP: 129.146.38.15
Operating System: Ubuntu 24.04 Linux
Resources: 2 CPUs, 15 GB RAM, 42 GB storage
Cost: $0/month (Oracle Cloud free tier)
```

**Services running:**
- `cryptobot.service` — CryptoBot (24/7)
- `putseller.service` — PutSeller (market hours)
- `callbuyer.service` — CallBuyer (market hours)
- `bot-watchdog.timer` — Health check every 2 minutes

---

## Tech Stack

For those interested in the technical details:

- **Language:** Python 3.12 (~56,000 lines across 200+ files)
- **AI/ML:** scikit-learn (machine learning models), PyTorch (neural networks)
- **Broker APIs:** Alpaca (stocks + options), Kraken (crypto futures)
- **Cloud:** Oracle Cloud Free Tier, Ubuntu 24.04, systemd services
- **Dashboard:** Flask web application
- **Process Management:** systemd + custom watchdog with auto-restart

---

## Current Status (April 2026)

| Bot | Status | Notes |
|-----|--------|-------|
| CryptoBot | Running 24/7 | Profitable after parameter overhaul |
| PutSeller | Running | 13 open positions, collecting premiums |
| CallBuyer | Running | Winning streak after bug fixes |
| AlpacaBot | Disabled | 0% allocation, code preserved |
| Cloud Server | Healthy | 7% memory usage, auto-restart active |

The system is in its "prove it" phase — collecting data and building a track record before moving to live trading with real money.

---

## Feedback Welcome!

If you have questions, suggestions, or spotted something that could be better — please open an Issue or leave a comment. Always looking to learn and improve.

---

**~56,000 lines of Python · 200+ files · 4 trading engines · 24/7 cloud operation · Built from scratch**

Last Updated: April 10, 2026
