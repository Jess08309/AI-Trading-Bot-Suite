# 🚀 Laptop Quickstart Guide

Everything you need to go from a fresh clone to running a bot on your laptop in paper mode.

---

## Prerequisites

- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **Git** — [git-scm.com](https://git-scm.com/)
- **Alpaca paper trading account** — [app.alpaca.markets](https://app.alpaca.markets/paper/dashboard/overview) (free)

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/Jess08309/AI-Trading-Bot-Suite.git
cd AI-Trading-Bot-Suite
```

---

## Step 2 — Pick a bot

| Bot | What it trades | Market hours |
|-----|---------------|--------------|
| **CryptoBot** | 18 spot + 8 futures crypto pairs | 24/7 |
| **PutSeller** | Credit spreads on 225+ stocks/ETFs | Market hours only |
| **CallBuyer** | Momentum calls on 149 stocks | Market hours only |
| **AlpacaBot** | Options scalping (paused) | Market hours only |

**Recommended for first run: CryptoBot** — trades 24/7 so you'll see activity right away on your laptop.

---

## Step 3 — Create your `.env` file

Navigate into your chosen bot's folder and copy the example:

```bash
# Example for CryptoBot
cd CryptoBot
cp .env.example .env
```

Open `.env` in any text editor and paste in your Alpaca **paper** keys:

```
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ **Never commit `.env` to git.** It's already in `.gitignore` — just don't override that.

---

## Step 4 — Install dependencies

```bash
# Still inside the bot folder (e.g. CryptoBot)
pip install -r requirements.txt
```

If you prefer a virtual environment (recommended):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Step 5 — Run it

### CryptoBot
```bash
cd CryptoBot
python cryptotrades/main.py
```

### PutSeller
```bash
cd PutSeller
python main.py
```

### CallBuyer
```bash
cd CallBuyer
python main.py
```

### AlpacaBot (test connection only — strategy paused)
```bash
cd AlpacaBot
python test_connection.py
```

---

## Step 6 — Verify it's working

You should see output like:
```
Paper Trading: True
Account connected!
  Equity:       $100,000.00
  Cash:         $100,000.00
```

If you see `Connection FAILED`, double-check your `.env` keys.

---

## What happens next

- **Paper mode** is the default for all bots — it trades **fake money**. Nothing real happens until you set `ALPACA_PAPER=false`.
- Your laptop needs to be **awake and online** to keep trading. Close the lid = bot stops.
- Logs are written to each bot's `logs/` folder.

---

## Switching to live trading (later)

When you're ready (after weeks of paper results you're happy with):

1. Get **live** Alpaca keys from [app.alpaca.markets](https://app.alpaca.markets)
2. Update your `.env`: set `ALPACA_PAPER=false` and swap in live keys
3. Start small — bots already have capital allocation limits built in

---

## Laptop vs. cloud

| | Laptop | VPS/Cloud |
|---|---|---|
| Cost | Free | ~$5-10/month |
| Runs 24/7 | ❌ only when awake | ✅ always on |
| Best for | Paper testing + learning | Live trading later |

For paper trading and learning, your laptop is perfect. When you go live, a cheap VPS keeps the bot running around the clock.
