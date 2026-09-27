<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/Alpaca-Paper%20Trading-FFD700?logo=alpaca&logoColor=black" alt="Alpaca Paper Trading" />
  <img src="https://img.shields.io/badge/Status-Paper%20Trading-blue" alt="Status: Paper Trading" />
  <img src="https://img.shields.io/badge/Live%20Orders-Disabled%20(Safety%20Default)-red" alt="Live Orders Disabled" />
</p>

# 🤖 AI Trading Bot Suite

**This repository contains CryptoBot, an automated crypto trading system.**

CryptoBot is a 24/7 machine-learning-driven cryptocurrency trading bot trading spot and (simulated) futures markets. It runs in **paper trading mode only — no real capital is at risk.**

> **2026-09-27 — Portfolio consolidation:** this repository previously hosted four independent trading bots (CryptoBot, PutSeller, CallBuyer, AlpacaBot) sharing one paper trading account. None of the other three ever cleared the fleet's go-live bar, and per an explicit decision to focus on a single strategy, **PutSeller, CallBuyer, and AlpacaBot have been retired and removed from this repository.** CryptoBot now runs with 100% of the account's trading capital. Full history of the prior four-bot architecture is preserved on the `archive/four-bot-portfolio-2026-09-27` branch and in git history — nothing was permanently lost, just removed from the active codebase.

---

## What's here

- **`CryptoBot/`** — the live bot. See [CryptoBot/README.md](CryptoBot/README.md) for strategy details, risk management, and how to run it.
- **`lean_workspace/lean.json`** — generic QuantConnect LEAN CLI configuration (retained for any future research use; the three per-bot LEAN projects that used to live here were removed along with their bots).
- **`tools/`** — fleet-wide reporting utilities (`portfolio_report.py`), now scoped to CryptoBot only.
- **`docs/`** — governance documentation (`GO_LIVE_CRITERIA.md`), the bar CryptoBot must clear before any real capital would ever be considered.

## Safety posture

- **Live trading is disabled.** All trading is against Alpaca's paper trading API (crypto spot) and Kraken's demo futures endpoint (fully simulated, no real orders placed).
- **No profitability representation.** Paper trading results are a research tool, not proof of future profitability.
- Internal risk management (position sizing caps, stop-loss/take-profit, circuit breakers on consecutive losses/daily loss/drawdown) is unchanged by the capital consolidation — see [CryptoBot/README.md](CryptoBot/README.md) for specifics.

## Local setup

```bash
git clone https://github.com/Jess08309/AI-Trading-Bot-Suite.git
cd AI-Trading-Bot-Suite/CryptoBot/cryptotrades
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in your own Alpaca (+ optional Kraken demo) credentials
python3 main.py
```

### Production service management (cloud VM)

Runs as a single systemd service on a DigitalOcean droplet:

```bash
sudo systemctl status cryptobot
sudo journalctl -u cryptobot -f --no-pager
```

---

## ⚠️ Disclaimer

This software is experimental and intended strictly for educational and research purposes. Nothing in this repository constitutes financial, investment, legal, or tax advice. Algorithmic trading involves substantial risk of loss. Past performance and simulated/paper trading results do not guarantee future profitability. Provided "as is", without warranty of any kind.

## 👤 Author

[Jess08309](https://github.com/Jess08309)
