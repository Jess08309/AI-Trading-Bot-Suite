# CryptoBot

**This repository contains CryptoBot, an automated crypto trading system.**

CryptoBot is a 24/7 machine-learning-driven spot + futures crypto trading bot. It previously ran alongside three other strategies (PutSeller, CallBuyer, AlpacaBot) as part of a four-bot portfolio; as of 2026-09-27 the portfolio was fully consolidated down to CryptoBot alone, which now runs with 100% of the account's trading capital. The other three bots' code has been removed from this repository (full history is preserved on the `archive/four-bot-portfolio-2026-09-27` branch and in git history).

---

## What it does

CryptoBot trades a watchlist of ~16 spot crypto pairs (BTC, ETH, SOL, ADA, AVAX, DOGE, LINK, XRP, LTC, UNI, BCH, DOT, MATIC, NEAR, AAVE, PAXG) plus a set of simulated perpetual futures contracts, on a ~5-minute decision cycle with a 20-second risk-check loop in between.

- **Signal generation:** a rolling-retrained `GradientBoostingClassifier` scores each symbol using RSI, MACD histogram, TRIX, ATR, Bollinger %B, mean-reversion, and volume-ratio features, combined with trend/sentiment/correlation filters.
- **Risk management (unchanged by the capital consolidation):** per-position sizing cap (12% max), max concurrent positions per asset class, stop-loss / take-profit / trailing-stop, a stale-position decay exit, max-hold-time forced exits, a consecutive-loss circuit breaker, a daily-loss circuit breaker, and a max-drawdown circuit breaker.
- **Broker:** real spot orders are placed on Alpaca's crypto paper-trading API (shared paper account, no real money). Futures are fully simulated against Kraken's demo endpoint (`KRAKEN_FUTURES_ENABLE_REAL_ORDERS=False`) — no real futures orders are ever placed.

## Capital

As of the 2026-09-27 consolidation, CryptoBot's internal spot capital base (`INITIAL_SPOT_BALANCE` in `cryptotrades/core/trading_engine.py`) is **$100,000** — up from a self-imposed $5,000 cap that existed only because capital was being conceptually reserved for the other three bots. CryptoBot's own risk management (position size %, circuit breakers, per-symbol/asset-class caps) was **not** loosened — only the capital base it sizes positions against changed.

## Running it

```bash
cd CryptoBot/cryptotrades
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in Alpaca + (optional) Kraken Futures demo credentials
python3 main.py
```

In production this runs as the `cryptobot` systemd service on a DigitalOcean droplet.

## Status

Paper trading only. No real capital is at risk. This is a research/track-record-building project, not a live trading service.

---

**Disclaimer:** experimental software for educational/research purposes only. Not financial advice. Trading involves substantial risk of loss. Past/simulated performance does not guarantee future results.

## Coinbase setup

Coinbase Advanced Trade is an opt-in **spot** broker. Create a CDP API key at
[coinbase.com/settings/api](https://coinbase.com/settings/api) with **View + Trade**
permissions, **no Transfer** permission. IP-allowlist the bot server's fixed
outbound IP. Install `CryptoBot/requirements.txt` (or
`CryptoBot/cryptotrades/requirements.txt`) before enabling it.

Set these variables in your existing private environment (never commit keys):

```dotenv
ENABLE_COINBASE=true
COINBASE_API_KEY=organizations/YOUR_ORG/apiKeys/YOUR_KEY
COINBASE_API_SECRET=<your CDP private key>
PAPER_TRADING=true
```

P-256 PEM keys use ES256 JWTs; Ed25519 PEM or base64 CDP keys use EdDSA JWTs.
For PEM values in dotenv, use a quoted multiline value or quoted `\n` escapes.
Do not use legacy Coinbase Exchange HMAC credentials.

When enabled with both credentials present, Coinbase is primary; otherwise the
existing Alpaca behavior is retained. Startup logs name the active spot broker.
Coinbase auth errors stop startup rather than silently moving exposure to Alpaca.
`BTC/USD` maps to `BTC-USD` (likewise for the existing spot universe); Coinbase
may not list every symbol, and unavailable products are not traded. Futures and
backtest tools are unchanged.

**Paper mode uses live read-only data/auth checks and logs intended orders; it
never submits or cancels Coinbase orders.** Live mode additionally requires the
bot's existing `LIVE_TRADING_CONFIRM` safeguard. Close tracked positions on their
original broker before changing brokers, or close live Coinbase positions before
switching to paper mode.

An ambiguous live order stops the bot and leaves
`cryptotrades/data/state/coinbase_pending_order.json`. Reconcile the recorded
client order ID with Coinbase and the bot's position ledger before removing that
marker and restarting; do not delete it blindly.
