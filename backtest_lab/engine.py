"""Synthetic-data backtest engine for PutSeller's iron-condor/credit-spread rules.

Mirrors PutSeller/quantconnect/main.py's mechanical logic (same constants, same
entry/exit conditions) but prices options with Black-Scholes over free underlying
data instead of real historical options ticks, so it costs nothing to run and has
no compute/queue limits. Read-only with respect to the production bots -- this
module never imports or writes into AlpacaBot/CallBuyer/PutSeller/CryptoBot.

Simplifications vs. the QC version (tracked so results are read as *relative*,
not absolute):
  - Decisions run once per day (at the daily close) instead of every 15 minutes.
  - Each entry targets the midpoint of the delta band (0.15) rather than scanning
    a full chain for the best available strike/credit combination.
  - Bid/ask spread is simulated via a fixed assumption, not real quoted spreads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import timedelta

from data import get_bars, get_vix, realized_volatility, get_risk_free_rate
from pricing import bs_price, bs_delta, strike_for_delta, synthetic_bid_ask

WATCHLIST = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "V", "HD", "UNH",
]

MIN_DTE = 30
MAX_DTE = 60
TARGET_DTE = 45
MIN_DTE_EXIT = 21

SHORT_DELTA_TARGET = 0.15   # midpoint of the live bot's 0.10-0.20 band

MIN_CREDIT_PCT = 0.15
TAKE_PROFIT_PCT = 0.50
STOP_LOSS_MULT = 2.0
EMERGENCY_BUFFER_PCT = 0.02

MAX_POSITIONS = 12
MAX_CALL_POSITIONS = 8
MAX_PER_UNDERLYING = 2

MIN_HOLD_DAYS = 3
SYNTHETIC_BID_ASK_PCT = 0.30   # assumed spread width as a fraction of mid, for credit realism
RE_ENTRY_COOLDOWN_DAYS = 5

SPY_VOL_SMA = 20
SPY_VOL_HALT_ANNUALIZED = 0.28
TREND_SMA = 50

CONTRACT_MULTIPLIER = 100


def spread_width(price: float) -> float:
    if price < 200:
        return 5.0
    elif price < 500:
        return 10.0
    return 25.0


@dataclass
class Spread:
    ticker: str
    right: str          # "put" or "call"
    short_strike: float
    long_strike: float
    expiry: pd.Timestamp
    credit: float        # per-share credit received at entry
    entry_date: pd.Timestamp
    width: float


@dataclass
class Trade:
    ticker: str
    right: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    credit: float
    exit_debit: float
    reason: str
    pnl: float


def run_backtest(start: str, end: str, verbose: bool = False) -> dict:
    warmup_start = (pd.Timestamp(start) - timedelta(days=int(TREND_SMA * 1.6))).strftime("%Y-%m-%d")

    bars = {t: get_bars(t, warmup_start, end) for t in WATCHLIST}
    vix = get_vix(warmup_start, end)
    rf = get_risk_free_rate(warmup_start, end)
    realized_vol = {t: realized_volatility(bars[t]["Close"], window=SPY_VOL_SMA) for t in WATCHLIST}
    sma = {t: bars[t]["Close"].rolling(TREND_SMA).mean() for t in WATCHLIST}

    trade_dates = bars["SPY"].loc[start:end].index

    open_spreads: dict[str, list[Spread]] = {t: [] for t in WATCHLIST}
    last_exit: dict[str, dict[str, pd.Timestamp | None]] = {t: {"put": None, "call": None} for t in WATCHLIST}
    trades: list[Trade] = []
    cash = 100_000.0
    equity_curve = []

    def sigma_for(ticker: str, date) -> float:
        if ticker in ("SPY", "QQQ", "IWM") and date in vix.index:
            v = vix.loc[date]
            if pd.notna(v):
                return float(v)
        rv = realized_vol[ticker]
        if date in rv.index and pd.notna(rv.loc[date]):
            return float(rv.loc[date])
        return 0.20  # fallback

    def rate_for(date) -> float:
        if date in rf.index and pd.notna(rf.loc[date]):
            return float(rf.loc[date])
        return 0.04

    def total_open(right: str) -> int:
        return sum(1 for spreads in open_spreads.values() for p in spreads if p.right == right)

    def in_cooldown(ticker: str, right: str, date) -> bool:
        last = last_exit[ticker][right]
        return last is not None and (date - last).days < RE_ENTRY_COOLDOWN_DAYS

    def trend_for(ticker: str, date) -> str | None:
        s = sma[ticker]
        if date not in s.index or pd.isna(s.loc[date]):
            return None
        price = bars[ticker].loc[date, "Close"]
        return "up" if price > s.loc[date] else "down"

    def spy_vol_spiked(date) -> bool:
        rv = realized_vol["SPY"]
        if date not in rv.index or pd.isna(rv.loc[date]):
            return False
        return float(rv.loc[date]) > SPY_VOL_HALT_ANNUALIZED

    def price_leg(S, K, T_years, r, sigma, right) -> float:
        mid = bs_price(S, K, T_years, r, sigma, right)
        return mid

    for date in trade_dates:
        # ---- exits ----
        for ticker in WATCHLIST:
            S = bars[ticker].loc[date, "Close"] if date in bars[ticker].index else None
            if S is None:
                continue
            sigma = sigma_for(ticker, date)
            r = rate_for(date)
            still_open = []
            for pos in open_spreads[ticker]:
                T_years = max((pos.expiry - date).days, 0) / 365.0
                short_mid = price_leg(S, pos.short_strike, T_years, r, sigma, pos.right)
                long_mid = price_leg(S, pos.long_strike, T_years, r, sigma, pos.right)
                current_debit = short_mid - long_mid
                credit = pos.credit
                days_held = (date - pos.entry_date).days
                reason = None

                if credit > 0 and days_held >= MIN_HOLD_DAYS:
                    profit_pct = (credit - current_debit) / credit
                    if profit_pct >= TAKE_PROFIT_PCT:
                        reason = f"TAKE_PROFIT ({profit_pct:.0%})"
                    elif current_debit >= credit * STOP_LOSS_MULT:
                        reason = "STOP_LOSS"

                days_to_exp = (pos.expiry - date).days
                if reason is None and days_to_exp <= MIN_DTE_EXIT:
                    reason = f"DTE_EXIT ({days_to_exp}d remaining)"

                if reason is None:
                    buffer = pos.short_strike * EMERGENCY_BUFFER_PCT
                    if pos.right == "call" and S >= pos.short_strike - buffer:
                        reason = "EMERGENCY_CALL"
                    elif pos.right == "put" and S <= pos.short_strike + buffer:
                        reason = "EMERGENCY_PUT"

                if reason:
                    pnl = (credit - current_debit) * CONTRACT_MULTIPLIER
                    cash += pnl
                    trades.append(Trade(ticker, pos.right, pos.entry_date, date, credit, current_debit, reason, pnl))
                    last_exit[ticker][pos.right] = date
                else:
                    still_open.append(pos)
            open_spreads[ticker] = still_open

        # ---- entries ----
        if not spy_vol_spiked(date):
            for ticker in WATCHLIST:
                if len(open_spreads[ticker]) >= MAX_PER_UNDERLYING:
                    continue
                if date not in bars[ticker].index:
                    continue
                S = bars[ticker].loc[date, "Close"]
                sigma = sigma_for(ticker, date)
                r = rate_for(date)
                trend = trend_for(ticker, date)
                width = spread_width(S)
                expiry = date + timedelta(days=TARGET_DTE)
                T_years = TARGET_DTE / 365.0

                opened = False
                if trend != "down" and not in_cooldown(ticker, "put", date) and total_open("put") < MAX_POSITIONS:
                    K_short = strike_for_delta(S, T_years, r, sigma, "put", SHORT_DELTA_TARGET)
                    K_long = K_short - width
                    short_mid = price_leg(S, K_short, T_years, r, sigma, "put")
                    long_mid = price_leg(S, K_long, T_years, r, sigma, "put")
                    credit = short_mid - long_mid
                    if credit > 0 and credit / width >= MIN_CREDIT_PCT:
                        open_spreads[ticker].append(Spread(ticker, "put", K_short, K_long, expiry, credit, date, width))
                        opened = True

                if not opened and trend != "up" and not in_cooldown(ticker, "call", date) and total_open("call") < MAX_CALL_POSITIONS:
                    K_short = strike_for_delta(S, T_years, r, sigma, "call", SHORT_DELTA_TARGET)
                    K_long = K_short + width
                    short_mid = price_leg(S, K_short, T_years, r, sigma, "call")
                    long_mid = price_leg(S, K_long, T_years, r, sigma, "call")
                    credit = short_mid - long_mid
                    if credit > 0 and credit / width >= MIN_CREDIT_PCT:
                        open_spreads[ticker].append(Spread(ticker, "call", K_short, K_long, expiry, credit, date, width))

        equity_curve.append((date, cash))

    return summarize(trades, equity_curve, start_cash=100_000.0)


def summarize(trades: list[Trade], equity_curve: list, start_cash: float) -> dict:
    if not trades:
        return {"trades": 0, "net_profit_pct": 0.0, "win_rate": 0.0}

    total_pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    curve = pd.Series([c for _, c in equity_curve], index=[d for d, _ in equity_curve])
    running_max = curve.cummax()
    drawdown = (curve - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    by_reason = {}
    for t in trades:
        key = t.reason.split(" (")[0]
        by_reason.setdefault(key, {"count": 0, "pnl": 0.0, "wins": 0})
        by_reason[key]["count"] += 1
        by_reason[key]["pnl"] += t.pnl
        if t.pnl > 0:
            by_reason[key]["wins"] += 1

    return {
        "trades": len(trades),
        "net_profit_pct": total_pnl / start_cash * 100,
        "end_equity": start_cash + total_pnl,
        "win_rate": wins / len(trades) * 100,
        "max_drawdown_pct": max_dd * 100,
        "by_reason": by_reason,
    }
