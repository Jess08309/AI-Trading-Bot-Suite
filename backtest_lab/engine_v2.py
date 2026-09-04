"""Synthetic-pricing PutSeller backtest, v2: same Black-Scholes option model as
engine.py, but the exit-check loop runs on REAL Alpaca 15-minute intraday
underlying bars instead of once-daily closes. This targets the gap v1 exposed:
EMERGENCY exits (triggered by the underlying touching near a short strike) need
intraday granularity to fire realistically -- daily closes almost never show it.

Entries still happen once per day (at the day's last 15-min bar) since the live
bot's entry cadence isn't the part that was under-modeled.
"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from datetime import timedelta, date as date_cls

from data import get_vix, get_risk_free_rate
from alpaca_data import get_minute_bars
from pricing import bs_price, strike_for_delta

WATCHLIST = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "V", "HD", "UNH",
]

MIN_DTE = 30
MAX_DTE = 60
TARGET_DTE = 45
MIN_DTE_EXIT = 21

SHORT_DELTA_TARGET = 0.15

MIN_CREDIT_PCT = 0.15
TAKE_PROFIT_PCT = 0.50
STOP_LOSS_MULT = 2.0
EMERGENCY_BUFFER_PCT = 0.02

MAX_POSITIONS = 12
MAX_CALL_POSITIONS = 8
MAX_PER_UNDERLYING = 2

MIN_HOLD_DAYS = 3
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
    right: str
    short_strike: float
    long_strike: float
    expiry: pd.Timestamp
    credit: float
    entry_date: date_cls
    width: float


@dataclass
class Trade:
    ticker: str
    right: str
    entry_date: date_cls
    exit_date: date_cls
    credit: float
    exit_debit: float
    reason: str
    pnl: float


def run_backtest(start: str, end: str) -> dict:
    warmup_start = (pd.Timestamp(start) - timedelta(days=int(TREND_SMA * 1.6))).strftime("%Y-%m-%d")

    # daily_close/sma/realized_vol are derived from tz-aware intraday bars, but VIX/rate
    # data (yfinance) is tz-naive -- strip tz here so all daily-level lookups use plain
    # Timestamps consistently (mismatched tz-aware vs naive lookups silently miss and
    # fall back to defaults, which previously starved high-vol names of any credit).
    intraday = {t: get_minute_bars(t, warmup_start, end, timeframe="15Min") for t in WATCHLIST}
    daily_close = {t: intraday[t]["close"].resample("1D").last().dropna().tz_localize(None) for t in WATCHLIST}
    sma = {t: daily_close[t].rolling(TREND_SMA).mean() for t in WATCHLIST}
    vix = get_vix(warmup_start, end)
    rf = get_risk_free_rate(warmup_start, end)

    import numpy as np
    realized_vol = {}
    for t in WATCHLIST:
        r = np.log(daily_close[t] / daily_close[t].shift(1))
        realized_vol[t] = r.rolling(SPY_VOL_SMA).std() * np.sqrt(252)

    trading_days = sorted(set(idx.date() for idx in daily_close["SPY"].index))
    trading_days = [d for d in trading_days if pd.Timestamp(start).date() <= d <= pd.Timestamp(end).date()]

    open_spreads: dict[str, list[Spread]] = {t: [] for t in WATCHLIST}
    last_exit: dict[str, dict[str, date_cls | None]] = {t: {"put": None, "call": None} for t in WATCHLIST}
    trades: list[Trade] = []
    cash = 100_000.0
    equity_curve = []

    def sigma_for(ticker: str, d: date_cls) -> float:
        ts = pd.Timestamp(d)
        if ticker in ("SPY", "QQQ", "IWM") and ts in vix.index:
            v = vix.loc[ts]
            if pd.notna(v):
                return float(v)
        rv = realized_vol[ticker]
        if ts in rv.index and pd.notna(rv.loc[ts]):
            return float(rv.loc[ts])
        return 0.20

    def rate_for(d: date_cls) -> float:
        ts = pd.Timestamp(d)
        if ts in rf.index and pd.notna(rf.loc[ts]):
            return float(rf.loc[ts])
        return 0.04

    def total_open(right: str) -> int:
        return sum(1 for spreads in open_spreads.values() for p in spreads if p.right == right)

    def in_cooldown(ticker: str, right: str, d: date_cls) -> bool:
        last = last_exit[ticker][right]
        return last is not None and (d - last).days < RE_ENTRY_COOLDOWN_DAYS

    def trend_for(ticker: str, d: date_cls) -> str | None:
        ts = pd.Timestamp(d)
        s = sma[ticker]
        if ts not in s.index or pd.isna(s.loc[ts]):
            return None
        price = daily_close[ticker].loc[ts]
        return "up" if price > s.loc[ts] else "down"

    def spy_vol_spiked(d: date_cls) -> bool:
        ts = pd.Timestamp(d)
        rv = realized_vol["SPY"]
        if ts not in rv.index or pd.isna(rv.loc[ts]):
            return False
        return float(rv.loc[ts]) > SPY_VOL_HALT_ANNUALIZED

    def check_exit(pos: Spread, S: float, sigma: float, r: float, d: date_cls):
        T_years = max((pos.expiry.date() - d).days, 0) / 365.0
        short_mid = bs_price(S, pos.short_strike, T_years, r, sigma, pos.right)
        long_mid = bs_price(S, pos.long_strike, T_years, r, sigma, pos.right)
        current_debit = short_mid - long_mid
        credit = pos.credit
        days_held = (d - pos.entry_date).days

        if credit > 0 and days_held >= MIN_HOLD_DAYS:
            profit_pct = (credit - current_debit) / credit
            if profit_pct >= TAKE_PROFIT_PCT:
                return f"TAKE_PROFIT ({profit_pct:.0%})", current_debit
            if current_debit >= credit * STOP_LOSS_MULT:
                return "STOP_LOSS", current_debit

        days_to_exp = (pos.expiry.date() - d).days
        if days_to_exp <= MIN_DTE_EXIT:
            return f"DTE_EXIT ({days_to_exp}d remaining)", current_debit

        buffer = pos.short_strike * EMERGENCY_BUFFER_PCT
        if pos.right == "call" and S >= pos.short_strike - buffer:
            return "EMERGENCY_CALL", current_debit
        if pos.right == "put" and S <= pos.short_strike + buffer:
            return "EMERGENCY_PUT", current_debit
        return None, current_debit

    def best_spread(S: float, T_years: float, r: float, sigma: float, right: str, width: float):
        """Mirrors the real QC code's _find_credit_spread: scans the whole 0.10-0.20
        delta band and picks the LOWEST-delta (safest, furthest OTM) strike that still
        clears MIN_CREDIT_PCT, rather than maximizing credit_pct -- maximizing credit
        was found to systematically pick the riskier, closer-to-money edge of the band."""
        best = None
        best_delta = 2.0
        for delta_target in (0.10, 0.12, 0.14, 0.16, 0.18, 0.20):
            K_short = strike_for_delta(S, T_years, r, sigma, right, delta_target)
            K_long = K_short - width if right == "put" else K_short + width
            credit = bs_price(S, K_short, T_years, r, sigma, right) - bs_price(S, K_long, T_years, r, sigma, right)
            if credit <= 0:
                continue
            credit_pct = credit / width
            if credit_pct < MIN_CREDIT_PCT:
                continue
            if delta_target < best_delta:
                best_delta = delta_target
                best = (K_short, K_long, credit)
        if best is None:
            return None
        K_short, K_long, credit = best
        return K_short, K_long, credit

    for d in trading_days:
        # ---- intraday exit scan (real 15-min bars) ----
        for ticker in WATCHLIST:
            if not open_spreads[ticker]:
                continue
            day_bars = intraday[ticker][intraday[ticker].index.date == d]
            if day_bars.empty:
                continue
            r = rate_for(d)
            sigma = sigma_for(ticker, d)

            for bar_time, row in day_bars.iterrows():
                if not open_spreads[ticker]:
                    break
                S = float(row["close"])
                still_open = []
                for pos in open_spreads[ticker]:
                    reason, current_debit = check_exit(pos, S, sigma, r, d)
                    if reason:
                        pnl = (pos.credit - current_debit) * CONTRACT_MULTIPLIER
                        cash += pnl
                        trades.append(Trade(ticker, pos.right, pos.entry_date, d, pos.credit, current_debit, reason, pnl))
                        last_exit[ticker][pos.right] = d
                    else:
                        still_open.append(pos)
                open_spreads[ticker] = still_open

        # ---- entries (once per day, at day's last real intraday price) ----
        if not spy_vol_spiked(d):
            for ticker in WATCHLIST:
                if len(open_spreads[ticker]) >= MAX_PER_UNDERLYING:
                    continue
                day_bars = intraday[ticker][intraday[ticker].index.date == d]
                if day_bars.empty:
                    continue
                S = float(day_bars["close"].iloc[-1])
                sigma = sigma_for(ticker, d)
                r = rate_for(d)
                trend = trend_for(ticker, d)
                width = spread_width(S)
                expiry = pd.Timestamp(d) + timedelta(days=TARGET_DTE)
                T_years = TARGET_DTE / 365.0

                opened = False
                if trend != "down" and not in_cooldown(ticker, "put", d) and total_open("put") < MAX_POSITIONS:
                    result = best_spread(S, T_years, r, sigma, "put", width)
                    if result:
                        K_short, K_long, credit = result
                        open_spreads[ticker].append(Spread(ticker, "put", K_short, K_long, expiry, credit, d, width))
                        opened = True

                if not opened and trend != "up" and not in_cooldown(ticker, "call", d) and total_open("call") < MAX_CALL_POSITIONS:
                    result = best_spread(S, T_years, r, sigma, "call", width)
                    if result:
                        K_short, K_long, credit = result
                        open_spreads[ticker].append(Spread(ticker, "call", K_short, K_long, expiry, credit, d, width))

        equity_curve.append((d, cash))

    return summarize(trades, equity_curve, start_cash=100_000.0), trades


def summarize(trades: list, equity_curve: list, start_cash: float) -> dict:
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
