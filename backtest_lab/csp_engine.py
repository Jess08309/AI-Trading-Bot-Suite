"""Cash-Secured Put (CSP) backtest engine -- "sweet spot" naked put-selling
framework (Options Trading IQ), 20-25 delta / 30-45 DTE.

IMPORTANT -- naming collision flagged, read before touching this file:
This is a NEW, DISTINCT strategy from engine.py / engine_v2.py in this same
folder, which model PutSeller's actual LIVE bot (bull-put + bear-call CREDIT
SPREADS, i.e. an iron condor -- see PutSeller/core/put_engine.py, which
self-identifies in its own docstrings/prints as "IronCondor v2.0", and is
currently deployed as the `putseller` systemd service). This file does not
touch, import, or replace either of those -- it is a fresh, additive module
for a single-leg, undefined-risk, NAKED put-selling strategy being evaluated
as a possible eventual replacement for CallBuyer (a deployment decision NOT
made by this backtest -- see the report this engine produces and the PR body
for the open question on where this should eventually live).

Read-only with respect to every production bot -- only imports the existing
shared data.py/pricing.py helpers (unmodified), same convention already
established by engine.py/engine_v2.py in this folder.

Simplifications vs. a real broker/chain (tracked so results are read as
*relative*, not a guarantee -- same spirit as engine.py's own documented
simplifications):
  - Decisions run once per day at the daily close (no intraday granularity).
  - Expiration is a flat +37-calendar-day offset (midpoint of 30-45 DTE) from
    entry, not snapped to a real listed monthly/weekly expiry -- matches the
    exact simplification already used by engine.py/engine_v2.py in this same
    folder (their TARGET_DTE offset) rather than introducing a different
    convention. (A real-monthly-expiry version was considered, but standard
    3rd-Friday expiries are ~30-31 days apart and can fall outside a strict
    30-45 DTE window depending on the entry date, which would have silently
    suppressed entries on some days -- the flat offset avoids that.)
  - Strike targets the midpoint of the 20-25 delta band (22.5) via a single
    Black-Scholes inversion, not a scan of a real listed strike chain.
  - "IV Rank" is proxied from a rolling annualized realized-volatility series
    (no free historical IV surface exists) -- current value ranked against its
    own trailing 252-trading-day [min, max], the standard IV Rank formula
    applied to that proxy: (current - 252d low) / (252d high - 252d low) * 100.
  - Fills: Black-Scholes mid price minus $0.05/share when WE are on the
    disadvantaged side -- sell-to-open at (mid - 0.05), buy-to-close at
    (mid + 0.05). A flat $0.10/share round-trip cost, applied to every fill,
    per the task's literal "mid price minus $0.05 slippage" instruction
    extended symmetrically to both legs of the round trip.
  - No intraday resampling happens anywhere in this file (all data is already
    daily bars; all rolling/window stats use pandas time-indexed `.rolling()`),
    so the positional-vs-calendar-time resampling bug class that affected
    AlpacaBot's intraday configs does not apply here.
  - Only 1 position per underlying at a time (not stated explicitly in the
    spec, but implied by "sell 1 put" per entry plus the position/sector caps
    only being meaningful if a single name can't be re-entered indefinitely).

EXIT RULES UPDATED (2026-09-26): the original 21-DTE exit was found to force
early closes before theta acceleration could play out (-$3,515 across 89
trades in the first backtest). Per explicit instruction, DTE_EXIT is now 10
(not 21) -- profit target, stop loss, and expiration-handling are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import timedelta, date as date_cls

from data import get_bars, get_risk_free_rate
from pricing import bs_price, bs_delta, strike_for_delta

CONTRACT_MULTIPLIER = 100
SLIPPAGE_PER_SHARE = 0.05

# ---- Universe: liquid, optionable, historically ~$20-100 stocks, tagged with
# an approximate sector for the "max 3 per sector" rule. ----
UNIVERSE = {
    "INTC": "TECH", "CSCO": "TECH", "PYPL": "TECH", "HPQ": "TECH", "DELL": "TECH",
    "BAC": "FINANCIALS", "WFC": "FINANCIALS", "C": "FINANCIALS", "USB": "FINANCIALS",
    "RF": "FINANCIALS", "FITB": "FINANCIALS",
    "OXY": "ENERGY", "KMI": "ENERGY", "SLB": "ENERGY", "DVN": "ENERGY", "HAL": "ENERGY",
    "PFE": "HEALTHCARE", "CVS": "HEALTHCARE", "BMY": "HEALTHCARE", "GILD": "HEALTHCARE",
    "T": "TELECOM", "VZ": "TELECOM",
    "KO": "CONSUMER", "GM": "CONSUMER", "UBER": "CONSUMER", "DAL": "CONSUMER",
    "CCL": "CONSUMER", "NCLH": "CONSUMER",
    "CSX": "INDUSTRIALS",
}
WATCHLIST = list(UNIVERSE.keys())

# ---- Universe filters ----
PRICE_MIN = 20.0
PRICE_MAX = 100.0
MIN_AVG_VOLUME = 1_000_000
AVG_VOLUME_WINDOW = 20
IV_RANK_MIN = 30.0
IV_RANK_LOOKBACK_DAYS = 252
VOL_WINDOW = 20                    # realized-vol lookback used as the IV proxy

# ---- Entry ----
MIN_DTE = 30
MAX_DTE = 45
TARGET_DTE = 37                    # midpoint; flat offset, see module docstring
SHORT_PUT_DELTA_TARGET = 0.225     # midpoint of the 20-25 delta band

# ---- Position sizing / portfolio limits ----
MAX_POSITION_PCT_EQUITY = 0.05     # 5% of equity secured per trade
MAX_POSITIONS = 20
MAX_PER_SECTOR = 3
MAX_PORTFOLIO_DELTA = 100.0        # shares-equivalent units (per-contract delta * 100)

# ---- Exits (checked in this literal priority order every day) ----
TAKE_PROFIT_PCT = 0.50
DTE_EXIT = 10                      # was 21 -- forced early closes before theta acceleration
STOP_LOSS_PCT_BELOW_STRIKE = 0.10

START_CASH = 100_000.0


@dataclass
class PutPosition:
    ticker: str
    sector: str
    strike: float
    expiry: date_cls
    entry_date: date_cls
    credit: float           # per-share credit received at entry, after slippage


@dataclass
class Trade:
    ticker: str
    sector: str
    entry_date: date_cls
    exit_date: date_cls
    strike: float
    credit: float
    exit_debit: float
    reason: str
    pnl: float


def run_backtest(start: str, end: str, verbose: bool = False) -> dict:
    warmup_start = (pd.Timestamp(start) - timedelta(days=int(IV_RANK_LOOKBACK_DAYS * 1.6))).strftime("%Y-%m-%d")

    raw_bars = {t: get_bars(t, warmup_start, end) for t in WATCHLIST}
    bars = {t: df for t, df in raw_bars.items() if not df.empty and len(df) > IV_RANK_LOOKBACK_DAYS + VOL_WINDOW}
    missing = set(WATCHLIST) - set(bars)
    if missing and verbose:
        print(f"Skipping (insufficient history): {sorted(missing)}")

    rf = get_risk_free_rate(warmup_start, end)

    iv_proxy: dict[str, pd.Series] = {}
    iv_rank: dict[str, pd.Series] = {}
    avg_volume: dict[str, pd.Series] = {}
    for t, df in bars.items():
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        vol = log_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)
        iv_proxy[t] = vol
        roll_min = vol.rolling(IV_RANK_LOOKBACK_DAYS).min()
        roll_max = vol.rolling(IV_RANK_LOOKBACK_DAYS).max()
        rank = (vol - roll_min) / (roll_max - roll_min) * 100
        iv_rank[t] = rank.replace([np.inf, -np.inf], np.nan)
        avg_volume[t] = df["Volume"].rolling(AVG_VOLUME_WINDOW).mean()

    all_days = sorted(set().union(*[set(df.index) for df in bars.values()]))
    trading_days = [d for d in all_days if pd.Timestamp(start) <= d <= pd.Timestamp(end)]

    open_positions: dict[str, list[PutPosition]] = {t: [] for t in bars}
    trades: list[Trade] = []
    cash = START_CASH
    equity_curve: list[tuple] = []

    def rate_for(d) -> float:
        if d in rf.index and pd.notna(rf.loc[d]):
            return float(rf.loc[d])
        return 0.04

    def sigma_for(t, d) -> float:
        s = iv_proxy[t]
        if d in s.index and pd.notna(s.loc[d]):
            return float(s.loc[d])
        return 0.30

    def total_open_positions() -> int:
        return sum(len(v) for v in open_positions.values())

    def sector_open_count(sector: str) -> int:
        return sum(len(v) for t, v in open_positions.items() if UNIVERSE[t] == sector)

    def mark_to_market(d) -> float:
        """Cash plus unrealized P&L of every currently open position, valued at the
        Black-Scholes mid (no slippage -- nothing is actually being closed)."""
        unrealized = 0.0
        for t, positions in open_positions.items():
            if not positions or d not in bars[t].index:
                continue
            S = float(bars[t].loc[d, "Close"])
            sigma = sigma_for(t, d)
            r = rate_for(d)
            for pos in positions:
                T_years = max((pos.expiry - d.date()).days, 0) / 365.0
                mid = bs_price(S, pos.strike, T_years, r, sigma, "put")
                unrealized += (pos.credit - mid) * CONTRACT_MULTIPLIER
        return cash + unrealized

    def portfolio_delta(d) -> float:
        total = 0.0
        for t, positions in open_positions.items():
            if not positions or d not in bars[t].index:
                continue
            S = float(bars[t].loc[d, "Close"])
            sigma = sigma_for(t, d)
            r = rate_for(d)
            for pos in positions:
                T_years = max((pos.expiry - d.date()).days, 0) / 365.0
                total += -bs_delta(S, pos.strike, T_years, r, sigma, "put") * CONTRACT_MULTIPLIER
        return total

    for d in trading_days:
        # ---- exits (priority order: profit target, DTE, stop loss, expiration) ----
        for t, df in bars.items():
            if not open_positions[t] or d not in df.index:
                continue
            S = float(df.loc[d, "Close"])
            sigma = sigma_for(t, d)
            r = rate_for(d)
            still_open = []
            for pos in open_positions[t]:
                dte_remaining = (pos.expiry - d.date()).days
                T_years = max(dte_remaining, 0) / 365.0
                mid = bs_price(S, pos.strike, T_years, r, sigma, "put")
                buy_to_close = mid + SLIPPAGE_PER_SHARE
                credit = pos.credit
                profit_pct = (credit - buy_to_close) / credit if credit > 0 else 0.0

                reason = None
                if profit_pct >= TAKE_PROFIT_PCT:
                    reason = f"PROFIT_TARGET ({profit_pct:.0%})"
                elif dte_remaining <= DTE_EXIT:
                    reason = f"DTE_EXIT ({dte_remaining}d remaining)"
                elif S <= pos.strike * (1 - STOP_LOSS_PCT_BELOW_STRIKE):
                    reason = "STOP_LOSS (10% below strike)"
                elif dte_remaining <= 0:
                    # Expected to be unreachable given DTE_EXIT always closes
                    # first -- kept for spec-completeness/safety, same pattern as
                    # Config G's documented-dead exit branches (AlpacaBot research).
                    if S > pos.strike:
                        reason = "EXPIRED_OTM"
                        buy_to_close = 0.0
                    else:
                        reason = "EXPIRED_ITM_BUYBACK"

                if reason:
                    pnl = (credit - buy_to_close) * CONTRACT_MULTIPLIER
                    cash += pnl
                    trades.append(Trade(t, UNIVERSE[t], pos.entry_date, d.date(), pos.strike, credit, buy_to_close, reason, pnl))
                else:
                    still_open.append(pos)
            open_positions[t] = still_open

        # ---- entries ----
        entry_equity = mark_to_market(d)     # fixed snapshot for all of today's sizing checks
        if total_open_positions() < MAX_POSITIONS:
            for t, df in bars.items():
                if total_open_positions() >= MAX_POSITIONS:
                    break
                if open_positions[t] or d not in df.index:
                    continue
                sector = UNIVERSE[t]
                if sector_open_count(sector) >= MAX_PER_SECTOR:
                    continue

                S = float(df.loc[d, "Close"])
                if not (PRICE_MIN <= S <= PRICE_MAX):
                    continue
                vol_avg = avg_volume[t].loc[d] if d in avg_volume[t].index else np.nan
                if pd.isna(vol_avg) or vol_avg < MIN_AVG_VOLUME:
                    continue
                ivr = iv_rank[t].loc[d] if d in iv_rank[t].index else np.nan
                if pd.isna(ivr) or ivr < IV_RANK_MIN:
                    continue

                sigma = sigma_for(t, d)
                r = rate_for(d)
                T_years = TARGET_DTE / 365.0
                K = strike_for_delta(S, T_years, r, sigma, "put", SHORT_PUT_DELTA_TARGET)

                collateral = K * CONTRACT_MULTIPLIER
                if collateral > MAX_POSITION_PCT_EQUITY * entry_equity:
                    continue

                mid = bs_price(S, K, T_years, r, sigma, "put")
                sell_to_open = max(0.0, mid - SLIPPAGE_PER_SHARE)
                if sell_to_open <= 0:
                    continue

                new_delta = -bs_delta(S, K, T_years, r, sigma, "put") * CONTRACT_MULTIPLIER
                if portfolio_delta(d) + new_delta > MAX_PORTFOLIO_DELTA:
                    continue

                expiry = d.date() + timedelta(days=TARGET_DTE)
                open_positions[t].append(PutPosition(t, sector, K, expiry, d.date(), sell_to_open))

        equity_curve.append((d, mark_to_market(d)))

    return summarize(trades, equity_curve, rf, start_cash=START_CASH)


def summarize(trades: list[Trade], equity_curve: list, rf_series: pd.Series, start_cash: float) -> dict:
    curve = pd.Series([e for _, e in equity_curve], index=[d for d, _ in equity_curve])
    end_equity = float(curve.iloc[-1]) if len(curve) else start_cash
    net_profit_pct = (end_equity - start_cash) / start_cash * 100

    if not trades:
        return {
            "trades": 0, "net_profit_pct": net_profit_pct, "end_equity": end_equity,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "max_drawdown_pct": 0.0, "sharpe": 0.0, "by_reason": {}, "by_sector": {},
        }

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    win_rate = len(wins) / len(trades) * 100
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    running_max = curve.cummax()
    drawdown = (curve - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    daily_ret = curve.pct_change().dropna()
    rf_daily = rf_series.reindex(daily_ret.index).ffill().fillna(0.04) / 252
    excess = daily_ret - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    by_reason: dict = {}
    by_sector: dict = {}
    for t in trades:
        key = t.reason.split(" (")[0]
        by_reason.setdefault(key, {"count": 0, "pnl": 0.0, "wins": 0})
        by_reason[key]["count"] += 1
        by_reason[key]["pnl"] += t.pnl
        if t.pnl > 0:
            by_reason[key]["wins"] += 1
        by_sector.setdefault(t.sector, {"count": 0, "pnl": 0.0, "wins": 0})
        by_sector[t.sector]["count"] += 1
        by_sector[t.sector]["pnl"] += t.pnl
        if t.pnl > 0:
            by_sector[t.sector]["wins"] += 1

    return {
        "trades": len(trades),
        "net_profit_pct": net_profit_pct,
        "end_equity": end_equity,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "by_reason": by_reason,
        "by_sector": by_sector,
    }
