#!/usr/bin/env python3
"""
Weekly Time-Series Momentum Backtest (research spike, issue #32)
=================================================================
RESEARCH ONLY — nothing in this file is wired into the live trading engine.

Strategy v1 (pre-registered in issue #32; lookback is the only free parameter):
  - Universe:  every symbol in the daily-close file with enough history
               (>= lookback + one test window of non-missing closes).
  - Signal:    lookback return (default 12 weeks = 84 days) > 0 -> long
               candidate; candidates ranked by that return (absolute momentum).
  - Portfolio: hold the top N=3; rebalance every 7 trading days (rows).
  - Sizing:    equal *risk* budget per position, i.e. notional inversely
               proportional to 20-day realized vol, hard-capped at 10% of
               equity per asset (utils/risk_sizing.calculate_risk_capped_size,
               with the 1-sigma move over one rebalance period as the "stop").
  - Execution: signal on close of day T, filled at close of day T+1 through
               utils/execution_model.execution_price (slippage) + spot fee.
               Fee/slippage defaults are resolved by SpotBacktester from the
               live config, so SIM_REALISM_PROFILE is respected (default
               "strict" = the most conservative costs).
  - Long-only spot. No shorting, no leverage, no futures.

Validation:
  - Walk-forward (WalkForwardValidator pattern: train -> test -> step). The
    strategy is deterministic, so the "train" window is only the signal
    warm-up; every test window is evaluated strictly out-of-sample and in
    sequence. All lookbacks share the same OOS windows.
  - Monte Carlo (tools/walk_forward.MonteCarloEngine) over the trade list.
  - Lookback sensitivity sweep {8, 12, 16, 20} weeks — ALL are reported,
    none is selected.
  - Benchmarks: buy-and-hold BTC over the same OOS period, and the scalper
    baseline from data/state/baseline_report.json when present.
  - Pre-registered acceptance + kill criteria are evaluated, not optimized.

Data:
  data/historical/historical_prices_2yr.csv (columns: timestamp, symbol, price)
  — the same file tools/walk_forward.py loads. It is gitignored
  (droplet-only). To regenerate it with alpaca-py (already a dependency;
  historical crypto bars need no API keys):

      from datetime import datetime, timedelta, timezone
      from alpaca.data.historical import CryptoHistoricalDataClient
      from alpaca.data.requests import CryptoBarsRequest
      from alpaca.data.timeframe import TimeFrame
      client = CryptoHistoricalDataClient()
      req = CryptoBarsRequest(
          symbol_or_symbols=["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD",
                             "LTC/USD", "AVAX/USD", ...],
          timeframe=TimeFrame.Day,
          start=datetime.now(timezone.utc) - timedelta(days=730))
      bars = client.get_crypto_bars(req).df.reset_index()
      (bars[["timestamp", "symbol", "close"]]
           .rename(columns={"close": "price"})
           .to_csv("data/historical/historical_prices_2yr.csv", index=False))

Usage:
    cd CryptoBot
    python tools/momentum_backtest.py
    python tools/momentum_backtest.py --data cryptotrades/tests/fixtures/momentum_prices_fixture.csv
    python tools/momentum_backtest.py --lookbacks 8 12 16 20 --test-days 28

Output:
    data/backtest/momentum_backtest_results.json
"""

import sys
import json
import math
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — mirror tools/walk_forward.py
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "cryptotrades"))

from cryptotrades.utils.backtester import SpotBacktester, BacktestTrade
from cryptotrades.utils.execution_model import execution_price
from cryptotrades.utils.risk_sizing import calculate_risk_capped_size
from cryptotrades.utils.technical_indicators import rate_of_change

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HISTORICAL_CSV = ROOT_DIR / "data" / "historical" / "historical_prices_2yr.csv"
BASELINE_JSON = ROOT_DIR / "data" / "state" / "baseline_report.json"
OUTPUT_JSON = ROOT_DIR / "data" / "backtest" / "momentum_backtest_results.json"

LOOKBACK_WEEKS_SWEEP = [8, 12, 16, 20]
PRIMARY_LOOKBACK_WEEKS = 12
TOP_N = 3
REBALANCE_DAYS = 7
VOL_WINDOW = 20
RISK_PCT = 0.01            # utils/risk_sizing default per-position risk budget
NOTIONAL_CAP_PCT = 0.10    # hard cap: 10% of equity notional per asset
MIN_ORDER_USD = 1.0        # Alpaca crypto minimum order notional
STARTING_EQUITY = 10_000.0
TEST_DAYS_DEFAULT = 28     # 4 weekly rebalances per OOS window
STEP_DAYS_DEFAULT = 28     # non-overlapping OOS windows
PERIODS_PER_YEAR = 365     # crypto trades every day
NUM_SIMS_DEFAULT = 10_000

# Pre-registered thresholds (issue #32) — DO NOT TUNE.
ACCEPT_MIN_OOS_SHARPE = 0.8
ACCEPT_MIN_POSITIVE_WINDOW_FRAC = 0.70
ACCEPT_MAX_DRAWDOWN_PCT = 40.0
KILL_MIN_TRADES = 30

SEPARATOR = "=" * 72


@dataclass
class MomentumConfig:
    lookback_weeks: int = PRIMARY_LOOKBACK_WEEKS
    top_n: int = TOP_N
    rebalance_days: int = REBALANCE_DAYS
    vol_window: int = VOL_WINDOW
    risk_pct: float = RISK_PCT
    notional_cap_pct: float = NOTIONAL_CAP_PCT
    min_order_usd: float = MIN_ORDER_USD
    starting_equity: float = STARTING_EQUITY
    slippage_bps: float = 0.0
    fee_rate: float = 0.0

    @property
    def lookback_days(self) -> int:
        return int(self.lookback_weeks) * 7


def resolve_execution_costs() -> Dict[str, float]:
    """Spot slippage (bps) + fee rate exactly as SpotBacktester resolves them
    from the live config (respects SIM_REALISM_PROFILE). Costs are always on."""
    bt = SpotBacktester(market_predictor=None, enable_execution_costs=True)
    return {"slippage_bps": float(bt.spot_slippage_bps),
            "fee_rate": float(bt.spot_fee_rate)}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_daily_prices(path: Path) -> pd.DataFrame:
    """Load a daily-close CSV (timestamp, symbol, price) -> wide frame
    indexed by date with one column per symbol (NaN where missing)."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "symbol", "price"]].dropna()
    df["date"] = df["timestamp"].dt.normalize()
    df = df.sort_values(["symbol", "timestamp"])
    # One close per symbol per day (last print of the day).
    df = df.groupby(["date", "symbol"], as_index=False)["price"].last()
    wide = df.pivot(index="date", columns="symbol", values="price").sort_index()
    wide = wide.where(wide > 0)
    return wide


def find_btc_symbol(symbols: Sequence[str]) -> Optional[str]:
    """Locate BTC in common symbol spellings (BTC/USD, BTC-USD, BTCUSD, XBTUSD, BTC)."""
    for sym in symbols:
        norm = "".join(ch for ch in str(sym).upper() if ch.isalnum())
        if norm in ("BTC", "BTCUSD", "XBTUSD", "BTCUSDT", "BTCUSDC"):
            return sym
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Signal + sizing primitives
# ═══════════════════════════════════════════════════════════════════════════

def momentum_returns(prices: Sequence[float], lookback_days: int) -> np.ndarray:
    """Lookback total return p[i] / p[i - lookback] - 1 (fraction).
    NaN during warm-up or when either endpoint is missing / non-positive.
    Reuses technical_indicators.rate_of_change."""
    arr = np.asarray(prices, dtype=float)
    out = np.full(len(arr), np.nan)
    if lookback_days <= 0 or len(arr) <= lookback_days:
        return out
    roc = rate_of_change(arr.tolist(), lookback_days) / 100.0
    base = arr[:-lookback_days]
    cur = arr[lookback_days:]
    ok = np.isfinite(base) & np.isfinite(cur) & (base > 0) & (cur > 0)
    tail = np.where(ok, roc[lookback_days:], np.nan)
    out[lookback_days:] = tail
    return out


def realized_vol(prices: Sequence[float], window: int = VOL_WINDOW) -> np.ndarray:
    """Rolling std (ddof=1) of daily log returns over the last `window`
    returns. Daily (not annualized). NaN until `window` returns exist."""
    arr = np.asarray(prices, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = pd.Series(np.log(arr)).diff()
    return log_ret.rolling(window, min_periods=window).std(ddof=1).to_numpy()


def select_top_n(scores: Dict[str, float], n: int = TOP_N) -> List[str]:
    """Keep symbols with finite momentum > 0, rank descending, take top n.
    Ties broken alphabetically for determinism."""
    candidates = [(sym, float(v)) for sym, v in scores.items()
                  if v is not None and math.isfinite(float(v)) and float(v) > 0]
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    return [sym for sym, _ in candidates[:max(0, int(n))]]


def vol_scaled_notional(equity: float, price: float, daily_vol: float,
                        hold_days: int = REBALANCE_DAYS,
                        risk_pct: float = RISK_PCT,
                        notional_cap_pct: float = NOTIONAL_CAP_PCT) -> float:
    """Notional ($) for one position: inverse to realized vol, hard-capped.

    Uses utils/risk_sizing.calculate_risk_capped_size with the 1-sigma move
    over one holding period (daily_vol * sqrt(hold_days)) as the stop
    distance, so notional = risk_pct * equity / (daily_vol * sqrt(hold_days))
    capped at notional_cap_pct * equity. Unknown/zero vol -> 0 (can't size).
    """
    if (equity <= 0 or price <= 0 or daily_vol is None
            or not math.isfinite(daily_vol) or daily_vol <= 0):
        return 0.0
    stop_distance = daily_vol * math.sqrt(max(1, int(hold_days)))
    stop_price = price * (1.0 - stop_distance)
    sized = calculate_risk_capped_size(
        equity=equity, entry_price=price, stop_price=stop_price,
        risk_pct=risk_pct, notional_cap_pct=notional_cap_pct)
    return float(sized["notional"])


def compute_targets(prices: np.ndarray, symbols: Sequence[str], t: int,
                    equity: float, cfg: MomentumConfig,
                    eligible: Optional[Sequence[bool]] = None) -> Dict[str, float]:
    """Target notional per symbol from information available at close of t.

    Only prices[:t + 1] are read — anything after t cannot influence the
    result (no look-ahead by construction).
    """
    hist = np.asarray(prices, dtype=float)[: t + 1]
    lb = cfg.lookback_days
    scores: Dict[str, float] = {}
    vols: Dict[str, float] = {}
    for j, sym in enumerate(symbols):
        if eligible is not None and not eligible[j]:
            continue
        col = hist[:, j]
        if len(col) <= max(lb, cfg.vol_window):
            continue
        mom = momentum_returns(col[-(lb + 1):], lb)[-1]
        vol = realized_vol(col[-(cfg.vol_window + 1):], cfg.vol_window)[-1]
        if not (math.isfinite(mom) and math.isfinite(vol)):
            continue
        scores[sym] = float(mom)
        vols[sym] = float(vol)

    targets: Dict[str, float] = {}
    for sym in select_top_n(scores, cfg.top_n):
        j = list(symbols).index(sym)
        notional = vol_scaled_notional(
            equity, float(hist[-1, j]), vols[sym], cfg.rebalance_days,
            cfg.risk_pct, cfg.notional_cap_pct)
        if notional >= cfg.min_order_usd:
            targets[sym] = notional
    return targets


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio simulator
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MomentumBacktestResult:
    lookback_weeks: int
    equity: np.ndarray             # equity at each close (len T)
    start_idx: int                 # first signal day
    trades: List[BacktestTrade]
    rebalances: List[Dict]
    fees_paid: float
    slippage_cost: float


def run_momentum_backtest(prices: np.ndarray, symbols: Sequence[str],
                          cfg: MomentumConfig, start_idx: int,
                          eligible: Optional[Sequence[bool]] = None
                          ) -> MomentumBacktestResult:
    """Simulate the strategy. First signal at close of start_idx, then every
    cfg.rebalance_days rows; orders fill at the NEXT close. All positions are
    liquidated (with costs) at the final close."""
    P = np.asarray(prices, dtype=float)
    T, S = P.shape
    P_mark = pd.DataFrame(P).ffill().to_numpy()  # valuation only
    sym_idx = {s: j for j, s in enumerate(symbols)}

    cash = float(cfg.starting_equity)
    units = np.zeros(S)
    open_pos: Dict[str, Dict] = {}
    trades: List[BacktestTrade] = []
    rebalances: List[Dict] = []
    equity = np.full(T, float(cfg.starting_equity))
    fees_paid = 0.0
    slip_cost = 0.0
    pending: Optional[Dict[str, float]] = None
    pending_signal_idx = -1
    fills: List[Dict] = []

    def _sell(sym: str, qty: float, t: int, mid: float) -> None:
        nonlocal cash, fees_paid, slip_cost
        j = sym_idx[sym]
        qty = min(qty, units[j])
        if qty <= 0:
            return
        px = execution_price(mid, "SELL", cfg.slippage_bps)
        gross = qty * px
        fee = gross * cfg.fee_rate
        cash += gross - fee
        fees_paid += fee
        slip_cost += qty * (mid - px)
        units[j] -= qty
        fills.append({"symbol": sym, "side": "SELL", "qty": qty, "price": px, "idx": t})
        pos = open_pos[sym]
        pos["sold_units"] += qty
        pos["sell_gross"] += gross
        pos["sell_net"] += gross - fee

    def _buy(sym: str, notional: float, t: int, mid: float) -> None:
        nonlocal cash, fees_paid, slip_cost
        j = sym_idx[sym]
        notional = min(notional, cash / (1.0 + cfg.fee_rate))
        if notional < cfg.min_order_usd:
            return
        px = execution_price(mid, "BUY", cfg.slippage_bps)
        qty = notional / px
        fee = notional * cfg.fee_rate
        cash -= notional + fee
        fees_paid += fee
        slip_cost += qty * (px - mid)
        units[j] += qty
        fills.append({"symbol": sym, "side": "BUY", "qty": qty, "price": px, "idx": t})
        pos = open_pos.setdefault(sym, {
            "entry_idx": t, "bought_units": 0.0, "buy_gross": 0.0,
            "buy_cost": 0.0, "sold_units": 0.0, "sell_gross": 0.0,
            "sell_net": 0.0})
        pos["bought_units"] += qty
        pos["buy_gross"] += notional
        pos["buy_cost"] += notional + fee

    def _close_trade(sym: str, t: int, reason: str) -> None:
        pos = open_pos.pop(sym)
        units[sym_idx[sym]] = 0.0
        pnl = pos["sell_net"] - pos["buy_cost"]
        trades.append(BacktestTrade(
            symbol=sym, side="spot", direction="long",
            entry_price=pos["buy_gross"] / pos["bought_units"],
            exit_price=pos["sell_gross"] / max(pos["sold_units"], 1e-18),
            entry_idx=int(pos["entry_idx"]), exit_idx=int(t),
            amount=float(pos["bought_units"]), cost=float(pos["buy_cost"]),
            pnl_pct=float(pnl / pos["buy_cost"] * 100.0) if pos["buy_cost"] > 0 else 0.0,
            pnl_usd=float(pnl), exit_reason=reason))

    for t in range(T):
        # 1) Fill orders generated at the previous close (T+1 execution).
        if pending is not None and t == pending_signal_idx + 1:
            fills = []
            # Exits first (frees cash), then resizes/entries.
            for sym in sorted(open_pos):
                mid = P[t, sym_idx[sym]]
                if sym not in pending and math.isfinite(mid):
                    _sell(sym, units[sym_idx[sym]], t, mid)
                    _close_trade(sym, t, "MOMENTUM_EXIT")
            for sym in sorted(pending):
                j = sym_idx[sym]
                mid = P[t, j]
                if not math.isfinite(mid):
                    continue  # no print today -> skip until next rebalance
                delta = pending[sym] - units[j] * mid
                if delta <= -cfg.min_order_usd and sym in open_pos:
                    _sell(sym, -delta / mid, t, mid)
                elif delta >= cfg.min_order_usd:
                    _buy(sym, delta, t, mid)
            rebalances.append({"signal_idx": pending_signal_idx, "exec_idx": t,
                               "targets": {k: round(v, 2) for k, v in pending.items()},
                               "fills": fills})
            pending = None

        # 2) Mark to market at close.
        marks = np.where(np.isfinite(P_mark[t]), P_mark[t], 0.0)
        equity[t] = cash + float(np.dot(units, marks))

        # 3) Final close: liquidate everything with costs.
        if t == T - 1:
            fills = []  # final liquidation fills are not part of a rebalance
            for sym in sorted(open_pos):
                _sell(sym, units[sym_idx[sym]], t, float(P_mark[t, sym_idx[sym]]))
                _close_trade(sym, t, "END_OF_DATA")
            equity[t] = cash
            break

        # 4) Signal at close of t -> orders for t + 1.
        if t >= start_idx and (t - start_idx) % cfg.rebalance_days == 0:
            pending = compute_targets(P, symbols, t, equity[t], cfg, eligible)
            pending_signal_idx = t

    return MomentumBacktestResult(
        lookback_weeks=cfg.lookback_weeks, equity=equity, start_idx=start_idx,
        trades=trades, rebalances=rebalances,
        fees_paid=round(fees_paid, 2), slippage_cost=round(slip_cost, 2))


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def equity_metrics(equity: Sequence[float]) -> Dict[str, float]:
    """Sharpe (annualized, 365d, rf=0), max drawdown %, total return % from
    an equity curve whose first point is the base (pre-period close)."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return {"sharpe": 0.0, "max_drawdown_pct": 0.0, "total_return_pct": 0.0,
                "days": 0}
    rets = eq[1:] / eq[:-1] - 1.0
    std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = float(np.mean(rets) / std * math.sqrt(PERIODS_PER_YEAR)) if std > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max((peak - eq) / peak) * 100.0)
    return {"sharpe": round(sharpe, 3), "max_drawdown_pct": round(max_dd, 2),
            "total_return_pct": round(float((eq[-1] / eq[0] - 1.0) * 100.0), 2),
            "days": int(len(rets))}


def trade_stats(trades: Sequence[BacktestTrade]) -> Dict[str, float]:
    pnl = np.array([t.pnl_usd for t in trades], dtype=float)
    if len(pnl) == 0:
        return {"num_trades": 0, "win_rate": 0.0, "total_pnl_usd": 0.0,
                "avg_trade_pct": 0.0, "profit_factor": 0.0}
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    return {"num_trades": int(len(pnl)),
            "win_rate": round(float(np.mean(pnl > 0)), 4),
            "total_pnl_usd": round(float(pnl.sum()), 2),
            "avg_trade_pct": round(float(np.mean([t.pnl_pct for t in trades])), 3),
            "profit_factor": round(gp / gl, 3) if gl > 0 else float("inf") if gp > 0 else 0.0}


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward harness
# ═══════════════════════════════════════════════════════════════════════════

class MomentumWalkForward:
    """Rolling train/test/step evaluation (WalkForwardValidator pattern).

    The strategy has nothing to fit, so `train_days` is only the signal
    warm-up. One continuous simulation starts trading at the first test
    day; each test window is then scored on its own slice of the equity
    curve, sequentially and strictly out-of-sample.
    """

    def __init__(self, train_days: int, test_days: int = TEST_DAYS_DEFAULT,
                 step_days: int = STEP_DAYS_DEFAULT):
        self.train_days = int(train_days)
        self.test_days = int(test_days)
        self.step_days = int(step_days)

    def windows(self, n_days: int) -> List[tuple]:
        out = []
        start = self.train_days
        while start + self.test_days <= n_days:
            out.append((start, start + self.test_days - 1))
            start += self.step_days
        return out

    def run(self, prices: pd.DataFrame, cfg: MomentumConfig) -> Dict:
        dates = prices.index
        symbols = list(prices.columns)
        wins = self.windows(len(dates))
        if not wins:
            return {"error": f"Not enough data: {len(dates)} days < train "
                             f"{self.train_days} + test {self.test_days}"}
        last = wins[-1][1]
        P = prices.to_numpy(dtype=float)[: last + 1]
        # Universe screen: >= lookback + one test window of real closes.
        min_obs = cfg.lookback_days + self.test_days
        counts = np.isfinite(P).sum(axis=0)
        eligible = [bool(c >= min_obs) for c in counts]
        universe = [s for s, ok in zip(symbols, eligible) if ok]

        res = run_momentum_backtest(P, symbols, cfg, self.train_days - 1, eligible)
        eq = res.equity

        per_window = []
        for (s, e) in wins:
            m = equity_metrics(eq[s - 1: e + 1])
            w_trades = [t for t in res.trades if s <= t.exit_idx <= e]
            ts = trade_stats(w_trades)
            per_window.append({
                "window_start": str(pd.Timestamp(dates[s]).date()),
                "window_end": str(pd.Timestamp(dates[e]).date()),
                "net_pnl_usd": round(float(eq[e] - eq[s - 1]), 2),
                "return_pct": m["total_return_pct"],
                "sharpe": m["sharpe"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "trades_closed": ts["num_trades"],
                "win_rate": ts["win_rate"],
            })

        oos = equity_metrics(eq[self.train_days - 1: last + 1])
        ts_all = trade_stats(res.trades)
        positive = [w["net_pnl_usd"] > 0 for w in per_window]
        held = {}
        for r in res.rebalances:
            for sym in r["targets"]:
                held[sym] = held.get(sym, 0) + 1
        aggregate = {
            "lookback_weeks": cfg.lookback_weeks,
            "lookback_days": cfg.lookback_days,
            "train_window_days": self.train_days,
            "test_window_days": self.test_days,
            "step_days": self.step_days,
            "oos_start": str(pd.Timestamp(dates[self.train_days]).date()),
            "oos_end": str(pd.Timestamp(dates[last]).date()),
            "total_windows": len(per_window),
            "universe_size": len(universe),
            "oos_sharpe": oos["sharpe"],
            "oos_max_drawdown_pct": oos["max_drawdown_pct"],
            "oos_total_return_pct": oos["total_return_pct"],
            "oos_days": oos["days"],
            "pct_windows_positive": round(float(np.mean(positive)), 4),
            "num_trades": ts_all["num_trades"],
            "win_rate": ts_all["win_rate"],
            "avg_trade_pct": ts_all["avg_trade_pct"],
            "profit_factor": ts_all["profit_factor"],
            "total_trade_pnl_usd": ts_all["total_pnl_usd"],
            "fees_paid_usd": res.fees_paid,
            "slippage_cost_usd": res.slippage_cost,
            "rebalances": len(res.rebalances),
            "rebalances_in_cash": sum(1 for r in res.rebalances if not r["targets"]),
            "selection_counts": dict(sorted(held.items(), key=lambda kv: -kv[1])),
        }
        return {"aggregate": aggregate, "universe": universe,
                "per_window": per_window, "result": res}


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def buy_and_hold_benchmark(prices: pd.DataFrame, symbol: str, train_days: int,
                           windows: List[tuple], cfg: MomentumConfig) -> Dict:
    """Buy `symbol` at the first OOS close (same T+1 fill as the strategy,
    100% of equity, same costs), hold, sell at the last OOS close."""
    col = prices[symbol].ffill().to_numpy(dtype=float)
    s0, last = train_days, windows[-1][1]
    if not (math.isfinite(col[s0]) and math.isfinite(col[last])):
        return {"error": f"{symbol} has no price at OOS start/end"}
    eq = np.full(last + 1, cfg.starting_equity)
    buy_px = execution_price(col[s0], "BUY", cfg.slippage_bps)
    qty = cfg.starting_equity / (1.0 + cfg.fee_rate) / buy_px
    eq[s0:last + 1] = qty * col[s0:last + 1]
    sell_px = execution_price(col[last], "SELL", cfg.slippage_bps)
    eq[last] = qty * sell_px * (1.0 - cfg.fee_rate)
    m = equity_metrics(eq[s0 - 1: last + 1])
    positive = [eq[e] - eq[s - 1] > 0 for (s, e) in windows]
    return {"symbol": symbol, "oos_sharpe": m["sharpe"],
            "oos_max_drawdown_pct": m["max_drawdown_pct"],
            "oos_total_return_pct": m["total_return_pct"],
            "pct_windows_positive": round(float(np.mean(positive)), 4)}


def load_scalper_baseline(path: Path) -> Dict:
    """Summarize backtest/run_baseline.py output if present."""
    if not path.exists():
        return {"available": False, "path": str(path),
                "note": "baseline_report.json not found (gitignored, droplet-only). "
                        "Generate with: SIM_REALISM_PROFILE=strict python3 backtest/run_baseline.py"}
    with open(path) as f:
        rep = json.load(f)
    per_sym = rep.get("per_symbol", {}) or {}
    summary = {}
    for side in ("spot", "futures"):
        rows = [v for v in per_sym.values() if v.get("side") == side]
        if not rows:
            continue
        summary[side] = {
            "symbols": len(rows),
            "num_trades": int(sum(r.get("num_trades", 0) for r in rows)),
            "total_return_usd": round(float(sum(r.get("total_return_usd", 0.0) for r in rows)), 2),
            "mean_sharpe": round(float(np.mean([r.get("sharpe_ratio", 0.0) for r in rows])), 3),
            "worst_max_drawdown_pct": round(float(max(r.get("max_drawdown_pct", 0.0) for r in rows)), 2),
        }
    return {"available": True, "path": str(path),
            "sim_realism_profile": rep.get("sim_realism_profile"),
            "summary": summary,
            "note": "Scalper baseline is per-symbol on 1-min data over a different "
                    "period; compare direction/magnitude, not exact numbers."}


# ═══════════════════════════════════════════════════════════════════════════
# Pre-registered criteria
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_acceptance(sweep: Dict[int, Dict], primary_weeks: int,
                        btc: Optional[Dict], baseline: Optional[Dict]) -> Dict:
    """Evaluate issue #32's pre-registered acceptance + kill criteria.
    `sweep` maps lookback_weeks -> walk-forward `aggregate` dict."""
    p = sweep[primary_weeks]
    sharpes = {w: sweep[w]["oos_sharpe"] for w in sorted(sweep)}

    def _row(cid, name, threshold, value, ok):
        return {"id": cid, "name": name, "threshold": threshold,
                "value": value, "status": "PASS" if ok else "FAIL"}

    criteria = [
        _row("A1", f"OOS Sharpe ({primary_weeks}w, all windows combined)",
             f"> {ACCEPT_MIN_OOS_SHARPE}", p["oos_sharpe"],
             p["oos_sharpe"] > ACCEPT_MIN_OOS_SHARPE),
        _row("A2", "Test windows with positive net P&L after costs",
             f">= {ACCEPT_MIN_POSITIVE_WINDOW_FRAC:.0%}", p["pct_windows_positive"],
             p["pct_windows_positive"] >= ACCEPT_MIN_POSITIVE_WINDOW_FRAC),
        _row("A3", "Max drawdown (OOS)", f"< {ACCEPT_MAX_DRAWDOWN_PCT:.0f}%",
             p["oos_max_drawdown_pct"],
             p["oos_max_drawdown_pct"] < ACCEPT_MAX_DRAWDOWN_PCT),
        _row("A4", "Sharpe > 0 for every lookback " + str(sorted(sweep)),
             "all > 0", sharpes, all(v > 0 for v in sharpes.values())),
    ]
    comparison = {"strategy_oos_sharpe": p["oos_sharpe"],
                  "strategy_oos_return_pct": p["oos_total_return_pct"],
                  "strategy_oos_max_drawdown_pct": p["oos_max_drawdown_pct"]}
    if btc and "error" not in btc:
        comparison.update({
            "btc_buy_hold_sharpe": btc["oos_sharpe"],
            "btc_buy_hold_return_pct": btc["oos_total_return_pct"],
            "btc_buy_hold_max_drawdown_pct": btc["oos_max_drawdown_pct"],
            "beats_btc_on_sharpe": p["oos_sharpe"] > btc["oos_sharpe"],
        })
    comparison["scalper_baseline_available"] = bool(baseline and baseline.get("available"))
    criteria.append({"id": "A5", "name": "Honest comparison vs BTC buy-and-hold + scalper baseline",
                     "threshold": "reported", "value": comparison,
                     "status": "REPORTED"})

    kills = [_row("K1", "Total trades in full OOS period (statistical significance)",
                  f">= {KILL_MIN_TRADES}", p["num_trades"],
                  p["num_trades"] >= KILL_MIN_TRADES)]

    failed = [c["id"] for c in criteria + kills if c["status"] == "FAIL"]
    verdict = ("NOT DEPLOYABLE" if failed else
               "PASSES PRE-REGISTERED CRITERIA (paper-trade candidate only)")
    return {"criteria": criteria, "kill_criteria": kills,
            "failed": failed, "verdict": verdict}


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_report(results: Dict) -> None:
    sweep = results["sweep"]
    primary = results["config"]["primary_lookback_weeks"]
    print(f"\n{SEPARATOR}")
    print("  LOOKBACK SENSITIVITY (all reported — none selected)")
    print(SEPARATOR)
    print(f"  {'Lookback':>9s} {'Sharpe':>8s} {'MaxDD%':>8s} {'Ret%':>9s} "
          f"{'Win%':>7s} {'Trades':>7s} {'Win-wnd%':>9s} {'Fees$':>9s}")
    for wk in sorted(sweep, key=int):
        a = sweep[wk]["aggregate"]
        print(f"  {str(wk) + 'w':>9s} {a['oos_sharpe']:>8.3f} "
              f"{a['oos_max_drawdown_pct']:>8.2f} {a['oos_total_return_pct']:>9.2f} "
              f"{a['win_rate']:>7.1%} {a['num_trades']:>7d} "
              f"{a['pct_windows_positive']:>9.1%} {a['fees_paid_usd']:>9.2f}")

    a = sweep[str(primary)]["aggregate"]
    print(f"\n{SEPARATOR}")
    print(f"  WALK-FORWARD — PRIMARY LOOKBACK {primary}w "
          f"({a['oos_start']} -> {a['oos_end']}, {a['total_windows']} windows)")
    print(SEPARATOR)
    print(f"  Universe: {a['universe_size']} symbols | Train/Test/Step: "
          f"{a['train_window_days']}d / {a['test_window_days']}d / {a['step_days']}d")
    print(f"  {'Window':>23s} {'Ret%':>8s} {'Sharpe':>8s} {'MaxDD%':>8s} "
          f"{'Trades':>7s} {'Win%':>7s}")
    for w in sweep[str(primary)]["per_window"]:
        print(f"  {w['window_start']}..{w['window_end']} {w['return_pct']:>8.2f} "
              f"{w['sharpe']:>8.3f} {w['max_drawdown_pct']:>8.2f} "
              f"{w['trades_closed']:>7d} {w['win_rate']:>7.1%}")

    print(f"\n{SEPARATOR}")
    print("  BENCHMARKS")
    print(SEPARATOR)
    btc = results["benchmarks"].get("btc_buy_and_hold") or {}
    if "error" in btc or not btc:
        print(f"  BTC buy-and-hold: unavailable ({btc.get('error', 'BTC not in data')})")
    else:
        print(f"  BTC buy-and-hold ({btc['symbol']}): Sharpe {btc['oos_sharpe']:.3f} | "
              f"MaxDD {btc['oos_max_drawdown_pct']:.2f}% | Return "
              f"{btc['oos_total_return_pct']:.2f}% | Win-wnd {btc['pct_windows_positive']:.1%}")
    base = results["benchmarks"]["scalper_baseline"]
    if base.get("available"):
        for side, s in base["summary"].items():
            print(f"  Scalper baseline ({side}): {s['num_trades']} trades | "
                  f"P&L ${s['total_return_usd']:.2f} | mean Sharpe {s['mean_sharpe']:.3f}")
    else:
        print(f"  Scalper baseline: {base['note']}")

    mc = results.get("monte_carlo")
    if mc:
        print(f"\n  Monte Carlo ({mc['num_simulations']:,} bootstraps of "
              f"{mc['num_trades_per_sim']} trades, {primary}w): "
              f"P(profit) {mc['probability_of_profit']:.1%} | median terminal "
              f"${mc['median_terminal_pnl_usd']:.2f} | CVaR5 ${mc['cvar_5pct_usd']:.2f}")

    acc = results["acceptance"]
    print(f"\n{SEPARATOR}")
    print("  PRE-REGISTERED ACCEPTANCE / KILL CRITERIA (issue #32)")
    print(SEPARATOR)
    for c in acc["criteria"] + acc["kill_criteria"]:
        print(f"  [{c['status']:>8s}] {c['id']} {c['name']} ({c['threshold']}): {_fmt(c['value'])}")
    print(f"\n  VERDICT: {acc['verdict']}")
    if results.get("data_warning"):
        print(f"  WARNING: {results['data_warning']}")


def save_results(results: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {path}")


def _monte_carlo(trades: List[BacktestTrade], sims: int) -> Optional[Dict]:
    """Reuse tools/walk_forward.MonteCarloEngine over the trade list."""
    if not trades:
        return None
    sys.path.insert(0, str(TOOLS_DIR))
    from walk_forward import MonteCarloEngine
    df = pd.DataFrame([asdict(t) for t in trades])
    mc = MonteCarloEngine(df, num_sims=sims).run()
    # Its Sharpe/ruin fields assume scalper trade frequency + config balances;
    # keep only the frequency-independent ones.
    keep = ("num_simulations", "num_trades_per_sim", "probability_of_profit",
            "expected_value_per_trade_usd", "cvar_5pct_usd", "mean_terminal_pnl_usd",
            "median_terminal_pnl_usd", "std_terminal_pnl_usd", "mean_max_drawdown_usd",
            "mean_profit_factor")
    out = {k: mc[k] for k in keep}
    out["terminal_pnl_percentiles"] = {k: v["terminal_pnl"]
                                       for k, v in mc["confidence_intervals"].items()}
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_study(prices: pd.DataFrame, lookbacks: Sequence[int], primary: int,
              test_days: int, step_days: int, train_days: Optional[int],
              starting_equity: float, top_n: int, sims: int,
              btc_symbol: Optional[str], baseline_path: Path) -> Dict:
    lookbacks = sorted(set(int(w) for w in lookbacks) | {int(primary)})
    costs = resolve_execution_costs()
    if train_days is None:
        # Common OOS start for every lookback: longest lookback + 1 day.
        train_days = max(lookbacks) * 7 + 1
    wf = MomentumWalkForward(train_days, test_days, step_days)

    sweep, sweep_aggs, primary_res = {}, {}, None
    for wk in lookbacks:
        cfg = MomentumConfig(lookback_weeks=wk, top_n=top_n,
                             starting_equity=starting_equity, **costs)
        out = wf.run(prices, cfg)
        if "error" in out:
            raise ValueError(out["error"])
        sweep[str(wk)] = {"aggregate": out["aggregate"], "per_window": out["per_window"]}
        sweep_aggs[wk] = out["aggregate"]
        if wk == primary:
            primary_res = out

    cfg_p = MomentumConfig(lookback_weeks=primary, top_n=top_n,
                           starting_equity=starting_equity, **costs)
    windows = wf.windows(len(prices.index))
    btc_sym = btc_symbol or find_btc_symbol(prices.columns)
    btc = (buy_and_hold_benchmark(prices, btc_sym, train_days, windows, cfg_p)
           if btc_sym in prices.columns else {"error": "BTC not found in data"})
    baseline = load_scalper_baseline(baseline_path)

    trades = primary_res["result"].trades
    results = {
        "config": {
            "lookbacks_weeks": lookbacks, "primary_lookback_weeks": primary,
            "top_n": top_n, "rebalance_days": REBALANCE_DAYS,
            "vol_window": VOL_WINDOW, "risk_pct": RISK_PCT,
            "notional_cap_pct": NOTIONAL_CAP_PCT, "starting_equity": starting_equity,
            "train_days": train_days, "test_days": test_days, "step_days": step_days,
            "execution": "signal on close T, fill on close T+1",
            **costs,
        },
        "data": {"days": int(len(prices.index)), "symbols": list(prices.columns),
                 "first_date": str(prices.index[0].date()),
                 "last_date": str(prices.index[-1].date())},
        "sweep": sweep,
        "benchmarks": {"btc_buy_and_hold": btc, "scalper_baseline": baseline},
        "monte_carlo": _monte_carlo(trades, sims) if sims > 0 else None,
        "primary_trades": [asdict(t) for t in trades],
    }
    results["acceptance"] = evaluate_acceptance(sweep_aggs, primary, btc, baseline)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Weekly time-series momentum backtest (issue #32 research spike)")
    parser.add_argument("--data", type=Path, default=HISTORICAL_CSV,
                        help="Daily closes CSV (timestamp, symbol, price)")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--baseline", type=Path, default=BASELINE_JSON)
    parser.add_argument("--lookbacks", type=int, nargs="+", default=LOOKBACK_WEEKS_SWEEP,
                        help="Lookbacks in weeks to sweep (default: 8 12 16 20)")
    parser.add_argument("--primary-lookback", type=int, default=PRIMARY_LOOKBACK_WEEKS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--train-days", type=int, default=None,
                        help="Warm-up before first OOS day (default: max lookback*7 + 1)")
    parser.add_argument("--test-days", type=int, default=TEST_DAYS_DEFAULT)
    parser.add_argument("--step-days", type=int, default=STEP_DAYS_DEFAULT)
    parser.add_argument("--starting-equity", type=float, default=STARTING_EQUITY)
    parser.add_argument("--sims", type=int, default=NUM_SIMS_DEFAULT,
                        help="Monte Carlo simulations (0 to skip)")
    parser.add_argument("--btc-symbol", type=str, default=None)
    args = parser.parse_args(argv)

    start_time = datetime.now()
    print(f"\n{'#' * 72}")
    print("  WEEKLY TIME-SERIES MOMENTUM BACKTEST (issue #32 — research only)")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 72}")

    if not args.data.exists():
        print(f"\n  ERROR: daily price file not found at {args.data}")
        print("  It is gitignored/droplet-only. See this module's docstring for how to")
        print("  regenerate it, or pass --data <csv> (columns: timestamp, symbol, price).")
        return 1

    prices = load_daily_prices(args.data)
    print(f"\n  Loaded {prices.shape[1]} symbols x {prices.shape[0]} days from {args.data}")

    results = run_study(prices, args.lookbacks, args.primary_lookback,
                        args.test_days, args.step_days, args.train_days,
                        args.starting_equity, args.top_n, args.sims,
                        args.btc_symbol, args.baseline)
    results["run_timestamp"] = start_time.isoformat()
    results["data"]["path"] = str(args.data)
    if "fixture" in args.data.name.lower():
        results["data_warning"] = ("SYNTHETIC FIXTURE DATA — pipeline check only; "
                                   "these numbers are NOT evidence for or against the strategy.")

    print_report(results)
    results["elapsed_seconds"] = round((datetime.now() - start_time).total_seconds(), 1)
    save_results(results, args.output)
    print(f"\n{SEPARATOR}\n  Completed in {results['elapsed_seconds']:.1f}s\n{SEPARATOR}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
