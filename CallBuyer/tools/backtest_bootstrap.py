"""
CallBuyer ML Bootstrap Backtest
================================
CallBuyer's ML model needs ML_WARMUP_TRADES (15) completed trades before it
even starts training, and only had 4 real ones after a week of paper trading.
This script generates realistic (features, outcome) training rows by
replaying the LIVE entry/exit rules against real historical stock prices, so
the model can be pretrained instead of waiting on live trade volume.

What's REAL vs APPROXIMATED:
  - Entry signal: 100% real — reuses the live CallBuyerFeatureEngine
    (build_features/compute_rule_score) and MetaLearner.evaluate(), so
    entries are gated by the actual current strategy thresholds.
  - Underlying prices: 100% real (Alpaca historical daily bars).
  - Option prices: APPROXIMATED via Black-Scholes. Real historical options
    chains (with real bid/ask/IV) aren't available, so IV is proxied by
    trailing 20-day realized volatility (same formula as
    CallBuyerAPI.calculate_hv20) scaled by VOL_RISK_PREMIUM_MULT, since
    implied vol historically runs above realized vol. IV is held constant
    for the life of each simulated trade rather than re-estimated daily.
  - Regime detection and morning/afternoon time-of-day adjustments are
    skipped (treated as neutral) — the live bot filters slightly more than
    this backtest does.
  - Only one simulated position per symbol at a time.

Treat the resulting model as a warm start, not a finished product — it will
keep retraining on real live outcomes as they accumulate (ML_RETRAIN_TRADES
= 15) and should gradually be dominated by real data.

Usage (from the CallBuyer/ directory):
    python tools/backtest_bootstrap.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import shutil
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import numpy as np

from core.config import CallBuyerConfig
from core.feature_engine import CallBuyerFeatureEngine, FEATURE_NAMES
from core.meta_learner import MetaLearner
from core.ml_model import CallBuyerMLModel

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

YEARS_OF_HISTORY = 3
RISK_FREE_RATE = 0.04
VOL_RISK_PREMIUM_MULT = 1.15   # IV typically runs ~15% above realized vol
MIN_OPTION_PRICE = 0.05
MIN_BARS_FOR_FEATURES = 50
HISTORICAL_CACHE_DIR = "data/historical"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(spot: float, strike: float, t_years: float, iv: float,
                   r: float = RISK_FREE_RATE) -> float:
    """Black-Scholes European call price; intrinsic value at/after expiry."""
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.0)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)


def realized_vol(closes: np.ndarray, window: int = 20) -> float:
    """Annualized realized vol from trailing daily log returns (same formula
    as CallBuyerAPI.calculate_hv20)."""
    if len(closes) < window + 1:
        window = len(closes) - 1
    if window < 5:
        return 0.30
    returns = np.diff(np.log(closes[-(window + 1):]))
    return float(np.std(returns) * np.sqrt(252))


class _Bar:
    """Minimal stand-in with the attrs build_features()/compute_rule_score() need."""
    __slots__ = ("open", "high", "low", "close", "volume", "timestamp")

    def __init__(self, o, h, l, c, v, ts):
        self.open, self.high, self.low, self.close, self.volume, self.timestamp = o, h, l, c, v, ts


def _cache_path(symbol: str) -> str:
    return os.path.join(HISTORICAL_CACHE_DIR, f"{symbol}_daily.csv")


def load_or_download_bars(client, symbol: str, start, end) -> List[_Bar]:
    """Load cached daily bars from disk, or download+cache from Alpaca."""
    path = _cache_path(symbol)
    if os.path.exists(path):
        bars = []
        with open(path) as f:
            next(f)  # header
            for line in f:
                ts, o, h, l, c, v = line.strip().split(",")
                bars.append(_Bar(float(o), float(h), float(l), float(c), float(v),
                                  datetime.fromisoformat(ts)))
        return bars

    # Free-tier Alpaca market data subscriptions only permit the IEX feed,
    # not full-market SIP data — using the default (SIP) 403s on historical bars.
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                            start=start, end=end, feed=DataFeed.IEX)
    resp = client.get_stock_bars(req)
    try:
        raw = list(resp[symbol])
    except (KeyError, IndexError):
        raw = []

    os.makedirs(HISTORICAL_CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for b in raw:
            f.write(f"{b.timestamp.isoformat()},{b.open},{b.high},{b.low},{b.close},{b.volume}\n")

    return [_Bar(float(b.open), float(b.high), float(b.low), float(b.close),
                  float(b.volume), b.timestamp) for b in raw]


def simulate_symbol(symbol: str, bars: List[_Bar], spy_bars: List[_Bar],
                     spy_date_index: Dict[date, int],
                     features_engine: CallBuyerFeatureEngine,
                     meta: MetaLearner, config: CallBuyerConfig) -> List[dict]:
    """Walk forward through daily bars, simulating entries/exits for one symbol."""
    outcomes = []
    n = len(bars)
    i = MIN_BARS_FOR_FEATURES
    max_hold_days = config.TARGET_DTE

    while i < n - 1:
        window = bars[:i + 1]
        bar_date = window[-1].timestamp.date()
        spy_idx = spy_date_index.get(bar_date)
        spy_window = spy_bars[:spy_idx + 1] if spy_idx is not None else None

        feature_vec = features_engine.build_features(daily_bars=window, spy_bars=spy_window)
        if feature_vec is None:
            i += 1
            continue

        rsi = feature_vec[0] * 100
        if rsi < config.MIN_RSI or rsi > config.MAX_RSI:
            i += 1
            continue

        rule_score = features_engine.compute_rule_score(feature_vec)
        confidence, should_trade, reason = meta.evaluate(
            rule_score=rule_score, ml_proba=0.5, ml_active=False,
        )
        if not should_trade:
            i += 1
            continue

        # ── Simulate entry ──
        closes = np.array([float(b.close) for b in window])
        entry_price = closes[-1]
        strike = round(entry_price * (1 + config.TARGET_OTM_PCT), 0)
        if strike <= 0:
            i += 1
            continue

        iv = realized_vol(closes) * VOL_RISK_PREMIUM_MULT
        iv = max(0.15, min(iv, 1.5))
        dte_days = config.TARGET_DTE
        entry_option_price = bs_call_price(entry_price, strike, dte_days / 365.0, iv)
        if entry_option_price < MIN_OPTION_PRICE:
            i += 1
            continue

        # ── Simulate exit, walking forward day by day ──
        high_water = entry_option_price
        exit_j = None
        exit_pnl_pct = None
        last_j = min(i + dte_days, n - 1)
        for j in range(i + 1, last_j + 1):
            days_elapsed = j - i
            remaining_t = max(dte_days - days_elapsed, 0) / 365.0
            spot_j = float(bars[j].close)
            option_price_j = bs_call_price(spot_j, strike, remaining_t, iv)
            pnl_pct = (option_price_j - entry_option_price) / entry_option_price
            high_water = max(high_water, option_price_j)

            if remaining_t <= 0:
                exit_j, exit_pnl_pct = j, pnl_pct
                break
            if pnl_pct >= config.TAKE_PROFIT_PCT:
                exit_j, exit_pnl_pct = j, pnl_pct
                break
            if pnl_pct <= config.STOP_LOSS_PCT:
                exit_j, exit_pnl_pct = j, pnl_pct
                break
            if (dte_days - days_elapsed) <= config.MIN_DTE_EXIT:
                exit_j, exit_pnl_pct = j, pnl_pct
                break
            if high_water > entry_option_price * 1.30:
                drawdown = (option_price_j - high_water) / high_water if high_water > 0 else 0
                if drawdown <= -config.TRAILING_STOP_PCT:
                    exit_j, exit_pnl_pct = j, pnl_pct
                    break

        if exit_j is None:
            exit_j, exit_pnl_pct = last_j, pnl_pct  # ran out of bars, mark-to-market

        outcomes.append({
            "symbol": symbol,
            "timestamp": window[-1].timestamp.isoformat(),
            "features": feature_vec.tolist(),
            "feature_names": FEATURE_NAMES,
            "outcome": 1 if exit_pnl_pct > 0 else 0,
            "pnl_pct": exit_pnl_pct * 100,
            "completed": True,
            "source": "backtest_bootstrap",
        })

        i = exit_j + 1  # resume scanning after this trade closes

    return outcomes


def main():
    config = CallBuyerConfig()
    if not config.has_keys:
        print("ERROR: ALPACA_API_KEY/ALPACA_API_SECRET not set — cannot fetch historical bars.")
        return

    client = StockHistoricalDataClient(api_key=config.API_KEY, secret_key=config.API_SECRET)
    end = datetime.now()
    start = end - timedelta(days=365 * YEARS_OF_HISTORY)

    print(f"=== Downloading/loading {YEARS_OF_HISTORY}y daily bars ===")
    symbols = list(config.WATCHLIST)
    spy_bars = load_or_download_bars(client, "SPY", start, end)
    spy_date_index = {b.timestamp.date(): idx for idx, b in enumerate(spy_bars)}
    print(f"  SPY: {len(spy_bars)} bars")

    features_engine = CallBuyerFeatureEngine()
    meta = MetaLearner(state_dir="data/state")  # reads current live thresholds, doesn't write

    all_outcomes = []
    for symbol in symbols:
        bars = load_or_download_bars(client, symbol, start, end)
        if len(bars) < MIN_BARS_FOR_FEATURES + 5:
            print(f"  {symbol}: only {len(bars)} bars — skipping")
            continue
        outcomes = simulate_symbol(symbol, bars, spy_bars, spy_date_index,
                                    features_engine, meta, config)
        wins = sum(o["outcome"] for o in outcomes)
        print(f"  {symbol}: {len(bars)} bars -> {len(outcomes)} simulated trades "
              f"({wins}W/{len(outcomes) - wins}L)")
        all_outcomes.extend(outcomes)

    if not all_outcomes:
        print("No simulated trades generated — nothing to bootstrap with.")
        return

    total = len(all_outcomes)
    wins = sum(o["outcome"] for o in all_outcomes)
    avg_pnl = sum(o["pnl_pct"] for o in all_outcomes) / total
    print(f"\n=== Simulated {total} trades total: {wins}W/{total - wins}L "
          f"({wins / total:.1%} win rate), avg pnl {avg_pnl:+.1f}% ===")

    # ── Merge into the live features_log.json (back up first) ──
    features_file = os.path.join("data", "state", "features_log.json")
    os.makedirs(os.path.dirname(features_file), exist_ok=True)
    existing = []
    if os.path.exists(features_file):
        backup = features_file + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(features_file, backup)
        print(f"Backed up existing features log to {backup}")
        with open(features_file) as f:
            existing = json.load(f)

    merged = existing + all_outcomes
    if len(merged) > 1000:
        merged = merged[-1000:]
    with open(features_file, "w") as f:
        json.dump(merged, f, indent=1)
    print(f"Wrote {len(merged)} total training rows to {features_file} "
          f"({len(existing)} pre-existing + {len(all_outcomes)} backtested)")

    # ── Train the model on the merged data ──
    print("\n=== Training model ===")
    ml = CallBuyerMLModel(models_dir=config.MODEL_DIR, state_dir="data/state")
    accepted = ml.train()
    print(f"Model {'ACCEPTED' if accepted else 'below quality gate'}: "
          f"accuracy={ml.accuracy:.1%} trained_on={ml.trained_on}")


if __name__ == "__main__":
    main()
