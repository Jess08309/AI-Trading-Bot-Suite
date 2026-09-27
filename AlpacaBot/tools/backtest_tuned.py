"""
AlpacaBot Strategy-Improvement Research -- Comparative Backtest (STEP 2/3)
============================================================================
RESEARCH ONLY. New file; does not modify core/scanner.py, core/indicators.py,
or any existing tools/backtest*.py. Run with AlpacaBot's own venv, from the
AlpacaBot/ directory (so its .env and relative data paths resolve), while the
alpacabot systemd service is stopped.

Reuses the Black-Scholes options P&L simulation from tools/backtest_mtf.py
(bs_price/estimate_iv/norm_cdf copied verbatim below) and the real 14-
indicator rule-based signal logic from core/scanner.py::_generate_signal
(Config A) plus a trimmed 4-indicator variant (Config B/C).

STEP 1 recap (full matrix in docs/indicator_correlation.md):
  Real correlation on 5,476 samples of trailing-6mo SPY 10-min bars shows
  RSI / MACD histogram / BB position are NOT mutually redundant (max
  pairwise |r| = 0.78, under the 0.8 threshold) -- the STOP condition does
  not trigger. One genuine 5-way redundancy cluster was found instead:
  bb_position / zscore / williams_r / cci / stochastic (all |r| >= 0.8 --
  mean-reversion/oscillator family carrying the same information). Trimmed
  set for Configs B/C chosen from real data, not assumed: rsi, macd_hist,
  bb_position (one representative of the big cluster), and volatility_ratio
  (the most orthogonal 4th indicator available -- max |r| = 0.03 against the
  other 3, versus 0.20-0.60 for every other candidate).

Config A (BASELINE): reproduces the current production rule-based strategy
  as closely as a closes-only backtest can:
    - 10-min bars, all 14 indicators, current core/config.py thresholds
      (MIN_SIGNAL_SCORE, per-symbol DTE map, ITM strike targeting,
      stop-loss/take-profit/trailing, MAX_POSITION_PCT sizing).
    - PLUS the live ML confidence/agreement gate (OptionsMLModel, the actual
      saved data/models/options_model.joblib artifact, MIN_ML_CONFIDENCE
      =0.55) -- added per explicit owner request, to see whether the ML
      filter was screening out bad trades or the rules themselves are the
      problem. predict() is called exactly as trading_engine.py calls it
      (prices-only, no timestamp/day_open/prev_close -- the live call site
      doesn't pass those either, so this is a faithful replay of the actual
      invocation, not an enhancement of it).
    - Sentiment / SPY-regime-filter / meta-learner ensemble / put-win-rate
      auto-disable are NOT modeled: these need live external data feeds or
      rolling multi-day trade-history state that cannot be replayed from
      historical bars alone. No existing backtest tool in this repo models
      them either.
Config B (SCALP-TUNED): 2-min bars (resampled from 1-min), trimmed
  4-indicator scoring, 15-min trend-agreement filter, DTE=1 (fixed --
  matches config.py's MIN_DTE floor and this repo's own "1DTE scalp"
  precedent in backtest_mtf.py).
Config C (SWING-TUNED): 1-hour bars (resampled from 1-min), trimmed
  4-indicator scoring, daily-chart trend-agreement filter, DTE=10 (fixed --
  comfortably beyond the max hold to limit theta burn), 2-5 day target
  holds (a MIN_HOLD guard blocks the time-based MAX_HOLD exit before 2 days
  elapse; real risk stops -- stop-loss/take-profit/trailing -- remain
  always-on regardless of MIN_HOLD, since disabling risk controls to force
  a hold period would not be a safe design).

All 3 configs share: universe, date range, sizing formula, MAX_POSITIONS,
and the Black-Scholes pricing/IV-estimation -- only bar timeframe,
indicator set, trend filter, and DTE differ (the intentional experiment
variables). Universe is SPY/QQQ/AAPL/MSFT/NVDA per the task spec; note
core/scanner.py's SCANNER_UNIVERSE comment block documents SPY/QQQ/MSFT as
historically-eliminated DROP symbols (unprofitable, excluded from the live
scanner's real universe) -- kept here anyway since the task explicitly
requires this universe for controlled comparison.

Usage (from AlpacaBot/, with its own venv):
  .venv/bin/python3 tools/backtest_tuned.py [--symbols SPY,QQQ,...] [--days 200] [--force-refresh]
"""
import sys, os, math, argparse, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from collections import defaultdict

from core.indicators import (
    compute_all_indicators,
    rsi as rsi_series, macd as macd_series,
    bollinger_bands as bb_series, volatility_ratio as volr_series,
)
from core.config import Config, SYMBOL_DTE_MAP, DEFAULT_DTE
from tools.bar_cache import fetch_1min_cached

try:
    from utils.ml_model import OptionsMLModel
    _ML_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - defensive only
    OptionsMLModel = None
    _ML_IMPORT_ERROR = e

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = Config()

# =============================================================
#  SHARED CONFIG (identical across A/B/C unless noted)
# =============================================================

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
HISTORY_DAYS = 200                              # >=3 months of tradeable bars after warm-up
INITIAL_BALANCE = cfg.INITIAL_BALANCE            # current production default ($50,000)
MAX_POSITIONS = cfg.MAX_POSITIONS                # 3
MAX_POSITION_PCT = cfg.MAX_POSITION_PCT          # 0.15
STOP_LOSS = cfg.STOP_LOSS_PCT                    # -0.20
TAKE_PROFIT = cfg.TAKE_PROFIT_PCT                # +0.50
TRAILING_STOP = cfg.TRAILING_STOP_PCT            # 0.12
TRAILING_TRIGGER = cfg.TRAILING_TRIGGER          # 0.15
TARGET_ITM_PCT = cfg.TARGET_ITM_PCT              # 0.05
MIN_ML_CONFIDENCE = 0.55                         # matches trading_engine.py's hardcoded constant
DTE_EXIT_BUFFER_DAYS = 0.15                      # force-close when this close to expiry

MODE_LABELS = {
    "A_baseline": "A: BASELINE (10-min, 14-ind, +ML gate)",
    "B_scalp": "B: SCALP-TUNED (2-min, 4-ind, 15m trend filter)",
    "C_swing": "C: SWING-TUNED (1-hour, 4-ind, daily trend filter)",
}


# =============================================================
#  BLACK-SCHOLES (copied verbatim from tools/backtest_mtf.py)
# =============================================================

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, sigma, opt_type, r=0.05):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.01) if opt_type == "call" else max(K - S, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def estimate_iv(closes, window=30, bars_per_day=78):
    if len(closes) < window + 1:
        return 0.25
    rets = np.diff(np.log(closes[-window - 1:]))
    hv = float(np.std(rets) * np.sqrt(bars_per_day * 252))
    return max(0.12, hv * 1.15)


def select_strike(S, direction, itm_pct=TARGET_ITM_PCT):
    """Single-strike ITM-target simplification (mirrors options_handler.py's
    TARGET_ITM_PCT concept -- no real chain data available in a closes-only
    backtest, same class of simplification backtest_mtf.py already uses for
    its own OTM strike selection)."""
    offset = S * itm_pct
    if direction == "call":
        return round(S - offset, 2)
    return round(S + offset, 2)


# =============================================================
#  SIGNAL GENERATION -- Config A: full 14 indicators
#  (copied verbatim from core/scanner.py::_generate_signal, with the
#  MIN_SIGNAL_SCORE threshold read from live config instead of hardcoded)
# =============================================================

def generate_signal_full(chunk, min_signal_score):
    indicators = compute_all_indicators(chunk)
    bull, bear = 0, 0

    rsi = indicators.get("rsi", 50)
    if rsi < 25:
        bull += 2
    elif 35 < rsi < 55:
        bull += 1
    elif rsi > 75:
        bear += 2
    elif 50 < rsi < 65:
        bear += 1

    macd_h = indicators.get("macd_hist", 0)
    if macd_h > 0:
        bull += 1
        if macd_h > 0.1:
            bull += 1
    elif macd_h < 0:
        bear += 1
        if macd_h < -0.1:
            bear += 1

    stoch = indicators.get("stochastic", 50)
    if stoch < 20:
        bull += 1
    elif stoch > 80:
        bear += 1

    bb = indicators.get("bb_position", 0.5)
    if bb < 0.10:
        bull += 2
    elif bb > 0.90:
        bear += 2
    elif bb < 0.30:
        bull += 1
    elif bb > 0.70:
        bear += 1

    atr_n = indicators.get("atr_normalized", 0)
    if atr_n > 0.005:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    cci_val = indicators.get("cci", 0)
    if cci_val < -100:
        bull += 1
    elif cci_val > 100:
        bear += 1

    roc_val = indicators.get("roc", 0)
    if roc_val > 0.3:
        bull += 1
    elif roc_val < -0.3:
        bear += 1

    wr = indicators.get("williams_r", -50)
    if wr > -20:
        bear += 1
    elif wr < -80:
        bull += 1

    vol_r = indicators.get("volatility_ratio", 1.0)
    if vol_r > 1.3:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    zs = indicators.get("zscore", 0)
    if zs < -2.0:
        bull += 1
    elif zs > 2.0:
        bear += 1

    ts = indicators.get("trend_strength", 0)
    if ts > 25:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    pc1 = indicators.get("price_change_1", 0)
    pc5 = indicators.get("price_change_5", 0)
    if pc1 > 0.001:
        bull += 1
    elif pc1 < -0.001:
        bear += 1
    if pc5 > 0.003:
        bull += 1
    elif pc5 < -0.003:
        bear += 1

    direction, score = None, 0
    if bull >= min_signal_score and bull > bear + 1:
        direction, score = "call", bull
    elif bear >= min_signal_score and bear > bull + 1:
        direction, score = "put", bear
    if direction is None:
        return None, 0, indicators

    pc20 = indicators.get("price_change_20", 0) if len(chunk) > 20 else 0
    if direction == "put" and pc20 > 0.015 and pc5 > 0.005:
        return None, 0, indicators
    if direction == "call" and pc20 < -0.015 and pc5 < -0.005:
        return None, 0, indicators

    return direction, score, indicators


# =============================================================
#  SIGNAL GENERATION -- Config B/C: trimmed 4-indicator set
#  rsi, macd_hist, bb_position, volatility_ratio -- chosen from the real
#  STEP 1 correlation results (see module docstring). Same per-indicator
#  point values as the full system; threshold recalibrated for the smaller
#  max-achievable score with only 4 indicators (documented judgment call).
# =============================================================

TRIMMED_MIN_SIGNAL_SCORE = 3   # out of a max of ~7 achievable with 4 indicators


def compute_trimmed_indicators(chunk):
    rsi_vals = rsi_series(chunk)
    _, _, macd_hist = macd_series(chunk)
    _, _, _, bb_pct = bb_series(chunk)
    vol_r = volr_series(chunk)
    return {
        "rsi": rsi_vals[-1], "macd_hist": macd_hist[-1],
        "bb_position": bb_pct[-1], "volatility_ratio": vol_r[-1],
    }


def generate_signal_trimmed(chunk, min_signal_score=TRIMMED_MIN_SIGNAL_SCORE):
    if len(chunk) < 30:
        return None, 0, {}
    indicators = compute_trimmed_indicators(chunk)
    bull, bear = 0, 0

    rsi = indicators["rsi"]
    if rsi < 25:
        bull += 2
    elif 35 < rsi < 55:
        bull += 1
    elif rsi > 75:
        bear += 2
    elif 50 < rsi < 65:
        bear += 1

    macd_h = indicators["macd_hist"]
    if macd_h > 0:
        bull += 1
        if macd_h > 0.1:
            bull += 1
    elif macd_h < 0:
        bear += 1
        if macd_h < -0.1:
            bear += 1

    bb = indicators["bb_position"]
    if bb < 0.10:
        bull += 2
    elif bb > 0.90:
        bear += 2
    elif bb < 0.30:
        bull += 1
    elif bb > 0.70:
        bear += 1

    vol_r = indicators["volatility_ratio"]
    if vol_r > 1.3:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    direction, score = None, 0
    if bull >= min_signal_score and bull > bear + 1:
        direction, score = "call", bull
    elif bear >= min_signal_score and bear > bull + 1:
        direction, score = "put", bear
    return direction, score, indicators


def trend_bias_trimmed(chunk):
    """Lightweight trend-agreement filter for Config B (15-min bars) and
    Config C (daily bars). Returns 'call' (bullish), 'put' (bearish), or
    None (flat -- no directional constraint applied)."""
    if len(chunk) < 25:
        return None
    ind = compute_trimmed_indicators(chunk)
    bull, bear = 0, 0
    if ind["rsi"] < 45:
        bull += 1
    elif ind["rsi"] > 55:
        bear += 1
    if ind["macd_hist"] > 0:
        bull += 1
    elif ind["macd_hist"] < 0:
        bear += 1
    if ind["bb_position"] < 0.40:
        bull += 1
    elif ind["bb_position"] > 0.60:
        bear += 1
    if bull >= 2 and bull > bear:
        return "call"
    if bear >= 2 and bear > bull:
        return "put"
    return None


def get_dte_for_symbol_A(symbol):
    return SYMBOL_DTE_MAP.get(symbol, DEFAULT_DTE)


# =============================================================
#  ML GATE -- Config A only
# =============================================================

class MLGate:
    """Wraps the real, saved OptionsMLModel for offline backtest replay.
    Calls .predict(chunk) exactly as trading_engine.py does (no timestamp/
    day_open/prev_close -- the live call site doesn't pass them either)."""

    def __init__(self):
        self.ready = False
        self.model = None
        if OptionsMLModel is None:
            print(f"  ML model import failed ({_ML_IMPORT_ERROR}) -- "
                  f"Config A will run WITHOUT the ML gate")
            return
        m = OptionsMLModel(model_dir=os.path.join(_ROOT, "data", "models"))
        if m.load_model() and m.model is not None:
            self.model = m
            self.ready = True
            print(f"  ML model loaded OK (test_accuracy={m.test_accuracy:.1%})")
        else:
            print("  ML model failed to load -- Config A will run WITHOUT the ML gate")

    def check(self, chunk, direction):
        """Returns (allowed, ml_confidence, ml_direction). Mirrors
        trading_engine.py's exact two hard gates (confidence + direction
        agreement) only -- sentiment/regime/meta-learner are out of scope."""
        if not self.ready:
            return True, 0.5, 0.5
        pred = self.model.predict(chunk)
        ml_conf = pred["confidence"]
        ml_dir = pred["direction"]
        ml_agrees = (direction == "call" and ml_dir > 0.5) or (direction == "put" and ml_dir < 0.5)
        if ml_conf >= MIN_ML_CONFIDENCE and not ml_agrees:
            return False, ml_conf, ml_dir
        if ml_conf < MIN_ML_CONFIDENCE:
            return False, ml_conf, ml_dir
        return True, ml_conf, ml_dir


# =============================================================
#  RESAMPLING (generalizes backtest_mtf.py's resample_to_10min pattern)
# =============================================================

def resample_intraday(df, factor):
    """Take every `factor`-th 1-min bar's close+timestamp."""
    sub = df.iloc[factor - 1::factor]
    closes = sub["close"].values.astype(float)
    ts = sub["timestamp"].values.astype("datetime64[ns]")
    return closes, ts


def resample_daily(df):
    """Group 1-min bars by calendar date, take each date's last close --
    derived from the same 1-min dataset (not a separate API fetch) so its
    calendar window matches every other series exactly."""
    d = df.copy()
    d["_date"] = d["timestamp"].dt.date
    last = d.groupby("_date", as_index=False).last()
    closes = last["close"].values.astype(float)
    ts = last["timestamp"].values.astype("datetime64[ns]")
    return closes, ts


def align_index_before(primary_ts, trend_ts, i, buffer_minutes):
    """Index into trend_ts of the latest trend bar guaranteed fully CLOSED
    before primary_ts[i] -- subtracts one full trend-bar interval since a
    bar's own timestamp marks its START (not yet complete/knowable until one
    interval later). Returns None if no such bar exists yet."""
    cutoff = primary_ts[i] - np.timedelta64(buffer_minutes, "m")
    pos = np.searchsorted(trend_ts, cutoff, side="right") - 1
    return int(pos) if pos >= 0 else None


# =============================================================
#  GENERIC EVENT-LOOP BACKTESTER (shared by Configs A/B/C)
#  Same architecture as backtest_mtf.py::run_single_backtest (mark-to-
#  market -> exits in priority order -> circuit breakers -> signal
#  generation -> price + size + open), generalized over bar timeframe /
#  indicator set / trend filter / DTE rather than hardcoded per-mode.
# =============================================================

def run_config(name, price, bars_per_day, lookback, signal_fn, dte_fn,
                max_hold_days_fn, min_hold_days=0, trend=None,
                trend_buffer_minutes=0, ml_gate=None, cooldown_bars=6,
                signal_check_interval=2, symbols=None):
    """
    price: {symbol: {"close": np.ndarray, "ts": np.ndarray[datetime64]}}
    trend: {symbol: {"close": np.ndarray, "ts": np.ndarray[datetime64]}} or None
    dte_fn / max_hold_days_fn: symbol -> value
    symbols: explicit list of symbols to trade (defaults to module SYMBOLS if None)
    """
    syms = symbols if symbols is not None else SYMBOLS
    max_bars = max(len(p["close"]) for p in price.values())
    warmup = lookback + 5
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    positions = []
    trades = []
    daily_balances = [INITIAL_BALANCE]
    consec_losses = 0
    cooldowns = {}
    ml_stats = {"checked": 0, "blocked_conf": 0, "blocked_disagree": 0}

    for bar_idx in range(warmup, max_bars):
        for pos in positions:
            sym = pos["symbol"]
            closes = price[sym]["close"]
            actual_bar = min(bar_idx, len(closes) - 1)
            S = closes[actual_bar]
            bars_held = bar_idx - pos["entry_bar"]
            days_elapsed = bars_held / bars_per_day
            remaining_dte = max(0.1, (pos["dte"] - days_elapsed)) / 365.0
            val = bs_price(S, pos["strike"], remaining_dte, pos["iv"], pos["type"])
            pos["current_value"] = val
            pos["peak_value"] = max(pos.get("peak_value", pos["premium"]), val)
            pos["bars_held"] = bars_held
            pos["days_elapsed"] = days_elapsed

        to_exit = []
        for i, pos in enumerate(positions):
            pnl_pct = (pos["current_value"] - pos["premium"]) / pos["premium"]
            remaining_days = pos["dte"] - pos["days_elapsed"]
            min_hold_ok = pos["days_elapsed"] >= pos["min_hold_days"]
            reason = None

            if pnl_pct <= STOP_LOSS:
                reason = "STOP_LOSS"
            elif pnl_pct >= TAKE_PROFIT:
                reason = "TAKE_PROFIT"
            elif pos["peak_value"] > pos["premium"] * (1 + TRAILING_TRIGGER):
                drop = (pos["current_value"] - pos["peak_value"]) / pos["peak_value"]
                if drop <= -TRAILING_STOP:
                    reason = "TRAILING_STOP"
            elif remaining_days <= DTE_EXIT_BUFFER_DAYS:
                reason = "DTE_EXIT"
            elif min_hold_ok and pos["days_elapsed"] >= pos["max_hold_days"]:
                reason = "MAX_HOLD"

            if reason:
                to_exit.append((i, reason))

        for i, reason in sorted(to_exit, reverse=True):
            pos = positions.pop(i)
            pnl_per = pos["current_value"] - pos["premium"]
            pnl = pnl_per * 100 * pos["qty"]
            balance += pnl
            consec_losses = consec_losses + 1 if pnl < 0 else 0
            peak_balance = max(peak_balance, balance)
            cooldowns[pos["symbol"]] = bar_idx + cooldown_bars
            trades.append({
                "symbol": pos["symbol"], "type": pos["type"],
                "entry_bar": pos["entry_bar"], "exit_bar": bar_idx,
                "bars_held": pos["bars_held"], "days_held": pos["days_elapsed"],
                "entry_price": pos["entry_price"], "strike": pos["strike"],
                "dte": pos["dte"], "premium": pos["premium"],
                "exit_value": pos["current_value"], "qty": pos["qty"],
                "pnl": pnl, "pnl_pct": pnl_per / pos["premium"],
                "exit_reason": reason, "score": pos["score"],
            })

        unrealized = sum((p["current_value"] - p["premium"]) * 100 * p["qty"] for p in positions)
        daily_balances.append(balance + unrealized)

        dd = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
        if consec_losses >= 5 or dd <= -0.15:
            consec_losses = max(0, consec_losses - 1)
            continue

        if bar_idx % signal_check_interval != 0:
            continue
        if len(positions) >= MAX_POSITIONS:
            continue

        for sym in syms:
            if len(positions) >= MAX_POSITIONS:
                break
            if any(p["symbol"] == sym for p in positions):
                continue
            if bar_idx < cooldowns.get(sym, 0):
                continue
            if sym not in price:
                continue

            closes = price[sym]["close"]
            ts = price[sym]["ts"]
            if bar_idx >= len(closes):
                continue
            chunk = closes[max(0, bar_idx - lookback):bar_idx + 1]
            if len(chunk) < 30:
                continue

            direction, score, _ = signal_fn(chunk)
            if direction is None:
                continue

            if trend is not None:
                t = trend.get(sym)
                if t is not None:
                    t_idx = align_index_before(ts, t["ts"], bar_idx, trend_buffer_minutes)
                    if t_idx is None:
                        continue
                    t_chunk = t["close"][max(0, t_idx - 25):t_idx + 1]
                    bias = trend_bias_trimmed(t_chunk)
                    if bias is not None and bias != direction:
                        continue

            if ml_gate is not None:
                ml_stats["checked"] += 1
                allowed, ml_conf, _ = ml_gate.check(chunk, direction)
                if not allowed:
                    if ml_conf < MIN_ML_CONFIDENCE:
                        ml_stats["blocked_conf"] += 1
                    else:
                        ml_stats["blocked_disagree"] += 1
                    continue

            S = closes[bar_idx]
            iv = estimate_iv(closes[:bar_idx + 1], bars_per_day=bars_per_day)
            K = select_strike(S, direction)
            dte = dte_fn(sym)
            T = dte / 365.0
            premium = bs_price(S, K, T, iv, direction)
            if premium < 0.05:
                continue

            cost_per = premium * 100
            max_spend = balance * MAX_POSITION_PCT
            if cost_per > max_spend:
                continue
            qty = max(1, int(max_spend / cost_per))

            positions.append({
                "symbol": sym, "type": direction, "entry_bar": bar_idx,
                "entry_price": S, "strike": K, "iv": iv, "dte": dte,
                "premium": premium, "current_value": premium,
                "peak_value": premium, "qty": qty, "score": score,
                "bars_held": 0, "days_elapsed": 0.0,
                "max_hold_days": max_hold_days_fn(sym), "min_hold_days": min_hold_days,
            })

    for pos in positions:
        sym = pos["symbol"]
        closes = price[sym]["close"]
        final_bar = min(max_bars - 1, len(closes) - 1)
        S = closes[final_bar]
        bars_held = (max_bars - 1) - pos["entry_bar"]
        days_elapsed = bars_held / bars_per_day
        remaining = max(0.1, (pos["dte"] - days_elapsed)) / 365.0
        val = bs_price(S, pos["strike"], remaining, pos["iv"], pos["type"])
        pnl = (val - pos["premium"]) * 100 * pos["qty"]
        balance += pnl
        trades.append({
            "symbol": sym, "type": pos["type"], "entry_bar": pos["entry_bar"],
            "exit_bar": max_bars - 1, "bars_held": bars_held, "days_held": days_elapsed,
            "entry_price": pos["entry_price"], "strike": pos["strike"], "dte": pos["dte"],
            "premium": pos["premium"], "exit_value": val, "qty": pos["qty"],
            "pnl": pnl, "pnl_pct": (val - pos["premium"]) / pos["premium"],
            "exit_reason": "END_OF_TEST", "score": pos["score"],
        })

    return {
        "mode": name, "balance": balance, "pnl": balance - INITIAL_BALANCE,
        "trades": trades, "daily_balances": daily_balances, "ml_stats": ml_stats,
    }


# =============================================================
#  REPORTING (adapted from backtest_mtf.py's print_mode_report/print_comparison)
# =============================================================

def print_mode_report(result):
    trades = result["trades"]
    label = MODE_LABELS[result["mode"]]
    pnl = result["pnl"]
    n = len(trades)
    print(f"\n{'=' * 78}\n  {label}\n{'=' * 78}")
    print(f"  Final balance: ${result['balance']:,.2f}  |  P&L: ${pnl:+,.2f} ({pnl / INITIAL_BALANCE:+.1%})")
    print(f"  Trades: {n}")

    if result.get("ml_stats", {}).get("checked"):
        s = result["ml_stats"]
        print(f"  ML gate: {s['checked']} signals checked, "
              f"{s['blocked_conf']} blocked (low confidence), "
              f"{s['blocked_disagree']} blocked (direction disagreement)")

    if n == 0:
        print("  No trades generated.")
        return

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n * 100
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gp / gl if gl > 0 else float("inf")
    avg_win = gp / len(wins) if wins else 0.0
    avg_loss = -gl / len(losses) if losses else 0.0
    avg_hold_days = sum(t["days_held"] for t in trades) / n

    peak, bal, max_dd = INITIAL_BALANCE, INITIAL_BALANCE, 0.0
    for t in sorted(trades, key=lambda x: x["exit_bar"]):
        bal += t["pnl"]
        peak = max(peak, bal)
        max_dd = min(max_dd, (bal - peak) / peak)

    print(f"  Win rate: {wr:.1f}%  |  Profit factor: {pf:.2f}  |  Max drawdown: {max_dd:.1%}")
    print(f"  Avg win: ${avg_win:+,.2f}  |  Avg loss: ${avg_loss:+,.2f}  |  Avg hold: {avg_hold_days:.2f} days")

    calls = [t for t in trades if t["type"] == "call"]
    puts = [t for t in trades if t["type"] == "put"]
    print(f"  Calls: {len(calls)} ({sum(t['pnl'] for t in calls):+,.0f})  |  "
          f"Puts: {len(puts)} ({sum(t['pnl'] for t in puts):+,.0f})")

    reasons = defaultdict(int)
    for t in trades:
        reasons[t["exit_reason"]] += 1
    print(f"  Exit reasons: {dict(reasons)}")

    print("\n  Example trades:")
    examples = sorted(trades, key=lambda x: -abs(x["pnl"]))[:3]
    for t in examples:
        print(f"    {t['symbol']} {t['type'].upper()} | entry ${t['entry_price']:.2f} strike ${t['strike']:.2f} "
              f"DTE{t['dte']} | premium ${t['premium']:.2f} -> ${t['exit_value']:.2f} | "
              f"qty {t['qty']} | held {t['days_held']:.1f}d | P&L ${t['pnl']:+,.2f} ({t['exit_reason']})")


def print_comparison(results):
    print(f"\n{'=' * 92}\n  COMPARISON TABLE\n{'=' * 92}")
    print(f"  {'Config':<45} {'Final':>10} {'P&L':>10} {'P&L%':>7} {'#':>4} {'WR':>5} {'PF':>6} {'AvgHold':>9} {'MaxDD':>7}")
    print(f"  {'-' * 90}")
    for r in results:
        trades = r["trades"]
        n = len(trades)
        pnl = r["pnl"]
        pct = pnl / INITIAL_BALANCE * 100
        wr = (len([t for t in trades if t["pnl"] > 0]) / n * 100) if n else 0.0
        gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        pf = gp / gl if gl > 0 else float("inf")
        avg_hold = sum(t["days_held"] for t in trades) / n if n else 0.0
        peak, bal, max_dd = INITIAL_BALANCE, INITIAL_BALANCE, 0.0
        for t in sorted(trades, key=lambda x: x["exit_bar"]):
            bal += t["pnl"]
            peak = max(peak, bal)
            max_dd = min(max_dd, (bal - peak) / peak)
        label = MODE_LABELS[r["mode"]]
        marker = " <-- BEST" if r is max(results, key=lambda x: x["pnl"]) else ""
        print(f"  {label:<45} ${r['balance']:>9,.0f} ${pnl:>+9,.0f} {pct:>+6.1f}% "
              f"{n:>4} {wr:>4.0f}% {pf:>5.2f} {avg_hold:>8.2f}d {max_dd:>6.1%}{marker}")

    best = max(results, key=lambda x: x["pnl"])
    print(f"\n  WINNER: {MODE_LABELS[best['mode']]}  (${best['pnl']:+,.0f}, {best['pnl'] / INITIAL_BALANCE:+.1%})")
    baseline = next((r for r in results if r["mode"] == "A_baseline"), None)
    if baseline:
        for r in results:
            if r["mode"] == "A_baseline":
                continue
            diff = r["pnl"] - baseline["pnl"]
            print(f"  {MODE_LABELS[r['mode']]} vs baseline: ${diff:+,.0f}")
    print(f"\n{'=' * 92}")


# =============================================================
#  MAIN
# =============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, default=",".join(SYMBOLS))
    p.add_argument("--days", type=int, default=HISTORY_DAYS)
    p.add_argument("--force-refresh", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print("=" * 78)
    print("  AlpacaBot Strategy-Improvement Comparative Backtest")
    print("  Config A (baseline+ML) vs B (scalp-tuned) vs C (swing-tuned)")
    print(f"  Universe: {', '.join(symbols)} | Balance: ${INITIAL_BALANCE:,.0f} | Days: {args.days}")
    print("=" * 78)

    print(f"\n[1/3] Fetching 1-min bars (~{args.days}d, cached, IEX feed)...")
    data_1min = {}
    for sym in symbols:
        df = fetch_1min_cached(sym, days=args.days, force=args.force_refresh)
        data_1min[sym] = df
        days_covered = df["timestamp"].astype(str).str[:10].nunique()
        print(f"  {sym}: {len(df):,} 1-min bars ({days_covered} trading days)")

    print("\n[2/3] Deriving resampled series (2m/10m/15m/1h/daily) from 1-min data...")
    data_2min, data_10min, data_15min, data_1hour, data_daily = {}, {}, {}, {}, {}
    for sym in symbols:
        df = data_1min[sym]
        c, t = resample_intraday(df, 2); data_2min[sym] = {"close": c, "ts": t}
        c, t = resample_intraday(df, 10); data_10min[sym] = {"close": c, "ts": t}
        c, t = resample_intraday(df, 15); data_15min[sym] = {"close": c, "ts": t}
        c, t = resample_intraday(df, 60); data_1hour[sym] = {"close": c, "ts": t}
        c, t = resample_daily(df); data_daily[sym] = {"close": c, "ts": t}
        print(f"  {sym}: {len(data_2min[sym]['close']):,} 2m | {len(data_10min[sym]['close']):,} 10m | "
              f"{len(data_15min[sym]['close']):,} 15m | {len(data_1hour[sym]['close']):,} 1h | "
              f"{len(data_daily[sym]['close']):,} daily")

    print("\n[3/3] Loading ML model for Config A gate...")
    ml_gate = MLGate()

    results = []

    print("\n" + "=" * 78)
    print("  Running Config A: BASELINE (10-min, 14 indicators, +ML gate)")
    print("=" * 78)
    results.append(run_config(
        "A_baseline", data_10min, bars_per_day=39, lookback=cfg.LOOKBACK_BARS,
        signal_fn=lambda chunk: generate_signal_full(chunk, cfg.MIN_SIGNAL_SCORE),
        dte_fn=get_dte_for_symbol_A, max_hold_days_fn=cfg.get_max_hold_days,
        min_hold_days=0, trend=None, ml_gate=ml_gate,
        cooldown_bars=cfg.COOLDOWN_BARS, signal_check_interval=cfg.SIGNAL_CHECK_BARS,
        symbols=symbols,
    ))

    print("\n" + "=" * 78)
    print("  Running Config B: SCALP-TUNED (2-min, 4 indicators, 15m trend filter)")
    print("=" * 78)
    results.append(run_config(
        "B_scalp", data_2min, bars_per_day=195, lookback=50,
        signal_fn=generate_signal_trimmed, dte_fn=lambda sym: 1,
        max_hold_days_fn=lambda sym: 1, min_hold_days=0,
        trend=data_15min, trend_buffer_minutes=15, ml_gate=None,
        cooldown_bars=12, signal_check_interval=3,
        symbols=symbols,
    ))

    print("\n" + "=" * 78)
    print("  Running Config C: SWING-TUNED (1-hour, 4 indicators, daily trend filter)")
    print("=" * 78)
    results.append(run_config(
        "C_swing", data_1hour, bars_per_day=6.5, lookback=50,
        signal_fn=generate_signal_trimmed, dte_fn=lambda sym: 10,
        max_hold_days_fn=lambda sym: 5, min_hold_days=2,
        trend=data_daily, trend_buffer_minutes=1440, ml_gate=None,
        cooldown_bars=4, signal_check_interval=1,
        symbols=symbols,
    ))

    for r in results:
        print_mode_report(r)
    print_comparison(results)


if __name__ == "__main__":
    main()
