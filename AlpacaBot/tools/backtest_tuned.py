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
  set for Configs B/C/D/G chosen from real data, not assumed: rsi, macd_hist,
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
Config D (ML HYBRID, optional -- only runs when --configs includes "D"):
  Same timeframe/trend-filter/DTE/holds as Config C, but the bull/bear
  point-scoring is replaced with a GradientBoostingClassifier (mirrors
  CryptoBot's cryptotrades/utils/market_predictor.py::MarketPredictor
  pattern: GradientBoostingClassifier, TimeSeriesSplit CV for a reported
  diagnostic accuracy, predict_proba-based confidence) trained on the same
  4 indicators to predict 3-bars-ahead direction. The model is trained
  ONCE on a --pretrain-start/--pretrain-end window strictly BEFORE all
  walk-forward test windows, then frozen and reused unchanged across every
  test window -- same no-look-ahead methodology already established in this
  repo for CryptoBot's pipeline v4 frozen gate model. MIN confidence 0.55.
Config G (GAMMA-SCALP, optional -- only runs when --configs includes "G"):
  5-min bars (resampled from 1-min), same trimmed 4-indicator scoring as
  B/C/D, 1-hour trend-agreement filter (only calls allowed in an uptrend,
  only puts in a downtrend -- flat/no-trend allows either), 1-7 DTE options
  (entry DTE fixed at 7, the top of the stated range, so there is room left
  before the 3-DTE hard exit -- documented assumption since a closes-only
  backtest can't select from a real listed-expiration chain). ATM strikes
  (0% ITM/OTM offset, NOT the 5%-ITM convention shared by A/B/C/D) --
  switched from the initial ITM reuse after discovering 2% position sizing
  combined with a 5%-ITM strike made every symbol except NVDA unaffordable
  (a single contract's baked-in intrinsic value alone exceeded the $1,000
  budget on a $50k account for SPY/QQQ/AAPL/MSFT); ATM is also the more
  correct convention for gamma scalping anyway (maximum gamma sits at the
  money, whereas deep ITM has more intrinsic value and less convexity).
  Exit rules are Config-G-specific (NOT the shared A/B/C/D stop-loss/take-
  profit/trailing-stop constants): stop-loss -40%, take-profit +75%, a
  theta stop (exits once cumulative P&L% <= -4% * days_held -- a scaling-
  allowance proxy for "losing value faster than acceptable time decay",
  since isolating theta from delta P&L isn't possible with a single BS
  price per bar), and a hard exit at 3 DTE remaining. Trailing-stop and
  MAX_HOLD are both disabled for Config G (not part of its stated rules).
  Adds an IV-percentile entry filter (only enter when the estimate_iv()
  proxy's percentile rank within its own trailing ~60-trading-day history
  is below 40 -- "buy options when they're statistically cheap", standard
  for a long-premium strategy; reuses core/indicators.py::iv_percentile
  unmodified). Position sizing is 2% of balance per trade (vs 15% for
  A/B/C/D). Does NOT implement continuous delta-hedging/rebalancing (the
  literal mechanics of "gamma scalping") -- this is a directional long-
  option config with short DTE and gamma-scalp-style exit rules, consistent
  with how Configs B/C/D are already directional, not spread/hedged,
  strategies in this same backtest framework.

All configs share: universe, date range, and the Black-Scholes pricing/
IV-estimation -- only bar timeframe, indicator set, trend filter, DTE,
exit rules, strike selection, and position sizing differ per-config (the
intentional experiment variables, each documented above). Universe is
SPY/QQQ/AAPL/MSFT/NVDA per the task spec; note core/scanner.py's
SCANNER_UNIVERSE comment block documents SPY/QQQ/MSFT as historically-
eliminated DROP symbols (unprofitable, excluded from the live scanner's
real universe) -- kept here anyway since the task explicitly requires this
universe for controlled comparison.

IMPORTANT (fixed 2026-09-27): resample_intraday() now does real calendar-
time resampling (pandas .resample() on a DatetimeIndex) instead of an
earlier positional "every Nth row" version. The positional version could
drift out of wall-clock alignment whenever the underlying 1-min data had
gaps or a different starting offset between two separate fetches. All
results in this file's history from before this fix should be treated as
unreliable.

Usage (from AlpacaBot/, with its own venv):
  .venv/bin/python3 tools/backtest_tuned.py [--symbols SPY,QQQ,...] [--days 200] [--force-refresh]
  .venv/bin/python3 tools/backtest_tuned.py --start-date 2025-08-04 --end-date 2026-03-02 --cache-label 1min_oos
    (explicit [start,end) window -- e.g. for out-of-sample/walk-forward
    validation on a non-overlapping historical period; cached separately
    via --cache-label so it never collides with the default trailing-days
    cache or with other windows)
  .venv/bin/python3 tools/backtest_tuned.py --start-date 2024-03-01 --end-date 2024-09-01 --cache-label wf_windowA \\
      --configs A,B,C,D --pretrain-start 2022-09-01 --pretrain-end 2024-03-01 --pretrain-cache-label 1min_pretrain
    (adds Config D -- trains the frozen ML hybrid model on the pretrain
    window once, then backtests it on the --start-date/--end-date window)
  .venv/bin/python3 tools/backtest_tuned.py --start-date 2024-03-01 --end-date 2024-09-01 --cache-label wf_windowA --configs G
    (Config G -- 5-min gamma-scalp variant; no pretrain needed)
"""
import sys, os, math, argparse, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from datetime import datetime

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

from core.indicators import (
    compute_all_indicators, iv_percentile,
    rsi as rsi_series, macd as macd_series,
    bollinger_bands as bb_series, volatility_ratio as volr_series,
)
from core.config import Config, SYMBOL_DTE_MAP, DEFAULT_DTE
from tools.bar_cache import fetch_1min_cached, fetch_1min_window_cached

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
    "D_ml_hybrid": "D: ML HYBRID (1-hour, GBM classifier, daily trend filter)",
    "G_gamma_scalp": "G: GAMMA-SCALP (5-min, 1-7 DTE, ATM, IV<40pct, 2% risk)",
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
    its own OTM strike selection). itm_pct=0.0 gives an ATM strike (used by
    Config G)."""
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
#  SIGNAL GENERATION -- Config B/C/D/G: trimmed 4-indicator set
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
    """Lightweight trend-agreement filter for Config B (15-min bars),
    Config C/D (daily bars), and Config G (1-hour bars). Returns 'call'
    (bullish), 'put' (bearish), or None (flat -- no directional constraint
    applied)."""
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
#  CONFIG D: ML HYBRID -- GradientBoostingClassifier on the same 4
#  indicators, mirroring CryptoBot's market_predictor.py pattern (GBM +
#  TimeSeriesSplit CV for a diagnostic accuracy). Frozen model trained
#  ONCE on a pretrain window strictly BEFORE any walk-forward test window
#  -- no look-ahead, same methodology as this repo's CryptoBot pipeline v4
#  frozen gate model.
# =============================================================

ML_HYBRID_MIN_CONFIDENCE = 0.55
ML_HYBRID_HORIZON = 3   # predict 3 bars ahead (3 hours, on Config D's 1-hour bars)


def build_indicator_feature_matrix(closes):
    """Full-series indicator computation (not just the latest value) for
    rsi/macd_hist/bb_position/volatility_ratio -- used for classifier
    training, where every historical bar needs its own feature row."""
    rsi_vals = rsi_series(closes)
    _, _, macd_hist = macd_series(closes)
    _, _, _, bb_pct = bb_series(closes)
    vol_r = volr_series(closes)
    X = np.column_stack([rsi_vals, macd_hist, bb_pct, vol_r])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def build_training_labels(closes, horizon=ML_HYBRID_HORIZON):
    """1 if close[t+horizon] > close[t] else 0. Last `horizon` bars are
    unlabeled (no future price yet) -- marked -1 and filtered out by callers."""
    y = np.full(len(closes), -1, dtype=int)
    if len(closes) > horizon:
        y[:-horizon] = (closes[horizon:] > closes[:-horizon]).astype(int)
    return y


def train_ml_hybrid_model(pretrain_data, warmup=50):
    """Train the frozen Config D model across all symbols' pretrain 1-hour
    closes. pretrain_data: {symbol: {"close": array, "ts": array}}.

    TimeSeriesSplit CV score is reported as a diagnostic only; the model
    actually used for backtesting is refit on the FULL pretrain set (still
    entirely before any of the walk-forward test windows -- refitting on
    100% of an already-frozen, already-past dataset introduces no leakage
    into the test windows that follow it).
    """
    X_parts, y_parts = [], []
    for sym, d in pretrain_data.items():
        closes = d["close"]
        if len(closes) < warmup + ML_HYBRID_HORIZON + 10:
            continue
        X = build_indicator_feature_matrix(closes)
        y = build_training_labels(closes)
        mask = y >= 0
        mask[:warmup] = False
        X_parts.append(X[mask])
        y_parts.append(y[mask])

    if not X_parts:
        raise RuntimeError("Not enough pretrain data to train the ML hybrid model")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    print(f"  ML hybrid: training on {len(X_all):,} samples "
          f"({np.mean(y_all):.1%} UP) from the pretrain window")

    n_splits = min(5, max(2, len(X_all) // 500))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    for train_idx, test_idx in tscv.split(X_all):
        m = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_split=20, min_samples_leaf=10,
            random_state=42,
        )
        m.fit(X_all[train_idx], y_all[train_idx])
        cv_scores.append(m.score(X_all[test_idx], y_all[test_idx]))
    print(f"  ML hybrid: TimeSeriesSplit CV accuracy = {np.mean(cv_scores):.1%} "
          f"(avg {n_splits} folds, diagnostic only)")

    final_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_split=20, min_samples_leaf=10,
        random_state=42,
    )
    final_model.fit(X_all, y_all)
    return final_model


class MLHybridSignal:
    """Config D's signal generator: same 4 indicators as Config C, but
    direction/confidence come from the frozen GradientBoostingClassifier
    instead of the bull/bear point-scoring system."""

    def __init__(self, model):
        self.model = model

    def __call__(self, chunk):
        if len(chunk) < 30:
            return None, 0, {}
        ind = compute_trimmed_indicators(chunk)
        feats = np.array([[ind["rsi"], ind["macd_hist"], ind["bb_position"],
                            ind["volatility_ratio"]]])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.predict_proba(feats)[0]
        up_prob = float(proba[1])
        confidence = max(up_prob, 1.0 - up_prob)
        if confidence < ML_HYBRID_MIN_CONFIDENCE:
            return None, 0, ind
        direction = "call" if up_prob > 0.5 else "put"
        score = int(round(confidence * 10))  # cosmetic -- reporting/example-trades only
        return direction, score, ind


# =============================================================
#  CONFIG G: GAMMA-SCALP -- IV-percentile entry filter
#  Only enter when the estimate_iv() proxy's percentile rank within its own
#  trailing history is below a threshold ("buy cheap vol"). Reuses
#  core/indicators.py::iv_percentile unmodified.
# =============================================================

IV_PERCENTILE_LOOKBACK_BARS = 4680   # ~60 trading days at 78 5-min bars/day
IV_PERCENTILE_MAX = 40               # only enter when IV percentile < this


def compute_iv_series(closes, bars_per_day, window=30):
    """Full-series estimate_iv() readings (one per bar), for IV-percentile
    filtering. Same estimate_iv() formula used for option pricing -- a
    historical-vol-based IV PROXY, not real market-quoted IV (no real
    options-chain IV data available in a closes-only backtest)."""
    n = len(closes)
    ivs = np.full(n, 0.25)
    for i in range(window + 1, n):
        ivs[i] = estimate_iv(closes[:i + 1], window=window, bars_per_day=bars_per_day)
    return ivs


def make_iv_percentile_filter(iv_series_by_symbol, lookback_bars=IV_PERCENTILE_LOOKBACK_BARS,
                               max_percentile=IV_PERCENTILE_MAX):
    """Returns an entry_filter_fn(sym, bar_idx) -> bool that only allows
    entry when the current IV's percentile rank within the trailing
    `lookback_bars` window is below `max_percentile`."""
    def _filter(sym, bar_idx):
        ivs = iv_series_by_symbol.get(sym)
        if ivs is None or bar_idx >= len(ivs):
            return True
        history = ivs[max(0, bar_idx - lookback_bars):bar_idx]
        if len(history) < 30:
            return True
        pctl = iv_percentile(ivs[bar_idx], history)
        return pctl < max_percentile
    return _filter


# =============================================================
#  RESAMPLING -- real calendar-time resampling (pandas .resample() on a
#  DatetimeIndex). FIXED 2026-09-27: an earlier version took every Nth
#  ROW (positional), which could drift out of true wall-clock alignment
#  whenever the underlying 1-min data had gaps or a different starting
#  offset between two separate fetches -- confirmed via a stark result
#  discrepancy between two overlapping-but-not-identical test windows.
# =============================================================

def resample_intraday(df, freq):
    """Resample 1-min bars to `freq` (pandas offset alias, e.g. '2min',
    '10min', '1h') using the LAST close within each real calendar interval.
    Bars are labeled/closed on the LEFT edge (bar timestamp = interval
    START), matching Alpaca's own bar convention and this module's
    align_index_before assumption that a bar's timestamp marks its start.
    Intervals with no underlying 1-min data are dropped (not silently
    filled/misaligned)."""
    d = df.set_index("timestamp").sort_index()
    resampled = d["close"].resample(freq, label="left", closed="left").last().dropna()
    closes = resampled.values.astype(float)
    ts = resampled.index.values.astype("datetime64[ns]")
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
#  GENERIC EVENT-LOOP BACKTESTER (shared by Configs A/B/C/D/G)
#  Same architecture as backtest_mtf.py::run_single_backtest (mark-to-
#  market -> exits in priority order -> circuit breakers -> signal
#  generation -> price + size + open), generalized over bar timeframe /
#  indicator set / trend filter / DTE / exit rules / sizing / strike
#  selection / entry filter rather than hardcoded per-mode. All new
#  optional parameters default to the shared module constants so Configs
#  A/B/C/D are completely unaffected.
# =============================================================

def run_config(name, price, bars_per_day, lookback, signal_fn, dte_fn,
                max_hold_days_fn, min_hold_days=0, trend=None,
                trend_buffer_minutes=0, ml_gate=None, cooldown_bars=6,
                signal_check_interval=2, symbols=None,
                stop_loss=None, take_profit=None, trailing_stop=None,
                trailing_trigger=None, enable_trailing_stop=True,
                dte_exit_buffer_days=None, theta_stop_pct_per_day=None,
                max_position_pct=None, entry_filter_fn=None,
                strike_itm_pct=None):
    """
    price: {symbol: {"close": np.ndarray, "ts": np.ndarray[datetime64]}}
    trend: {symbol: {"close": np.ndarray, "ts": np.ndarray[datetime64]}} or None
    dte_fn / max_hold_days_fn: symbol -> value
    symbols: explicit list of symbols to trade (defaults to module SYMBOLS if None)
    stop_loss/take_profit/trailing_stop/trailing_trigger/dte_exit_buffer_days:
        per-call overrides of the shared module constants (None = use the
        module default, preserving A/B/C/D's existing behavior exactly).
    enable_trailing_stop: set False to disable the trailing-stop check
        entirely (Config G's exit rules don't include one).
    theta_stop_pct_per_day: if set, adds an exit when
        pnl_pct <= -theta_stop_pct_per_day * days_elapsed (a scaling loss
        allowance proxying "losing value faster than acceptable time
        decay"). None (default) disables this check -- only Config G uses it.
    max_position_pct: overrides MAX_POSITION_PCT for this run (None = use
        the module default).
    entry_filter_fn: optional callable(sym, bar_idx) -> bool, checked after
        the signal+trend filters and before the ML gate/pricing. Return
        False to skip this entry (used by Config G's IV-percentile filter).
    strike_itm_pct: overrides TARGET_ITM_PCT for this run's strike selection
        (None = use the module default; Config G passes 0.0 for ATM).
    """
    syms = symbols if symbols is not None else SYMBOLS
    sl = STOP_LOSS if stop_loss is None else stop_loss
    tp = TAKE_PROFIT if take_profit is None else take_profit
    ts_pct = TRAILING_STOP if trailing_stop is None else trailing_stop
    tt_pct = TRAILING_TRIGGER if trailing_trigger is None else trailing_trigger
    dte_buf = DTE_EXIT_BUFFER_DAYS if dte_exit_buffer_days is None else dte_exit_buffer_days
    pos_pct = MAX_POSITION_PCT if max_position_pct is None else max_position_pct
    itm_pct = TARGET_ITM_PCT if strike_itm_pct is None else strike_itm_pct

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

            if pnl_pct <= sl:
                reason = "STOP_LOSS"
            elif pnl_pct >= tp:
                reason = "TAKE_PROFIT"
            elif (theta_stop_pct_per_day is not None and pos["days_elapsed"] > 0
                  and pnl_pct <= -theta_stop_pct_per_day * pos["days_elapsed"]):
                reason = "THETA_STOP"
            elif enable_trailing_stop and pos["peak_value"] > pos["premium"] * (1 + tt_pct):
                drop = (pos["current_value"] - pos["peak_value"]) / pos["peak_value"]
                if drop <= -ts_pct:
                    reason = "TRAILING_STOP"
            elif remaining_days <= dte_buf:
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

            if entry_filter_fn is not None and not entry_filter_fn(sym, bar_idx):
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
            K = select_strike(S, direction, itm_pct=itm_pct)
            dte = dte_fn(sym)
            T = dte / 365.0
            premium = bs_price(S, K, T, iv, direction)
            if premium < 0.05:
                continue

            cost_per = premium * 100
            max_spend = balance * pos_pct
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
    p.add_argument("--start-date", type=str, default=None,
                    help="Explicit window start YYYY-MM-DD (e.g. out-of-sample/"
                         "walk-forward validation on a non-overlapping period). "
                         "Requires --end-date.")
    p.add_argument("--end-date", type=str, default=None,
                    help="Explicit window end YYYY-MM-DD, exclusive. Requires --start-date.")
    p.add_argument("--cache-label", type=str, default="1min_oos",
                    help="Cache file label for an explicit --start-date/--end-date window "
                         "(kept separate from the default trailing-days cache so different "
                         "historical periods never collide/overwrite each other).")
    p.add_argument("--configs", type=str, default="A,B,C",
                    help="Comma list of configs to run, e.g. 'A,B,C,D,G'. D requires "
                         "--pretrain-start/--pretrain-end.")
    p.add_argument("--pretrain-start", type=str, default=None,
                    help="Config D only: ML hybrid pretrain window start YYYY-MM-DD. "
                         "Must end before --start-date to avoid look-ahead leakage.")
    p.add_argument("--pretrain-end", type=str, default=None,
                    help="Config D only: ML hybrid pretrain window end YYYY-MM-DD, exclusive.")
    p.add_argument("--pretrain-cache-label", type=str, default="1min_pretrain",
                    help="Cache label for the Config D pretrain window.")
    return p.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    use_window = bool(args.start_date and args.end_date)
    configs_to_run = {c.strip().upper() for c in args.configs.split(",")}

    print("=" * 78)
    print("  AlpacaBot Strategy-Improvement Comparative Backtest")
    print(f"  Configs requested: {sorted(configs_to_run)}")
    if use_window:
        print(f"  Universe: {', '.join(symbols)} | Balance: ${INITIAL_BALANCE:,.0f} | "
              f"Window: {args.start_date} to {args.end_date} (label={args.cache_label})")
    else:
        print(f"  Universe: {', '.join(symbols)} | Balance: ${INITIAL_BALANCE:,.0f} | Days: {args.days}")
    print("=" * 78)

    window_desc = f"window {args.start_date} to {args.end_date}" if use_window else f"~{args.days}d trailing"
    print(f"\n[1/3] Fetching 1-min bars ({window_desc}, cached, IEX feed)...")
    data_1min = {}
    for sym in symbols:
        if use_window:
            start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            df = fetch_1min_window_cached(sym, start_dt, end_dt, label=args.cache_label,
                                           force=args.force_refresh)
        else:
            df = fetch_1min_cached(sym, days=args.days, force=args.force_refresh)
        data_1min[sym] = df
        days_covered = df["timestamp"].astype(str).str[:10].nunique()
        print(f"  {sym}: {len(df):,} 1-min bars ({days_covered} trading days)")

    print("\n[2/3] Deriving resampled series (5m/2m/10m/15m/1h/daily) from 1-min data...")
    data_5min, data_2min, data_10min, data_15min, data_1hour, data_daily = {}, {}, {}, {}, {}, {}
    for sym in symbols:
        df = data_1min[sym]
        c, t = resample_intraday(df, "5min"); data_5min[sym] = {"close": c, "ts": t}
        c, t = resample_intraday(df, "2min"); data_2min[sym] = {"close": c, "ts": t}
        c, t = resample_intraday(df, "10min"); data_10min[sym] = {"close": c, "ts": t}
        c, t = resample_intraday(df, "15min"); data_15min[sym] = {"close": c, "ts": t}
        c, t = resample_intraday(df, "1h"); data_1hour[sym] = {"close": c, "ts": t}
        c, t = resample_daily(df); data_daily[sym] = {"close": c, "ts": t}
        print(f"  {sym}: {len(data_5min[sym]['close']):,} 5m | {len(data_2min[sym]['close']):,} 2m | "
              f"{len(data_10min[sym]['close']):,} 10m | {len(data_15min[sym]['close']):,} 15m | "
              f"{len(data_1hour[sym]['close']):,} 1h | {len(data_daily[sym]['close']):,} daily")

    results = []

    if "A" in configs_to_run:
        print("\n[3/3] Loading ML model for Config A gate...")
        ml_gate = MLGate()
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

    if "B" in configs_to_run:
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

    if "C" in configs_to_run:
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

    if "D" in configs_to_run:
        if not (args.pretrain_start and args.pretrain_end):
            raise SystemExit("Config D requires --pretrain-start and --pretrain-end")
        pre_start = datetime.strptime(args.pretrain_start, "%Y-%m-%d")
        pre_end = datetime.strptime(args.pretrain_end, "%Y-%m-%d")
        if use_window:
            test_start = datetime.strptime(args.start_date, "%Y-%m-%d")
            if pre_end > test_start:
                raise SystemExit("Config D pretrain window must end before --start-date "
                                  "(no look-ahead into the test window)")

        print(f"\nFetching Config D pretrain 1-min bars ({args.pretrain_start} to "
              f"{args.pretrain_end}, cached, IEX feed)...")
        pretrain_1hour = {}
        for sym in symbols:
            pdf = fetch_1min_window_cached(sym, pre_start, pre_end,
                                            label=args.pretrain_cache_label,
                                            force=args.force_refresh)
            c, t = resample_intraday(pdf, "1h")
            pretrain_1hour[sym] = {"close": c, "ts": t}
            print(f"  {sym}: {len(c):,} pretrain 1-hour bars")

        print("\nTraining Config D's frozen ML hybrid model...")
        frozen_model = train_ml_hybrid_model(pretrain_1hour)
        ml_signal = MLHybridSignal(frozen_model)

        print("\n" + "=" * 78)
        print("  Running Config D: ML HYBRID (1-hour, GBM classifier, daily trend filter)")
        print("=" * 78)
        results.append(run_config(
            "D_ml_hybrid", data_1hour, bars_per_day=6.5, lookback=50,
            signal_fn=ml_signal, dte_fn=lambda sym: 10,
            max_hold_days_fn=lambda sym: 5, min_hold_days=2,
            trend=data_daily, trend_buffer_minutes=1440, ml_gate=None,
            cooldown_bars=4, signal_check_interval=1,
            symbols=symbols,
        ))

    if "G" in configs_to_run:
        print("\nComputing IV-percentile series for Config G's entry filter (5-min bars)...")
        iv_series_by_symbol = {}
        for sym in symbols:
            iv_series_by_symbol[sym] = compute_iv_series(data_5min[sym]["close"], bars_per_day=78)
            print(f"  {sym}: {len(iv_series_by_symbol[sym]):,} IV readings")
        iv_filter = make_iv_percentile_filter(iv_series_by_symbol)

        print("\n" + "=" * 78)
        print("  Running Config G: GAMMA-SCALP (5-min, 1-7 DTE, ATM, IV<40pct, 2% risk)")
        print("=" * 78)
        results.append(run_config(
            "G_gamma_scalp", data_5min, bars_per_day=78, lookback=50,
            signal_fn=generate_signal_trimmed, dte_fn=lambda sym: 7,
            max_hold_days_fn=lambda sym: 999, min_hold_days=0,
            trend=data_1hour, trend_buffer_minutes=60,
            ml_gate=None, cooldown_bars=12, signal_check_interval=1,
            symbols=symbols,
            stop_loss=-0.40, take_profit=0.75,
            enable_trailing_stop=False,
            dte_exit_buffer_days=3.0, theta_stop_pct_per_day=0.04,
            max_position_pct=0.02, entry_filter_fn=iv_filter,
            strike_itm_pct=0.0,
        ))

    for r in results:
        print_mode_report(r)
    print_comparison(results)


if __name__ == "__main__":
    main()
