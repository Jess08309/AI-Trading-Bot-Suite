"""
AlpacaBot Walk-Forward Backtester with Monte Carlo Simulation
==============================================================
Proves whether the scalp options strategy has a genuine edge by:
  1. Walk-Forward Optimization (WFO) — train/test on sliding windows
  2. Monte Carlo Simulation — 10,000 resamplings of OOS trades
  3. Comprehensive statistical report with confidence intervals

Walk-Forward layout:
  |---Train 1---|--Test 1--|
                |---Train 2---|--Test 2--|
                              |---Train 3---|--Test 3--|

All test-window trades are "out-of-sample" (OOS) — the model has never
seen this data during training. This is the gold standard for proving edge.

Usage:
  python tools/walk_forward.py                     # all watchlist symbols
  python tools/walk_forward.py --symbols SPY,QQQ   # custom symbols
  python tools/walk_forward.py --balance 50000      # custom starting balance
  python tools/walk_forward.py --mc-sims 50000      # more Monte Carlo runs
  python tools/walk_forward.py --no-fetch           # skip API fetch, use cached CSV
"""
import sys, os, argparse, warnings, math, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from core.config import Config, SYMBOL_DTE_MAP, DEFAULT_DTE
from core.indicators import compute_all_indicators
from utils.feature_engine import OptionsFeatureEngine, FEATURE_NAMES
from utils.ml_model import OptionsMLModel

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Walk-forward windows (in trading days)
TRAIN_WINDOW_DAYS = 60    # 60 trading days (~3 months) for training
TEST_WINDOW_DAYS = 20     # 20 trading days (~1 month) out-of-sample
STEP_SIZE_DAYS = 20       # Slide forward 20 days each step

# Bar settings (10-min candles)
BARS_PER_DAY = 39         # 9:30-16:00 = 390 min / 10 min = 39
TIMEFRAME = "10Min"       # Alpaca timeframe string
LOOKBACK = 50             # Bars for indicator computation

# Options parameters
OTM_PCT = 0.015           # 1.5% OTM
MIN_DTE = 1               # NEVER 0DTE

# Risk / position management
MAX_RISK_PER_TRADE = 0.04   # 4% per trade (matches live config)
MAX_POSITIONS = 5
STOP_LOSS_PCT = -0.15        # -15% of premium
TAKE_PROFIT_PCT = 0.15       # +15% of premium
TRAILING_STOP_PCT = 0.12     # 12% from peak
TRAILING_TRIGGER = 0.06      # Trigger trailing at +6%
COOLDOWN_BARS = 6            # ~1 hour cooldown after exit

# Signal thresholds
MIN_SIGNAL_SCORE = 3
MIN_ML_CONFIDENCE = 0.58

# Monte Carlo
MC_SIMULATIONS = 10_000
MC_DRAWDOWN_THRESHOLD = 0.15  # Report P(DD > 15%)

# Data fetch
FETCH_MONTHS = 6  # 6 months of historical data


# ═══════════════════════════════════════════════════════════
#  BLACK-SCHOLES OPTION PRICING
# ═══════════════════════════════════════════════════════════

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, sigma, opt_type, r=0.05):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.01) if opt_type == "call" else max(K - S, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def estimate_iv(closes, window=30):
    """Estimate IV from recent returns, annualized."""
    if len(closes) < window + 1:
        return 0.25
    rets = np.diff(np.log(closes[-window - 1:]))
    hv = float(np.std(rets) * np.sqrt(BARS_PER_DAY * 252))
    return max(0.12, hv * 1.15)


# ═══════════════════════════════════════════════════════════
#  SIGNAL GENERATION (14 indicators — identical to live bot)
# ═══════════════════════════════════════════════════════════

def generate_signal(prices_chunk, indicators):
    """
    Rule-based signal using all 14 indicators.
    Returns: (direction, score) — 'call'/'put'/None, 0-14
    """
    bull, bear = 0, 0

    # 1. RSI
    rsi = indicators.get("rsi", 50)
    if rsi < 25:
        bull += 2
    elif 35 < rsi < 55:
        bull += 1
    elif rsi > 75:
        bear += 2
    elif 50 < rsi < 65:
        bear += 1

    # 2. MACD histogram
    macd_h = indicators.get("macd_hist", 0)
    if macd_h > 0:
        bull += 1
        if macd_h > 0.1:
            bull += 1
    elif macd_h < 0:
        bear += 1
        if macd_h < -0.1:
            bear += 1

    # 3. Stochastic
    stoch = indicators.get("stochastic", 50)
    if stoch < 20:
        bull += 1
    elif stoch > 80:
        bear += 1

    # 4. Bollinger Band position
    bb = indicators.get("bb_position", 0.5)
    if bb < 0.10:
        bull += 2
    elif bb > 0.90:
        bear += 2
    elif bb < 0.30:
        bull += 1
    elif bb > 0.70:
        bear += 1

    # 5. ATR normalized
    atr_n = indicators.get("atr_normalized", 0)
    if atr_n > 0.005:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    # 6. CCI
    cci_val = indicators.get("cci", 0)
    if cci_val < -100:
        bull += 1
    elif cci_val > 100:
        bear += 1

    # 7. ROC
    roc_val = indicators.get("roc", 0)
    if roc_val > 0.3:
        bull += 1
    elif roc_val < -0.3:
        bear += 1

    # 8. Williams %R
    wr = indicators.get("williams_r", -50)
    if wr > -20:
        bear += 1
    elif wr < -80:
        bull += 1

    # 9. Volatility Ratio
    vol_r = indicators.get("volatility_ratio", 1.0)
    if vol_r > 1.3:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    # 10. Z-Score
    zs = indicators.get("zscore", 0)
    if zs < -2.0:
        bull += 1
    elif zs > 2.0:
        bear += 1

    # 11. Trend Strength
    ts = indicators.get("trend_strength", 0)
    if ts > 25:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    # 12-14. Price changes
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

    # Decision
    if bull >= MIN_SIGNAL_SCORE and bull > bear + 1:
        return "call", bull
    elif bear >= MIN_SIGNAL_SCORE and bear > bull + 1:
        return "put", bear

    return None, 0


# ═══════════════════════════════════════════════════════════
#  DATA FETCHING (Alpaca API)
# ═══════════════════════════════════════════════════════════

def fetch_historical_bars(symbols: List[str], months: int = 6,
                          timeframe: str = TIMEFRAME,
                          config: Optional[Config] = None) -> Dict[str, pd.DataFrame]:
    """Fetch historical bars from Alpaca API and cache to CSV.

    Returns: {symbol: DataFrame with columns [timestamp, open, high, low, close, volume]}
    """
    if config is None:
        config = Config()

    if not config.has_keys:
        print("  ERROR: No API keys found. Set ALPACA_API_KEY / ALPACA_API_SECRET in .env")
        return {}

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        print("  ERROR: alpaca-py not installed. Run: pip install alpaca-py")
        return {}

    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)

    # Map timeframe string
    tf_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, "Min") if hasattr(TimeFrame, '__call__') else TimeFrame.Minute,
        "10Min": TimeFrame(10, "Min") if hasattr(TimeFrame, '__call__') else TimeFrame.Minute,
    }

    end_dt = datetime.now() - timedelta(minutes=20)  # slight lag for data availability
    start_dt = end_dt - timedelta(days=months * 30)

    all_data = {}
    os.makedirs("data/historical", exist_ok=True)

    for sym in symbols:
        cache_path = f"data/historical/{sym}_10min.csv"

        # Check cache freshness (use cached if < 1 day old)
        if os.path.exists(cache_path):
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            if age_hours < 24:
                print(f"  {sym}: Using cached data ({age_hours:.0f}h old)")
                df = pd.read_csv(cache_path)
                all_data[sym] = df
                continue

        print(f"  {sym}: Fetching {months}mo of {timeframe} bars from Alpaca...")
        try:
            # Build request depending on alpaca-py version
            try:
                from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                tf = TimeFrame(10, TimeFrameUnit.Minute)
            except (ImportError, TypeError):
                tf = TimeFrame.Minute  # fallback

            request = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
                limit=None,
            )
            bars = client.get_stock_bars(request)
            df = bars.df

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(sym, level='symbol')

            df = df.reset_index()
            # Normalize column names
            col_map = {}
            for c in df.columns:
                cl = str(c).lower()
                if 'time' in cl or 'date' in cl:
                    col_map[c] = 'timestamp'
                elif cl == 'open':
                    col_map[c] = 'open'
                elif cl == 'high':
                    col_map[c] = 'high'
                elif cl == 'low':
                    col_map[c] = 'low'
                elif cl == 'close':
                    col_map[c] = 'close'
                elif cl in ('volume', 'vol'):
                    col_map[c] = 'volume'
            df = df.rename(columns=col_map)

            required = ['close']
            if not all(c in df.columns for c in required):
                print(f"  {sym}: Missing required columns, skipping")
                continue

            # Save cache
            df.to_csv(cache_path, index=False)
            all_data[sym] = df
            print(f"  {sym}: {len(df):,} bars fetched")

        except Exception as e:
            print(f"  {sym}: Fetch error: {e}")
            # Fall back to cached
            if os.path.exists(cache_path):
                print(f"  {sym}: Falling back to cached data")
                df = pd.read_csv(cache_path)
                all_data[sym] = df

    return all_data


def load_cached_bars(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load bars from cached CSV files only (no API calls)."""
    all_data = {}
    for sym in symbols:
        # Try 10min first, then 5min
        for suffix in ["_10min.csv", "_5min.csv"]:
            path = f"data/historical/{sym}{suffix}"
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_data[sym] = df
                bars_per = BARS_PER_DAY if "10min" in suffix else 78
                days = len(df) / bars_per
                print(f"  {sym}: {len(df):,} bars ({days:.0f} days) from {suffix}")
                break
        else:
            print(f"  {sym}: No cached data found")
    return all_data


# ═══════════════════════════════════════════════════════════
#  WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════

class WalkForwardEngine:
    """
    Walk-Forward Optimization engine.

    Divides data into overlapping train/test windows, trains an ML model
    on each training window, then runs the strategy on the subsequent
    test window. All test-window trades are Out-Of-Sample (OOS).
    """

    def __init__(self, initial_balance: float = 100_000,
                 symbols: Optional[List[str]] = None,
                 train_days: int = TRAIN_WINDOW_DAYS,
                 test_days: int = TEST_WINDOW_DAYS,
                 step_days: int = STEP_SIZE_DAYS):
        self.initial_balance = initial_balance
        self.symbols = symbols or []
        self.train_bars = train_days * BARS_PER_DAY
        self.test_bars = test_days * BARS_PER_DAY
        self.step_bars = step_days * BARS_PER_DAY
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

        # Feature engine (same as live bot)
        self.feature_engine = OptionsFeatureEngine(lookback=LOOKBACK, prediction_horizon=6)

        # OOS results
        self.oos_trades: List[Dict] = []
        self.window_results: List[Dict] = []

    def run(self, all_prices: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Execute walk-forward optimization across all windows.

        Args:
            all_prices: {symbol: close_prices_array}

        Returns:
            List of all OOS trades
        """
        if not all_prices:
            print("  ERROR: No price data!")
            return []

        max_bars = max(len(p) for p in all_prices.values())
        min_bars_needed = self.train_bars + self.test_bars
        if max_bars < min_bars_needed:
            print(f"  ERROR: Need {min_bars_needed} bars, only have {max_bars}")
            return []

        # Calculate number of windows
        n_windows = 0
        start = 0
        while start + self.train_bars + self.test_bars <= max_bars:
            n_windows += 1
            start += self.step_bars

        print(f"\n  Walk-Forward Configuration:")
        print(f"    Train window:  {self.train_days} days ({self.train_bars} bars)")
        print(f"    Test window:   {self.test_days} days ({self.test_bars} bars)")
        print(f"    Step size:     {self.step_days} days ({self.step_bars} bars)")
        print(f"    Total bars:    {max_bars:,}")
        print(f"    Windows:       {n_windows}")
        print(f"    Symbols:       {', '.join(self.symbols)}")
        print()

        # Run each window
        window_start = 0
        window_num = 0

        while window_start + self.train_bars + self.test_bars <= max_bars:
            window_num += 1
            train_start = window_start
            train_end = window_start + self.train_bars
            test_start = train_end
            test_end = min(train_end + self.test_bars, max_bars)

            print(f"  Window {window_num}/{n_windows}: "
                  f"Train [{train_start}-{train_end}] "
                  f"Test [{test_start}-{test_end}]", end="")

            # ── Phase 1: Train ML model on training window ──
            ml_model = OptionsMLModel(
                model_dir=f"data/models/wf_window_{window_num}",
                min_accuracy=0.50,  # Accept slightly lower for WF (smaller data)
            )
            price_dict_train = {}
            for sym, prices in all_prices.items():
                seg = prices[train_start:train_end]
                if len(seg) > 100:
                    price_dict_train[sym] = seg

            ml_ready = False
            if price_dict_train:
                ml_ready = ml_model.train(price_dict_train, min_samples=100, stride=3)

            # ── Phase 2: Run strategy on test window (OOS) ──
            window_trades = self._run_test_window(
                all_prices, test_start, test_end, ml_model if ml_ready else None
            )

            # Record results
            n_trades = len(window_trades)
            window_pnl = sum(t["pnl"] for t in window_trades)
            n_wins = len([t for t in window_trades if t["pnl"] > 0])
            win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0

            print(f" -> {n_trades} trades, "
                  f"${window_pnl:+,.0f}, "
                  f"{win_rate:.0f}% WR, "
                  f"ML={'OK' if ml_ready else 'NO'} "
                  f"({ml_model.test_accuracy:.0%})" if ml_ready else "")

            self.oos_trades.extend(window_trades)
            self.window_results.append({
                "window": window_num,
                "train_start": int(train_start),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
                "n_trades": n_trades,
                "pnl": float(window_pnl),
                "win_rate": float(win_rate),
                "ml_ready": ml_ready,
                "ml_accuracy": float(ml_model.test_accuracy) if ml_ready else 0.0,
            })

            window_start += self.step_bars

        print(f"\n  Walk-forward complete: {len(self.oos_trades)} total OOS trades "
              f"across {window_num} windows")
        return self.oos_trades

    def _run_test_window(self, all_prices: Dict[str, np.ndarray],
                         test_start: int, test_end: int,
                         ml_model: Optional[OptionsMLModel]) -> List[Dict]:
        """
        Run the scalp strategy on a test window. All trades are OOS.
        Uses a fixed starting balance for each window (not cumulative)
        so we measure edge, not compounding.
        """
        balance = self.initial_balance
        peak_balance = self.initial_balance
        positions: List[Dict] = []
        trades: List[Dict] = []
        cooldowns: Dict[str, int] = {}
        consec_losses = 0

        for bar_idx in range(test_start, test_end):
            # ── Mark-to-market ──
            for pos in positions:
                sym = pos["symbol"]
                if bar_idx >= len(all_prices.get(sym, [])):
                    continue
                S = all_prices[sym][bar_idx]
                bars_held = bar_idx - pos["entry_bar"]
                days_elapsed = bars_held / BARS_PER_DAY
                remaining_dte = max(0.01, (pos["dte"] - days_elapsed)) / 365.0
                val = bs_price(S, pos["strike"], remaining_dte, pos["iv"], pos["type"])
                pos["current_value"] = val
                pos["peak_value"] = max(pos.get("peak_value", pos["premium"]), val)
                pos["bars_held"] = bars_held

            # ── Check exits ──
            max_hold = min(pos.get("dte", 2) * BARS_PER_DAY - BARS_PER_DAY, 3 * BARS_PER_DAY)
            max_hold = max(BARS_PER_DAY, max_hold) if positions else BARS_PER_DAY

            to_exit = []
            for i, pos in enumerate(positions):
                pnl_pct = (pos["current_value"] - pos["premium"]) / pos["premium"]
                reason = None

                if pnl_pct <= STOP_LOSS_PCT:
                    reason = "STOP_LOSS"
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    reason = "TAKE_PROFIT"
                elif pos["peak_value"] > pos["premium"] * (1 + TRAILING_TRIGGER):
                    drop = (pos["current_value"] - pos["peak_value"]) / pos["peak_value"]
                    if drop <= -TRAILING_STOP_PCT:
                        reason = "TRAILING_STOP"
                elif pos["bars_held"] >= max_hold:
                    reason = "MAX_HOLD"
                elif pos.get("dte", 2) > 1 and pos["bars_held"] >= (pos["dte"] - 1) * BARS_PER_DAY:
                    reason = "DTE_EXIT"

                if reason:
                    to_exit.append((i, reason))

            for i, reason in sorted(to_exit, reverse=True):
                pos = positions.pop(i)
                pnl_per = pos["current_value"] - pos["premium"]
                pnl = pnl_per * 100 * pos["qty"]
                pnl_pct = pnl_per / pos["premium"] if pos["premium"] > 0 else 0
                balance += pnl
                consec_losses = consec_losses + 1 if pnl < 0 else 0
                peak_balance = max(peak_balance, balance)
                cooldowns[pos["symbol"]] = bar_idx + COOLDOWN_BARS

                trades.append({
                    "symbol": pos["symbol"],
                    "type": pos["type"],
                    "entry_bar": pos["entry_bar"],
                    "exit_bar": bar_idx,
                    "bars_held": pos["bars_held"],
                    "days_held": pos["bars_held"] / BARS_PER_DAY,
                    "entry_price": pos["entry_price"],
                    "exit_price": float(all_prices[pos["symbol"]][min(bar_idx, len(all_prices[pos["symbol"]]) - 1)]),
                    "strike": pos["strike"],
                    "dte": pos["dte"],
                    "premium": pos["premium"],
                    "exit_value": pos["current_value"],
                    "qty": pos["qty"],
                    "pnl": float(pnl),
                    "pnl_pct": float(pnl_pct),
                    "exit_reason": reason,
                    "score": pos["score"],
                    "ml_conf": pos.get("ml_conf", 0),
                    "oos": True,
                })

            # ── Circuit breaker ──
            dd = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
            if consec_losses >= 5 or dd <= -0.15:
                consec_losses = max(0, consec_losses - 1)
                continue

            # Only scan every 3 bars (~30 min for 10-min candles)
            if bar_idx % 3 != 0:
                continue

            if len(positions) >= MAX_POSITIONS:
                continue

            # ── Generate signals ──
            for sym in self.symbols:
                if len(positions) >= MAX_POSITIONS:
                    break
                if any(p["symbol"] == sym for p in positions):
                    continue
                if bar_idx < cooldowns.get(sym, 0):
                    continue

                prices = all_prices.get(sym)
                if prices is None or bar_idx < LOOKBACK or bar_idx >= len(prices):
                    continue

                chunk = prices[bar_idx - LOOKBACK:bar_idx + 1]
                indicators = compute_all_indicators(chunk)
                if not indicators:
                    continue

                direction, score = generate_signal(chunk, indicators)
                if direction is None:
                    continue

                # ── ML gate (if model available) ──
                ml_conf = 0.0
                if ml_model is not None:
                    ml_pred = ml_model.predict(chunk)
                    ml_conf = ml_pred["confidence"]
                    ml_dir = ml_pred["direction"]
                    ml_agrees = (
                        (direction == "call" and ml_dir > 0.5) or
                        (direction == "put" and ml_dir < 0.5)
                    )
                    # Block if ML low confidence or disagrees
                    if ml_conf < MIN_ML_CONFIDENCE:
                        continue
                    if not ml_agrees and ml_conf < 0.70:
                        continue

                # ── Open position ──
                S = prices[bar_idx]
                iv = estimate_iv(prices[:bar_idx + 1])
                target_dte = SYMBOL_DTE_MAP.get(sym, DEFAULT_DTE)
                target_dte = max(MIN_DTE, target_dte)

                K = round(S * (1 + OTM_PCT), 2) if direction == "call" else round(S * (1 - OTM_PCT), 2)
                T = target_dte / 365.0
                premium = bs_price(S, K, T, iv, direction)

                if premium < 0.05:
                    continue

                cost_per = premium * 100
                max_spend = balance * MAX_RISK_PER_TRADE
                if cost_per > max_spend:
                    continue

                qty = max(1, int(max_spend / cost_per))

                positions.append({
                    "symbol": sym,
                    "type": direction,
                    "entry_bar": bar_idx,
                    "entry_price": float(S),
                    "strike": K,
                    "iv": iv,
                    "dte": target_dte,
                    "premium": premium,
                    "current_value": premium,
                    "peak_value": premium,
                    "qty": qty,
                    "score": score,
                    "bars_held": 0,
                    "ml_conf": ml_conf,
                })

        # Close remaining positions at end of window
        for pos in list(positions):
            sym = pos["symbol"]
            final_bar = min(test_end - 1, len(all_prices.get(sym, [])) - 1)
            if final_bar < 0:
                continue
            S = all_prices[sym][final_bar]
            bars_held = final_bar - pos["entry_bar"]
            days_elapsed = bars_held / BARS_PER_DAY
            remaining = max(0.01, (pos["dte"] - days_elapsed)) / 365.0
            val = bs_price(S, pos["strike"], remaining, pos["iv"], pos["type"])
            pnl = (val - pos["premium"]) * 100 * pos["qty"]
            pnl_pct = (val - pos["premium"]) / pos["premium"] if pos["premium"] > 0 else 0
            balance += pnl

            trades.append({
                "symbol": sym,
                "type": pos["type"],
                "entry_bar": pos["entry_bar"],
                "exit_bar": final_bar,
                "bars_held": bars_held,
                "days_held": bars_held / BARS_PER_DAY,
                "entry_price": pos["entry_price"],
                "exit_price": float(S),
                "strike": pos["strike"],
                "dte": pos["dte"],
                "premium": pos["premium"],
                "exit_value": float(val),
                "qty": pos["qty"],
                "pnl": float(pnl),
                "pnl_pct": float(pnl_pct),
                "exit_reason": "END_OF_WINDOW",
                "score": pos["score"],
                "ml_conf": pos.get("ml_conf", 0),
                "oos": True,
            })

        return trades


# ═══════════════════════════════════════════════════════════
#  MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    Monte Carlo simulation of trade sequences.

    Takes a set of OOS trades, resamples them randomly with replacement
    N times, and computes equity curve statistics for each path.
    This gives probability distributions of outcomes.
    """

    def __init__(self, n_simulations: int = MC_SIMULATIONS,
                 initial_balance: float = 100_000):
        self.n_simulations = n_simulations
        self.initial_balance = initial_balance

    def run(self, trades: List[Dict]) -> Dict:
        """
        Run Monte Carlo simulation on trade P&Ls.

        Args:
            trades: List of trade dicts (must have 'pnl' key)

        Returns:
            Dict with simulation statistics.
        """
        if not trades:
            return self._empty_result()

        pnls = np.array([t["pnl"] for t in trades], dtype=np.float64)
        n_trades = len(pnls)

        print(f"\n  Running {self.n_simulations:,} Monte Carlo simulations "
              f"on {n_trades} OOS trades...")

        rng = np.random.default_rng(seed=42)

        # Pre-allocate results
        final_pnls = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)
        sharpe_ratios = np.zeros(self.n_simulations)

        start_time = time.time()

        for sim in range(self.n_simulations):
            # Random resample with replacement
            sampled_indices = rng.integers(0, n_trades, size=n_trades)
            sampled_pnls = pnls[sampled_indices]

            # Build equity curve
            equity = self.initial_balance + np.cumsum(sampled_pnls)
            equity = np.insert(equity, 0, self.initial_balance)

            # Final P&L
            final_pnls[sim] = equity[-1] - self.initial_balance

            # Max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max
            max_drawdowns[sim] = float(np.min(drawdowns))

            # Sharpe ratio (per-trade basis, annualized assuming ~5 trades/day)
            if np.std(sampled_pnls) > 1e-10:
                trades_per_year = 252 * 5  # rough estimate
                sharpe_ratios[sim] = (np.mean(sampled_pnls) / np.std(sampled_pnls)
                                      * np.sqrt(trades_per_year))
            else:
                sharpe_ratios[sim] = 0.0

        elapsed = time.time() - start_time
        print(f"  Monte Carlo complete in {elapsed:.1f}s")

        # Compute statistics
        prob_profit = float(np.mean(final_pnls > 0) * 100)
        prob_dd_15 = float(np.mean(max_drawdowns < -MC_DRAWDOWN_THRESHOLD) * 100)

        result = {
            "n_simulations": self.n_simulations,
            "n_trades": n_trades,
            "probability_of_profit_pct": round(prob_profit, 1),
            "probability_dd_gt_15pct": round(prob_dd_15, 1),
            "expected_pnl_median": round(float(np.median(final_pnls)), 2),
            "expected_pnl_mean": round(float(np.mean(final_pnls)), 2),
            "pnl_5th_percentile": round(float(np.percentile(final_pnls, 5)), 2),
            "pnl_25th_percentile": round(float(np.percentile(final_pnls, 25)), 2),
            "pnl_50th_percentile": round(float(np.percentile(final_pnls, 50)), 2),
            "pnl_75th_percentile": round(float(np.percentile(final_pnls, 75)), 2),
            "pnl_95th_percentile": round(float(np.percentile(final_pnls, 95)), 2),
            "max_dd_5th_percentile": round(float(np.percentile(max_drawdowns, 5)), 4),
            "max_dd_median": round(float(np.median(max_drawdowns)), 4),
            "max_dd_95th_percentile": round(float(np.percentile(max_drawdowns, 95)), 4),
            "sharpe_5th_percentile": round(float(np.percentile(sharpe_ratios, 5)), 2),
            "sharpe_median": round(float(np.median(sharpe_ratios)), 2),
            "sharpe_95th_percentile": round(float(np.percentile(sharpe_ratios, 95)), 2),
        }
        return result

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "n_simulations": 0, "n_trades": 0,
            "probability_of_profit_pct": 0, "probability_dd_gt_15pct": 0,
            "expected_pnl_median": 0, "expected_pnl_mean": 0,
            "pnl_5th_percentile": 0, "pnl_25th_percentile": 0,
            "pnl_50th_percentile": 0, "pnl_75th_percentile": 0,
            "pnl_95th_percentile": 0,
            "max_dd_5th_percentile": 0, "max_dd_median": 0,
            "max_dd_95th_percentile": 0,
            "sharpe_5th_percentile": 0, "sharpe_median": 0,
            "sharpe_95th_percentile": 0,
        }


# ═══════════════════════════════════════════════════════════
#  OOS STATISTICS
# ═══════════════════════════════════════════════════════════

def compute_oos_stats(trades: List[Dict], initial_balance: float) -> Dict:
    """Compute comprehensive out-of-sample statistics."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0, "profit_factor": 0,
            "sharpe": 0, "max_drawdown": 0, "total_pnl": 0,
            "avg_win": 0, "avg_loss": 0, "expectancy": 0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(trades) * 100
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0
    expectancy = total_pnl / len(trades)

    # Equity curve for drawdown + Sharpe
    equity = [initial_balance]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)

    # Max drawdown
    peak = equity[0]
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = (e - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # Sharpe (per-trade, annualized)
    pnl_arr = np.array(pnls)
    if np.std(pnl_arr) > 1e-10:
        trades_per_year = 252 * 5
        sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # By symbol breakdown
    by_symbol = defaultdict(lambda: {"n": 0, "pnl": 0, "wins": 0})
    for t in trades:
        by_symbol[t["symbol"]]["n"] += 1
        by_symbol[t["symbol"]]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_symbol[t["symbol"]]["wins"] += 1

    # By direction
    calls = [t for t in trades if t["type"] == "call"]
    puts = [t for t in trades if t["type"] == "put"]

    # By exit reason
    by_reason = defaultdict(lambda: {"n": 0, "pnl": 0})
    for t in trades:
        by_reason[t["exit_reason"]]["n"] += 1
        by_reason[t["exit_reason"]]["pnl"] += t["pnl"]

    return {
        "n_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_hold_days": round(float(np.mean([t["days_held"] for t in trades])), 1),
        "by_symbol": dict(by_symbol),
        "n_calls": len(calls),
        "n_puts": len(puts),
        "call_pnl": round(sum(t["pnl"] for t in calls), 2) if calls else 0,
        "put_pnl": round(sum(t["pnl"] for t in puts), 2) if puts else 0,
        "by_reason": dict(by_reason),
    }


# ═══════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════

def print_report(oos_stats: Dict, mc_results: Dict, window_results: List[Dict],
                 initial_balance: float):
    """Print comprehensive walk-forward + Monte Carlo report."""
    print()
    print("=" * 70)
    print("  WALK-FORWARD VALIDATION REPORT")
    print("=" * 70)

    # OOS Summary
    print(f"\n  === Out-of-Sample Performance ===")
    print(f"  OOS Trades:       {oos_stats['n_trades']:>10}")
    print(f"  OOS Win Rate:     {oos_stats['win_rate']:>9.1f}%  "
          f"({oos_stats['n_wins']}W / {oos_stats['n_losses']}L)")
    print(f"  OOS Profit Factor:{oos_stats['profit_factor']:>10.2f}")
    print(f"  OOS Sharpe Ratio: {oos_stats['sharpe']:>10.2f}")
    print(f"  OOS Max Drawdown: {oos_stats['max_drawdown']:>9.1%}")
    print(f"  OOS Total P&L:    ${oos_stats['total_pnl']:>+10,.2f}  "
          f"({oos_stats['total_pnl']/initial_balance:>+.1%})")
    print(f"  OOS Expectancy:   ${oos_stats['expectancy']:>+10.2f} per trade")
    print(f"  Avg Win:          ${oos_stats['avg_win']:>10.2f}")
    print(f"  Avg Loss:         ${oos_stats['avg_loss']:>10.2f}")
    print(f"  Avg Hold:         {oos_stats['avg_hold_days']:>9.1f} days")

    # Direction breakdown
    print(f"\n  Calls: {oos_stats['n_calls']} trades  ${oos_stats['call_pnl']:>+,.2f}")
    print(f"  Puts:  {oos_stats['n_puts']} trades  ${oos_stats['put_pnl']:>+,.2f}")

    # By symbol
    if oos_stats.get("by_symbol"):
        print(f"\n  {'Sym':<6} {'#':>4} {'P&L':>10} {'WR':>6}")
        print(f"  {'─' * 30}")
        for sym, data in sorted(oos_stats["by_symbol"].items(),
                                key=lambda x: -x[1]["pnl"]):
            wr = data["wins"] / data["n"] * 100 if data["n"] > 0 else 0
            print(f"  {sym:<6} {data['n']:>4} ${data['pnl']:>+9,.2f} {wr:>5.0f}%")

    # Exit reasons
    if oos_stats.get("by_reason"):
        print(f"\n  Exit Reasons:")
        for reason, data in sorted(oos_stats["by_reason"].items(),
                                   key=lambda x: -x[1]["n"]):
            avg = data["pnl"] / data["n"] if data["n"] > 0 else 0
            print(f"    {reason:<18} {data['n']:>3}  ${data['pnl']:>+9,.2f}  "
                  f"(avg ${avg:>+.2f})")

    # Window-by-window
    if window_results:
        print(f"\n  === Window-by-Window Results ===")
        print(f"  {'Win':>4} {'Trades':>6} {'P&L':>10} {'WR':>6} {'ML Acc':>7}")
        print(f"  {'─' * 40}")
        for wr in window_results:
            print(f"  {wr['window']:>4} {wr['n_trades']:>6} ${wr['pnl']:>+9,.0f} "
                  f"{wr['win_rate']:>5.0f}% "
                  f"{wr['ml_accuracy']:>6.0%}" if wr['ml_ready'] else
                  f"  {wr['window']:>4} {wr['n_trades']:>6} ${wr['pnl']:>+9,.0f} "
                  f"{wr['win_rate']:>5.0f}%    N/A")
        profitable_windows = len([w for w in window_results if w["pnl"] > 0])
        total_windows = len(window_results)
        print(f"\n  Profitable windows: {profitable_windows}/{total_windows} "
              f"({profitable_windows/total_windows*100:.0f}%)")

    # Monte Carlo
    print(f"\n  === Monte Carlo Simulation ({mc_results['n_simulations']:,} runs) ===")
    print(f"  Probability of Profit:    {mc_results['probability_of_profit_pct']:>6.1f}%")
    print(f"  Probability of >15% DD:   {mc_results['probability_dd_gt_15pct']:>6.1f}%")
    print()
    print(f"  Expected P&L (median):    ${mc_results['expected_pnl_median']:>+12,.2f}")
    print(f"  Expected P&L (mean):      ${mc_results['expected_pnl_mean']:>+12,.2f}")
    print()
    print(f"   5th percentile P&L:      ${mc_results['pnl_5th_percentile']:>+12,.2f}  (worst case)")
    print(f"  25th percentile P&L:      ${mc_results['pnl_25th_percentile']:>+12,.2f}")
    print(f"  50th percentile P&L:      ${mc_results['pnl_50th_percentile']:>+12,.2f}  (median)")
    print(f"  75th percentile P&L:      ${mc_results['pnl_75th_percentile']:>+12,.2f}")
    print(f"  95th percentile P&L:      ${mc_results['pnl_95th_percentile']:>+12,.2f}  (best case)")
    print()
    print(f"  Max Drawdown (5th pct):   {mc_results['max_dd_5th_percentile']:>9.1%}")
    print(f"  Max Drawdown (median):    {mc_results['max_dd_median']:>9.1%}")
    print(f"  Max Drawdown (95th pct):  {mc_results['max_dd_95th_percentile']:>9.1%}")
    print()
    print(f"  Sharpe Ratio (5th pct):   {mc_results['sharpe_5th_percentile']:>10.2f}")
    print(f"  Sharpe Ratio (median):    {mc_results['sharpe_median']:>10.2f}")
    print(f"  Sharpe Ratio (95th pct):  {mc_results['sharpe_95th_percentile']:>10.2f}")

    # Verdict
    print(f"\n  {'=' * 50}")
    edge_score = 0
    verdicts = []

    if mc_results["probability_of_profit_pct"] >= 70:
        edge_score += 2
        verdicts.append(f"P(profit) {mc_results['probability_of_profit_pct']:.0f}% >= 70%")
    elif mc_results["probability_of_profit_pct"] >= 55:
        edge_score += 1
        verdicts.append(f"P(profit) {mc_results['probability_of_profit_pct']:.0f}% >= 55%")

    if oos_stats["profit_factor"] >= 1.3:
        edge_score += 2
        verdicts.append(f"PF {oos_stats['profit_factor']:.2f} >= 1.30")
    elif oos_stats["profit_factor"] >= 1.1:
        edge_score += 1
        verdicts.append(f"PF {oos_stats['profit_factor']:.2f} >= 1.10")

    if oos_stats["win_rate"] >= 50:
        edge_score += 1
        verdicts.append(f"WR {oos_stats['win_rate']:.0f}% >= 50%")

    if mc_results["probability_dd_gt_15pct"] <= 20:
        edge_score += 1
        verdicts.append(f"P(DD>15%) {mc_results['probability_dd_gt_15pct']:.0f}% <= 20%")

    if mc_results["pnl_5th_percentile"] > 0:
        edge_score += 2
        verdicts.append(f"5th pct P&L ${mc_results['pnl_5th_percentile']:+,.0f} > $0")

    if window_results:
        profitable_windows = len([w for w in window_results if w["pnl"] > 0])
        total_windows = len(window_results)
        if profitable_windows / total_windows >= 0.6:
            edge_score += 1
            verdicts.append(f"{profitable_windows}/{total_windows} profitable windows")

    if edge_score >= 7:
        verdict = "STRONG EDGE CONFIRMED — Ready for live paper trading"
    elif edge_score >= 5:
        verdict = "MODERATE EDGE — Paper trade with caution"
    elif edge_score >= 3:
        verdict = "WEAK EDGE — Needs further tuning"
    else:
        verdict = "NO EDGE DETECTED — Do NOT go live"

    print(f"  VERDICT: {verdict}")
    print(f"  Edge score: {edge_score}/10")
    for v in verdicts:
        print(f"    + {v}")
    print(f"  {'=' * 50}")


def save_results(oos_stats: Dict, mc_results: Dict, window_results: List[Dict],
                 trades: List[Dict], filepath: str):
    """Save all results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Make trades JSON-serializable (remove numpy types)
    clean_trades = []
    for t in trades:
        clean = {}
        for k, v in t.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            else:
                clean[k] = v
        clean_trades.append(clean)

    results = {
        "timestamp": datetime.now().isoformat(),
        "oos_statistics": oos_stats,
        "monte_carlo": mc_results,
        "window_results": window_results,
        "trades": clean_trades,
    }

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {filepath}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AlpacaBot Walk-Forward Backtester + Monte Carlo Simulation"
    )
    parser.add_argument("--balance", type=float, default=100_000,
                        help="Starting balance (default: 100000)")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols (default: config watchlist)")
    parser.add_argument("--train-days", type=int, default=TRAIN_WINDOW_DAYS,
                        help=f"Training window in days (default: {TRAIN_WINDOW_DAYS})")
    parser.add_argument("--test-days", type=int, default=TEST_WINDOW_DAYS,
                        help=f"Test window in days (default: {TEST_WINDOW_DAYS})")
    parser.add_argument("--step-days", type=int, default=STEP_SIZE_DAYS,
                        help=f"Step size in days (default: {STEP_SIZE_DAYS})")
    parser.add_argument("--mc-sims", type=int, default=MC_SIMULATIONS,
                        help=f"Monte Carlo simulations (default: {MC_SIMULATIONS})")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip API fetch, use cached CSVs only")
    parser.add_argument("--output", type=str,
                        default="data/backtest/walk_forward_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # ── Setup ──
    print("=" * 70)
    print("  AlpacaBot Walk-Forward Backtester + Monte Carlo Simulation")
    print("=" * 70)

    config = Config()

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = config.WATCHLIST
    print(f"\n  Symbols: {', '.join(symbols)}")
    print(f"  Balance: ${args.balance:,.0f}")

    # ── Step 1: Load / fetch data ──
    print(f"\n  Step 1: Loading historical data...")
    if args.no_fetch:
        raw_data = load_cached_bars(symbols)
    else:
        raw_data = fetch_historical_bars(symbols, months=FETCH_MONTHS, config=config)
        # Fall back to cached for any missing symbols
        missing = [s for s in symbols if s not in raw_data]
        if missing:
            print(f"  Fetching failed for {missing}, trying cached...")
            cached = load_cached_bars(missing)
            raw_data.update(cached)

    # Extract close price arrays
    all_prices: Dict[str, np.ndarray] = {}
    for sym, df in raw_data.items():
        if isinstance(df, pd.DataFrame) and "close" in df.columns:
            all_prices[sym] = df["close"].values.astype(np.float64)
        elif isinstance(df, np.ndarray):
            all_prices[sym] = df.astype(np.float64)

    active_symbols = [s for s in symbols if s in all_prices]
    if not active_symbols:
        print("  ERROR: No data available for any symbol!")
        sys.exit(1)

    print(f"\n  Active symbols: {', '.join(active_symbols)}")
    for sym in active_symbols:
        days = len(all_prices[sym]) / BARS_PER_DAY
        print(f"    {sym}: {len(all_prices[sym]):,} bars ({days:.0f} trading days)")

    # ── Step 2: Walk-Forward Optimization ──
    print(f"\n  Step 2: Walk-Forward Optimization...")
    wf_engine = WalkForwardEngine(
        initial_balance=args.balance,
        symbols=active_symbols,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    oos_trades = wf_engine.run(all_prices)

    if not oos_trades:
        print("\n  ERROR: No OOS trades generated. Data may be insufficient.")
        print("  Try reducing --train-days or --test-days, or adding more symbols.")
        sys.exit(1)

    # ── Step 3: Compute OOS statistics ──
    print(f"\n  Step 3: Computing OOS statistics...")
    oos_stats = compute_oos_stats(oos_trades, args.balance)

    # ── Step 4: Monte Carlo simulation ──
    print(f"\n  Step 4: Monte Carlo simulation...")
    mc_sim = MonteCarloSimulator(
        n_simulations=args.mc_sims,
        initial_balance=args.balance,
    )
    mc_results = mc_sim.run(oos_trades)

    # ── Step 5: Print report ──
    print_report(oos_stats, mc_results, wf_engine.window_results, args.balance)

    # ── Step 6: Save results ──
    save_results(oos_stats, mc_results, wf_engine.window_results,
                 oos_trades, args.output)

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
