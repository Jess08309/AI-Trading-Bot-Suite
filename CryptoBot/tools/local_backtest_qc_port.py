"""
Local backtest of CryptoBot's rules-only QC port logic, using REAL historical
hourly crypto bars pulled for free from Alpaca's market-data API (same
credentials the live spot bot already uses) instead of QuantConnect's cloud
(which now gates backtests behind a credit-card verification step the user
does not want to complete).

Base rules mirror CryptoBot/quantconnect/main.py's CryptoBotMomentumAlgorithm:
calculate_trend (20-bar slope), RSI(14)/MAX_RSI_LONG=68, the 0-10 mechanical
rule score, the exit ladder (stop-loss, take-profit, trailing stop, max-hold),
and the circuit breaker. SPOT-LONG-ONLY, same as the QC port (no short side
without Kraken futures data).

Several thresholds (MIN_RULE_SCORE, MIN_TREND_SLOPE, TAKE_PROFIT_PCT,
STOP_LOSS_PCT) and an added longer-horizon regime filter (REGIME_SMA_PERIOD)
were re-tuned via grid search (see optimize_backtest*.py) against this same
2021-2026 dataset, improving net profit from -88.5% (QC-port-as-written) to
-15.5% -- still a net loser. This is a real, load-bearing finding, not a bug:
a rules-only mechanical strategy without CryptoBot's live ML confidence gate
does not have a positive edge on this data, no matter how it's tuned. Treat
the "improved" config as a local optimum on backtested history, not a proven
edge -- it hasn't been validated out-of-sample.

ADA/USD is excluded: Alpaca's free crypto data only goes back to ~mid-2026 for
that symbol (no multi-year history available), unlike BTC/ETH/LTC/BCH which
have hourly bars back to at least 2021.

Usage:
  python3 CryptoBot/tools/local_backtest_qc_port.py
"""
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, "cryptotrades", ".env"))

API_KEY = os.environ["ALPACA_API_KEY"]
API_SECRET = os.environ["ALPACA_API_SECRET"]
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}
DATA_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"

CACHE_DIR = os.path.join(BASE_DIR, "data", "backtest_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD"]  # ADA excluded, see docstring
START = "2021-06-01"
END = "2026-08-01"

# --- Rule constants ---
# Base values ported verbatim from CryptoBot/quantconnect/main.py; MIN_RULE_SCORE,
# MIN_TREND_SLOPE, TAKE_PROFIT_PCT, STOP_LOSS_PCT, and REGIME_SMA_PERIOD were
# re-tuned via grid search (optimize_backtest*.py) against this same 2021-2026
# dataset -- baseline (QC-port-as-written) values are noted alongside each.
TREND_LOOKBACK = 20
MIN_TREND_SLOPE = 0.0015  # was 0.0005; stronger trend required to enter
MAX_RSI_LONG = 68.0
MIN_RULE_SCORE = 8.0  # was 5.0; only the highest-quality setups qualify

MAX_POSITION_PCT = 0.12
MAX_POSITIONS = 4

STOP_LOSS_PCT = -1.0  # was -1.5; tighter stop for a better reward:risk ratio
TAKE_PROFIT_PCT = 2.5  # was 1.5
TRAILING_STOP_PCT = 0.8
TRAILING_ACTIVATE_PCT = 0.6

MAX_HOLD_FLAT_BAND_PCT = 0.5
MAX_HOLD_HOURS = 8.0
MAX_HOLD_FORCED_HOURS = 12.0

CB_MAX_CONSECUTIVE_LOSSES = 5
CB_DAILY_LOSS_LIMIT_PCT = -4.0
CB_MAX_DRAWDOWN_PCT = -8.0
CB_COOLDOWN_MINUTES = 60  # matches live bot's utils/circuit_breaker.py -- pause-then-retry, not a permanent halt

FEE_RATE = 0.001  # 0.10% per side, approximating Alpaca crypto commission
INITIAL_CASH = 10000.0

# Optional longer-horizon regime filter (not in the original QC port): require
# close > SMA(REGIME_SMA_PERIOD) to avoid buying momentum blips inside a
# broader downtrend. Grid search (optimize_backtest_v3.py/_v4.py) found 4000
# hours (~167 days) as the local optimum -- net profit improved monotonically
# up to 4000 then got worse again at 6000-10000, so this isn't just "trade
# less = better", it's a genuine local optimum. 0 disables it entirely.
REGIME_SMA_PERIOD = 4000


def fetch_bars(symbol: str, start: str, end: str) -> list:
    safe_symbol = symbol.replace("/", "")
    cache_path = os.path.join(CACHE_DIR, f"{safe_symbol}_1Hour_{start}_{end}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    all_bars = []
    params = {
        "symbols": symbol,
        "timeframe": "1Hour",
        "start": f"{start}T00:00:00Z",
        "end": f"{end}T00:00:00Z",
        "limit": 10000,
    }
    page_token = None
    while True:
        if page_token:
            params["page_token"] = page_token
        for attempt in range(5):
            r = requests.get(DATA_URL, headers=HEADERS, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        d = r.json()
        bars = d.get("bars", {}).get(symbol, [])
        all_bars.extend(bars)
        page_token = d.get("next_page_token")
        if not page_token:
            break

    with open(cache_path, "w") as f:
        json.dump(all_bars, f)
    print(f"  {symbol}: fetched {len(all_bars)} bars")
    return all_bars


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-12
    rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
    return float(100.0 - 100.0 / (1.0 + rs))


def _trend(closes: np.ndarray, lookback: int, min_slope: float) -> tuple:
    if len(closes) < lookback:
        return "SIDE", 0.0
    recent = closes[-lookback:]
    x = np.arange(len(recent))
    slope, _ = np.polyfit(x, recent, 1)
    normalized_slope = slope / np.mean(recent)
    if normalized_slope > min_slope:
        return "UP", normalized_slope
    elif normalized_slope < -min_slope:
        return "DOWN", normalized_slope
    return "SIDE", normalized_slope


def _macd_histogram(closes: np.ndarray) -> float:
    def _ema(data, period):
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    if len(closes) < 35:
        return 0.0
    # EMA(26) converges within ~4x its period; a 150-bar trailing window is
    # numerically indistinguishable from using full history but keeps this
    # O(1) per call instead of O(n), which matters since history grows to
    # tens of thousands of bars over a multi-year backtest.
    window = closes[-150:] if len(closes) > 150 else closes
    ema12 = _ema(window, 12)
    ema26 = _ema(window, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line[-9:], 9)[-1]
    return float(macd_line[-1] - signal_line)


def compute_rule_score(rsi, trend, slope, macd_hist, close, sma20) -> float:
    score = 0.0
    if trend == "UP":
        score += 3.0 if slope > 0.0015 else 2.0
    if 45 <= rsi <= 60:
        score += 3.0
    elif 60 < rsi <= 68:
        score += 2.0
    elif 38 <= rsi < 45:
        score += 1.0
    if macd_hist > 0:
        score += 2.0
    if close > sma20:
        score += 2.0
    return min(score, 10.0)


def load_events(symbols=SYMBOLS, start=START, end=END, verbose=True):
    events = []  # (datetime, symbol, close)
    for sym in symbols:
        bars = fetch_bars(sym, start, end)
        for b in bars:
            ts = datetime.strptime(b["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            events.append((ts, sym, float(b["c"])))
    events.sort(key=lambda e: (e[0], e[1]))
    if verbose:
        print(f"Total events across {len(symbols)} symbols: {len(events)}")
    return events


def run_backtest(events, symbols=SYMBOLS, params=None) -> dict:
    """Runs the rules-only QC-port simulation over pre-loaded events.

    `params` overrides any of the module-level rule constants (e.g.
    {"MIN_RULE_SCORE": 6.5, "TAKE_PROFIT_PCT": 2.0}) for parameter sweeps
    without mutating global state or re-fetching data.
    """
    p = dict(
        TREND_LOOKBACK=TREND_LOOKBACK, MIN_TREND_SLOPE=MIN_TREND_SLOPE,
        MAX_RSI_LONG=MAX_RSI_LONG, MIN_RULE_SCORE=MIN_RULE_SCORE,
        MAX_POSITION_PCT=MAX_POSITION_PCT, MAX_POSITIONS=MAX_POSITIONS,
        STOP_LOSS_PCT=STOP_LOSS_PCT, TAKE_PROFIT_PCT=TAKE_PROFIT_PCT,
        TRAILING_STOP_PCT=TRAILING_STOP_PCT, TRAILING_ACTIVATE_PCT=TRAILING_ACTIVATE_PCT,
        MAX_HOLD_FLAT_BAND_PCT=MAX_HOLD_FLAT_BAND_PCT, MAX_HOLD_HOURS=MAX_HOLD_HOURS,
        MAX_HOLD_FORCED_HOURS=MAX_HOLD_FORCED_HOURS,
        CB_MAX_CONSECUTIVE_LOSSES=CB_MAX_CONSECUTIVE_LOSSES, CB_DAILY_LOSS_LIMIT_PCT=CB_DAILY_LOSS_LIMIT_PCT,
        CB_MAX_DRAWDOWN_PCT=CB_MAX_DRAWDOWN_PCT, CB_COOLDOWN_MINUTES=CB_COOLDOWN_MINUTES,
        FEE_RATE=FEE_RATE, INITIAL_CASH=INITIAL_CASH, REGIME_SMA_PERIOD=REGIME_SMA_PERIOD,
    )
    if params:
        p.update(params)

    cash = p["INITIAL_CASH"]
    positions = {}          # symbol -> {entry_price, entry_time, max_pnl_pct, qty}
    last_price = {}         # symbol -> latest known price
    history = {s: [] for s in symbols}
    consecutive_losses = 0
    paused_until = None  # datetime, cleared once cooldown elapses
    day_start_equity = p["INITIAL_CASH"]
    peak_equity = p["INITIAL_CASH"]
    current_day = None
    trades = []
    equity_curve = []

    def equity():
        return cash + sum(positions[s]["qty"] * last_price.get(s, positions[s]["entry_price"]) for s in positions)

    def is_paused(now):
        nonlocal peak_equity
        eq = equity()
        if eq > peak_equity:
            peak_equity = eq
        daily_pnl_pct = (eq / day_start_equity - 1.0) * 100.0 if day_start_equity else 0.0
        # Daily loss limit is day-scoped (blocks regardless of cooldown timer).
        if daily_pnl_pct <= p["CB_DAILY_LOSS_LIMIT_PCT"]:
            return True
        return paused_until is not None and now < paused_until

    def check_trip(now):
        """Evaluate trip conditions once at trade-close time, matching the live
        bot's CircuitBreaker.record_trade() -- arms a cooldown pause, does NOT
        permanently block."""
        nonlocal peak_equity, paused_until
        eq = equity()
        if eq > peak_equity:
            peak_equity = eq
        drawdown_pct = (eq / peak_equity - 1.0) * 100.0 if peak_equity else 0.0
        if consecutive_losses >= p["CB_MAX_CONSECUTIVE_LOSSES"] or drawdown_pct <= p["CB_MAX_DRAWDOWN_PCT"]:
            paused_until = now + timedelta(minutes=p["CB_COOLDOWN_MINUTES"])

    for t, symbol, close in events:
        last_price[symbol] = close
        if current_day is None:
            current_day = t.date()
        if t.date() != current_day:
            current_day = t.date()
            day_start_equity = equity()

        # --- exits (always run, regardless of circuit breaker, matching QC port) ---
        if symbol in positions:
            pos = positions[symbol]
            entry = pos["entry_price"]
            pnl_pct = (close / entry - 1.0) * 100.0
            if pnl_pct > pos["max_pnl_pct"]:
                pos["max_pnl_pct"] = pnl_pct
            hours_held = (t - pos["entry_time"]).total_seconds() / 3600.0

            exit_reason = None
            if pnl_pct <= p["STOP_LOSS_PCT"]:
                exit_reason = "STOP_LOSS"
            elif pnl_pct >= p["TAKE_PROFIT_PCT"]:
                exit_reason = "TAKE_PROFIT"
            elif pos["max_pnl_pct"] >= p["TRAILING_ACTIVATE_PCT"] and \
                    (pos["max_pnl_pct"] - pnl_pct) >= p["TRAILING_STOP_PCT"]:
                exit_reason = "TRAILING_STOP"
            elif hours_held >= p["MAX_HOLD_FORCED_HOURS"]:
                exit_reason = "MAX_HOLD_FORCED"
            elif hours_held >= p["MAX_HOLD_HOURS"] and abs(pnl_pct) < p["MAX_HOLD_FLAT_BAND_PCT"]:
                exit_reason = "MAX_HOLD_FLAT"

            if exit_reason:
                proceeds = pos["qty"] * close
                fee = proceeds * p["FEE_RATE"]
                cash += proceeds - fee
                trades.append({
                    "symbol": symbol, "entry_time": pos["entry_time"].isoformat(),
                    "exit_time": t.isoformat(), "entry_price": entry, "exit_price": close,
                    "pnl_pct": pnl_pct, "exit_reason": exit_reason,
                })
                if pnl_pct < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                del positions[symbol]
                check_trip(t)

        # --- entries ---
        regime_period = p["REGIME_SMA_PERIOD"]
        min_hist = max(p["TREND_LOOKBACK"] + 20, regime_period)
        if symbol not in positions and len(positions) < p["MAX_POSITIONS"] and not is_paused(t):
            hist = history[symbol]
            if len(hist) >= min_hist:
                # No indicator below looks back further than 150 bars; slicing
                # bounds the per-bar numpy conversion to O(1) instead of O(len(hist)),
                # which matters since hist grows to tens of thousands of bars.
                window = max(150, regime_period)
                closes_arr = np.array(hist[-window:]) if len(hist) > window else np.array(hist)
                regime_ok = True
                if regime_period:
                    regime_ok = closes_arr[-1] > float(np.mean(closes_arr[-regime_period:]))
                trend, slope = _trend(closes_arr, p["TREND_LOOKBACK"], p["MIN_TREND_SLOPE"])
                if regime_ok and trend == "UP":
                    rsi = _rsi(closes_arr, 14)
                    if rsi <= p["MAX_RSI_LONG"]:
                        macd_hist = _macd_histogram(closes_arr)
                        sma20 = float(np.mean(closes_arr[-20:]))
                        score = compute_rule_score(rsi, trend, slope, macd_hist, closes_arr[-1], sma20)
                        if score >= p["MIN_RULE_SCORE"]:
                            eq = equity()
                            target_value = eq * p["MAX_POSITION_PCT"]
                            qty = target_value / close
                            cost = qty * close
                            fee = cost * p["FEE_RATE"]
                            cash -= (cost + fee)
                            positions[symbol] = {
                                "entry_price": close, "entry_time": t, "max_pnl_pct": 0.0, "qty": qty,
                            }

        history[symbol].append(close)
        equity_curve.append(equity())

    # liquidate anything still open at the end
    for symbol, pos in list(positions.items()):
        close = last_price[symbol]
        pnl_pct = (close / pos["entry_price"] - 1.0) * 100.0
        proceeds = pos["qty"] * close
        fee = proceeds * p["FEE_RATE"]
        cash += proceeds - fee
        trades.append({
            "symbol": symbol, "entry_time": pos["entry_time"].isoformat(),
            "exit_time": "END", "entry_price": pos["entry_price"], "exit_price": close,
            "pnl_pct": pnl_pct, "exit_reason": "END_OF_BACKTEST",
        })

    final_equity = cash
    pnls = [tr["pnl_pct"] for tr in trades]
    wins = [pl for pl in pnls if pl > 0]
    losses = [pl for pl in pnls if pl <= 0]

    peak = equity_curve[0] if equity_curve else p["INITIAL_CASH"]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (v / peak - 1.0) * 100.0 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    from collections import Counter
    return {
        "params": p,
        "initial_cash": p["INITIAL_CASH"], "final_equity": final_equity,
        "net_profit_pct": (final_equity / p["INITIAL_CASH"] - 1.0) * 100.0,
        "total_trades": len(trades),
        "win_rate_pct": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "avg_win_pct": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss_pct": (sum(losses) / len(losses)) if losses else 0.0,
        "max_drawdown_pct": max_dd,
        "exit_reasons": dict(Counter(tr["exit_reason"] for tr in trades)),
        "trades_per_symbol": dict(Counter(tr["symbol"] for tr in trades)),
        "trades": trades,
    }


def main():
    print("Fetching real historical hourly bars from Alpaca (free tier)...")
    events = load_events()
    result = run_backtest(events)

    print("\n=== CryptoBot Local Backtest Results (Alpaca real hourly data, rules-only QC-port logic) ===")
    print(f"Symbols: {SYMBOLS}  |  Period: {START} to {END}")
    print(f"Initial cash: ${result['initial_cash']:,.2f}  |  Final equity: ${result['final_equity']:,.2f}")
    print(f"Net Profit: {result['net_profit_pct']:.3f}%")
    print(f"Total trades: {result['total_trades']}")
    if result["total_trades"]:
        print(f"Win rate: {result['win_rate_pct']:.1f}%")
        print(f"Avg win: {result['avg_win_pct']:.3f}%")
        print(f"Avg loss: {result['avg_loss_pct']:.3f}%")
    print(f"Max drawdown: {result['max_drawdown_pct']:.2f}%")
    print("Exit reasons:", result["exit_reasons"])
    print("Trades per symbol:", result["trades_per_symbol"])

    out_path = os.path.join(BASE_DIR, "data", "state", "local_qc_port_backtest_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "symbols": SYMBOLS, "start": START, "end": END,
            "initial_cash": result["initial_cash"], "final_equity": result["final_equity"],
            "net_profit_pct": result["net_profit_pct"],
            "total_trades": result["total_trades"], "win_rate_pct": result["win_rate_pct"],
            "max_drawdown_pct": result["max_drawdown_pct"], "trades": result["trades"],
        }, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
