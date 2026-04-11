"""
AlpacaBot DTE Sweep Backtester
===============================
Tests every symbol at every DTE it actually supports (from live API discovery).
Finds the OPTIMAL DTE per symbol, then produces a final recommendation:
which symbols to trade, and at what DTE.

This fixes the critical flaw in the original backtest: it forced 1DTE on
everything, but most symbols don't even offer 1DTE options. Many have
2DTE minimum, some 7DTE, etc.

Usage:
  python tools/backtest_dte_sweep.py
  python tools/backtest_dte_sweep.py --days 365
"""
import sys, os, math, argparse, warnings, time, json, builtins
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Force unbuffered prints
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from core.indicators import compute_all_indicators

# ═══════════════════════════════════════════════════════════
#  UNIVERSE (full 61 for re-evaluation)
# ═══════════════════════════════════════════════════════════

TIER1 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
TIER2 = ["AMD", "NFLX", "CRM", "AVGO", "ORCL", "ADBE", "INTC",
         "QCOM", "MU", "SHOP", "UBER", "COIN", "PYPL", "SNOW"]
TIER3 = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
         "ARKK", "SMH", "SOXX", "GLD", "SLV", "EEM", "TLT"]
TIER4 = ["BA", "JPM", "GS", "V", "MA", "WMT", "HD", "COST",
         "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "XOM",
         "CVX", "DIS", "NKE", "SBUX", "F", "GM", "RIVN",
         "PLTR", "SOFI", "MARA", "RIOT"]

TIER_MAP = {}
for s in TIER1: TIER_MAP[s] = "TIER1"
for s in TIER2: TIER_MAP[s] = "TIER2"
for s in TIER3: TIER_MAP[s] = "TIER3"
for s in TIER4: TIER_MAP[s] = "TIER4"

ALL_SYMBOLS = list(dict.fromkeys(TIER1 + TIER2 + TIER3 + TIER4))

# ═══════════════════════════════════════════════════════════
#  DTE MAP — from live API probe (tools/backtests/dte_discovery.json)
#  Fallback: if no discovery data, use typical patterns
# ═══════════════════════════════════════════════════════════

# These are the DTE values we'll test for each symbol.
# Key insight: options expiring on a specific weekday always have the
# SAME set of DTEs when viewed from each weekday. E.g., Friday-expiring
# options are always 2DTE when probed on Wednesday, but 1DTE on Thursday.
# We test a representative range and the backtest simulates holding across days.

# Symbols with daily expirations (0DTE available) — test 0, 1, 2
DAILY_EXPIRY = {"SPY", "QQQ", "IWM", "AAPL", "MSFT", "AMZN", "GOOGL",
                "META", "TSLA", "AVGO", "GLD", "SLV", "TLT"}

# Default weekly DTE options for everything else
# Most have expiries at 2, 7/9, 14/16 days — we test representative values
DEFAULT_DTES = [2, 7]

# Full DTE test matrix
def get_test_dtes(symbol: str) -> list:
    """Return list of DTEs to test for a given symbol."""
    if symbol in DAILY_EXPIRY:
        return [1, 2, 5, 7]   # test daily symbols at 1, 2, 5, 7
    else:
        return [2, 7]          # weekly symbols: test 2DTE and 7DTE

# Also try loading actual DTE data from probe
DTE_DISCOVERY = {}
dte_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "backtests", "dte_discovery.json")
if os.path.exists(dte_path):
    with open(dte_path) as f:
        DTE_DISCOVERY = json.load(f)

def get_real_dtes(symbol: str) -> list:
    """Get DTEs from actual API probe, or fall back to defaults."""
    if symbol in DTE_DISCOVERY:
        dtes = DTE_DISCOVERY[symbol].get("dtes", [])
        # Filter: never test 0DTE, always test at least 1
        valid = sorted(set(d for d in dtes if d >= 1))
        # Also add some common test points if not present
        if not valid:
            return get_test_dtes(symbol)
        # Cap at max 4 DTEs to test (keep runtime manageable)
        # Prioritize: min, 7, max, and a midpoint
        if len(valid) > 4:
            result = [valid[0]]               # min DTE
            if 7 in valid: result.append(7)    # weekly
            mid = valid[len(valid) // 2]
            if mid not in result: result.append(mid)
            if valid[-1] not in result: result.append(valid[-1])
            return sorted(set(result))
        return valid
    return get_test_dtes(symbol)


# ═══════════════════════════════════════════════════════════
#  BACKTEST CONFIG
# ═══════════════════════════════════════════════════════════

INITIAL_BALANCE = 25_000
OTM_PCT = 0.015
MAX_RISK_PER_TRADE = 0.06
STOP_LOSS = -0.20
TAKE_PROFIT = 0.25
TRAILING_STOP = 0.15
TRAILING_TRIGGER = 0.08
LOOKBACK = 50
MIN_SIGNAL_SCORE = 3
COOLDOWN_BARS = 6
BARS_PER_DAY = 39
SIGNAL_CHECK_INTERVAL = 2
HIST_DAYS = 365

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "historical")


# ═══════════════════════════════════════════════════════════
#  BLACK-SCHOLES
# ═══════════════════════════════════════════════════════════

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, sigma, opt_type, r=0.05):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.01) if opt_type == "call" else max(K - S, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def estimate_iv(closes, window=30):
    if len(closes) < window + 1:
        return 0.25
    rets = np.diff(np.log(closes[-window - 1:]))
    hv = float(np.std(rets) * np.sqrt(BARS_PER_DAY * 252))
    return max(0.12, hv * 1.15)


# ═══════════════════════════════════════════════════════════
#  SIGNAL GENERATION (identical to live engine)
# ═══════════════════════════════════════════════════════════

def generate_signal(chunk, indicators):
    bull, bear = 0, 0

    rsi = indicators.get("rsi", 50)
    if rsi < 25:      bull += 2
    elif 35 < rsi < 55: bull += 1
    elif rsi > 75:     bear += 2
    elif 50 < rsi < 65: bear += 1

    macd_h = indicators.get("macd_hist", 0)
    if macd_h > 0:
        bull += 1
        if macd_h > 0.1: bull += 1
    elif macd_h < 0:
        bear += 1
        if macd_h < -0.1: bear += 1

    stoch = indicators.get("stochastic", 50)
    if stoch < 20:   bull += 1
    elif stoch > 80: bear += 1

    bb = indicators.get("bb_position", 0.5)
    if bb < 0.10:    bull += 2
    elif bb > 0.90:  bear += 2
    elif bb < 0.30:  bull += 1
    elif bb > 0.70:  bear += 1

    atr_n = indicators.get("atr_normalized", 0)
    if atr_n > 0.005:
        if bull > bear:   bull += 1
        elif bear > bull: bear += 1

    cci_val = indicators.get("cci", 0)
    if cci_val < -100: bull += 1
    elif cci_val > 100: bear += 1

    roc_val = indicators.get("roc", 0)
    if roc_val > 0.3:   bull += 1
    elif roc_val < -0.3: bear += 1

    wr = indicators.get("williams_r", -50)
    if wr > -20:   bear += 1
    elif wr < -80: bull += 1

    vol_r = indicators.get("volatility_ratio", 1.0)
    if vol_r > 1.3:
        if bull > bear:   bull += 1
        elif bear > bull: bear += 1

    zs = indicators.get("zscore", 0)
    if zs < -2.0: bull += 1
    elif zs > 2.0: bear += 1

    ts = indicators.get("trend_strength", 0)
    if ts > 25:
        if bull > bear:   bull += 1
        elif bear > bull: bear += 1

    pc1 = indicators.get("price_change_1", 0)
    pc5 = indicators.get("price_change_5", 0)
    if pc1 > 0.001:   bull += 1
    elif pc1 < -0.001: bear += 1
    if pc5 > 0.003:   bull += 1
    elif pc5 < -0.003: bear += 1

    if bull >= MIN_SIGNAL_SCORE and bull > bear + 1:
        return "call", bull, bear
    elif bear >= MIN_SIGNAL_SCORE and bear > bull + 1:
        return "put", bull, bear
    return None, bull, bear


# ═══════════════════════════════════════════════════════════
#  DATA FETCHING (reuses cached CSVs from prior backtest)
# ═══════════════════════════════════════════════════════════

def fetch_all_bars(symbols, days):
    """Load cached bar data or fetch from Alpaca."""
    from core.config import Config
    from core.api_client import AlpacaAPI
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    cfg = Config()
    api = AlpacaAPI(cfg)
    api.connect()
    tf = TimeFrame(10, TimeFrameUnit.Minute)

    all_data = {}
    cached = fetched = failed = 0

    for i, sym in enumerate(symbols):
        csv_path = os.path.join(DATA_DIR, f"{sym}_10min.csv")

        if os.path.exists(csv_path):
            age_hours = (time.time() - os.path.getmtime(csv_path)) / 3600
            if age_hours < 48:  # 48h cache for sweep
                df = pd.read_csv(csv_path)
                if len(df) > 100:
                    all_data[sym] = df["close"].values
                    cached += 1
                    continue

        try:
            bars = api.get_bars(sym, tf, days=days)
            if not bars or len(bars) < 50:
                failed += 1
                continue
            rows = [{"timestamp": str(b.timestamp), "open": float(b.open),
                     "high": float(b.high), "low": float(b.low),
                     "close": float(b.close), "volume": int(b.volume)} for b in bars]
            df = pd.DataFrame(rows)
            os.makedirs(DATA_DIR, exist_ok=True)
            df.to_csv(csv_path, index=False)
            all_data[sym] = df["close"].values
            fetched += 1
            time.sleep(0.1)
        except Exception as e:
            print(f"    {sym}: FAILED - {e}")
            failed += 1

    print(f"    Data ready: {len(all_data)} symbols "
          f"({cached} cached, {fetched} fetched, {failed} failed)")
    return all_data


# ═══════════════════════════════════════════════════════════
#  PER-SYMBOL BACKTESTER (DTE-aware)
# ═══════════════════════════════════════════════════════════

def backtest_symbol_at_dte(symbol, closes, dte):
    """
    Run isolated backtest on a single symbol at a specific DTE.
    Key DTE-dependent logic:
      - Option premium: higher DTE = more time value = more expensive
      - Theta decay: scales with sqrt(T), slower for higher DTE
      - Max hold: scales with DTE (can hold overnight on 2DTE+)
      - DTE exit: exit before expiry to avoid exercise risk
    """
    if len(closes) < LOOKBACK + 50:
        return None

    # DTE-dependent parameters
    max_hold_bars = int(BARS_PER_DAY * max(1, dte - 0.5))  # hold up to DTE-0.5 days
    dte_exit_bars = int(BARS_PER_DAY * max(0.5, dte - 0.5))  # exit 0.5 days before expiry

    train_end = int(len(closes) * 0.4)

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    position = None
    trades = []
    cooldown_until = 0
    consec_losses = 0
    signals_generated = 0

    for bar_idx in range(train_end, len(closes)):
        S = closes[bar_idx]

        # Mark-to-market
        if position is not None:
            bars_held = bar_idx - position["entry_bar"]
            days_elapsed = bars_held / BARS_PER_DAY
            remaining_dte = max(0.01, (dte - days_elapsed)) / 365.0
            val = bs_price(S, position["strike"], remaining_dte, position["iv"], position["type"])
            position["current_value"] = val
            position["peak_value"] = max(position.get("peak_value", position["premium"]), val)
            position["bars_held"] = bars_held

            pnl_pct = (val - position["premium"]) / position["premium"]
            reason = None

            # Standard exits
            if pnl_pct <= STOP_LOSS:
                reason = "STOP_LOSS"
            elif pnl_pct >= TAKE_PROFIT:
                reason = "TAKE_PROFIT"
            elif position["peak_value"] > position["premium"] * (1 + TRAILING_TRIGGER):
                drop = (val - position["peak_value"]) / position["peak_value"]
                if drop <= -TRAILING_STOP:
                    reason = "TRAILING_STOP"
            elif bars_held >= max_hold_bars:
                reason = "MAX_HOLD"
            elif bars_held >= dte_exit_bars:
                reason = "DTE_EXIT"

            if reason:
                pnl_per = val - position["premium"]
                pnl = pnl_per * 100 * position["qty"]
                balance += pnl
                if pnl < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0
                peak_balance = max(peak_balance, balance)
                cooldown_until = bar_idx + COOLDOWN_BARS

                trades.append({
                    "type": position["type"],
                    "entry_bar": position["entry_bar"],
                    "exit_bar": bar_idx,
                    "bars_held": bars_held,
                    "days_held": bars_held / BARS_PER_DAY,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": reason,
                })
                position = None

        if position is not None:
            continue
        if bar_idx < cooldown_until:
            continue
        dd = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
        if consec_losses >= 5 or dd <= -0.15:
            consec_losses = max(0, consec_losses - 1)
            continue
        if bar_idx % SIGNAL_CHECK_INTERVAL != 0:
            continue
        if bar_idx < LOOKBACK:
            continue

        chunk = closes[bar_idx - LOOKBACK:bar_idx + 1]
        indicators = compute_all_indicators(chunk)
        direction, bull, bear = generate_signal(chunk, indicators)

        if direction is None:
            continue

        signals_generated += 1
        iv = estimate_iv(closes[:bar_idx + 1])
        K = round(S * (1 + OTM_PCT), 2) if direction == "call" else round(S * (1 - OTM_PCT), 2)
        T = dte / 365.0
        premium = bs_price(S, K, T, iv, direction)

        if premium < 0.05:
            continue

        cost_per = premium * 100
        max_spend = balance * MAX_RISK_PER_TRADE
        if cost_per > max_spend:
            continue

        qty = max(1, int(max_spend / cost_per))

        position = {
            "type": direction,
            "entry_bar": bar_idx,
            "entry_price": S,
            "strike": K,
            "iv": iv,
            "premium": premium,
            "current_value": premium,
            "peak_value": premium,
            "qty": qty,
            "bars_held": 0,
        }

    # Close any open position at end
    if position is not None:
        S = closes[-1]
        bars_held = len(closes) - 1 - position["entry_bar"]
        days_elapsed = bars_held / BARS_PER_DAY
        remaining = max(0.01, (dte - days_elapsed)) / 365.0
        val = bs_price(S, position["strike"], remaining, position["iv"], position["type"])
        pnl = (val - position["premium"]) * 100 * position["qty"]
        balance += pnl
        trades.append({
            "type": position["type"],
            "entry_bar": position["entry_bar"],
            "exit_bar": len(closes) - 1,
            "bars_held": bars_held,
            "days_held": bars_held / BARS_PER_DAY,
            "pnl": pnl,
            "pnl_pct": (val - position["premium"]) / position["premium"],
            "exit_reason": "END",
        })

    if not trades:
        return None

    pnl_total = balance - INITIAL_BALANCE
    w = [t for t in trades if t["pnl"] > 0]
    l = [t for t in trades if t["pnl"] <= 0]
    wr = len(w) / len(trades) * 100
    gp = sum(t["pnl"] for t in w) if w else 0
    gl = abs(sum(t["pnl"] for t in l)) if l else 0
    pf = gp / gl if gl > 0 else float("inf")

    running = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    max_dd = 0
    for t in trades:
        running += t["pnl"]
        peak = max(peak, running)
        dd = (running - peak) / peak
        max_dd = min(max_dd, dd)

    avg_hold = np.mean([t["days_held"] for t in trades])
    calls = [t for t in trades if t["type"] == "call"]
    puts = [t for t in trades if t["type"] == "put"]

    return {
        "pnl": pnl_total,
        "pnl_pct": pnl_total / INITIAL_BALANCE * 100,
        "trades": len(trades),
        "win_rate": wr,
        "profit_factor": min(pf, 999),
        "max_dd": max_dd,
        "avg_hold": avg_hold,
        "calls_pnl": sum(t["pnl"] for t in calls),
        "puts_pnl": sum(t["pnl"] for t in puts),
        "calls_n": len(calls),
        "puts_n": len(puts),
    }


# ═══════════════════════════════════════════════════════════
#  DTE SWEEP
# ═══════════════════════════════════════════════════════════

def sweep_symbol(symbol, closes):
    """Test all available DTEs for a symbol, return best result + all results."""
    test_dtes = get_real_dtes(symbol)

    results = {}
    for dte in test_dtes:
        result = backtest_symbol_at_dte(symbol, closes, dte)
        if result is not None:
            results[dte] = result

    if not results:
        return None, {}

    # Find best DTE by P&L (with PF tiebreaker)
    best_dte = max(results.keys(), key=lambda d: (results[d]["pnl"], results[d]["profit_factor"]))

    return best_dte, results


# ═══════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════

def print_results(all_results):
    """Print comprehensive DTE sweep results."""

    print(f"\n{'=' * 120}")
    print(f"  DTE SWEEP RESULTS  |  {HIST_DAYS} days  |  10-min bars  |  Per-symbol optimal DTE  |  ${INITIAL_BALANCE:,}")
    print(f"  SL: {STOP_LOSS:.0%} | TP: {TAKE_PROFIT:+.0%} | Trail: {TRAILING_STOP:.0%}")
    print(f"{'=' * 120}")

    # ── Per-symbol comparison table ──
    print(f"\n  {'Sym':<6} {'Tier':<6} ", end="")
    # Column headers for each DTE tested
    all_dtes_tested = sorted(set(dte for _, results in all_results for dte in results.keys()))
    for dte in all_dtes_tested:
        print(f"{'%dDTE P&L' % dte:>12} {'PF':>5} ", end="")
    print(f"  {'BEST':>5} {'P&L':>10} {'PF':>5} {'WR':>5} {'#Tr':>4} {'Hold':>5} {'Verdict':>8}")
    print(f"  {'-' * (14 + 17 * len(all_dtes_tested) + 52)}")

    # Sort by best DTE P&L
    sorted_results = sorted(all_results,
                            key=lambda x: max((r["pnl"] for r in x[1].values()), default=-99999),
                            reverse=True)

    keep = []
    maybe = []
    drop = []

    for symbol, results in sorted_results:
        if not results:
            continue
        tier = TIER_MAP.get(symbol, "?")
        best_dte = max(results.keys(), key=lambda d: (results[d]["pnl"], results[d]["profit_factor"]))
        best = results[best_dte]

        # Classify
        if best["pnl"] > 0 and best["profit_factor"] >= 1.2:
            verdict = "KEEP"
            keep.append((symbol, best_dte, best))
        elif best["pnl"] > 0:
            verdict = "MAYBE"
            maybe.append((symbol, best_dte, best))
        else:
            verdict = "DROP"
            drop.append((symbol, best_dte, best))

        # Print DTE comparison
        print(f"  {symbol:<6} {tier:<6} ", end="")
        for dte in all_dtes_tested:
            if dte in results:
                r = results[dte]
                pf_s = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else "inf"
                marker = " *" if dte == best_dte else "  "
                print(f"${r['pnl']:>+9,.0f}{marker} {pf_s:>5} ", end="")
            else:
                print(f"{'---':>12} {'---':>5} ", end="")

        pf_s = f"{best['profit_factor']:.2f}" if best['profit_factor'] < 100 else "inf"
        print(f"  {best_dte:>4}d ${best['pnl']:>+9,.0f} {pf_s:>5} "
              f"{best['win_rate']:>4.0f}% {best['trades']:>4} {best['avg_hold']:>4.1f}d "
              f"{'  ' + verdict:>8}")

    # ── Summary ──
    print(f"\n{'=' * 120}")
    print(f"  SUMMARY")
    print(f"{'=' * 120}")
    print(f"  KEEP  ({len(keep):>2} symbols, PF >= 1.2): ", end="")
    if keep:
        total_keep_pnl = sum(b["pnl"] for _, _, b in keep)
        print(f"combined P&L = ${total_keep_pnl:+,.0f}")
        keep.sort(key=lambda x: x[2]["pnl"], reverse=True)
        for sym, dte, b in keep:
            pf_s = f"{b['profit_factor']:.2f}" if b['profit_factor'] < 100 else "inf"
            print(f"    {sym:<6} @ {dte}DTE  ${b['pnl']:>+9,.0f} ({b['pnl_pct']:>+.0f}%)  "
                  f"PF {pf_s}  WR {b['win_rate']:.0f}%  {b['trades']} trades  "
                  f"hold {b['avg_hold']:.1f}d  DD {b['max_dd']:.1%}")
    else:
        print("(none)")

    print(f"\n  MAYBE ({len(maybe):>2} symbols, PF 1.0-1.2): ", end="")
    if maybe:
        total_maybe_pnl = sum(b["pnl"] for _, _, b in maybe)
        print(f"combined P&L = ${total_maybe_pnl:+,.0f}")
        maybe.sort(key=lambda x: x[2]["pnl"], reverse=True)
        for sym, dte, b in maybe:
            pf_s = f"{b['profit_factor']:.2f}" if b['profit_factor'] < 100 else "inf"
            print(f"    {sym:<6} @ {dte}DTE  ${b['pnl']:>+9,.0f} ({b['pnl_pct']:>+.0f}%)  "
                  f"PF {pf_s}  WR {b['win_rate']:.0f}%  {b['trades']} trades")
    else:
        print("(none)")

    print(f"\n  DROP  ({len(drop):>2} symbols): ", end="")
    if drop:
        total_drop_pnl = sum(b["pnl"] for _, _, b in drop)
        print(f"combined P&L = ${total_drop_pnl:+,.0f}")
        drop.sort(key=lambda x: x[2]["pnl"])
        print(f"    {', '.join(s for s, _, _ in drop)}")
    else:
        print("(none)")

    # ── DTE impact analysis ──
    print(f"\n{'=' * 120}")
    print(f"  DTE IMPACT ANALYSIS")
    print(f"{'=' * 120}")

    # Compare 1DTE vs 2DTE vs 7DTE averages
    for dte in sorted(all_dtes_tested):
        syms_at_dte = [(sym, results[dte]) for sym, results in all_results if dte in results]
        if not syms_at_dte:
            continue
        avg_pnl = np.mean([r["pnl"] for _, r in syms_at_dte])
        avg_pf = np.mean([min(r["profit_factor"], 10) for _, r in syms_at_dte])
        avg_wr = np.mean([r["win_rate"] for _, r in syms_at_dte])
        prof = len([1 for _, r in syms_at_dte if r["pnl"] > 0])
        total = sum(r["pnl"] for _, r in syms_at_dte)
        print(f"  {dte:>2}DTE: {len(syms_at_dte):>2} symbols tested | "
              f"{prof}/{len(syms_at_dte)} profitable | "
              f"avg P&L ${avg_pnl:>+8,.0f} | avg PF {avg_pf:.2f} | "
              f"avg WR {avg_wr:.0f}% | total ${total:>+10,.0f}")

    # ── Symbols that CHANGED verdict with DTE optimization ──
    # Load old 1DTE results for comparison
    old_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "backtests", "scanner_universe_results.json")
    if os.path.exists(old_results_path):
        with open(old_results_path) as f:
            old_data = json.load(f)
        old_map = {r["symbol"]: r for r in old_data}

        print(f"\n{'=' * 120}")
        print(f"  VERDICT CHANGES vs 1DTE BACKTEST")
        print(f"{'=' * 120}")

        changed = []
        for sym, results in all_results:
            if not results:
                continue
            best_dte = max(results.keys(), key=lambda d: (results[d]["pnl"], results[d]["profit_factor"]))
            best = results[best_dte]
            old = old_map.get(sym)
            if old:
                old_verdict = old.get("verdict", "?")
                new_verdict = "KEEP" if best["pnl"] > 0 and best["profit_factor"] >= 1.2 \
                    else "MAYBE" if best["pnl"] > 0 else "DROP"
                if old_verdict != new_verdict:
                    direction = "UPGRADED" if (new_verdict == "KEEP" or
                                               (new_verdict == "MAYBE" and old_verdict == "DROP")) \
                        else "DOWNGRADED"
                    changed.append((sym, old_verdict, new_verdict, old.get("pnl", 0),
                                    best["pnl"], best_dte, direction))

        if changed:
            changed.sort(key=lambda x: x[4], reverse=True)
            print(f"  {'Symbol':<7} {'Old':>6} {'New':>6} {'Old P&L':>10} {'New P&L':>10} {'DTE':>5} {'Change':>10}")
            print(f"  {'-' * 58}")
            for sym, old_v, new_v, old_pnl, new_pnl, dte, direction in changed:
                print(f"  {sym:<7} {old_v:>6} {new_v:>6} ${old_pnl:>+9,.0f} "
                      f"${new_pnl:>+9,.0f} {dte:>4}d {direction:>10}")
        else:
            print("  No verdict changes detected.")

    # ── Generate per-symbol DTE config ──
    print(f"\n{'=' * 120}")
    print(f"  RECOMMENDED DTE CONFIG (for bot)")
    print(f"{'=' * 120}")

    dte_config = {}
    for sym, dte, b in keep + maybe:
        dte_config[sym] = dte

    if dte_config:
        print(f"\n  SYMBOL_DTE_MAP = {{")
        for sym, dte in sorted(dte_config.items()):
            print(f"      \"{sym}\": {dte},")
        print(f"  }}")
    print()

    # ── Save results ──
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "backtests", "dte_sweep_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    save_data = []
    for symbol, results in sorted_results:
        if not results:
            continue
        best_dte = max(results.keys(), key=lambda d: (results[d]["pnl"], results[d]["profit_factor"]))
        best = results[best_dte]
        entry = {
            "symbol": symbol,
            "tier": TIER_MAP.get(symbol, "?"),
            "best_dte": best_dte,
            "best_pnl": round(best["pnl"], 2),
            "best_pnl_pct": round(best["pnl_pct"], 2),
            "best_pf": round(best["profit_factor"], 2),
            "best_wr": round(best["win_rate"], 1),
            "best_trades": best["trades"],
            "best_avg_hold": round(best["avg_hold"], 2),
            "best_max_dd": round(best["max_dd"], 4),
            "best_calls_pnl": round(best.get("calls_pnl", 0), 2),
            "best_puts_pnl": round(best.get("puts_pnl", 0), 2),
            "verdict": "KEEP" if best["pnl"] > 0 and best["profit_factor"] >= 1.2
                       else "MAYBE" if best["pnl"] > 0 else "DROP",
            "all_dtes": {str(d): {
                "pnl": round(r["pnl"], 2),
                "pnl_pct": round(r["pnl_pct"], 2),
                "pf": round(r["profit_factor"], 2),
                "wr": round(r["win_rate"], 1),
                "trades": r["trades"],
            } for d, r in results.items()},
        }
        save_data.append(entry)

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to: {results_path}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    global HIST_DAYS

    parser = argparse.ArgumentParser(description="DTE Sweep Backtester")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    args = parser.parse_args()

    HIST_DAYS = args.days

    symbols = ALL_SYMBOLS
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Show what we'll test
    print("=" * 120)
    print(f"  AlpacaBot DTE Sweep Backtest")
    print(f"  {len(symbols)} symbols | {HIST_DAYS} days | 10-min bars | Multi-DTE | ${INITIAL_BALANCE:,}")
    print(f"  SL: {STOP_LOSS:.0%} | TP: {TAKE_PROFIT:+.0%} | Trail: {TRAILING_STOP:.0%}")
    print("=" * 120)

    print(f"\n  DTE test matrix:")
    dte_counts = defaultdict(int)
    for sym in symbols:
        dtes = get_real_dtes(sym)
        for d in dtes:
            dte_counts[d] += 1
        print(f"    {sym:<6}: test DTEs {dtes}")
    print(f"\n  DTE coverage: ", end="")
    for d in sorted(dte_counts.keys()):
        print(f"{d}d={dte_counts[d]}sym  ", end="")
    print()

    total_tests = sum(len(get_real_dtes(s)) for s in symbols)
    print(f"  Total backtests to run: {total_tests}")

    # Phase 1: Data
    print(f"\n  Phase 1: Loading {HIST_DAYS}-day 10-min bars for {len(symbols)} symbols...")
    t0 = time.time()
    all_data = fetch_all_bars(symbols, HIST_DAYS)
    dl_time = time.time() - t0
    print(f"  Data loaded in {dl_time:.1f}s")

    # Phase 2: DTE sweep
    print(f"\n  Phase 2: Running DTE sweep on {len(all_data)} symbols...")
    t0 = time.time()
    all_results = []

    for i, (sym, closes) in enumerate(all_data.items()):
        best_dte, results = sweep_symbol(sym, closes)
        if results:
            all_results.append((sym, results))
            # Quick progress
            if best_dte:
                best = results[best_dte]
                dtes_tested = sorted(results.keys())
                status = f"best={best_dte}DTE ${best['pnl']:+,.0f}"
            else:
                status = "no trades"
            if (i + 1) % 10 == 0 or i == len(all_data) - 1:
                elapsed = time.time() - t0
                print(f"    [{i+1}/{len(all_data)}] {elapsed:.1f}s elapsed")
        else:
            print(f"    {sym}: insufficient data")

    bt_time = time.time() - t0
    print(f"  Sweep complete in {bt_time:.1f}s")

    # Phase 3: Report
    print_results(all_results)

    total_time = dl_time + bt_time
    print(f"\n  Total runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
