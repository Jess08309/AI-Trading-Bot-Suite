"""
AlpacaBot Multi-Timeframe Backtester
=====================================
Tests multi-timeframe confirmation: use a fast timeframe for signals
and a slow timeframe as a filter (or vice versa).

Modes:
  1. 5-min only          (baseline — our proven +66.7%)
  2. 10-min only          (never tested)
  3. 5-min signal + 10-min confirmation
  4. 10-min signal + 5-min confirmation

Resamples 5-min bars → 10-min by aggregating every 2 bars.
All 14 indicators computed on both timeframes independently.
MSFT + NVDA only, 1DTE, calls+puts.

Usage:
  python tools/backtest_mtf.py
"""
import sys, os, math, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from collections import defaultdict
from core.indicators import compute_all_indicators

# ═══════════════════════════════════════════════════════════
#  CONFIG (matches our proven winner)
# ═══════════════════════════════════════════════════════════

SYMBOLS = ["MSFT", "NVDA"]
INITIAL_BALANCE = 25_000

# DTE
MIN_DTE = 1
DTE = 1

# Options
OTM_PCT = 0.015

# Risk
MAX_RISK_PER_TRADE = 0.06
MAX_POSITIONS = 3
STOP_LOSS = -0.20
TAKE_PROFIT = 0.25
TRAILING_STOP = 0.15
TRAILING_TRIGGER = 0.08

# Signal
LOOKBACK_5 = 50       # 50 5-min bars for indicators
LOOKBACK_10 = 50      # 50 10-min bars for indicators (need 50 for SMA-50, trend strength)
MIN_SIGNAL_SCORE = 3
COOLDOWN_BARS_5 = 12  # 1 hour in 5-min bars
COOLDOWN_BARS_10 = 6  # 1 hour in 10-min bars

BARS_PER_DAY_5 = 78   # 390 min / 5
BARS_PER_DAY_10 = 39  # 390 min / 10


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

def estimate_iv(closes, window=30, bars_per_day=78):
    if len(closes) < window + 1:
        return 0.25
    rets = np.diff(np.log(closes[-window - 1:]))
    hv = float(np.std(rets) * np.sqrt(bars_per_day * 252))
    return max(0.12, hv * 1.15)


# ═══════════════════════════════════════════════════════════
#  SIGNAL GENERATION (same 14 indicators)
# ═══════════════════════════════════════════════════════════

def generate_signal(prices_chunk, indicators):
    """
    Same scalp signal as backtest_scalp.py.
    Returns: (direction, score) -- 'call'/'put'/None, 0-10
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

    # 12-14. Price momentum
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


def get_confirmation_direction(indicators):
    """
    Lightweight check: is the confirming timeframe bullish or bearish?
    Returns 'call', 'put', or None (neutral).
    Uses a simplified subset to determine trend bias.
    """
    bull, bear = 0, 0

    rsi = indicators.get("rsi", 50)
    if rsi < 40:
        bull += 1
    elif rsi > 60:
        bear += 1

    macd_h = indicators.get("macd_hist", 0)
    if macd_h > 0:
        bull += 1
    elif macd_h < 0:
        bear += 1

    bb = indicators.get("bb_position", 0.5)
    if bb < 0.35:
        bull += 1
    elif bb > 0.65:
        bear += 1

    ts = indicators.get("trend_strength", 0)
    pc5 = indicators.get("price_change_5", 0)
    if pc5 > 0.001:
        bull += 1
    elif pc5 < -0.001:
        bear += 1

    cci_val = indicators.get("cci", 0)
    if cci_val < -50:
        bull += 1
    elif cci_val > 50:
        bear += 1

    if bull >= 3 and bull > bear:
        return "call"
    elif bear >= 3 and bear > bull:
        return "put"
    return None


# ═══════════════════════════════════════════════════════════
#  RESAMPLE 5-min → 10-min
# ═══════════════════════════════════════════════════════════

def resample_to_10min(df_5min):
    """Aggregate every 2 consecutive 5-min bars into 1 10-min bar."""
    closes = df_5min["close"].values
    # Take every other close (end of each 10-min window)
    closes_10 = closes[1::2]  # indices 1, 3, 5, ... (2nd bar of each pair)
    return closes_10


# ═══════════════════════════════════════════════════════════
#  CORE BACKTESTER
# ═══════════════════════════════════════════════════════════

def run_single_backtest(mode, all_data_5, all_data_10):
    """
    Run one backtest configuration.
    
    Modes:
      '5min_only'     - signals on 5-min, no confirmation
      '10min_only'    - signals on 10-min, no confirmation
      '5sig_10conf'   - signals on 5-min, 10-min must confirm direction
      '10sig_5conf'   - signals on 10-min, 5-min must confirm direction
    """
    # Choose primary timeframe params
    if mode in ('5min_only', '5sig_10conf'):
        primary_data = all_data_5
        bars_per_day = BARS_PER_DAY_5
        lookback = LOOKBACK_5
        cooldown_bars = COOLDOWN_BARS_5
        signal_check_interval = 3  # every 15 min
    else:
        primary_data = all_data_10
        bars_per_day = BARS_PER_DAY_10
        lookback = LOOKBACK_10
        cooldown_bars = COOLDOWN_BARS_10
        signal_check_interval = 2  # every 20 min (2 * 10min)

    max_bars = max(len(p) for p in primary_data.values())
    train_end = int(max_bars * 0.6)

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    positions = []
    trades = []
    daily_balances = [INITIAL_BALANCE]
    consec_losses = 0
    cooldowns = {}

    max_hold = max(bars_per_day, min(DTE * bars_per_day * 2, (DTE - 1) * bars_per_day))
    max_hold = max(bars_per_day, max_hold)

    signals_generated = 0
    signals_confirmed = 0
    signals_rejected = 0

    for bar_idx in range(train_end, max_bars):
        # Mark-to-market
        for pos in positions:
            sym = pos["symbol"]
            # Use 5-min data for pricing always (more granular)
            prices_5 = all_data_5[sym]
            if mode in ('5min_only', '5sig_10conf'):
                actual_bar = bar_idx
            else:
                # 10-min bar_idx → approximate 5-min bar for pricing
                actual_bar = min(bar_idx * 2 + 1, len(prices_5) - 1)

            if actual_bar >= len(prices_5):
                continue

            S = prices_5[actual_bar]
            bars_held = bar_idx - pos["entry_bar"]
            days_elapsed = bars_held / bars_per_day
            remaining_dte = max(0.1, (pos["dte"] - days_elapsed)) / 365.0
            val = bs_price(S, pos["strike"], remaining_dte, pos["iv"], pos["type"])
            pos["current_value"] = val
            pos["peak_value"] = max(pos.get("peak_value", pos["premium"]), val)
            pos["bars_held"] = bars_held

        # Check exits
        to_exit = []
        for i, pos in enumerate(positions):
            pnl_pct = (pos["current_value"] - pos["premium"]) / pos["premium"]
            reason = None

            if pnl_pct <= STOP_LOSS:
                reason = "STOP_LOSS"
            elif pnl_pct >= TAKE_PROFIT:
                reason = "TAKE_PROFIT"
            elif pos["peak_value"] > pos["premium"] * (1 + TRAILING_TRIGGER):
                drop = (pos["current_value"] - pos["peak_value"]) / pos["peak_value"]
                if drop <= -TRAILING_STOP:
                    reason = "TRAILING_STOP"
            elif pos["bars_held"] >= max_hold:
                reason = "MAX_HOLD"
            elif pos["bars_held"] >= (pos["dte"] - 1) * bars_per_day:
                reason = "DTE_EXIT"

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
                "bars_held": pos["bars_held"],
                "days_held": pos["bars_held"] / bars_per_day,
                "entry_price": pos["entry_price"],
                "strike": pos["strike"], "dte": pos["dte"],
                "premium": pos["premium"],
                "exit_value": pos["current_value"],
                "qty": pos["qty"],
                "pnl": pnl,
                "pnl_pct": (pos["current_value"] - pos["premium"]) / pos["premium"],
                "exit_reason": reason, "score": pos["score"],
            })

        # Track equity
        unrealized = sum(
            (p["current_value"] - p["premium"]) * 100 * p["qty"]
            for p in positions
        )
        daily_balances.append(balance + unrealized)

        # Circuit breakers
        dd = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
        if consec_losses >= 5 or dd <= -0.15:
            consec_losses = max(0, consec_losses - 1)
            continue

        # Signal check interval
        if bar_idx % signal_check_interval != 0:
            continue
        if len(positions) >= MAX_POSITIONS:
            continue

        # Generate signals
        for sym in SYMBOLS:
            if len(positions) >= MAX_POSITIONS:
                break
            if any(p["symbol"] == sym for p in positions):
                continue
            if bar_idx < cooldowns.get(sym, 0):
                continue

            prices_primary = primary_data[sym]
            if bar_idx < lookback or bar_idx >= len(prices_primary):
                continue

            chunk = prices_primary[bar_idx - lookback:bar_idx + 1]
            indicators = compute_all_indicators(chunk)
            direction, score = generate_signal(chunk, indicators)

            if direction is None:
                continue

            signals_generated += 1

            # ── Multi-timeframe confirmation ──
            if mode == '5sig_10conf':
                # Signal from 5-min, confirm on 10-min
                ten_idx = bar_idx // 2
                prices_10 = all_data_10[sym]
                if ten_idx < LOOKBACK_10 or ten_idx >= len(prices_10):
                    signals_rejected += 1
                    continue
                chunk_10 = prices_10[ten_idx - LOOKBACK_10:ten_idx + 1]
                ind_10 = compute_all_indicators(chunk_10)
                conf_dir = get_confirmation_direction(ind_10)
                if conf_dir != direction:
                    signals_rejected += 1
                    continue
                signals_confirmed += 1

            elif mode == '10sig_5conf':
                # Signal from 10-min, confirm on 5-min
                five_idx = min(bar_idx * 2 + 1, len(all_data_5[sym]) - 1)
                prices_5 = all_data_5[sym]
                if five_idx < LOOKBACK_5 or five_idx >= len(prices_5):
                    signals_rejected += 1
                    continue
                chunk_5 = prices_5[five_idx - LOOKBACK_5:five_idx + 1]
                ind_5 = compute_all_indicators(chunk_5)
                conf_dir = get_confirmation_direction(ind_5)
                if conf_dir != direction:
                    signals_rejected += 1
                    continue
                signals_confirmed += 1

            # Price the option (always use 5-min data for current price)
            if mode in ('5min_only', '5sig_10conf'):
                S = all_data_5[sym][bar_idx]
                iv = estimate_iv(all_data_5[sym][:bar_idx + 1], bars_per_day=BARS_PER_DAY_5)
            else:
                five_idx = min(bar_idx * 2 + 1, len(all_data_5[sym]) - 1)
                S = all_data_5[sym][five_idx]
                iv = estimate_iv(all_data_5[sym][:five_idx + 1], bars_per_day=BARS_PER_DAY_5)

            K = round(S * (1 + OTM_PCT), 2) if direction == "call" else round(S * (1 - OTM_PCT), 2)
            T = DTE / 365.0
            premium = bs_price(S, K, T, iv, direction)

            if premium < 0.05:
                continue

            cost_per = premium * 100
            max_spend = balance * MAX_RISK_PER_TRADE
            if cost_per > max_spend:
                continue

            qty = max(1, int(max_spend / cost_per))

            positions.append({
                "symbol": sym, "type": direction,
                "entry_bar": bar_idx, "entry_price": S,
                "strike": K, "iv": iv, "dte": DTE,
                "premium": premium, "current_value": premium,
                "peak_value": premium, "qty": qty,
                "score": score, "bars_held": 0,
            })

    # Close remaining positions
    for pos in positions:
        sym = pos["symbol"]
        if mode in ('5min_only', '5sig_10conf'):
            final_5 = min(max_bars - 1, len(all_data_5[sym]) - 1)
        else:
            final_5 = min((max_bars - 1) * 2 + 1, len(all_data_5[sym]) - 1)

        S = all_data_5[sym][final_5]
        bars_held = (max_bars - 1) - pos["entry_bar"]
        days_elapsed = bars_held / bars_per_day
        remaining = max(0.1, (pos["dte"] - days_elapsed)) / 365.0
        val = bs_price(S, pos["strike"], remaining, pos["iv"], pos["type"])
        pnl = (val - pos["premium"]) * 100 * pos["qty"]
        balance += pnl
        trades.append({
            "symbol": sym, "type": pos["type"],
            "entry_bar": pos["entry_bar"], "exit_bar": max_bars - 1,
            "bars_held": bars_held, "days_held": bars_held / bars_per_day,
            "entry_price": pos["entry_price"],
            "strike": pos["strike"], "dte": pos["dte"],
            "premium": pos["premium"], "exit_value": val, "qty": pos["qty"],
            "pnl": pnl,
            "pnl_pct": (val - pos["premium"]) / pos["premium"],
            "exit_reason": "END_OF_TEST", "score": pos["score"],
        })

    return {
        "mode": mode,
        "balance": balance,
        "pnl": balance - INITIAL_BALANCE,
        "trades": trades,
        "daily_balances": daily_balances,
        "signals_generated": signals_generated,
        "signals_confirmed": signals_confirmed,
        "signals_rejected": signals_rejected,
    }


# ═══════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════

MODE_LABELS = {
    '5min_only':   '5-min Only (baseline)',
    '10min_only':  '10-min Only',
    '5sig_10conf': '5-min Signal + 10-min Confirm',
    '10sig_5conf': '10-min Signal + 5-min Confirm',
}

def print_mode_report(result):
    mode = result["mode"]
    trades = result["trades"]
    daily_balances = result["daily_balances"]
    balance = result["balance"]
    label = MODE_LABELS[mode]

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")

    if not trades:
        print("  No trades executed!")
        return

    pnl = balance - INITIAL_BALANCE
    w = [t for t in trades if t["pnl"] > 0]
    l = [t for t in trades if t["pnl"] <= 0]
    wr = len(w) / len(trades) * 100

    gp = sum(t["pnl"] for t in w) if w else 0
    gl = abs(sum(t["pnl"] for t in l)) if l else 0
    pf = gp / gl if gl > 0 else float("inf")

    avg_w = gp / len(w) if w else 0
    avg_l = gl / len(l) if l else 0

    # Max drawdown
    peak_b = INITIAL_BALANCE
    max_dd = 0
    for b in daily_balances:
        peak_b = max(peak_b, b)
        dd = (b - peak_b) / peak_b
        max_dd = min(max_dd, dd)

    avg_hold = np.mean([t["days_held"] for t in trades])
    avg_bars = np.mean([t["bars_held"] for t in trades])

    # Sharpe
    if len(daily_balances) > 2:
        dr = np.diff(daily_balances) / np.array(daily_balances[:-1])
        sharpe = np.mean(dr) / (np.std(dr) + 1e-10) * np.sqrt(78 * 252)
    else:
        sharpe = 0

    sig_gen = result["signals_generated"]
    sig_conf = result["signals_confirmed"]
    sig_rej = result["signals_rejected"]

    print(f"\n  Start:          ${INITIAL_BALANCE:>10,.2f}")
    print(f"  End:            ${balance:>10,.2f}")
    print(f"  P&L:            ${pnl:>+10,.2f}  ({pnl/INITIAL_BALANCE:>+.1%})")
    print(f"  Trades:         {len(trades):>10}")
    print(f"  Win Rate:       {wr:>9.1f}%  ({len(w)}W / {len(l)}L)")
    print(f"  Profit Factor:  {pf:>10.2f}")
    print(f"  Avg Win:        ${avg_w:>10.2f}")
    print(f"  Avg Loss:       ${avg_l:>10.2f}")
    print(f"  Max Drawdown:   {max_dd:>9.1%}")
    print(f"  Sharpe:         {sharpe:>10.2f}")
    print(f"  Avg Hold:       {avg_hold:>9.1f}d ({avg_bars:.0f} bars)")

    if sig_gen > 0:
        print(f"\n  Signals:        {sig_gen} generated")
        if sig_conf > 0 or sig_rej > 0:
            print(f"  Confirmed:      {sig_conf}  ({sig_conf/(sig_conf+sig_rej)*100:.0f}%)")
            print(f"  Rejected:       {sig_rej}  ({sig_rej/(sig_conf+sig_rej)*100:.0f}%)")

    # By symbol
    print(f"\n  {'-' * 55}")
    print(f"  {'Sym':<6} {'#':>4} {'P&L':>10} {'WR':>6} {'PF':>6} {'Hold':>6}")
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)
    for sym in SYMBOLS:
        if sym not in by_sym:
            continue
        st = by_sym[sym]
        s_pnl = sum(t["pnl"] for t in st)
        s_w = len([t for t in st if t["pnl"] > 0])
        s_wr = s_w / len(st) * 100
        s_gp = sum(t["pnl"] for t in st if t["pnl"] > 0)
        s_gl = abs(sum(t["pnl"] for t in st if t["pnl"] <= 0))
        s_pf = s_gp / s_gl if s_gl > 0 else float("inf")
        s_ah = np.mean([t["days_held"] for t in st])
        print(f"  {sym:<6} {len(st):>4} ${s_pnl:>+9.2f} {s_wr:>5.0f}% {s_pf:>5.2f} {s_ah:>5.1f}d")

    # Calls vs Puts
    print(f"\n  {'-' * 55}")
    for typ in ["call", "put"]:
        tt = [t for t in trades if t["type"] == typ]
        if not tt:
            continue
        t_pnl = sum(t["pnl"] for t in tt)
        t_wr = len([t for t in tt if t["pnl"] > 0]) / len(tt) * 100
        print(f"  {typ.upper()+'S':<7} {len(tt):>4} trades  ${t_pnl:>+9.2f}  {t_wr:.0f}% WR")

    # Exit reasons
    print(f"\n  {'-' * 55}")
    reasons = defaultdict(lambda: {"n": 0, "pnl": 0})
    for t in trades:
        reasons[t["exit_reason"]]["n"] += 1
        reasons[t["exit_reason"]]["pnl"] += t["pnl"]
    for r, d in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
        avg = d["pnl"] / d["n"]
        print(f"  {r:<18} {d['n']:>3}  ${d['pnl']:>+9.2f}  (avg ${avg:>+.2f})")


def print_comparison(results):
    """Print a comparison table of all modes."""
    print(f"\n\n{'=' * 75}")
    print(f"  MULTI-TIMEFRAME COMPARISON  --  MSFT+NVDA, 1DTE, $25K")
    print(f"{'=' * 75}")
    print(f"  {'Mode':<35} {'Final':>9} {'P&L':>9} {'P&L%':>7} "
          f"{'#':>4} {'WR':>5} {'PF':>6} {'DD':>7}")
    print(f"  {'-' * 70}")

    for r in results:
        trades = r["trades"]
        pnl = r["pnl"]
        pct = pnl / INITIAL_BALANCE * 100
        n = len(trades)
        w = len([t for t in trades if t["pnl"] > 0])
        wr = w / max(1, n) * 100
        gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        pf = gp / gl if gl > 0 else float("inf")
        # Max drawdown
        peak_b = INITIAL_BALANCE
        max_dd = 0
        for b in r["daily_balances"]:
            peak_b = max(peak_b, b)
            dd = (b - peak_b) / peak_b
            max_dd = min(max_dd, dd)

        label = MODE_LABELS[r["mode"]]
        marker = " <-- BEST" if r == max(results, key=lambda x: x["pnl"]) else ""
        print(f"  {label:<35} ${r['balance']:>8,.0f} ${pnl:>+8,.0f} "
              f"{pct:>+6.1f}% {n:>4} {wr:>4.0f}% {pf:>5.2f} {max_dd:>6.1%}{marker}")

    best = max(results, key=lambda x: x["pnl"])
    worst = min(results, key=lambda x: x["pnl"])
    print(f"\n  WINNER: {MODE_LABELS[best['mode']]}")
    print(f"          ${best['pnl']:+,.0f} ({best['pnl']/INITIAL_BALANCE:+.1%})")

    # Compare MTF modes vs baseline
    baseline = next((r for r in results if r["mode"] == "5min_only"), None)
    if baseline:
        for r in results:
            if r["mode"] == "5min_only":
                continue
            diff = r["pnl"] - baseline["pnl"]
            print(f"\n  {MODE_LABELS[r['mode']]} vs baseline: ${diff:+,.0f}")

    print(f"\n{'=' * 75}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  AlpacaBot Multi-Timeframe Backtest")
    print("  MSFT + NVDA | 1DTE | $25K | Calls + Puts")
    print("=" * 65)

    # Load 5-min data
    all_data_5 = {}
    all_data_10 = {}

    for sym in SYMBOLS:
        path = f"data/historical/{sym}_5min.csv"
        if not os.path.exists(path):
            print(f"  SKIP {sym}: no 5-min data at {path}")
            continue
        df = pd.read_csv(path)
        closes_5 = df["close"].values
        all_data_5[sym] = closes_5
        closes_10 = resample_to_10min(df)
        all_data_10[sym] = closes_10
        days = len(set(str(t)[:10] for t in df["timestamp"]))
        print(f"  {sym}: {len(closes_5):,} 5-min bars -> {len(closes_10):,} 10-min bars ({days}d)")

    if len(all_data_5) < len(SYMBOLS):
        print("\n  Missing data for some symbols!")
        return

    # Run all 4 modes
    modes = ['5min_only', '10min_only', '5sig_10conf', '10sig_5conf']
    results = []

    for mode in modes:
        print(f"\n{'#' * 65}")
        print(f"  Running: {MODE_LABELS[mode]}")
        print(f"{'#' * 65}")
        r = run_single_backtest(mode, all_data_5, all_data_10)
        print_mode_report(r)
        results.append(r)

    # Final comparison
    print_comparison(results)


if __name__ == "__main__":
    main()
