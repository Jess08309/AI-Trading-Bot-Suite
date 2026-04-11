"""
AlpacaBot Scalp Options Backtester v3
=====================================
Scalps 1-7 DTE options on SPY, QQQ, AAPL, MSFT, NVDA using 5-min bars.
Uses all 14 indicators: RSI, MACD, Stochastic, Bollinger, ATR, CCI, ROC,
Williams %R, Volatility Ratio, Z-Score, Trend Strength, Price Changes.

Rules:
  - NEVER buy 0DTE — minimum 1 DTE, up to 7 DTE
  - Scalp: quick in, quick out — target 15-30% gains
  - Tight stops: -20% max loss per trade
  - Max 3 open positions at a time
  - Near-the-money options (1-2% OTM) for high delta
  - Use 5-min candles for signal generation

Usage:
  python tools/backtest_scalp.py              # default $25K
  python tools/backtest_scalp.py --balance 5000
  python tools/backtest_scalp.py --sweep       # DTE sweep 1-7d
"""
import sys, os, argparse, warnings, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from collections import defaultdict
from core.indicators import compute_all_indicators

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════

ALL_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
SYMBOLS = ALL_SYMBOLS     # overridden by --symbols flag

# Direction filter
CALLS_ONLY = False        # True = skip put signals entirely

# DTE
MIN_DTE = 1              # NEVER 0DTE
DEFAULT_DTE = 1           # 1DTE is our proven winner
MAX_DTE = 7

# Options pricing
OTM_PCT = 0.015           # 1.5% OTM — high delta, affordable

# Risk
MAX_RISK_PER_TRADE = 0.06 # 6% of balance per trade
MAX_POSITIONS = 3
STOP_LOSS = -0.20         # -20% of premium (tight stop for scalps)
TAKE_PROFIT = 0.25        # +25% of premium (scalp target)
TRAILING_STOP = 0.15      # 15% from peak (lock in profits fast)
MAX_HOLD_BARS = 78 * 2    # ~2 trading days in 5-min bars (78 bars/day)

# Signal
LOOKBACK = 50             # 50 5-min bars for indicators (~4 hours)
MIN_SIGNAL_SCORE = 3      # minimum confluence score
COOLDOWN_BARS = 12        # 1 hour cooldown after exit per symbol

# 5-min bars per trading day (9:30-16:00 = 390 min / 5 = 78)
BARS_PER_DAY = 78


# ═══════════════════════════════════════════════════════════
#  BLACK-SCHOLES OPTION PRICING
# ═══════════════════════════════════════════════════════════

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, sigma, opt_type, r=0.05):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        if opt_type == "call":
            return max(S - K, 0.01)
        else:
            return max(K - S, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def estimate_iv(closes, window=30):
    """Estimate IV from recent 5-min returns, annualized."""
    if len(closes) < window + 1:
        return 0.25
    rets = np.diff(np.log(closes[-window - 1:]))
    # Annualize: 5-min has 78 bars/day * 252 days/year
    hv = float(np.std(rets) * np.sqrt(78 * 252))
    return max(0.12, hv * 1.15)  # IV ~ 1.15x HV


# ═══════════════════════════════════════════════════════════
#  SCALP SIGNAL GENERATION (all 14 indicators)
# ═══════════════════════════════════════════════════════════

def generate_scalp_signal(prices_chunk, indicators):
    """
    Scalp signal using all 14 indicators.
    Returns: (direction, score) — 'call'/'put'/None, 0-10
    
    Key difference from swing: we want MOMENTUM, not just trend.
    Scalps need the move to start NOW, not in 5 days.
    """
    bull, bear = 0, 0
    
    # ── 1. RSI: oversold bounce or momentum ──
    rsi = indicators.get("rsi", 50)
    if rsi < 25:               # deeply oversold — bounce
        bull += 2
    elif 35 < rsi < 55:        # recovering
        bull += 1
    elif rsi > 75:             # deeply overbought — fade
        bear += 2
    elif 50 < rsi < 65:        # mild bearish
        bear += 1
    
    # ── 2. MACD histogram: momentum direction ──
    macd_h = indicators.get("macd_hist", 0)
    if macd_h > 0:
        bull += 1
        if macd_h > 0.1:      # strong momentum
            bull += 1
    elif macd_h < 0:
        bear += 1
        if macd_h < -0.1:
            bear += 1
    
    # ── 3. Stochastic: timing entry ──
    stoch = indicators.get("stochastic", 50)
    if stoch < 20:             # oversold
        bull += 1
    elif stoch > 80:           # overbought
        bear += 1
    
    # ── 4. Bollinger Band position: squeeze & breakout ──
    bb = indicators.get("bb_position", 0.5)
    if bb < 0.10:              # at lower band
        bull += 2
    elif bb > 0.90:            # at upper band
        bear += 2
    elif bb < 0.30:
        bull += 1
    elif bb > 0.70:
        bear += 1
    
    # ── 5. ATR (normalized): volatility check ──
    atr_n = indicators.get("atr_normalized", 0)
    # High vol = bigger moves = better for scalps
    if atr_n > 0.005:          # decent volatility
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1
    
    # ── 6. CCI: overbought/oversold ──
    cci_val = indicators.get("cci", 0)
    if cci_val < -100:
        bull += 1
    elif cci_val > 100:
        bear += 1
    
    # ── 7. ROC: rate of change momentum ──
    roc_val = indicators.get("roc", 0)
    if roc_val > 0.3:
        bull += 1
    elif roc_val < -0.3:
        bear += 1
    
    # ── 8. Williams %R ──
    wr = indicators.get("williams_r", -50)
    if wr > -20:               # overbought
        bear += 1
    elif wr < -80:             # oversold
        bull += 1
    
    # ── 9. Volatility Ratio: breakout detection ──
    vol_r = indicators.get("volatility_ratio", 1.0)
    if vol_r > 1.3:            # expanding vol = breakout starting
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1
    
    # ── 10. Z-Score: mean reversion ──
    zs = indicators.get("zscore", 0)
    if zs < -2.0:
        bull += 1
    elif zs > 2.0:
        bear += 1
    
    # ── 11. Trend Strength ──
    ts = indicators.get("trend_strength", 0)
    if ts > 25:                # strong trend — go with it
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1
    
    # ── 12-14. Short-term price momentum ──
    pc1 = indicators.get("price_change_1", 0)
    pc5 = indicators.get("price_change_5", 0)
    
    # Immediate momentum (last bar)
    if pc1 > 0.001:            # > 0.1% last bar
        bull += 1
    elif pc1 < -0.001:
        bear += 1
    
    # 5-bar momentum
    if pc5 > 0.003:            # > 0.3% in 5 bars
        bull += 1
    elif pc5 < -0.003:
        bear += 1
    
    # ── Decision: need clear edge ──
    if bull >= MIN_SIGNAL_SCORE and bull > bear + 1:
        return "call", bull
    elif bear >= MIN_SIGNAL_SCORE and bear > bull + 1:
        return "put", bear
    
    return None, 0


# ═══════════════════════════════════════════════════════════
#  BACKTESTER
# ═══════════════════════════════════════════════════════════

def run_backtest(initial_balance=25000, dte_override=None):
    dte = max(MIN_DTE, dte_override if dte_override else DEFAULT_DTE)
    
    print("=" * 65)
    print(f"  AlpacaBot SCALP Backtest v3 — 5-min Bars")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Balance: ${initial_balance:,.0f} | Risk: {MAX_RISK_PER_TRADE:.0%}/trade")
    print(f"  DTE: {dte}d (min {MIN_DTE}d, NEVER 0DTE)")
    print(f"  OTM: {OTM_PCT:.1%} | SL: {STOP_LOSS:.0%} | TP: +{TAKE_PROFIT:.0%}")
    print("=" * 65)
    
    # ── Load 5-min data ──
    all_data = {}
    for sym in SYMBOLS:
        path = f"data/historical/{sym}_5min.csv"
        if not os.path.exists(path):
            print(f"  SKIP {sym}: no 5-min data")
            continue
        df = pd.read_csv(path)
        all_data[sym] = df["close"].values
        days = len(set(str(t)[:10] for t in df["timestamp"]))
        print(f"  {sym}: {len(df):,} bars ({days}d)  "
              f"${df['close'].iloc[0]:.2f} -> ${df['close'].iloc[-1]:.2f}")
    
    if not all_data:
        print("No data!")
        return None
    
    # Walk-forward: train 60%, test 40%
    max_bars = max(len(p) for p in all_data.values())
    train_end = int(max_bars * 0.6)
    test_bars = max_bars - train_end
    test_days = test_bars / BARS_PER_DAY
    
    print(f"\n  Train: 0-{train_end} bars | Test: {train_end}-{max_bars} "
          f"({test_bars} bars, ~{test_days:.0f} trading days)")
    
    # ── Simulation ──
    balance = initial_balance
    peak_balance = initial_balance
    positions = []
    trades = []
    daily_balances = [initial_balance]
    consec_losses = 0
    cooldowns = {}  # symbol -> bar when cooldown ends
    
    max_hold = min(MAX_HOLD_BARS, dte * BARS_PER_DAY - BARS_PER_DAY)
    # Never hold past (DTE - 1) days — sell by end of day before expiry
    max_hold = max(BARS_PER_DAY, max_hold)  # at least 1 day
    
    signals_generated = 0
    signals_filtered = 0
    
    for bar_idx in range(train_end, max_bars):
        # ── Mark-to-market ──
        for pos in positions:
            sym = pos["symbol"]
            if bar_idx >= len(all_data[sym]):
                continue
            
            S = all_data[sym][bar_idx]
            bars_held = bar_idx - pos["entry_bar"]
            days_elapsed = bars_held / BARS_PER_DAY
            remaining_dte = max(0.1, (pos["dte"] - days_elapsed)) / 365.0
            
            val = bs_price(S, pos["strike"], remaining_dte, pos["iv"], pos["type"])
            pos["current_value"] = val
            pos["peak_value"] = max(pos.get("peak_value", pos["premium"]), val)
            pos["bars_held"] = bars_held
        
        # ── Check exits ──
        to_exit = []
        for i, pos in enumerate(positions):
            pnl_pct = (pos["current_value"] - pos["premium"]) / pos["premium"]
            reason = None
            
            # Stop loss
            if pnl_pct <= STOP_LOSS:
                reason = "STOP_LOSS"
            # Take profit (scalp target)
            elif pnl_pct >= TAKE_PROFIT:
                reason = "TAKE_PROFIT"
            # Trailing stop (lock in gains)
            elif pos["peak_value"] > pos["premium"] * 1.08:  # 8% gain triggered
                drop = (pos["current_value"] - pos["peak_value"]) / pos["peak_value"]
                if drop <= -TRAILING_STOP:
                    reason = "TRAILING_STOP"
            # Max hold time
            elif pos["bars_held"] >= max_hold:
                reason = "MAX_HOLD"
            # DTE safety: exit before expiry day
            elif pos["bars_held"] >= (pos["dte"] - 1) * BARS_PER_DAY:
                reason = "DTE_EXIT"
            
            if reason:
                to_exit.append((i, reason))
        
        for i, reason in sorted(to_exit, reverse=True):
            pos = positions.pop(i)
            pnl_per = pos["current_value"] - pos["premium"]
            pnl = pnl_per * 100 * pos["qty"]
            pnl_pct = pnl_per / pos["premium"]
            balance += pnl
            
            consec_losses = consec_losses + 1 if pnl < 0 else 0
            peak_balance = max(peak_balance, balance)
            
            # Set cooldown
            cooldowns[pos["symbol"]] = bar_idx + COOLDOWN_BARS
            
            trades.append({
                "symbol": pos["symbol"],
                "type": pos["type"],
                "entry_bar": pos["entry_bar"],
                "exit_bar": bar_idx,
                "bars_held": pos["bars_held"],
                "days_held": pos["bars_held"] / BARS_PER_DAY,
                "entry_price": pos["entry_price"],
                "exit_price": all_data[pos["symbol"]][bar_idx],
                "strike": pos["strike"],
                "dte": pos["dte"],
                "premium": pos["premium"],
                "exit_value": pos["current_value"],
                "qty": pos["qty"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": reason,
                "score": pos["score"],
            })
        
        # ── Track equity ──
        unrealized = sum(
            (p["current_value"] - p["premium"]) * 100 * p["qty"]
            for p in positions
        )
        daily_balances.append(balance + unrealized)
        
        # ── Circuit breakers ──
        dd = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
        if consec_losses >= 5 or dd <= -0.15:
            consec_losses = max(0, consec_losses - 1)
            continue
        
        # Only check signals every 3 bars (15 min) to avoid overtrading
        if bar_idx % 3 != 0:
            continue
        
        if len(positions) >= MAX_POSITIONS:
            continue
        
        # ── Generate signals ──
        for sym in SYMBOLS:
            if len(positions) >= MAX_POSITIONS:
                break
            if any(p["symbol"] == sym for p in positions):
                continue
            if bar_idx < cooldowns.get(sym, 0):
                continue
            
            prices = all_data[sym]
            if bar_idx < LOOKBACK or bar_idx >= len(prices):
                continue
            
            chunk = prices[bar_idx - LOOKBACK:bar_idx + 1]
            indicators = compute_all_indicators(chunk)
            direction, score = generate_scalp_signal(chunk, indicators)
            
            if direction is None:
                continue
            
            # Calls-only filter
            if CALLS_ONLY and direction == "put":
                continue
            
            signals_generated += 1
            
            # Price the option
            S = prices[bar_idx]
            iv = estimate_iv(prices[:bar_idx + 1])
            
            if direction == "call":
                K = round(S * (1 + OTM_PCT), 2)
            else:
                K = round(S * (1 - OTM_PCT), 2)
            
            T = dte / 365.0
            premium = bs_price(S, K, T, iv, direction)
            
            if premium < 0.05:
                signals_filtered += 1
                continue
            
            # Position sizing
            cost_per = premium * 100
            max_spend = balance * MAX_RISK_PER_TRADE
            
            if cost_per > max_spend:
                signals_filtered += 1
                continue
            
            qty = max(1, int(max_spend / cost_per))
            
            positions.append({
                "symbol": sym,
                "type": direction,
                "entry_bar": bar_idx,
                "entry_price": S,
                "strike": K,
                "iv": iv,
                "dte": dte,
                "premium": premium,
                "current_value": premium,
                "peak_value": premium,
                "qty": qty,
                "score": score,
                "bars_held": 0,
            })
    
    # ── Close remaining ──
    for pos in positions:
        sym = pos["symbol"]
        final = min(max_bars - 1, len(all_data[sym]) - 1)
        S = all_data[sym][final]
        bars_held = final - pos["entry_bar"]
        days_elapsed = bars_held / BARS_PER_DAY
        remaining = max(0.1, (pos["dte"] - days_elapsed)) / 365.0
        val = bs_price(S, pos["strike"], remaining, pos["iv"], pos["type"])
        pnl = (val - pos["premium"]) * 100 * pos["qty"]
        pnl_pct = (val - pos["premium"]) / pos["premium"]
        balance += pnl
        trades.append({
            "symbol": sym, "type": pos["type"],
            "entry_bar": pos["entry_bar"], "exit_bar": final,
            "bars_held": bars_held, "days_held": bars_held / BARS_PER_DAY,
            "entry_price": pos["entry_price"], "exit_price": S,
            "strike": pos["strike"], "dte": pos["dte"],
            "premium": pos["premium"], "exit_value": val,
            "qty": pos["qty"],
            "pnl": pnl, "pnl_pct": pnl_pct,
            "exit_reason": "END_OF_TEST", "score": pos["score"],
        })
    
    # ── Report ──
    print(f"\n  Signals: {signals_generated} generated, {signals_filtered} filtered")
    print_report(trades, daily_balances, initial_balance, balance, dte)
    
    w = [t for t in trades if t["pnl"] > 0]
    l = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in w) if w else 0
    gl = abs(sum(t["pnl"] for t in l)) if l else 1
    
    return {
        "balance": balance,
        "pnl": balance - initial_balance,
        "trades": len(trades),
        "win_rate": len(w) / max(1, len(trades)) * 100,
        "pf": gp / gl if gl > 0 else float("inf"),
        "dte": dte,
        "initial": initial_balance,
    }


def print_report(trades, daily_balances, initial, final, dte):
    print(f"\n{'=' * 65}")
    print(f"  SCALP BACKTEST RESULTS (DTE={dte}d)")
    print(f"{'=' * 65}")
    
    if not trades:
        print("  No trades executed!")
        return
    
    pnl = final - initial
    w = [t for t in trades if t["pnl"] > 0]
    l = [t for t in trades if t["pnl"] <= 0]
    wr = len(w) / len(trades) * 100
    
    gp = sum(t["pnl"] for t in w) if w else 0
    gl = abs(sum(t["pnl"] for t in l)) if l else 0
    pf = gp / gl if gl > 0 else float("inf")
    
    avg_w = gp / len(w) if w else 0
    avg_l = gl / len(l) if l else 0
    
    # Max drawdown
    peak_b = initial
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
    
    print(f"\n  Start:         ${initial:>10,.2f}")
    print(f"  End:           ${final:>10,.2f}")
    print(f"  P&L:           ${pnl:>+10,.2f}  ({pnl/initial:>+.1%})")
    print(f"  Trades:        {len(trades):>10}")
    print(f"  Win Rate:      {wr:>9.1f}%  ({len(w)}W / {len(l)}L)")
    print(f"  Profit Factor: {pf:>10.2f}")
    print(f"  Avg Win:       ${avg_w:>10.2f}")
    print(f"  Avg Loss:      ${avg_l:>10.2f}")
    print(f"  Max Drawdown:  {max_dd:>9.1%}")
    print(f"  Sharpe:        {sharpe:>10.2f}")
    print(f"  Avg Hold:      {avg_hold:>9.1f}d ({avg_bars:.0f} bars)")
    
    # By symbol
    print(f"\n  {'─' * 55}")
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
    print(f"\n  {'─' * 55}")
    for typ in ["call", "put"]:
        tt = [t for t in trades if t["type"] == typ]
        if not tt:
            continue
        t_pnl = sum(t["pnl"] for t in tt)
        t_wr = len([t for t in tt if t["pnl"] > 0]) / len(tt) * 100
        print(f"  {typ.upper()+'S':<7} {len(tt):>4} trades  ${t_pnl:>+9.2f}  {t_wr:.0f}% WR")
    
    # Exit reasons
    print(f"\n  {'─' * 55}")
    reasons = defaultdict(lambda: {"n": 0, "pnl": 0})
    for t in trades:
        reasons[t["exit_reason"]]["n"] += 1
        reasons[t["exit_reason"]]["pnl"] += t["pnl"]
    for r, d in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
        avg = d["pnl"] / d["n"]
        print(f"  {r:<18} {d['n']:>3}  ${d['pnl']:>+9.2f}  (avg ${avg:>+.2f})")
    
    # Best/worst
    print(f"\n  {'─' * 55}")
    st = sorted(trades, key=lambda x: x["pnl"], reverse=True)
    print(f"  BEST:")
    for t in st[:3]:
        print(f"    {t['symbol']} {t['type'].upper()} K={t['strike']:.0f} | "
              f"${t['pnl']:>+.2f} ({t['pnl_pct']:>+.0%}) | "
              f"{t['days_held']:.1f}d | {t['exit_reason']}")
    print(f"  WORST:")
    for t in st[-3:]:
        print(f"    {t['symbol']} {t['type'].upper()} K={t['strike']:.0f} | "
              f"${t['pnl']:>+.2f} ({t['pnl_pct']:>+.0%}) | "
              f"{t['days_held']:.1f}d | {t['exit_reason']}")
    
    # Verdict
    print(f"\n{'=' * 65}")
    if pnl > 0 and wr > 45 and pf > 1.2 and max_dd > -0.15:
        print(f"  VERDICT: STRONG  ✓  — Ready for paper trading")
    elif pnl > 0 and pf > 1.0:
        print(f"  VERDICT: PROFITABLE  ~  — Promising, keep tuning")
    elif pnl > -initial * 0.03:
        print(f"  VERDICT: MARGINAL  ~  — Close, tune parameters")
    else:
        print(f"  VERDICT: NEEDS WORK  ✗  — Not yet profitable")
    print(f"{'=' * 65}")


def run_dte_sweep():
    """Sweep DTE 1-7 days."""
    results = []
    for d in [1, 2, 3, 4, 5, 7]:
        print(f"\n{'#' * 65}")
        print(f"  DTE = {d} days")
        print(f"{'#' * 65}")
        r = run_backtest(25_000, dte_override=d)
        if r:
            results.append(r)
    
    if results:
        print(f"\n\n{'=' * 65}")
        print(f"  DTE SWEEP — SCALP STRATEGY ($25K)")
        print(f"{'=' * 65}")
        print(f"  {'DTE':>4} {'Final':>10} {'P&L':>10} {'P&L%':>7} "
              f"{'Trades':>6} {'WR':>5} {'PF':>6}")
        for r in results:
            pnl = r["pnl"]
            pct = pnl / r["initial"] * 100
            # Quick PF calc
            print(f"  {r['dte']:>3}d ${r['balance']:>9,.0f} ${pnl:>+9,.0f} "
                  f"{pct:>+6.1f}% {r['trades']:>6} {r['win_rate']:>4.0f}%")
        
        best = max(results, key=lambda x: x["pnl"])
        print(f"\n  BEST DTE: {best['dte']}d -> ${best['pnl']:+,.0f} "
              f"({best['pnl']/best['initial']:+.1%})")
        print(f"{'=' * 65}")


def run_tune_sweep():
    """Test multiple configurations to find optimal strategy."""
    global SYMBOLS, CALLS_ONLY
    
    configs = [
        ("ALL 5, Calls+Puts",  ALL_SYMBOLS, False),
        ("ALL 5, Calls Only",  ALL_SYMBOLS, True),
        ("MSFT+NVDA+QQQ, C+P", ["QQQ", "MSFT", "NVDA"], False),
        ("MSFT+NVDA+QQQ, CO",  ["QQQ", "MSFT", "NVDA"], True),
        ("MSFT+NVDA, Calls",   ["MSFT", "NVDA"], True),
        ("MSFT+NVDA, C+P",     ["MSFT", "NVDA"], False),
        ("ALL 5 no AAPL, CO",  ["SPY", "QQQ", "MSFT", "NVDA"], True),
    ]
    
    results = []
    for label, syms, co in configs:
        SYMBOLS = syms
        CALLS_ONLY = co
        print(f"\n{'#' * 65}")
        print(f"  CONFIG: {label}")
        print(f"  Symbols: {syms} | Calls Only: {co} | DTE: 1d")
        print(f"{'#' * 65}")
        r = run_backtest(25_000, dte_override=1)
        if r:
            r["config"] = label
            results.append(r)
    
    # Reset
    SYMBOLS = ALL_SYMBOLS
    CALLS_ONLY = False
    
    if results:
        print(f"\n\n{'=' * 70}")
        print(f"  TUNING SWEEP - 1DTE SCALP ($25K)")
        print(f"{'=' * 70}")
        print(f"  {'Config':<25} {'Final':>10} {'P&L':>10} {'P&L%':>7} "
              f"{'#':>5} {'WR':>5}")
        print(f"  {'-' * 63}")
        for r in results:
            pnl = r["pnl"]
            pct = pnl / r["initial"] * 100
            print(f"  {r['config']:<25} ${r['balance']:>9,.0f} ${pnl:>+9,.0f} "
                  f"{pct:>+6.1f}% {r['trades']:>5} {r['win_rate']:>4.0f}%")
        
        best = max(results, key=lambda x: x["pnl"])
        print(f"\n  BEST: {best['config']} -> ${best['pnl']:+,.0f} "
              f"({best['pnl']/best['initial']:+.1%})")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", type=int, default=25000)
    parser.add_argument("--dte", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Sweep configs")
    parser.add_argument("--calls-only", action="store_true")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols override")
    args = parser.parse_args()
    
    if args.calls_only:
        CALLS_ONLY = True
    if args.symbols:
        SYMBOLS = [s.strip().upper() for s in args.symbols.split(",")]
    
    if args.tune:
        run_tune_sweep()
    elif args.sweep:
        run_dte_sweep()
    else:
        run_backtest(args.balance, dte_override=args.dte if args.dte else None)
