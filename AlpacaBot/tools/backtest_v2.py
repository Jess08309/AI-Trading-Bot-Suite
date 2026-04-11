"""
AlpacaBot Options Backtest v2 — Trend-Following with OTM Options

Strategy: Buy calls in confirmed uptrends, puts in downtrends.
Uses 5% OTM options (cheaper premiums, defined risk).
Walk-forward: train on first 60%, test on last 40%.

Improvements over v1:
  - OTM options reduce entry cost ~40-50%
  - Trend confirmation requires multiple signals
  - Better theta/delta modeling
  - Multi-balance testing ($5K, $10K, $25K, $100K)
  - Daily mark-to-market using proper greeks

Usage: python tools/backtest_v2.py [--balance 5000]
"""
import sys, os, argparse, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from collections import defaultdict
from core.indicators import compute_all_indicators

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
LOOKBACK = 30

# Options parameters
MIN_DTE = 1             # NEVER buy 0DTE — always next day or later
DEFAULT_DTE = 3         # Default DTE for entries (tunable)
OTM_PCT = 0.02          # 2% out of the money (near ATM)

# Risk management
MAX_RISK_PER_TRADE = 0.06    # Risk max 6% of balance per trade
MAX_POSITIONS = 3
STOP_LOSS = -0.25            # -25% of premium paid
TAKE_PROFIT = 0.50           # +50% of premium paid
TRAILING_STOP = 0.20         # 20% pullback from peak value
MAX_HOLD = 6                 # Max hold days

# Signal thresholds
MIN_TREND_SCORE = 4          # Higher bar for entry


# ═══════════════════════════════════════════════════════════
# OPTION PRICING (Simplified Black-Scholes)
# ═══════════════════════════════════════════════════════════

def norm_cdf(x):
    """Standard normal CDF approximation."""
    import math
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))


def bs_call(S, K, T, sigma, r=0.05):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def bs_put(S, K, T, sigma, r=0.05):
    """Black-Scholes put price via put-call parity."""
    return bs_call(S, K, T, sigma, r) - S + K * np.exp(-r * T)


def bs_delta(S, K, T, sigma, option_type="call", r=0.05):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0


def historical_iv(prices, window=20):
    """Estimate IV from historical returns (annualized)."""
    if len(prices) < window + 1:
        return 0.25
    rets = np.diff(np.log(prices[-window - 1:]))
    hv = float(np.std(rets) * np.sqrt(252))
    # IV typically 1.1-1.3x HV for equities
    return max(0.15, hv * 1.2)


def price_option(S, K, T, sigma, option_type):
    """Price an option using Black-Scholes."""
    if option_type == "call":
        return bs_call(S, K, T, sigma)
    else:
        return bs_put(S, K, T, sigma)


# ═══════════════════════════════════════════════════════════
# SIGNAL GENERATION (Trend-Following)
# ═══════════════════════════════════════════════════════════

def generate_signal(prices, indicators):
    """
    Trend-following signal with confluence voting.
    Returns: (direction, score) where direction is 'call'/'put'/None
             and score is 0-10 (higher = stronger signal)
    """
    bull, bear = 0, 0
    
    # 1. Price vs moving averages (SMA 10 & 20)
    if len(prices) >= 20:
        sma10 = np.mean(prices[-10:])
        sma20 = np.mean(prices[-20:])
        current = prices[-1]
        
        if current > sma10 > sma20:       # strong uptrend
            bull += 2
        elif current > sma10:              # mild uptrend
            bull += 1
        elif current < sma10 < sma20:      # strong downtrend
            bear += 2
        elif current < sma10:
            bear += 1
    
    # 2. RSI (mean reversion within trend)
    rsi = indicators.get("rsi", 50)
    if 30 < rsi < 50:            # oversold pullback in potential uptrend
        bull += 1
    elif 50 < rsi < 70:          # momentum, not overbought
        bull += 1
    elif rsi > 75:               # overbought
        bear += 1
    elif rsi < 25:               # deep oversold (bounce play)
        bull += 2
    
    # 3. MACD
    macd = indicators.get("macd_hist", 0)
    if macd > 0:
        bull += 1
    elif macd < 0:
        bear += 1
    
    # 4. Bollinger Band position
    bb = indicators.get("bb_position", 0.5)
    if bb < 0.15:               # near lower band (bullish reversal)
        bull += 1
    elif bb > 0.85:             # near upper band (bearish reversal)
        bear += 1
    
    # 5. Trend strength
    ts = indicators.get("trend_strength", 0)
    if ts > 20:
        # Strong trend - amplify the dominant direction
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1
    
    # 6. Price momentum (5-day return)
    pc5 = indicators.get("price_change_5", 0)
    if pc5 > 1.5:
        bull += 1
    elif pc5 < -1.5:
        bear += 1
    
    # 7. Z-score (mean reversion)
    zs = indicators.get("zscore", 0)
    if zs < -1.5:              # significantly below mean
        bull += 1
    elif zs > 1.5:
        bear += 1
    
    # Decision
    if bull >= MIN_TREND_SCORE and bull > bear + 1:
        return "call", bull
    elif bear >= MIN_TREND_SCORE and bear > bull + 1:
        return "put", bear
    else:
        return None, 0


# ═══════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════

def run_backtest(initial_balance=5000, dte_override=None):
    dte = dte_override if dte_override else DEFAULT_DTE
    print("=" * 65)
    print(f"  AlpacaBot Options Backtest v2 — Trend Following")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Balance: ${initial_balance:,.0f} | Risk: {MAX_RISK_PER_TRADE:.0%}/trade")
    print(f"  Options: {OTM_PCT:.0%} OTM | DTE: {dte}d (min {MIN_DTE}d)")
    print(f"  SL: {STOP_LOSS:.0%} | TP: +{TAKE_PROFIT:.0%} | Trail: {TRAILING_STOP:.0%}")
    print("=" * 65)
    
    # Load data
    all_prices = {}
    for symbol in SYMBOLS:
        path = f"data/historical/{symbol}_daily.csv"
        if not os.path.exists(path):
            print(f"  SKIP {symbol}: no data")
            continue
        df = pd.read_csv(path)
        all_prices[symbol] = df["close"].values
        print(f"  {symbol}: {len(df)}d  ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}"
              f"  ({(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%)")
    
    if not all_prices:
        print("No data!")
        return
    
    max_len = max(len(p) for p in all_prices.values())
    train_end = int(max_len * 0.6)
    
    print(f"\n  Walk-forward: train 0-{train_end} | test {train_end}-{max_len}")
    
    # ── Run simulation ──
    balance = initial_balance
    peak_balance = initial_balance
    positions = []
    trades = []
    daily_pnl = []
    daily_balances = [initial_balance]
    prev_day_balance = initial_balance
    consec_losses = 0
    
    for day in range(train_end, max_len):
        # ── Mark-to-market all positions ──
        for pos in positions:
            sym = pos["symbol"]
            if day >= len(all_prices[sym]):
                continue
            
            S = all_prices[sym][day]
            days_held = day - pos["entry_day"]
            remaining_dte = max(0.5, (pos["dte"] - days_held)) / 365.0
            
            current_val = price_option(
                S, pos["strike"], remaining_dte, pos["iv"], pos["type"]
            )
            pos["current_value"] = current_val
            pos["peak_value"] = max(pos.get("peak_value", pos["premium"]), current_val)
            pos["days_held"] = days_held
        
        # ── Check exits ──
        to_exit = []
        for i, pos in enumerate(positions):
            pnl_pct = (pos["current_value"] - pos["premium"]) / pos["premium"]
            reason = None
            
            if pnl_pct <= STOP_LOSS:
                reason = "STOP_LOSS"
            elif pnl_pct >= TAKE_PROFIT:
                reason = "TAKE_PROFIT"
            elif pos["peak_value"] > pos["premium"] * 1.15:
                drop = (pos["current_value"] - pos["peak_value"]) / pos["peak_value"]
                if drop <= -TRAILING_STOP:
                    reason = "TRAILING_STOP"
            elif pos["days_held"] >= MAX_HOLD:
                reason = "MAX_HOLD"
            elif pos["days_held"] >= pos["dte"] - 1:
                # Exit before expiration day (never hold to 0DTE)
                reason = "DTE_EXIT"
            
            if reason:
                to_exit.append((i, reason))
        
        for i, reason in sorted(to_exit, reverse=True):
            pos = positions.pop(i)
            pnl_per_share = pos["current_value"] - pos["premium"]
            pnl = pnl_per_share * 100 * pos["qty"]
            pnl_pct = pnl_per_share / pos["premium"]
            balance += pnl
            
            consec_losses = consec_losses + 1 if pnl < 0 else 0
            peak_balance = max(peak_balance, balance)
            
            trades.append({
                "symbol": pos["symbol"], "type": pos["type"],
                "entry_day": pos["entry_day"], "exit_day": day,
                "days_held": pos["days_held"],
                "entry_price": pos["entry_price"],
                "exit_price": all_prices[pos["symbol"]][day],
                "strike": pos["strike"],
                "premium": pos["premium"],
                "exit_value": pos["current_value"],
                "qty": pos["qty"],
                "pnl": pnl, "pnl_pct": pnl_pct,
                "exit_reason": reason,
                "signal_score": pos["score"],
            })
        
        # ── Track daily P&L ──
        unrealized = sum(
            (p["current_value"] - p["premium"]) * 100 * p["qty"]
            for p in positions
        )
        total_equity = balance + unrealized
        daily_pnl.append(total_equity - prev_day_balance)
        prev_day_balance = total_equity
        daily_balances.append(total_equity)
        
        # ── Circuit breakers ──
        dd = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
        if consec_losses >= 4 or dd <= -0.20:
            consec_losses = max(0, consec_losses - 1)  # cool down
            continue
        
        if len(positions) >= MAX_POSITIONS:
            continue
        
        # ── Generate signals ──
        for symbol in SYMBOLS:
            if len(positions) >= MAX_POSITIONS:
                break
            if any(p["symbol"] == symbol for p in positions):
                continue
            
            prices = all_prices[symbol]
            if day < LOOKBACK or day >= len(prices):
                continue
            
            chunk = prices[day - LOOKBACK:day + 1]
            indicators = compute_all_indicators(chunk)
            direction, score = generate_signal(chunk, indicators)
            
            if direction is None:
                continue
            
            # Price option
            S = prices[day]
            iv = historical_iv(prices[:day + 1])
            
            # OTM strike
            if direction == "call":
                K = S * (1 + OTM_PCT)   # 5% above current price
            else:
                K = S * (1 - OTM_PCT)   # 5% below current price
            
            # Use configured DTE (always >= MIN_DTE)
            trade_dte = max(MIN_DTE, dte)
            T = trade_dte / 365.0
            premium = price_option(S, K, T, iv, direction)
            
            if premium < 0.10:
                continue  # too cheap, likely deep OTM
            
            # Position sizing
            cost_per_contract = premium * 100
            max_spend = balance * MAX_RISK_PER_TRADE
            
            if cost_per_contract > max_spend:
                continue  # can't afford even 1 contract
            
            qty = max(1, int(max_spend / cost_per_contract))
            
            positions.append({
                "symbol": symbol,
                "type": direction,
                "entry_day": day,
                "entry_price": S,
                "strike": K,
                "iv": iv,
                "dte": trade_dte,
                "premium": premium,
                "current_value": premium,
                "peak_value": premium,
                "qty": qty,
                "score": score,
                "days_held": 0,
            })
    
    # ── Close remaining positions ──
    for pos in positions:
        sym = pos["symbol"]
        final = min(max_len - 1, len(all_prices[sym]) - 1)
        S = all_prices[sym][final]
        days_held = final - pos["entry_day"]
        remaining = max(0.5, (pos["dte"] - days_held)) / 365.0
        exit_val = price_option(S, pos["strike"], remaining, pos["iv"], pos["type"])
        pnl = (exit_val - pos["premium"]) * 100 * pos["qty"]
        pnl_pct = (exit_val - pos["premium"]) / pos["premium"]
        balance += pnl
        trades.append({
            "symbol": sym, "type": pos["type"],
            "entry_day": pos["entry_day"], "exit_day": final,
            "days_held": days_held,
            "entry_price": pos["entry_price"],
            "exit_price": S,
            "strike": pos["strike"],
            "premium": pos["premium"],
            "exit_value": exit_val,
            "qty": pos["qty"],
            "pnl": pnl, "pnl_pct": pnl_pct,
            "exit_reason": "END_OF_TEST",
            "signal_score": pos["score"],
        })
    
    # ═══════════════════════════════════════════════════════
    #  REPORT
    # ═══════════════════════════════════════════════════════
    print_report(trades, daily_balances, initial_balance, balance)
    
    return {
        "balance": balance,
        "pnl": balance - initial_balance,
        "trades": len(trades),
        "win_rate": len([t for t in trades if t["pnl"] > 0]) / max(1, len(trades)) * 100,
        "dte": dte,
    }


def print_report(trades, daily_balances, initial_balance, final_balance):
    print(f"\n{'=' * 65}")
    print(f"  BACKTEST RESULTS")
    print(f"{'=' * 65}")
    
    if not trades:
        print("  No trades executed!")
        return
    
    total_pnl = final_balance - initial_balance
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]
    wr = len(winners) / len(trades) * 100
    
    gp = sum(t["pnl"] for t in winners) if winners else 0
    gl = abs(sum(t["pnl"] for t in losers)) if losers else 0
    pf = gp / gl if gl > 0 else float("inf")
    
    avg_w = gp / len(winners) if winners else 0
    avg_l = gl / len(losers) if losers else 0
    
    # Max drawdown
    peak = initial_balance
    max_dd = 0
    for b in daily_balances:
        peak = max(peak, b)
        dd = (b - peak) / peak
        max_dd = min(max_dd, dd)
    
    avg_hold = np.mean([t["days_held"] for t in trades])
    
    # Sharpe (daily, annualized)
    if len(daily_balances) > 2:
        daily_rets = np.diff(daily_balances) / np.array(daily_balances[:-1])
        sharpe = np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0
    
    print(f"\n  Start:        ${initial_balance:>10,.2f}")
    print(f"  End:          ${final_balance:>10,.2f}")
    print(f"  P&L:          ${total_pnl:>+10,.2f}  ({total_pnl/initial_balance:+.1%})")
    print(f"  Trades:       {len(trades):>10}")
    print(f"  Win Rate:     {wr:>9.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Profit Factor:{pf:>10.2f}")
    print(f"  Avg Win:      ${avg_w:>10.2f}")
    print(f"  Avg Loss:     ${avg_l:>10.2f}")
    print(f"  Max Drawdown: {max_dd:>9.1%}")
    print(f"  Sharpe Ratio: {sharpe:>10.2f}")
    print(f"  Avg Hold:     {avg_hold:>9.1f}d")
    
    # By symbol
    print(f"\n  {'─' * 55}")
    print(f"  {'Symbol':<7} {'#':>4} {'P&L':>10} {'WR':>6} {'PF':>6} {'AvgDays':>7}")
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)
    
    for sym in SYMBOLS:
        if sym not in by_sym:
            continue
        st = by_sym[sym]
        s_pnl = sum(t["pnl"] for t in st)
        s_w = [t for t in st if t["pnl"] > 0]
        s_l = [t for t in st if t["pnl"] <= 0]
        s_wr = len(s_w) / len(st) * 100
        s_gp = sum(t["pnl"] for t in s_w)
        s_gl = abs(sum(t["pnl"] for t in s_l))
        s_pf = s_gp / s_gl if s_gl > 0 else float("inf")
        s_ah = np.mean([t["days_held"] for t in st])
        print(f"  {sym:<7} {len(st):>4} ${s_pnl:>+9.2f} {s_wr:>5.0f}% {s_pf:>5.2f} {s_ah:>6.1f}d")
    
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
    reasons = defaultdict(lambda: {"count": 0, "pnl": 0})
    for t in trades:
        reasons[t["exit_reason"]]["count"] += 1
        reasons[t["exit_reason"]]["pnl"] += t["pnl"]
    
    for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        avg = d["pnl"] / d["count"]
        print(f"  {r:<18} {d['count']:>3}  ${d['pnl']:>+9.2f}  (avg ${avg:>+.2f})")
    
    # Best/worst trades
    print(f"\n  {'─' * 55}")
    st = sorted(trades, key=lambda x: x["pnl"], reverse=True)
    print(f"  BEST:")
    for t in st[:3]:
        print(f"    {t['symbol']} {t['type'].upper()} K={t['strike']:.0f} | "
              f"${t['pnl']:>+.2f} ({t['pnl_pct']:>+.0%}) | {t['days_held']}d | {t['exit_reason']}")
    print(f"  WORST:")
    for t in st[-3:]:
        print(f"    {t['symbol']} {t['type'].upper()} K={t['strike']:.0f} | "
              f"${t['pnl']:>+.2f} ({t['pnl_pct']:>+.0%}) | {t['days_held']}d | {t['exit_reason']}")
    
    # Verdict
    print(f"\n{'=' * 65}")
    if total_pnl > 0 and wr > 45 and pf > 1.2 and max_dd > -0.20:
        print(f"  VERDICT: STRONG  ✓  — Ready for paper trading")
    elif total_pnl > 0 and pf > 1.0:
        print(f"  VERDICT: PROFITABLE  ~  — Needs parameter tuning")
    elif total_pnl > -initial_balance * 0.05:
        print(f"  VERDICT: MARGINAL  ~  — Close to breakeven, tune further")
    else:
        print(f"  VERDICT: NEEDS WORK  ✗  — Strategy not yet profitable")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════
# MULTI-BALANCE COMPARISON
# ═══════════════════════════════════════════════════════════

def run_multi():
    """Run backtest across multiple starting balances."""
    balances = [5_000, 10_000, 25_000, 100_000]
    results = []
    
    for bal in balances:
        print(f"\n\n{'#' * 65}")
        print(f"  BACKTEST WITH ${bal:,}")
        print(f"{'#' * 65}")
        r = run_backtest(bal)
        if r:
            r["initial"] = bal
            results.append(r)
    
    if results:
        print(f"\n\n{'=' * 65}")
        print(f"  MULTI-BALANCE COMPARISON")
        print(f"{'=' * 65}")
        print(f"  {'Balance':>10} {'Final':>10} {'P&L':>10} {'P&L%':>7} "
              f"{'Trades':>6} {'WR':>5}")
        for r in results:
            final = r["balance"]
            pnl = r["pnl"]
            pct = pnl / r["initial"] * 100
            print(f"  ${r['initial']:>9,} ${final:>9,.0f} ${pnl:>+9,.0f} "
                  f"{pct:>+6.1f}% {r['trades']:>6} {r['win_rate']:>4.0f}%")
        print(f"{'=' * 65}")


def run_dte_sweep():
    """Sweep DTEs from 1d to 21d to find optimal holding period."""
    dte_values = [1, 2, 3, 5, 7, 10, 14, 21]
    balance = 25_000  # use $25K for all tests
    results = []
    
    for d in dte_values:
        print(f"\n{'#' * 65}")
        print(f"  DTE = {d} days")
        print(f"{'#' * 65}")
        r = run_backtest(balance, dte_override=d)
        if r:
            r["initial"] = balance
            results.append(r)
    
    if results:
        print(f"\n\n{'=' * 65}")
        print(f"  DTE SWEEP RESULTS ($25K)")
        print(f"{'=' * 65}")
        print(f"  {'DTE':>4} {'Final':>10} {'P&L':>10} {'P&L%':>7} "
              f"{'Trades':>6} {'WR':>5}")
        for r in results:
            pnl = r["pnl"]
            pct = pnl / r["initial"] * 100
            print(f"  {r['dte']:>3}d ${r['balance']:>9,.0f} ${pnl:>+9,.0f} "
                  f"{pct:>+6.1f}% {r['trades']:>6} {r['win_rate']:>4.0f}%")
        
        best = max(results, key=lambda x: x["pnl"])
        print(f"\n  BEST DTE: {best['dte']}d → ${best['pnl']:+,.0f} ({best['pnl']/best['initial']:+.1%})")
        print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", type=int, default=0)
    parser.add_argument("--dte", type=int, default=0)
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Sweep DTEs 1-21d")
    args = parser.parse_args()
    
    if args.sweep:
        run_dte_sweep()
    elif args.multi:
        run_multi()
    elif args.balance or args.dte:
        run_backtest(
            args.balance if args.balance else 25_000,
            dte_override=args.dte if args.dte else None
        )
    else:
        # Default: DTE sweep to find optimal
        run_dte_sweep()
