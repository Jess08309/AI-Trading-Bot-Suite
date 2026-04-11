"""
ITM vs OTM Strategy Comparison
================================
Replays the last N trading days of available data using both:
  - OLD settings: 4% OTM (what the bot was doing)
  - NEW settings: 3% ITM (your personal approach, what you traded manually)

Uses SPY and QQQ with 5-min bars + Black-Scholes pricing.

Usage:
  cd c:\\AlpacaBot
  python tools/compare_itm_vs_otm.py
  python tools/compare_itm_vs_otm.py --days 7 --balance 25000
"""
import sys
import os
import math
import argparse
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from core.indicators import compute_all_indicators

# ── Symbols to test ──────────────────────────────────────
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# ── Shared parameters (same for both modes) ──────────────
DTE           = 2          # 2 DTE proven sweet spot from bot history
MIN_DTE       = 1
BARS_PER_DAY  = 78         # 5-min bars per trading day
LOOKBACK      = 50         # bars for indicator calculation
MIN_SIGNAL    = 3          # minimum signal confluence score
MAX_POSITIONS = 3
MAX_RISK_PCT  = 0.06       # 6% of balance per trade
STOP_LOSS     = -0.20      # -20% of premium
TAKE_PROFIT   = 0.25       # +25% of premium
TRAILING_STOP = 0.15       # 15% trailing stop
COOLDOWN_BARS = 12         # 1hr cooldown after exit
IV_MULTIPLIER = 1.15       # IV = HV * 1.15

# ── Mode definitions ─────────────────────────────────────
MODES = {
    "OLD (4% OTM)": {
        "otm_pct": 0.04,    # positive = OTM
        "label": "OLD",
        "desc": "4% Out-of-The-Money (previous bot config)",
    },
    "NEW (3% ITM)": {
        "otm_pct": -0.03,   # negative = ITM
        "label": "NEW",
        "desc": "3% In-The-Money (your personal strategy)",
    },
}


# ── Black-Scholes ────────────────────────────────────────
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, sigma, opt_type, r=0.05):
    """Black-Scholes option price. Returns intrinsic value if near-expiry."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.01) if opt_type == "call" else max(K - S, 0.01)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if opt_type == "call":
            return max(0.01, S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))
        else:
            return max(0.01, K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1))
    except (ValueError, ZeroDivisionError):
        return max(S - K, 0.01) if opt_type == "call" else max(K - S, 0.01)


def bs_delta(S, K, T, sigma, opt_type, r=0.05):
    """Black-Scholes delta (directional exposure indicator)."""
    if T <= 0 or sigma <= 0:
        return 1.0 if opt_type == "call" else -1.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if opt_type == "call":
            return round(norm_cdf(d1), 3)
        else:
            return round(norm_cdf(d1) - 1, 3)
    except (ValueError, ZeroDivisionError):
        return 0.5


def estimate_iv(closes, window=30):
    """Estimate IV from recent returns, annualized."""
    if len(closes) < window + 1:
        return 0.25
    rets = np.diff(np.log(closes[-window - 1:]))
    hv = float(np.std(rets) * np.sqrt(BARS_PER_DAY * 252))
    return max(0.12, hv * IV_MULTIPLIER)


# ── Signal generation ─────────────────────────────────────
def generate_signal(prices_chunk, indicators):
    bull, bear = 0, 0

    rsi = indicators.get("rsi", 50)
    if rsi < 25:    bull += 2
    elif 35 < rsi < 55: bull += 1
    elif rsi > 75:  bear += 2
    elif 50 < rsi < 65: bear += 1

    macd_h = indicators.get("macd_hist", 0)
    if macd_h > 0.1: bull += 2
    elif macd_h > 0: bull += 1
    elif macd_h < -0.1: bear += 2
    elif macd_h < 0: bear += 1

    stoch = indicators.get("stochastic", 50)
    if stoch < 20:  bull += 1
    elif stoch > 80: bear += 1

    bb = indicators.get("bb_position", 0.5)
    if bb < 0.10:   bull += 2
    elif bb > 0.90: bear += 2
    elif bb < 0.30: bull += 1
    elif bb > 0.70: bear += 1

    atr_n = indicators.get("atr_normalized", 0)
    if atr_n > 0.005:
        if bull > bear: bull += 1
        elif bear > bull: bear += 1

    cci_val = indicators.get("cci", 0)
    if cci_val < -100: bull += 1
    elif cci_val > 100: bear += 1

    roc_val = indicators.get("roc", 0)
    if roc_val > 0.3: bull += 1
    elif roc_val < -0.3: bear += 1

    wr = indicators.get("williams_r", -50)
    if wr > -20:    bear += 1
    elif wr < -80:  bull += 1

    zs = indicators.get("zscore", 0)
    if zs < -2.0:   bull += 1
    elif zs > 2.0:  bear += 1

    pc1 = indicators.get("price_change_1", 0)
    if pc1 > 0.001:  bull += 1
    elif pc1 < -0.001: bear += 1

    pc5 = indicators.get("price_change_5", 0)
    if pc5 > 0.003:  bull += 1
    elif pc5 < -0.003: bear += 1

    if bull >= MIN_SIGNAL and bull > bear + 1:
        return "call", bull
    elif bear >= MIN_SIGNAL and bear > bull + 1:
        return "put", bear
    return None, 0


# ── Single-mode simulation ────────────────────────────────
def run_simulation(all_data, all_timestamps, start_bar, initial_balance, otm_pct, mode_label):
    """
    Run a simulation from start_bar to end of data.
    otm_pct > 0 = OTM, otm_pct < 0 = ITM.
    """
    max_bars = max(len(p) for p in all_data.values())
    balance = initial_balance
    peak_balance = initial_balance
    positions = []
    trades = []
    cooldowns = {}
    consec_losses = 0
    max_hold = max(BARS_PER_DAY, DTE * BARS_PER_DAY - BARS_PER_DAY)

    for bar_idx in range(start_bar, max_bars):
        # Mark-to-market
        for pos in positions:
            sym = pos["symbol"]
            if bar_idx >= len(all_data[sym]):
                continue
            S = all_data[sym][bar_idx]
            bars_held = bar_idx - pos["entry_bar"]
            days_elapsed = bars_held / BARS_PER_DAY
            remaining_dte = max(0.05, (pos["dte"] - days_elapsed)) / 365.0
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
            elif pos["peak_value"] > pos["premium"] * 1.08:
                drop = (pos["current_value"] - pos["peak_value"]) / pos["peak_value"]
                if drop <= -TRAILING_STOP:
                    reason = "TRAILING_STOP"
            elif pos["bars_held"] >= max_hold:
                reason = "MAX_HOLD"
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
            cooldowns[pos["symbol"]] = bar_idx + COOLDOWN_BARS
            trades.append({
                "symbol":     pos["symbol"],
                "type":       pos["type"],
                "direction":  "CALL" if pos["type"] == "call" else "PUT",
                "entry_bar":  pos["entry_bar"],
                "exit_bar":   bar_idx,
                "bars_held":  pos["bars_held"],
                "entry_price": pos["entry_price"],
                "strike":     pos["strike"],
                "delta":      pos["delta"],
                "premium":    pos["premium"],
                "exit_value": pos["current_value"],
                "qty":        pos["qty"],
                "pnl":        pnl,
                "pnl_pct":   pnl_pct,
                "reason":    reason,
                "score":     pos["score"],
            })

        # Circuit breakers
        dd = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
        if consec_losses >= 5 or dd <= -0.15:
            consec_losses = max(0, consec_losses - 1)
            continue

        if bar_idx % 3 != 0:
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
            prices = all_data[sym]
            if bar_idx < LOOKBACK or bar_idx >= len(prices):
                continue
            chunk = prices[bar_idx - LOOKBACK:bar_idx + 1]
            indicators = compute_all_indicators(chunk)
            direction, score = generate_signal(chunk, indicators)
            if direction is None:
                continue

            S = prices[bar_idx]
            iv = estimate_iv(prices[:bar_idx + 1])

            # Strike: OTM if otm_pct > 0, ITM if otm_pct < 0
            if direction == "call":
                K = round(S * (1 + otm_pct), 2)   # call: higher = OTM, lower = ITM
            else:
                K = round(S * (1 - otm_pct), 2)   # put: lower = OTM, higher = ITM

            T = DTE / 365.0
            premium = bs_price(S, K, T, iv, direction)
            delta = abs(bs_delta(S, K, T, iv, direction))

            if premium < 0.05:
                continue

            cost_per = premium * 100
            max_spend = balance * MAX_RISK_PCT
            if cost_per > max_spend:
                continue

            qty = max(1, int(max_spend / cost_per))
            positions.append({
                "symbol":       sym,
                "type":         direction,
                "entry_bar":    bar_idx,
                "entry_price":  S,
                "strike":       K,
                "iv":           iv,
                "dte":          DTE,
                "premium":      premium,
                "current_value": premium,
                "peak_value":   premium,
                "delta":        delta,
                "qty":          qty,
                "score":        score,
                "bars_held":    0,
            })

    # Close any remaining open positions
    for pos in positions:
        sym = pos["symbol"]
        final = min(max_bars - 1, len(all_data[sym]) - 1)
        S = all_data[sym][final]
        bars_held = final - pos["entry_bar"]
        remaining = max(0.05, (pos["dte"] - bars_held / BARS_PER_DAY)) / 365.0
        val = bs_price(S, pos["strike"], remaining, pos["iv"], pos["type"])
        pnl = (val - pos["premium"]) * 100 * pos["qty"]
        pnl_pct = (val - pos["premium"]) / pos["premium"]
        balance += pnl
        trades.append({
            "symbol":     sym, "type": pos["type"],
            "direction":  "CALL" if pos["type"] == "call" else "PUT",
            "entry_bar":  pos["entry_bar"], "exit_bar": final,
            "bars_held":  pos["bars_held"], "entry_price": pos["entry_price"],
            "strike":     pos["strike"], "delta": pos["delta"],
            "premium":    pos["premium"], "exit_value": val,
            "qty":        pos["qty"], "pnl": pnl, "pnl_pct": pnl_pct,
            "reason":     "END", "score": pos["score"],
        })

    return balance, trades


# ── Report builder ────────────────────────────────────────
def build_stats(trades, initial_balance, final_balance, mode_label, otm_pct):
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)   if wins   else 0
    gl = abs(sum(t["pnl"] for t in losses)) if losses else 0
    avg_delta = sum(t["delta"] for t in trades) / len(trades) if trades else 0
    avg_premium = sum(t["premium"] for t in trades) / len(trades) if trades else 0
    label = "ITM" if otm_pct < 0 else "OTM"
    moneyness = f"{abs(otm_pct):.0%} {label}"
    return {
        "mode":           mode_label,
        "moneyness":      moneyness,
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       len(wins) / max(1, len(trades)) * 100,
        "gross_profit":   gp,
        "gross_loss":     gl,
        "net_pnl":        final_balance - initial_balance,
        "return_pct":     (final_balance - initial_balance) / initial_balance * 100,
        "profit_factor":  gp / gl if gl > 0 else float("inf"),
        "avg_win":        gp / len(wins) if wins else 0,
        "avg_loss":       gl / len(losses) if losses else 0,
        "avg_delta":      avg_delta,
        "avg_premium":    avg_premium,
    }


def print_comparison(stats_old, stats_new, days_tested, initial):
    w = 62
    print()
    print("=" * w)
    print(f"  ITM vs OTM STRATEGY COMPARISON")
    print(f"  Last {days_tested} trading days | Balance: ${initial:,.0f}")
    print(f"  Symbols: {', '.join(SYMBOLS)} | DTE: {DTE}d")
    print("=" * w)
    print(f"  {'Metric':<26} {'OLD (4% OTM)':>14} {'NEW (3% ITM)':>14}  {'Diff':>6}")
    print(f"  {'-'*26} {'-'*14} {'-'*14}  {'-'*6}")

    rows = [
        ("Moneyness",       stats_old["moneyness"],           stats_new["moneyness"],          None,    False),
        ("Avg Delta",       f"{stats_old['avg_delta']:.2f}",  f"{stats_new['avg_delta']:.2f}", None,    False),
        ("Avg Premium ($)", f"${stats_old['avg_premium']:.2f}", f"${stats_new['avg_premium']:.2f}", None, False),
        ("Total Trades",    str(stats_old["total_trades"]),   str(stats_new["total_trades"]),  None,     False),
        ("Wins / Losses",   f"{stats_old['wins']}/{stats_old['losses']}",
                            f"{stats_new['wins']}/{stats_new['losses']}",                      None,     False),
        ("Win Rate",        f"{stats_old['win_rate']:.1f}%",  f"{stats_new['win_rate']:.1f}%",
                            stats_new['win_rate'] - stats_old['win_rate'],                              True),
        ("Profit Factor",   f"{stats_old['profit_factor']:.2f}", f"{stats_new['profit_factor']:.2f}",
                            stats_new['profit_factor'] - stats_old['profit_factor'],                    True),
        ("Avg Win ($)",     f"${stats_old['avg_win']:.0f}",   f"${stats_new['avg_win']:.0f}",
                            stats_new['avg_win'] - stats_old['avg_win'],                                True),
        ("Avg Loss ($)",    f"${stats_old['avg_loss']:.0f}",  f"${stats_new['avg_loss']:.0f}",
                            stats_new['avg_loss'] - stats_old['avg_loss'],                              False),
        ("Gross Profit",    f"${stats_old['gross_profit']:.0f}",f"${stats_new['gross_profit']:.0f}",
                            stats_new['gross_profit'] - stats_old['gross_profit'],                      True),
        ("Gross Loss",      f"${stats_old['gross_loss']:.0f}", f"${stats_new['gross_loss']:.0f}",
                            stats_new['gross_loss'] - stats_old['gross_loss'],                          False),
        ("Net P&L",         f"${stats_old['net_pnl']:+.0f}",  f"${stats_new['net_pnl']:+.0f}",
                            stats_new['net_pnl'] - stats_old['net_pnl'],                                True),
        ("Return %",        f"{stats_old['return_pct']:+.2f}%",f"{stats_new['return_pct']:+.2f}%",
                            stats_new['return_pct'] - stats_old['return_pct'],                          True),
    ]

    for metric, v_old, v_new, diff, higher_is_better in rows:
        if diff is None:
            diff_str = ""
        elif higher_is_better:
            arrow = "▲" if diff > 0 else ("▼" if diff < 0 else " ")
            diff_str = f"{arrow}{abs(diff):.1f}"
        else:
            # lower is better (e.g. losses, avg loss)
            arrow = "▼" if diff > 0 else ("▲" if diff < 0 else " ")
            diff_str = f"{arrow}{abs(diff):.1f}"
        print(f"  {metric:<26} {v_old:>14} {v_new:>14}  {diff_str:>6}")

    print("=" * w)

    # Verdict
    itm_wins = 0
    if stats_new["win_rate"] > stats_old["win_rate"]:       itm_wins += 1
    if stats_new["profit_factor"] > stats_old["profit_factor"]: itm_wins += 1
    if stats_new["net_pnl"] > stats_old["net_pnl"]:         itm_wins += 1
    if stats_new["avg_win"] > stats_old["avg_win"]:          itm_wins += 1

    verdict = "ITM appears STRONGER" if itm_wins >= 3 else \
              "MIXED — both have merits" if itm_wins == 2 else \
              "OTM appears STRONGER in this window"
    print(f"\n  VERDICT: {verdict} ({itm_wins}/4 metrics)\n")


def print_trade_table(trades, mode_label, max_rows=15):
    print(f"\n  {'─'*62}")
    print(f"  {mode_label} — Recent Trades (last {min(max_rows, len(trades))})")
    print(f"  {'─'*62}")
    if not trades:
        print("  No trades executed.")
        return
    print(f"  {'Sym':<6} {'Dir':<5} {'Strike':>8} {'Δ':>5} {'Prem':>6} "
          f"{'Exit':>6} {'P&L':>8} {'%':>7}  {'Reason':<18}")
    print(f"  {'-'*6} {'-'*5} {'-'*8} {'-'*5} {'-'*6} {'-'*6} {'-'*8} {'-'*7}  {'-'*18}")
    for t in trades[-max_rows:]:
        sign = "+" if t["pnl"] >= 0 else ""
        pm_sign = "+" if t["pnl_pct"] >= 0 else ""
        print(f"  {t['symbol']:<6} {t['direction']:<5} "
              f"${t['strike']:>7.2f} {t['delta']:>5.2f} "
              f"${t['premium']:>5.2f} ${t['exit_value']:>5.2f} "
              f"{sign}${t['pnl']:>7.0f} "
              f"{pm_sign}{t['pnl_pct']*100:>6.1f}%  "
              f"{t['reason']:<18}")


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ITM vs OTM Strategy Comparison")
    parser.add_argument("--balance",  type=float, default=25000,
                        help="Starting balance (default: $25,000)")
    parser.add_argument("--days",     type=int,   default=5,
                        help="Number of recent trading days to test (default: 5)")
    parser.add_argument("--trades",   action="store_true",
                        help="Show individual trade tables for each mode")
    args = parser.parse_args()

    print(f"\n  Loading data...")
    all_data = {}
    all_timestamps = {}
    for sym in SYMBOLS:
        for res in ["5min", "10min"]:
            path = f"data/historical/{sym}_{res}.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_data[sym] = df["close"].values
                all_timestamps[sym] = df.get("timestamp", pd.Series(range(len(df)))).values
                print(f"  {sym}: {len(df):,} bars of {res} data  "
                      f"(${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f})")
                break

    if not all_data:
        print("  ERROR: No historical data found. Run tools/download_historical.py first.")
        sys.exit(1)

    # Use the last N days
    max_bars = max(len(p) for p in all_data.values())
    start_bar = max(LOOKBACK, max_bars - args.days * BARS_PER_DAY)
    bars_used = max_bars - start_bar
    days_actual = bars_used / BARS_PER_DAY
    print(f"\n  Testing last {days_actual:.1f} trading days "
          f"({bars_used} bars, bars {start_bar}→{max_bars})\n")

    results = {}
    all_trades = {}
    for mode_label, cfg in MODES.items():
        bal, trades = run_simulation(
            all_data, all_timestamps, start_bar,
            args.balance, cfg["otm_pct"], mode_label
        )
        results[mode_label] = (bal, trades)
        all_trades[mode_label] = trades
        w = len([t for t in trades if t["pnl"] > 0])
        print(f"  {mode_label:20}: {len(trades):>3} trades | "
              f"{w}/{len(trades)} wins | "
              f"Net: ${bal - args.balance:+,.0f}")

    # Stats and comparison
    stats = {}
    for mode_label, cfg in MODES.items():
        bal, trades = results[mode_label]
        stats[mode_label] = build_stats(trades, args.balance, bal, mode_label, cfg["otm_pct"])

    old_key = "OLD (4% OTM)"
    new_key = "NEW (3% ITM)"
    print_comparison(stats[old_key], stats[new_key], int(days_actual), args.balance)

    if args.trades:
        for mode_label in MODES:
            print_trade_table(all_trades[mode_label], mode_label)


if __name__ == "__main__":
    main()
