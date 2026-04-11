"""
AlpacaBot Options Backtester
Simulates buying calls/puts based on ML + indicator signals on daily data.

Since we can't get historical options pricing, we simulate options P&L using
the Black-Scholes-approximated delta relationship:
  - ATM call/put premium ≈ underlying_price * IV * sqrt(DTE/365) * 0.4
  - P&L ≈ delta * underlying_move * 100 (per contract)
  - ATM delta ≈ 0.50, adjusted for moneyness

This gives realistic directional options P&L without needing historical chains.

Usage: python tools/backtest.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from core.indicators import compute_all_indicators

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
INITIAL_BALANCE = 5000.0
MAX_POSITION_PCT = 0.20       # 20% per trade
MIN_POSITION_PCT = 0.05
MAX_POSITIONS = 5
TARGET_DTE = 21               # 21-day options
STOP_LOSS_PCT = -0.30         # -30% of premium
TAKE_PROFIT_PCT = 0.60        # +60% of premium
TRAILING_STOP_PCT = 0.20      # 20% from peak
MAX_HOLD_DAYS = 7
MIN_ML_CONFIDENCE = 0.52      # Lowered: more trades
MAX_COST_PER_TRADE = 0.25     # Max 25% of balance per trade

# ── ML / Signal Config ──
MIN_RSI_LONG = 25.0
MAX_RSI_LONG = 70.0
MIN_RSI_SHORT = 30.0
MAX_RSI_SHORT = 75.0
LOOKBACK = 30                 # indicator lookback window

# Feature names
FEATURE_NAMES = [
    "rsi", "macd_hist", "bb_position", "stochastic", "atr_normalized",
    "cci", "roc", "williams_r", "volatility_ratio", "zscore",
    "trend_strength", "price_change_1", "price_change_5", "price_change_20",
]


# ═══════════════════════════════════════════════════════════
# OPTIONS PRICING SIMULATION
# ═══════════════════════════════════════════════════════════

def estimate_iv(prices, window=20):
    """Estimate implied volatility from historical returns."""
    if len(prices) < window + 1:
        return 0.25  # default 25%
    returns = np.diff(np.log(prices[-window - 1:])) 
    return float(np.std(returns) * np.sqrt(252))  # annualized


def estimate_premium(price, iv, dte=21):
    """Estimate ATM option premium using simplified Black-Scholes."""
    # ATM call/put ≈ price * IV * sqrt(T) * 0.4
    t = dte / 365.0
    premium = price * iv * np.sqrt(t) * 0.4
    return max(premium, 0.10)  # minimum $0.10


def simulate_option_pnl(entry_price, current_price, premium_paid, 
                         option_type, dte_at_entry, days_held):
    """
    Simulate option P&L based on underlying price movement.
    
    Uses delta approximation:
    - ATM delta ≈ 0.50
    - As price moves ITM, delta increases
    - Theta decay reduces value over time
    """
    # Price move as % of underlying
    price_move_pct = (current_price - entry_price) / entry_price
    
    # Directional P&L (delta effect)
    if option_type == "call":
        delta = 0.50 + price_move_pct * 5  # delta increases as price goes up
    else:  # put
        delta = -0.50 + price_move_pct * 5  # negative delta for puts
        price_move_pct = -price_move_pct     # puts profit from down moves
    
    delta = max(0.05, min(0.95, abs(delta)))
    
    # Intrinsic value gain/loss from underlying move
    intrinsic_change = abs(price_move_pct) * entry_price * delta
    if price_move_pct > 0:
        if option_type == "call":
            option_value_change = intrinsic_change
        else:
            option_value_change = -intrinsic_change
    else:
        if option_type == "call":
            option_value_change = -intrinsic_change
        else:
            option_value_change = intrinsic_change
    
    # Theta decay (accelerates near expiry)
    remaining_dte = max(1, dte_at_entry - days_held)
    theta_decay_pct = 1.0 - np.sqrt(remaining_dte / max(1, dte_at_entry))
    theta_cost = premium_paid * theta_decay_pct * 0.6  # 60% of theoretical
    
    # Current option value
    current_value = premium_paid + option_value_change - theta_cost
    current_value = max(0.01, current_value)  # can't go below $0.01
    
    return current_value


# ═══════════════════════════════════════════════════════════
# ML MODEL (TRAINED ON FIRST 60% OF DATA)
# ═══════════════════════════════════════════════════════════

def train_model(all_prices_dict, train_end_idx):
    """Train a simple ML model on the training portion of data."""
    from sklearn.ensemble import GradientBoostingClassifier
    
    X_all, y_all = [], []
    
    for symbol, prices in all_prices_dict.items():
        train_prices = prices[:train_end_idx]
        if len(train_prices) < LOOKBACK + 10:
            continue
            
        for i in range(LOOKBACK, len(train_prices) - 5):
            chunk = train_prices[i - LOOKBACK:i + 1]
            indicators = compute_all_indicators(chunk)
            features = [indicators.get(f, 0) for f in FEATURE_NAMES]
            X_all.append(features)
            
            # Label: did price go up in next 5 days?
            future_ret = (train_prices[i + 5] - train_prices[i]) / train_prices[i]
            y_all.append(1 if future_ret > 0.002 else 0)
    
    if len(X_all) < 50:
        print("  Not enough training data, using rules only")
        return None
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8, random_state=42,
    )
    model.fit(X, y)
    
    acc = model.score(X, y)
    up_pct = np.mean(y) * 100
    print(f"  ML Model: {len(X)} samples, train acc={acc:.1%}, up%={up_pct:.0f}%")
    
    return model


def predict(model, indicators):
    """Get direction and confidence from ML model or rules."""
    if model is not None:
        features = np.array([[indicators.get(f, 0) for f in FEATURE_NAMES]])
        try:
            proba = model.predict_proba(features)[0]
            if proba[1] > 0.5:
                return "call", float(proba[1])
            else:
                return "put", float(proba[0])
        except:
            pass
    
    # Fallback: rule-based
    return rule_signal(indicators)


def rule_signal(ind):
    """Rule-based signal using indicator confluence."""
    bull, bear = 0, 0
    
    rsi = ind.get("rsi", 50)
    if rsi < 35: bull += 2
    elif rsi < 45: bull += 1
    elif rsi > 65: bear += 2
    elif rsi > 55: bear += 1
    
    if ind.get("macd_hist", 0) > 0: bull += 1
    else: bear += 1
    
    bb = ind.get("bb_position", 0.5)
    if bb < 0.2: bull += 2
    elif bb > 0.8: bear += 2
    
    if ind.get("stochastic", 50) < 25: bull += 1
    elif ind.get("stochastic", 50) > 75: bear += 1
    
    if ind.get("cci", 0) < -100: bull += 1
    elif ind.get("cci", 0) > 100: bear += 1
    
    zs = ind.get("zscore", 0)
    if zs < -1.5: bull += 1
    elif zs > 1.5: bear += 1
    
    total = bull + bear
    if total == 0:
        return "call", 0.0
    
    if bull > bear:
        return "call", min(bull / (total + 2), 0.95)
    else:
        return "put", min(bear / (total + 2), 0.95)


# ═══════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════

def run_backtest():
    print("=" * 65)
    print("  AlpacaBot Options Backtest")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Balance: ${INITIAL_BALANCE:,.0f} | Position: {MAX_POSITION_PCT:.0%}")
    print(f"  SL/TP: {STOP_LOSS_PCT:.0%} / +{TAKE_PROFIT_PCT:.0%}")
    print(f"  DTE: {TARGET_DTE}d | Max Hold: {MAX_HOLD_DAYS}d")
    print("=" * 65)
    
    # ── Load Data ──
    all_prices = {}
    all_data = {}
    
    for symbol in SYMBOLS:
        path = f"data/historical/{symbol}_daily.csv"
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        df = pd.read_csv(path)
        all_prices[symbol] = df["close"].values
        all_data[symbol] = df
        print(f"  Loaded {symbol}: {len(df)} days, "
              f"${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
    
    if not all_prices:
        print("No data loaded!")
        return
    
    # ── Walk-Forward Split: Train 60%, Test 40% ──
    max_len = max(len(p) for p in all_prices.values())
    train_end = int(max_len * 0.6)
    
    print(f"\n  Train: days 0-{train_end} | Test: days {train_end}-{max_len}")
    print(f"  Training ML model...")
    
    model = train_model(all_prices, train_end)
    
    # ── Simulate Trading on Test Period ──
    print(f"\n{'─' * 65}")
    print(f"  RUNNING BACKTEST (test period: {max_len - train_end} days)")
    print(f"{'─' * 65}\n")
    
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    positions = []  # list of open positions
    trades = []     # completed trades
    daily_balances = []
    consecutive_losses = 0
    
    for day_idx in range(train_end, max_len):
        # Track daily balance
        # Mark-to-market open positions
        unrealized = 0
        for pos in positions:
            sym = pos["symbol"]
            if day_idx < len(all_prices[sym]):
                current_price = all_prices[sym][day_idx]
                days_held = day_idx - pos["entry_day"]
                option_value = simulate_option_pnl(
                    pos["entry_underlying"], current_price,
                    pos["premium"], pos["type"],
                    TARGET_DTE, days_held
                )
                pos["current_value"] = option_value
                pos["peak_value"] = max(pos.get("peak_value", pos["premium"]), option_value)
                unrealized += (option_value - pos["premium"]) * 100 * pos["qty"]
        
        daily_balances.append(balance + unrealized)
        
        # ── Check Exits ──
        exits = []
        for i, pos in enumerate(positions):
            sym = pos["symbol"]
            if day_idx >= len(all_prices[sym]):
                continue
                
            days_held = day_idx - pos["entry_day"]
            pnl_pct = (pos["current_value"] - pos["premium"]) / pos["premium"]
            
            exit_reason = None
            
            # Stop loss
            if pnl_pct <= STOP_LOSS_PCT:
                exit_reason = "STOP_LOSS"
            # Take profit
            elif pnl_pct >= TAKE_PROFIT_PCT:
                exit_reason = "TAKE_PROFIT"
            # Trailing stop
            elif pos["peak_value"] > pos["premium"] * 1.1:  # only after 10% gain
                drop = (pos["current_value"] - pos["peak_value"]) / pos["peak_value"]
                if drop <= -TRAILING_STOP_PCT:
                    exit_reason = "TRAILING_STOP"
            # Max hold
            elif days_held >= MAX_HOLD_DAYS:
                exit_reason = "MAX_HOLD"
            # DTE exit (approaching expiry)
            elif days_held >= TARGET_DTE - 3:
                exit_reason = "DTE_EXIT"
            
            if exit_reason:
                exits.append((i, exit_reason))
        
        # Process exits (reverse order to maintain indices)
        for i, reason in sorted(exits, reverse=True):
            pos = positions.pop(i)
            pnl = (pos["current_value"] - pos["premium"]) * 100 * pos["qty"]
            pnl_pct = (pos["current_value"] - pos["premium"]) / pos["premium"]
            balance += pnl
            
            if pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
            
            if balance > peak_balance:
                peak_balance = balance
            
            trades.append({
                "symbol": pos["symbol"],
                "type": pos["type"],
                "direction": "bullish" if pos["type"] == "call" else "bearish",
                "entry_day": pos["entry_day"],
                "exit_day": day_idx,
                "days_held": day_idx - pos["entry_day"],
                "entry_underlying": pos["entry_underlying"],
                "exit_underlying": all_prices[pos["symbol"]][day_idx],
                "premium_paid": pos["premium"],
                "exit_value": pos["current_value"],
                "qty": pos["qty"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": reason,
                "confidence": pos["confidence"],
            })
        
        # ── Check Circuit Breakers ──
        if consecutive_losses >= 4:
            continue  # skip new entries
        drawdown = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
        if drawdown <= -0.15:
            continue
        
        # ── Generate Signals & Open Positions ──
        # Check for new entries daily
        
        if len(positions) >= MAX_POSITIONS:
            continue
        
        for symbol in SYMBOLS:
            if len(positions) >= MAX_POSITIONS:
                break
            
            # Skip if already have position on this symbol
            if any(p["symbol"] == symbol for p in positions):
                continue
            
            prices = all_prices[symbol]
            if day_idx < LOOKBACK or day_idx >= len(prices):
                continue
            
            # Compute indicators
            chunk = prices[day_idx - LOOKBACK:day_idx + 1]
            indicators = compute_all_indicators(chunk)
            
            # Get signal
            option_type, confidence = predict(model, indicators)
            
            if confidence < MIN_ML_CONFIDENCE:
                continue
            
            # RSI filter
            rsi_val = indicators.get("rsi", 50)
            if option_type == "call":
                if rsi_val < MIN_RSI_LONG or rsi_val > MAX_RSI_LONG:
                    continue
            else:
                if rsi_val < MIN_RSI_SHORT or rsi_val > MAX_RSI_SHORT:
                    continue
            
            # Trend filter (softened)
            trend = indicators.get("trend_strength", 0)
            if trend < 10:
                confidence *= 0.90
                if confidence < MIN_ML_CONFIDENCE:
                    continue
            
            # Price and premium
            current_price = prices[day_idx]
            iv = estimate_iv(prices[:day_idx + 1])
            premium = estimate_premium(current_price, iv, TARGET_DTE)
            
            # Position sizing
            cost_per_contract = premium * 100
            
            # Skip if single contract costs more than 15% of balance
            if cost_per_contract > balance * MAX_COST_PER_TRADE:
                continue
            
            max_spend = balance * MAX_POSITION_PCT
            qty = max(1, int(max_spend / cost_per_contract))
            total_cost = cost_per_contract * qty
            
            # Hard cap: never spend more than 15% on one trade
            if total_cost > balance * MAX_COST_PER_TRADE:
                qty = max(1, int(balance * MAX_COST_PER_TRADE / cost_per_contract))
                if cost_per_contract * qty > balance * MAX_COST_PER_TRADE:
                    continue  # still too expensive
            
            positions.append({
                "symbol": symbol,
                "type": option_type,
                "entry_day": day_idx,
                "entry_underlying": current_price,
                "premium": premium,
                "current_value": premium,
                "peak_value": premium,
                "qty": qty,
                "confidence": confidence,
            })
    
    # ── Close remaining positions at end ──
    for pos in positions:
        sym = pos["symbol"]
        final_idx = min(max_len - 1, len(all_prices[sym]) - 1)
        days_held = final_idx - pos["entry_day"]
        final_value = simulate_option_pnl(
            pos["entry_underlying"], all_prices[sym][final_idx],
            pos["premium"], pos["type"], TARGET_DTE, days_held
        )
        pnl = (final_value - pos["premium"]) * 100 * pos["qty"]
        pnl_pct = (final_value - pos["premium"]) / pos["premium"]
        balance += pnl
        
        trades.append({
            "symbol": sym, "type": pos["type"],
            "direction": "bullish" if pos["type"] == "call" else "bearish",
            "entry_day": pos["entry_day"], "exit_day": final_idx,
            "days_held": days_held,
            "entry_underlying": pos["entry_underlying"],
            "exit_underlying": all_prices[sym][final_idx],
            "premium_paid": pos["premium"], "exit_value": final_value,
            "qty": pos["qty"], "pnl": pnl, "pnl_pct": pnl_pct,
            "exit_reason": "END_OF_TEST", "confidence": pos["confidence"],
        })
    
    # ═══════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════
    
    print("=" * 65)
    print("  BACKTEST RESULTS")
    print("=" * 65)
    
    if not trades:
        print("  No trades executed!")
        return
    
    total_pnl = sum(t["pnl"] for t in trades)
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(winners) / len(trades) * 100
    
    gross_profit = sum(t["pnl"] for t in winners) if winners else 0
    gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    avg_win = gross_profit / len(winners) if winners else 0
    avg_loss = gross_loss / len(losers) if losers else 0
    
    max_dd = 0
    peak_bal = INITIAL_BALANCE
    for b in daily_balances:
        if b > peak_bal:
            peak_bal = b
        dd = (b - peak_bal) / peak_bal
        if dd < max_dd:
            max_dd = dd
    
    avg_hold = np.mean([t["days_held"] for t in trades])
    
    print(f"\n  Starting Balance:  ${INITIAL_BALANCE:,.2f}")
    print(f"  Final Balance:     ${balance:,.2f}")
    print(f"  Total P&L:         ${total_pnl:+,.2f} ({total_pnl/INITIAL_BALANCE:.1%})")
    print(f"  Total Trades:      {len(trades)}")
    print(f"  Win Rate:          {win_rate:.1f}% ({len(winners)}W / {len(losers)}L)")
    print(f"  Profit Factor:     {pf:.2f}")
    print(f"  Avg Win:           ${avg_win:.2f}")
    print(f"  Avg Loss:          ${avg_loss:.2f}")
    print(f"  Max Drawdown:      {max_dd:.1%}")
    print(f"  Avg Hold:          {avg_hold:.1f} days")
    
    # ── By Symbol ──
    print(f"\n  {'─' * 55}")
    print(f"  BY SYMBOL:")
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)
    
    print(f"  {'Symbol':<8} {'Trades':>6} {'P&L':>10} {'WR':>6} {'PF':>6} {'Avg Hold':>8}")
    for sym in SYMBOLS:
        if sym not in by_sym:
            continue
        st = by_sym[sym]
        s_pnl = sum(t["pnl"] for t in st)
        s_win = [t for t in st if t["pnl"] > 0]
        s_lose = [t for t in st if t["pnl"] <= 0]
        s_wr = len(s_win) / len(st) * 100 if st else 0
        s_gp = sum(t["pnl"] for t in s_win)
        s_gl = abs(sum(t["pnl"] for t in s_lose))
        s_pf = s_gp / s_gl if s_gl > 0 else float("inf")
        s_ah = np.mean([t["days_held"] for t in st])
        print(f"  {sym:<8} {len(st):>6} ${s_pnl:>+9.2f} {s_wr:>5.0f}% {s_pf:>5.2f} {s_ah:>7.1f}d")
    
    # ── Calls vs Puts ──
    print(f"\n  {'─' * 55}")
    calls = [t for t in trades if t["type"] == "call"]
    puts = [t for t in trades if t["type"] == "put"]
    
    if calls:
        c_pnl = sum(t["pnl"] for t in calls)
        c_wr = len([t for t in calls if t["pnl"] > 0]) / len(calls) * 100
        print(f"  CALLS:  {len(calls)} trades, ${c_pnl:+.2f}, {c_wr:.0f}% WR")
    if puts:
        p_pnl = sum(t["pnl"] for t in puts)
        p_wr = len([t for t in puts if t["pnl"] > 0]) / len(puts) * 100
        print(f"  PUTS:   {len(puts)} trades, ${p_pnl:+.2f}, {p_wr:.0f}% WR")
    
    # ── Exit Reasons ──
    print(f"\n  {'─' * 55}")
    print(f"  EXIT REASONS:")
    reasons = defaultdict(lambda: {"count": 0, "pnl": 0})
    for t in trades:
        r = t["exit_reason"]
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t["pnl"]
    
    for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        avg_pnl = data["pnl"] / data["count"]
        print(f"  {reason:<20} {data['count']:>3} trades  ${data['pnl']:>+8.2f}  (avg ${avg_pnl:>+.2f})")
    
    # ── Top/Bottom Trades ──
    print(f"\n  {'─' * 55}")
    sorted_trades = sorted(trades, key=lambda x: x["pnl"], reverse=True)
    print(f"  BEST TRADES:")
    for t in sorted_trades[:3]:
        print(f"    {t['symbol']} {t['type'].upper()} | ${t['pnl']:+.2f} ({t['pnl_pct']:+.0%}) | "
              f"{t['days_held']}d | {t['exit_reason']}")
    
    print(f"  WORST TRADES:")
    for t in sorted_trades[-3:]:
        print(f"    {t['symbol']} {t['type'].upper()} | ${t['pnl']:+.2f} ({t['pnl_pct']:+.0%}) | "
              f"{t['days_held']}d | {t['exit_reason']}")
    
    # ── Verdict ──
    print(f"\n{'=' * 65}")
    if total_pnl > 0 and win_rate > 50 and pf > 1.0:
        print(f"  VERDICT: PROFITABLE  ✓")
        print(f"  Strategy shows edge. Ready for paper trading.")
    elif total_pnl > 0:
        print(f"  VERDICT: MARGINAL  ~")
        print(f"  Positive but needs tuning (WR or PF weak).")
    else:
        print(f"  VERDICT: NEEDS WORK  ✗")
        print(f"  Not profitable yet. Tune parameters before going live.")
    print(f"{'=' * 65}")
    
    return {
        "balance": balance, "pnl": total_pnl, "trades": len(trades),
        "win_rate": win_rate, "pf": pf, "max_dd": max_dd,
    }


if __name__ == "__main__":
    run_backtest()
