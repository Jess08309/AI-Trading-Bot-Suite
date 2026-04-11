"""
Quick stock-only backtest to validate signal quality.
No options — just buy/sell the underlying based on the same signals.
This tells us if the directions are right before adding options complexity.

Usage: python tools/backtest_stocks.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from collections import defaultdict

from core.indicators import compute_all_indicators

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
LOOKBACK = 30
INITIAL_BALANCE = 10_000
POSITION_SIZE = 0.15     # 15% of balance per trade
MAX_POSITIONS = 4
STOP_LOSS = -0.03        # -3% on underlying
TAKE_PROFIT = 0.04       # +4% on underlying
MAX_HOLD = 8
MIN_SCORE = 3


def generate_signal(prices, indicators):
    bull, bear = 0, 0
    
    if len(prices) >= 20:
        sma10 = np.mean(prices[-10:])
        sma20 = np.mean(prices[-20:])
        cur = prices[-1]
        if cur > sma10 > sma20: bull += 2
        elif cur > sma10: bull += 1
        elif cur < sma10 < sma20: bear += 2
        elif cur < sma10: bear += 1
    
    rsi = indicators.get("rsi", 50)
    if rsi < 30: bull += 2
    elif 30 < rsi < 50: bull += 1
    elif rsi > 70: bear += 2
    elif 50 < rsi < 70: bear += 1
    
    if indicators.get("macd_hist", 0) > 0: bull += 1
    else: bear += 1
    
    bb = indicators.get("bb_position", 0.5)
    if bb < 0.15: bull += 1
    elif bb > 0.85: bear += 1
    
    ts = indicators.get("trend_strength", 0)
    if ts > 20:
        if bull > bear: bull += 1
        elif bear > bull: bear += 1
    
    pc5 = indicators.get("price_change_5", 0)
    if pc5 > 1.5: bull += 1
    elif pc5 < -1.5: bear += 1
    
    zs = indicators.get("zscore", 0)
    if zs < -1.5: bull += 1
    elif zs > 1.5: bear += 1
    
    if bull >= MIN_SCORE and bull > bear + 1:
        return "long", bull
    elif bear >= MIN_SCORE and bear > bull + 1:
        return "short", bear
    return None, 0


def run():
    print("=" * 60)
    print("  Stock Signal Quality Test")
    print(f"  {', '.join(SYMBOLS)} | ${INITIAL_BALANCE:,} | {POSITION_SIZE:.0%}/trade")
    print("=" * 60)
    
    all_prices = {}
    for sym in SYMBOLS:
        path = f"data/historical/{sym}_daily.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        all_prices[sym] = df["close"].values
        print(f"  {sym}: {len(df)}d  ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
    
    max_len = max(len(p) for p in all_prices.values())
    test_start = int(max_len * 0.6)
    
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    positions = []
    trades = []
    signal_count = {"long": 0, "short": 0}
    
    for day in range(test_start, max_len):
        # Check exits
        exits = []
        for i, pos in enumerate(positions):
            sym = pos["symbol"]
            if day >= len(all_prices[sym]):
                continue
            price = all_prices[sym][day]
            ret = (price - pos["entry"]) / pos["entry"]
            if pos["direction"] == "short":
                ret = -ret
            
            days_held = day - pos["day"]
            reason = None
            if ret <= STOP_LOSS: reason = "STOP"
            elif ret >= TAKE_PROFIT: reason = "TP"
            elif days_held >= MAX_HOLD: reason = "MAX_HOLD"
            
            if reason:
                exits.append((i, reason, ret))
        
        for i, reason, ret in sorted(exits, reverse=True):
            pos = positions.pop(i)
            pnl = balance * pos["alloc"] * ret
            balance += pnl
            peak = max(peak, balance)
            trades.append({
                "symbol": pos["symbol"], "direction": pos["direction"],
                "pnl": pnl, "ret": ret, "days": day - pos["day"],
                "reason": reason, "score": pos["score"],
            })
        
        if len(positions) >= MAX_POSITIONS:
            continue
        
        for sym in SYMBOLS:
            if len(positions) >= MAX_POSITIONS:
                break
            if any(p["symbol"] == sym for p in positions):
                continue
            
            prices = all_prices[sym]
            if day < LOOKBACK or day >= len(prices):
                continue
            
            chunk = prices[day - LOOKBACK:day + 1]
            indicators = compute_all_indicators(chunk)
            direction, score = generate_signal(chunk, indicators)
            
            if direction is None:
                continue
            signal_count[direction] += 1
            
            positions.append({
                "symbol": sym, "direction": direction,
                "entry": prices[day], "day": day,
                "alloc": POSITION_SIZE, "score": score,
            })
    
    # Close remaining
    for pos in positions:
        sym = pos["symbol"]
        final = min(max_len - 1, len(all_prices[sym]) - 1)
        price = all_prices[sym][final]
        ret = (price - pos["entry"]) / pos["entry"]
        if pos["direction"] == "short": ret = -ret
        pnl = balance * pos["alloc"] * ret
        balance += pnl
        trades.append({
            "symbol": sym, "direction": pos["direction"],
            "pnl": pnl, "ret": ret, "days": final - pos["day"],
            "reason": "END", "score": pos["score"],
        })
    
    # Report
    print(f"\n{'=' * 60}")
    print(f"  RESULTS (test: {max_len - test_start} days)")
    print(f"{'=' * 60}")
    
    if not trades:
        print("  No trades!")
        return
    
    w = [t for t in trades if t["pnl"] > 0]
    l = [t for t in trades if t["pnl"] <= 0]
    wr = len(w) / len(trades) * 100
    total = balance - INITIAL_BALANCE
    gp = sum(t["pnl"] for t in w) if w else 0
    gl = abs(sum(t["pnl"] for t in l)) if l else 0
    pf = gp / gl if gl > 0 else 0
    
    print(f"  P&L:      ${total:+,.2f} ({total/INITIAL_BALANCE:+.1%})")
    print(f"  Trades:   {len(trades)}  ({signal_count['long']}L / {signal_count['short']}S signals)")
    print(f"  Win Rate: {wr:.1f}%  ({len(w)}W / {len(l)}L)")
    print(f"  PF:       {pf:.2f}")
    print(f"  Avg Win:  ${gp/max(1,len(w)):,.2f}  |  Avg Loss: ${gl/max(1,len(l)):,.2f}")
    
    # Direction accuracy
    long_trades = [t for t in trades if t["direction"] == "long"]
    short_trades = [t for t in trades if t["direction"] == "short"]
    
    if long_trades:
        l_wr = len([t for t in long_trades if t["ret"] > 0]) / len(long_trades) * 100
        l_pnl = sum(t["pnl"] for t in long_trades)
        print(f"\n  LONG:  {len(long_trades)} trades, {l_wr:.0f}% accurate, ${l_pnl:+,.2f}")
    if short_trades:
        s_wr = len([t for t in short_trades if t["ret"] > 0]) / len(short_trades) * 100
        s_pnl = sum(t["pnl"] for t in short_trades)
        print(f"  SHORT: {len(short_trades)} trades, {s_wr:.0f}% accurate, ${s_pnl:+,.2f}")
    
    # By symbol
    print(f"\n  {'Symbol':<7} {'#':>3} {'P&L':>9} {'WR':>5}")
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)
    for sym in SYMBOLS:
        if sym not in by_sym:
            continue
        st = by_sym[sym]
        s_pnl = sum(t["pnl"] for t in st)
        s_wr = len([t for t in st if t["pnl"] > 0]) / len(st) * 100
        print(f"  {sym:<7} {len(st):>3} ${s_pnl:>+8.2f} {s_wr:>4.0f}%")
    
    if total > 0:
        print(f"\n  Signal quality: GOOD — direction calls have edge")
        print(f"  Options were the problem, not the signals")
    else:
        print(f"\n  Signal quality: POOR — direction calls don't have edge")
        print(f"  Need to improve signal generation before adding options")


if __name__ == "__main__":
    run()
