import csv, json, os

os.chdir(r"D:\042021\CryptoBot")

# Read trade history
trades = []
with open("data/history/trade_history.csv") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 6:
            try:
                trades.append({
                    "time": parts[0],
                    "pair": parts[1],
                    "side": parts[2],
                    "price": float(parts[3]),
                    "amount": float(parts[4]),
                    "pnl": float(parts[5])
                })
            except ValueError:
                continue

# ALL v2 trades (after 09:37 when v2 engine launched)
v2 = [t for t in trades if t["time"] >= "2026-02-06T09:37"]
buys = [t for t in v2 if t["side"] in ("BUY", "SHORT")]
sells = [t for t in v2 if t["side"] in ("SELL", "CLOSE")]

# Split: inherited v1 closures vs pure v2 decisions
inherited = [t for t in sells if t["time"] < "2026-02-06T09:55"]
pure_v2 = [t for t in sells if t["time"] >= "2026-02-06T09:55"]

print("=" * 58)
print("   BOT v2.0 PERFORMANCE - 10+ HOURS OF RUNTIME")
print("=" * 58)
print(f"Total v2 actions: {len(v2)} ({len(buys)} opens, {len(sells)} closes)")
print()

# Inherited v1 closures
iw = [t for t in inherited if t["pnl"] > 0]
il = [t for t in inherited if t["pnl"] <= 0]
ipnl = sum(t["pnl"] for t in inherited)
print("--- INHERITED v1 CLOSURES (cleaning up old positions) ---")
for t in inherited:
    tag = "WIN" if t["pnl"] > 0 else "LOSS"
    ts = t["time"][11:19]
    side = t["side"]
    pair = t["pair"]
    pnl = t["pnl"]
    print(f"  {ts}  {side:6s} {pair:14s} P/L: ${pnl:+.2f} [{tag}]")
print(f"  Subtotal: {len(iw)}W/{len(il)}L = ${ipnl:+.2f}")
print()

# Pure v2 trades
pw = [t for t in pure_v2 if t["pnl"] > 0]
pl = [t for t in pure_v2 if t["pnl"] <= 0]
ppnl = sum(t["pnl"] for t in pure_v2)
print("--- PURE v2 TRADES (bot's own decisions) ---")
for t in pure_v2:
    tag = "WIN " if t["pnl"] > 0 else "LOSS"
    ts = t["time"][11:19]
    side = t["side"]
    pair = t["pair"]
    price = t["price"]
    pnl = t["pnl"]
    print(f"  {ts}  {side:6s} {pair:14s} @ ${price:>10.4f}  P/L: ${pnl:+.4f} [{tag}]")
wr = len(pw) / max(1, len(pure_v2)) * 100
print()
print(f"Results: {len(pw)}W / {len(pl)}L ({wr:.0f}% win rate)")
print(f"Realized P/L: ${ppnl:+.2f}")
if pw:
    avg_w = sum(t["pnl"] for t in pw) / len(pw)
    print(f"Avg Win:  ${avg_w:+.4f}")
if pl:
    avg_l = sum(t["pnl"] for t in pl) / len(pl)
    print(f"Avg Loss: ${avg_l:+.4f}")

# Current positions
print()
print("--- OPEN POSITIONS ---")
with open("data/state/positions.json") as f:
    spots = json.load(f)
for pair, pos in spots.items():
    rsi = pos.get("entry_rsi", 0)
    ens = pos.get("ensemble_pred", 0)
    print(f"  SPOT  {pair:14s} entry=${pos['entry_price']} RSI={rsi:.1f} Ensemble={ens:.1%}")

with open("data/state/futures_positions.json") as f:
    futs = json.load(f)
for pair, pos in futs.items():
    d = pos["direction"]
    ep = pos["entry_price"]
    cv = pos["contract_value"]
    lev = pos["leverage"]
    ml_c = pos.get("ml_confidence", 0)
    rsi = pos.get("entry_rsi", 0)
    extra = ""
    if ml_c:
        extra = f" ML={ml_c:.1%} RSI={rsi:.1f}"
    print(f"  {d:5s} {pair:14s} entry=${ep} val=${cv:.2f} {lev}x{extra}")

# Balance summary
with open("data/state/paper_balances.json") as f:
    bal = json.load(f)
spot_value = sum(pos["entry_price"] * pos["amount"] for pos in spots.values())
futures_value = sum(pos["contract_value"] for pos in futs.values())
total_cash = bal["spot"] + bal["futures"]
total_portfolio = total_cash + spot_value + futures_value

print()
print("--- PORTFOLIO SUMMARY ---")
print(f"Spot Cash:        ${bal['spot']:,.2f}")
print(f"Futures Cash:     ${bal['futures']:,.2f}")
print(f"Spot Holdings:    ${spot_value:,.2f} ({len(spots)} positions)")
print(f"Futures Notional: ${futures_value:,.2f} ({len(futs)} positions)")
print(f"TOTAL PORTFOLIO:  ${total_portfolio:,.2f}")
print(f"Started With:     $5,000.00")
print(f"Overall P/L:      ${total_portfolio - 5000:+,.2f} ({(total_portfolio/5000 - 1)*100:+.1f}%)")

# AI learning
with open("data/state/meta_learner.json") as f:
    ml = json.load(f)
print()
print("--- AI LEARNING STATUS ---")
for model, history in ml["model_history"].items():
    correct = sum(1 for h in history if h["correct"])
    n = len(history)
    pct = correct / max(1, n) * 100
    w = ml["model_weights"][model] * 100
    print(f"  {model:12s}: {correct}/{n} correct ({pct:.0f}%) weight={w:.1f}%")

with open("data/state/rl_agent.json") as f:
    rl = json.load(f)
print(f"  RL states:    {len(rl['q_table'])}/243 explored")
print(f"  RL updates:   {rl['total_updates']}")
print(f"  Explore rate: {rl['exploration_rate']*100:.1f}%")
print(f"  RL reward:    {rl['cumulative_reward']:+.4f}")

# Day-by-day comparison
from collections import defaultdict
daily = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0, "losses": 0})
for t in trades:
    day = t["time"][:10]
    if t["side"] in ("SELL", "CLOSE"):
        daily[day]["trades"] += 1
        daily[day]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            daily[day]["wins"] += 1
        else:
            daily[day]["losses"] += 1

print()
print("--- DAILY P/L HISTORY ---")
for day in sorted(daily.keys())[-5:]:
    d = daily[day]
    wr = d["wins"] / max(1, d["trades"]) * 100
    label = ""
    if day == "2026-02-06":
        label = "  <-- v2 upgrade day"
    print(f"  {day}: {d['trades']:3d} closed, {d['wins']}W/{d['losses']}L ({wr:.0f}%), P/L: ${d['pnl']:+.2f}{label}")
