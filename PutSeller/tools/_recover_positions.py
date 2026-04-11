"""
Recover PutSeller positions from Alpaca API.
Rebuilds positions.json by pairing short+long legs into credit spreads.
"""
import sys, json, os, re
from datetime import datetime, date
sys.path.insert(0, "C:/PutSeller")
os.chdir("C:/PutSeller")

from dotenv import load_dotenv
load_dotenv("C:/PutSeller/.env")

from alpaca.trading.client import TradingClient

tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"], paper=True)
positions = tc.get_all_positions()

# Parse OCC option symbols
def parse_occ(symbol: str):
    """Parse OCC symbol: UNDERLYING + YYMMDD + C/P + 8-digit strike"""
    m = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', str(symbol))
    if not m:
        return None
    underlying = m.group(1)
    date_str = m.group(2)  # YYMMDD
    opt_type = "call" if m.group(3) == "C" else "put"
    strike = int(m.group(4)) / 1000.0
    exp = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    return {
        "underlying": underlying,
        "expiration": exp,
        "type": opt_type,
        "strike": strike,
        "symbol": str(symbol),
    }

# Parse all positions
legs = []
for p in positions:
    parsed = parse_occ(str(p.symbol))
    if not parsed:
        print(f"SKIP non-option: {p.symbol}")
        continue
    parsed["qty"] = int(p.qty)
    parsed["side"] = "short" if int(p.qty) < 0 else "long"
    parsed["avg_price"] = float(p.avg_entry_price)
    parsed["market_value"] = float(p.market_value)
    parsed["unrealized_pl"] = float(p.unrealized_pl)
    legs.append(parsed)

print(f"Parsed {len(legs)} option legs")

# Group by (underlying, expiration, type)
from collections import defaultdict
groups = defaultdict(lambda: {"shorts": [], "longs": []})
for leg in legs:
    key = (leg["underlying"], leg["expiration"], leg["type"])
    if leg["side"] == "short":
        groups[key]["shorts"].append(leg)
    else:
        groups[key]["longs"].append(leg)

# Pair shorts with longs into spreads
recovered = {}
unmatched = []

for (underlying, expiration, opt_type), group in sorted(groups.items()):
    shorts = sorted(group["shorts"], key=lambda x: x["strike"])
    longs = sorted(group["longs"], key=lambda x: x["strike"])
    
    if not shorts or not longs:
        unmatched.extend(shorts + longs)
        continue
    
    # For credit spreads:
    # PUT spread: short HIGHER strike, long LOWER strike
    # CALL spread: short LOWER strike, long HIGHER strike
    
    # Match by qty where possible
    used_longs = set()
    for short in shorts:
        best_long = None
        for i, long in enumerate(longs):
            if i in used_longs:
                continue
            if abs(short["qty"]) != long["qty"]:
                continue
            # Pick the correct pairing:
            if opt_type == "put":
                # Short is higher strike, long is lower
                if long["strike"] < short["strike"]:
                    best_long = (i, long)
                    break
            else:
                # Short is lower strike, long is higher
                if long["strike"] > short["strike"]:
                    best_long = (i, long)
                    break
        
        if not best_long:
            # Try any matching long regardless of qty
            for i, long in enumerate(longs):
                if i in used_longs:
                    continue
                if opt_type == "put" and long["strike"] < short["strike"]:
                    best_long = (i, long)
                    break
                elif opt_type == "call" and long["strike"] > short["strike"]:
                    best_long = (i, long)
                    break
        
        if best_long:
            li, long = best_long
            used_longs.add(li)
            
            qty = abs(short["qty"])
            credit_per_share = short["avg_price"] - long["avg_price"]
            spread_width = abs(short["strike"] - long["strike"])
            max_loss_per_contract = (spread_width - credit_per_share) * 100
            total_credit = credit_per_share * qty * 100
            total_max_loss = max_loss_per_contract * qty
            
            # Calculate expiration DTE
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            dte_now = (exp_date - date.today()).days
            
            # Build position ID
            strike_int = int(short["strike"])
            base_id = f"{underlying}_{expiration}_{opt_type[0].upper()}{strike_int}"
            pos_id = base_id
            suffix = 2
            while pos_id in recovered:
                pos_id = f"{base_id}_{suffix}"
                suffix += 1
            
            recovered[pos_id] = {
                "underlying": underlying,
                "spread_type": opt_type,
                "short_symbol": short["symbol"],
                "long_symbol": long["symbol"],
                "short_strike": short["strike"],
                "long_strike": long["strike"],
                "spread_width": spread_width,
                "expiration": expiration,
                "dte_at_open": dte_now + 3,  # estimate: opened ~3 days ago
                "qty": qty,
                "credit_per_share": round(credit_per_share, 4),
                "total_credit": round(total_credit, 2),
                "max_loss_per_contract": round(max_loss_per_contract, 2),
                "max_loss_total": round(total_max_loss, 2),
                "max_profit_total": round(total_credit, 2),
                "order_id": "RECOVERED",
                "open_date": (date.today() - __import__('datetime').timedelta(days=3)).isoformat(),
                "open_time": "RECOVERED",
                "current_debit": round(credit_per_share, 4),
                "current_pnl_total": round(short["unrealized_pl"] + long["unrealized_pl"], 2),
                "current_pnl_pct": 0,
                "roc_annual": 0,
                "short_delta": None,
                "entry_iv": None,  # Can't recover — snapshot at entry time is lost
                "iv_premium": None,
                "features": None,
                "ml_confidence": None,
                "ml_rule_score": None,
            }
            
            # Calculate current P&L %
            if total_credit > 0:
                recovered[pos_id]["current_pnl_pct"] = round(
                    recovered[pos_id]["current_pnl_total"] / total_credit, 4
                )
        else:
            unmatched.append(short)

print(f"\nRecovered {len(recovered)} spreads:")
total_credit = 0
total_max_loss = 0
total_pnl = 0
for pid, pos in sorted(recovered.items()):
    pnl = pos["current_pnl_total"]
    total_credit += pos["total_credit"]
    total_max_loss += pos["max_loss_total"]
    total_pnl += pnl
    sign = "+" if pnl >= 0 else ""
    print(f"  {pid:40s} | {pos['spread_type']:4s} | qty={pos['qty']:2d} | "
          f"credit=${pos['total_credit']:7.0f} | max_loss=${pos['max_loss_total']:7.0f} | "
          f"pnl={sign}${pnl:.0f}")

print(f"\nTotals: credit=${total_credit:.0f} | max_loss=${total_max_loss:.0f} | pnl=${total_pnl:.0f}")

if unmatched:
    print(f"\nWARNING: {len(unmatched)} unmatched legs:")
    for leg in unmatched:
        print(f"  {leg['symbol']} | {leg['side']} | qty={leg['qty']} | strike={leg['strike']}")

# Write to positions.json
out_path = "C:/PutSeller/data/state/positions.json"
print(f"\nWriting {len(recovered)} positions to {out_path}")
with open(out_path, "w") as f:
    json.dump(recovered, f, indent=2, default=str)
print("DONE — positions recovered!")
