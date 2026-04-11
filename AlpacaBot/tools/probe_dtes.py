"""
Probe available DTEs for all scanner universe symbols.
Queries the Alpaca options API to find the next 30 days of
available expiration dates for each symbol, then reports
the minimum DTE, available intervals, and recommended target DTE.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from datetime import datetime, timedelta
from collections import defaultdict

# ── Config ──
API_KEY    = "PKBYFD2FSVABAIANHG3LACEVLD"
API_SECRET = ""
BASE_URL   = "https://paper-api.alpaca.markets"

# Load secret from config
from core.config import Config
cfg = Config()
API_SECRET = cfg.API_SECRET

# All 61 original symbols + the proven ones
ALL_SYMBOLS = [
    # TIER1 proven
    "NFLX", "SNOW", "MA", "GM", "LLY", "IWM",
    # MAYBE
    "UBER", "CVX", "ARKK", "SMH",
    # Original TIER1 (dropped at 1DTE, maybe good at other DTE)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    # Original TIER2
    "AMD", "CRM", "AVGO", "ORCL", "ADBE", "INTC",
    "QCOM", "MU", "SHOP", "COIN", "PYPL",
    # Original TIER3
    "SPY", "QQQ", "DIA", "XLF", "XLE", "XLK",
    "SOXX", "GLD", "SLV", "EEM", "TLT",
    # Original TIER4
    "BA", "JPM", "GS", "V", "WMT", "HD", "COST",
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "XOM",
    "DIS", "NKE", "SBUX", "F", "RIVN",
    "PLTR", "SOFI", "MARA", "RIOT",
]

# Remove dupes preserving order
SYMBOLS = list(dict.fromkeys(ALL_SYMBOLS))

HEADERS = {
    "APCA-API-KEY-ID": cfg.API_KEY,
    "APCA-API-SECRET-KEY": cfg.API_SECRET,
    "accept": "application/json",
}

def get_expirations(symbol: str, days_ahead: int = 30) -> list:
    """Get unique expiration dates for a symbol in the next N days."""
    today = datetime.now().date()
    exp_after = today.strftime("%Y-%m-%d")
    exp_before = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    url = f"{BASE_URL}/v2/options/contracts"
    params = {
        "underlying_symbols": symbol,
        "status": "active",
        "expiration_date_gte": exp_after,
        "expiration_date_lte": exp_before,
        "limit": 250,
    }

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        contracts = data.get("option_contracts", [])

        # Extract unique expiration dates
        expirations = sorted(set(c.get("expiration_date", "") for c in contracts))
        return expirations
    except Exception as e:
        return [f"ERROR: {e}"]


def compute_dtes(expirations: list) -> list:
    """Convert expiration date strings to DTEs (days to expiry)."""
    today = datetime.now().date()
    dtes = []
    for exp in expirations:
        if exp.startswith("ERROR"):
            continue
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte >= 0:
                dtes.append(dte)
        except:
            pass
    return sorted(dtes)


def main():
    print("=" * 80)
    print("  AlpacaBot DTE Discovery Probe")
    print(f"  {len(SYMBOLS)} symbols | Next 30 days | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print()

    results = {}
    errors = []

    for i, symbol in enumerate(SYMBOLS):
        expirations = get_expirations(symbol)
        dtes = compute_dtes(expirations)

        if not dtes:
            errors.append(symbol)
            status = "NO OPTIONS"
        else:
            min_dte = min(dtes)
            max_dte = max(dtes)
            status = f"DTEs: {dtes}"

        results[symbol] = {
            "expirations": expirations,
            "dtes": dtes,
            "min_dte": min(dtes) if dtes else None,
        }

        # Print live
        dte_str = ",".join(str(d) for d in dtes) if dtes else "NONE"
        print(f"  [{i+1:2d}/{len(SYMBOLS)}] {symbol:6s}  DTEs available: [{dte_str}]")

        # Rate limit: be gentle
        time.sleep(0.2)

    # ── Summary ──
    print()
    print("=" * 80)
    print("  DTE SUMMARY")
    print("=" * 80)

    # Group by DTE pattern
    dte_patterns = defaultdict(list)
    for sym, info in results.items():
        if info["dtes"]:
            # Classify: daily (has 0 or 1 DTE), weekly-ish, etc.
            min_d = info["min_dte"]
            has_daily = any(d <= 1 for d in info["dtes"])
            has_2day = 2 in info["dtes"]
            has_weekly = any(d in [5, 6, 7] for d in info["dtes"])

            if has_daily:
                category = "DAILY (0-1 DTE available)"
            elif has_2day:
                category = "2-DAY MIN"
            elif has_weekly:
                category = f"WEEKLY MIN ({min_d}d)"
            else:
                category = f"OTHER MIN ({min_d}d)"

            dte_patterns[category].append(sym)

    for cat in sorted(dte_patterns.keys()):
        syms = dte_patterns[cat]
        print(f"\n  {cat}:")
        print(f"    {', '.join(syms)}")

    if errors:
        print(f"\n  NO OPTIONS FOUND:")
        print(f"    {', '.join(errors)}")

    # ── Recommended DTE per symbol ──
    print()
    print("=" * 80)
    print("  RECOMMENDED TARGET DTE PER SYMBOL")
    print("=" * 80)
    print(f"  {'Symbol':8s} {'Min DTE':>8s} {'DTEs Available':40s} {'Recommended':>12s}")
    print(f"  {'─'*8} {'─'*8} {'─'*40} {'─'*12}")

    for sym, info in sorted(results.items(), key=lambda x: (x[1]["min_dte"] or 999)):
        if not info["dtes"]:
            continue
        dte_str = ",".join(str(d) for d in info["dtes"][:10])
        if len(info["dtes"]) > 10:
            dte_str += "..."
        min_d = info["min_dte"]

        # Recommend: use the smallest DTE >= 1 (never 0DTE)
        valid_dtes = [d for d in info["dtes"] if d >= 1]
        if valid_dtes:
            recommended = min(valid_dtes)
        else:
            recommended = min(info["dtes"])  # fallback

        print(f"  {sym:8s} {min_d:>8d} [{dte_str:38s}] {recommended:>10d}d")

    # ── Save JSON ──
    out_path = os.path.join(os.path.dirname(__file__), "backtests", "dte_discovery.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    save_data = {}
    for sym, info in results.items():
        valid_dtes = [d for d in info["dtes"] if d >= 1]
        recommended = min(valid_dtes) if valid_dtes else (min(info["dtes"]) if info["dtes"] else None)
        save_data[sym] = {
            "dtes": info["dtes"],
            "expirations": info["expirations"],
            "min_dte": info["min_dte"],
            "recommended_dte": recommended,
        }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {out_path}")
    print()


if __name__ == "__main__":
    main()
