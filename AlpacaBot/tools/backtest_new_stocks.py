"""
Backtest ALL new stock candidates across AlpacaBot, PutSeller, and CallBuyer.
Downloads 1 year of daily bars from Alpaca, computes quality metrics,
and reports which symbols are worth trading vs which are junk.

Usage:
  python tools/backtest_new_stocks.py
  python tools/backtest_new_stocks.py --bot alpacabot
  python tools/backtest_new_stocks.py --bot putseller
  python tools/backtest_new_stocks.py --bot callbuyer
"""
import sys, os, json, time, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
from core.config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ═══════════════════════════════════════════════════════════
#  SYMBOL DEFINITIONS — must match universe_scanner.py
# ═══════════════════════════════════════════════════════════

ALPACABOT_PROVEN = {
    "AVGO", "PYPL", "NFLX", "NKE", "SBUX", "F", "COST", "SHOP",
    "IWM", "META", "JPM", "AMZN", "DIS", "SMH", "RIVN", "XLK",
    "GM", "SOFI", "QCOM", "MARA", "MA", "AAPL",
}

PUTSELLER_PROVEN = {
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AMD", "CRM", "AVGO", "NFLX", "JPM", "V",
    "MA", "HD", "UNH", "PG", "COST",
}

CALLBUYER_PROVEN = {
    "NVDA", "TSLA", "AMD", "META", "AMZN", "GOOGL", "NFLX", "AVGO",
    "CRM", "SHOP", "SQ", "COIN", "QQQ", "SPY", "IWM", "JPM", "GS", "XOM",
}

# Full SEED_UNIVERSE from each scanner (deduped)
ALPACABOT_SEED = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN",
    "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SNPS", "CDNS", "NXPI",
    "ADI", "ON", "MCHP", "FTNT", "PANW", "CRWD", "ZS", "NET",
    "DDOG", "SNOW", "PLTR", "UBER", "ABNB", "DASH", "COIN", "SQ",
    "PYPL", "SHOP", "MELI", "SE", "RBLX", "U", "TTWO", "EA",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC",
    "SCHW", "BLK", "AXP", "COF", "DFS", "V", "MA", "FIS", "FISV",
    "ICE", "CME", "NDAQ", "SPGI", "MCO",
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    "DHR", "BMY", "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT",
    "SYK", "BSX", "EW", "ZTS", "DXCM", "ILMN", "MRNA", "CVS",
    "WMT", "COST", "HD", "LOW", "TGT", "TJX", "ROST",
    "DG", "DLTR", "NKE", "LULU", "SBUX", "MCD", "YUM", "CMG",
    "DPZ", "WYNN", "MGM", "LVS", "MAR", "HLT", "BKNG",
    "PG", "KO", "PEP", "MDLZ", "CL", "KMB", "GIS", "HSY",
    "CAT", "DE", "GE", "HON", "MMM", "BA", "LMT", "RTX", "NOC",
    "GD", "TDG", "ITW", "EMR", "ROK", "CARR", "OTIS", "UPS",
    "FDX", "CSX", "UNP", "NSC", "DAL", "UAL", "AAL", "LUV",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
    "OXY", "DVN", "HAL", "BKR", "FANG", "HES", "PXD",
    "NFLX", "DIS", "CMCSA", "CHTR", "TMUS", "VZ", "T",
    "PARA", "WBD", "SPOT", "ROKU",
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "GOLD",
    "AMT", "PLD", "EQIX", "SPG", "O", "DLR", "NEE", "DUK", "SO",
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
    "SMH", "ARKK", "GLD", "SLV", "TLT", "HYG", "EEM", "EFA",
    "XBI", "KWEB", "SOXX",
    "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM", "SOFI",
    "HOOD", "MARA", "RIOT", "BITF", "CLSK", "HUT",
    "SMCI", "ARM", "IONQ", "RGTI", "QUBT",
    "CELH", "ENPH", "SEDG", "FSLR", "RUN",
    "AI", "AFRM", "UPST", "OPEN", "RDFN", "Z", "ZG",
    "DKNG", "PENN", "CHWY", "W", "ETSY",
    "PATH", "CFLT", "MNDY", "GLBE", "BILL", "HUBS",
    "ZI", "PCOR", "TOST", "BRZE", "DOCN",
    "OKTA", "ESTC", "MDB", "TEAM", "NOW",
    "WDAY", "VEEV", "CPNG", "GRAB", "DUOL",
    "APP", "TTD", "PINS", "SNAP", "MTCH",
    "LYFT", "BMBL", "CVNA", "CPRT", "MNST",
    "NTES", "BIDU", "PDD", "JD", "BABA",
    "TSM", "ASML", "LSCC", "WOLF",
    "ACHR", "JOBY", "LILM", "EVTL", "BLDE",
]

PUTSELLER_SEED = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN",
    "MU", "AMAT", "LRCX", "MRVL", "SNPS", "CDNS", "NOW", "WDAY",
    "PANW", "CRWD", "FTNT", "NFLX",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC",
    "SCHW", "BLK", "AXP", "COF", "DFS", "V", "MA",
    "FIS", "FISV", "ICE", "CME", "SPGI", "MCO",
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    "DHR", "BMY", "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT",
    "SYK", "BSX", "EW", "ZTS", "CVS", "CI", "HUM", "MCK",
    "WMT", "COST", "PG", "KO", "PEP", "MDLZ", "CL", "KMB",
    "GIS", "K", "HSY", "MO", "PM", "STZ", "TAP", "SJM",
    "HD", "LOW", "TGT", "TJX", "ROST", "NKE", "LULU", "SBUX",
    "MCD", "YUM", "CMG", "DPZ", "MAR", "HLT", "BKNG",
    "CAT", "DE", "GE", "HON", "BA", "LMT", "RTX", "NOC",
    "GD", "ITW", "EMR", "UPS", "FDX", "CSX", "UNP", "NSC",
    "WM", "RSG", "CARR", "OTIS",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
    "OXY", "DVN", "HAL", "BKR", "FANG",
    "DIS", "CMCSA", "CHTR", "TMUS", "VZ", "T",
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX",
    "AMT", "PLD", "EQIX", "SPG", "O", "DLR",
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL",
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
    "XLU", "XLP", "XLI", "XLB", "XLRE", "XLC", "XLY",
    "SMH", "GLD", "SLV", "TLT", "HYG", "EEM", "EFA",
    "XBI", "IBB", "SOXX",
    "F", "GM", "RIVN", "LCID", "PLTR", "COIN", "SQ", "PYPL",
    "SHOP", "UBER", "ABNB", "DASH", "SOFI", "HOOD",
    "AI", "SMCI", "ARM", "MARA", "RIOT", "DKNG",
    "SNAP", "PINS", "ROKU", "SPOT", "RBLX",
    "ENPH", "FSLR", "RUN", "SEDG",
    "MRNA", "BNTX",
    "CVNA", "W", "CHWY", "ETSY",
]

CALLBUYER_SEED = [
    "NVDA", "TSLA", "AMD", "META", "AMZN", "GOOGL", "NFLX", "AVGO",
    "CRM", "SHOP", "SQ", "COIN", "SMCI", "ARM", "MRVL", "MU",
    "AMAT", "LRCX", "QCOM", "INTC", "ADBE", "ORCL", "NOW", "WDAY",
    "PLTR", "AI", "IONQ", "RGTI", "QUBT", "SNOW", "DDOG", "CRWD",
    "PANW", "ZS", "NET", "CFLT", "MDB", "ESTC", "S", "OKTA",
    "UBER", "ABNB", "DASH", "SOFI", "HOOD", "AFRM", "UPST",
    "CVNA", "CPNG", "GRAB", "DUOL", "APP", "TTD", "RBLX",
    "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM",
    "ENPH", "FSLR", "RUN", "SEDG", "PLUG", "BE",
    "MARA", "RIOT", "BITF", "CLSK", "HUT",
    "ACHR", "JOBY", "LILM", "EVTL",
    "MRNA", "BNTX", "CRSP", "BEAM", "EDIT", "NTLA",
    "XBI",
    "SNAP", "PINS", "MTCH", "BMBL", "ROKU", "SPOT",
    "W", "CHWY", "ETSY", "DKNG", "PENN", "CPRT", "MNST",
    "BABA", "JD", "PDD", "BIDU", "NTES", "KWEB",
    "TSM", "ASML", "SOXX",
    "JPM", "GS", "SCHW", "MS", "COF",
    "XOM", "CVX", "COP", "OXY", "SLB", "DVN", "FANG",
    "XLE",
    "QQQ", "SPY", "IWM", "SMH", "ARKK", "XLK",
    "LYFT", "OPEN", "RDFN", "Z", "CELH",
    "SE", "MELI", "PYPL", "LULU", "NKE",
    "BA", "CAT", "DE",
    "AAPL", "MSFT", "V", "MA", "HD", "UNH", "LLY",
    "COST", "WMT", "PG", "DIS", "CMCSA",
]

BOT_CONFIGS = {
    "alpacabot": {"seed": ALPACABOT_SEED, "proven": ALPACABOT_PROVEN, "label": "AlpacaBot (Scalp)"},
    "putseller": {"seed": PUTSELLER_SEED, "proven": PUTSELLER_PROVEN, "label": "PutSeller (Spreads)"},
    "callbuyer": {"seed": CALLBUYER_SEED, "proven": CALLBUYER_PROVEN, "label": "CallBuyer (Momentum)"},
}


def compute_metrics(closes: List[float], volumes: List[float]) -> Dict:
    """Compute Sharpe, return, drawdown, volatility from daily closes."""
    if len(closes) < 30:
        return {"data_points": len(closes), "total_return_pct": 0, "max_drawdown_pct": 0,
                "sharpe": 0, "annual_vol_pct": 0, "avg_daily_volume": 0, "grade": "F"}

    daily_returns = [(closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes))]
    mean_ret = statistics.mean(daily_returns)
    stdev = statistics.pstdev(daily_returns) if len(daily_returns) > 1 else 0.001

    total_return = (closes[-1] / closes[0] - 1.0) * 100.0

    peak = closes[0]
    max_dd = 0.0
    for p in closes:
        if p > peak:
            peak = p
        dd = (p / peak) - 1.0
        if dd < max_dd:
            max_dd = dd

    sharpe = (mean_ret / stdev) * (252 ** 0.5) if stdev > 0 else 0
    annual_vol = stdev * (252 ** 0.5) * 100.0
    avg_vol = statistics.mean(volumes) if volumes else 0

    # Grade: A/B/C/D/F based on composite quality
    grade = "F"
    if sharpe >= 1.0 and max_dd > -25 and total_return > 10:
        grade = "A"
    elif sharpe >= 0.5 and max_dd > -35 and total_return > 0:
        grade = "B"
    elif sharpe >= 0.0 and max_dd > -50:
        grade = "C"
    elif sharpe >= -0.5 and max_dd > -60:
        grade = "D"

    return {
        "data_points": len(closes),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "annual_vol_pct": round(annual_vol, 2),
        "avg_daily_volume": round(avg_vol, 0),
        "grade": grade,
    }


def download_and_score(client, symbols: List[str], days: int = 365) -> Dict[str, Dict]:
    """Download daily bars and compute metrics for each symbol."""
    end = datetime.now()
    start = end - timedelta(days=days)
    results = {}
    total = len(symbols)

    # Batch download in groups of 50 (Alpaca limit)
    batch_size = 50
    for batch_start in range(0, total, batch_size):
        batch = symbols[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} symbols...", end=" ", flush=True)

        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = client.get_stock_bars(req)

            fetched = 0
            for sym in batch:
                bar_list = bars.data.get(sym, [])
                if len(bar_list) < 30:
                    results[sym] = {"data_points": len(bar_list), "grade": "F",
                                    "error": f"insufficient data ({len(bar_list)} bars)"}
                    continue

                closes = [float(b.close) for b in bar_list]
                volumes = [float(b.volume) for b in bar_list]
                results[sym] = compute_metrics(closes, volumes)
                fetched += 1

            print(f"{fetched} OK")

        except Exception as e:
            print(f"ERROR: {e}")
            for sym in batch:
                if sym not in results:
                    results[sym] = {"grade": "F", "error": str(e)}

        time.sleep(0.3)  # Rate limiting

    return results


def run_bot_backtest(client, bot_name: str, days: int):
    """Run backtest for one bot's symbol universe."""
    cfg = BOT_CONFIGS[bot_name]
    seed = list(dict.fromkeys(cfg["seed"]))  # dedup
    proven = cfg["proven"]
    label = cfg["label"]

    # Only test NEW symbols (not in proven set)
    new_symbols = [s for s in seed if s not in proven]

    print(f"\n{'=' * 70}")
    print(f"  {label} — BACKTESTING {len(new_symbols)} NEW SYMBOLS")
    print(f"  (Proven: {len(proven)} | Seed total: {len(seed)} | New: {len(new_symbols)})")
    print(f"  Period: {days} days of daily bars")
    print(f"{'=' * 70}")

    if not new_symbols:
        print("  No new symbols to test!")
        return {}

    results = download_and_score(client, new_symbols, days)

    # Categorize
    grades = {"A": [], "B": [], "C": [], "D": [], "F": []}
    for sym, metrics in sorted(results.items()):
        g = metrics.get("grade", "F")
        grades[g].append((sym, metrics))

    # Print results
    print(f"\n{'─' * 70}")
    print(f"  RESULTS: {label}")
    print(f"{'─' * 70}")
    for grade_label in ["A", "B", "C", "D", "F"]:
        items = grades[grade_label]
        if not items:
            continue

        grade_desc = {
            "A": "EXCELLENT — safe to trade",
            "B": "GOOD — likely profitable",
            "C": "MEDIOCRE — proceed with caution",
            "D": "POOR — risky, probably not worth it",
            "F": "FAIL — do NOT trade",
        }[grade_label]

        print(f"\n  Grade {grade_label} ({len(items)}): {grade_desc}")
        print(f"  {'Symbol':<8} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'Vol%':>8} {'AvgVol':>12}")
        print(f"  {'─' * 56}")

        # Sort within grade by Sharpe
        items.sort(key=lambda x: x[1].get("sharpe", -99), reverse=True)
        for sym, m in items:
            if "error" in m:
                print(f"  {sym:<8} ERROR: {m['error']}")
            else:
                print(f"  {sym:<8} {m['total_return_pct']:>+7.1f}% {m['max_drawdown_pct']:>+7.1f}% "
                      f"{m['sharpe']:>+7.3f} {m['annual_vol_pct']:>7.1f}% {m['avg_daily_volume']:>11,.0f}")

    # Summary
    total_new = len(results)
    a_count = len(grades["A"])
    b_count = len(grades["B"])
    c_count = len(grades["C"])
    d_count = len(grades["D"])
    f_count = len(grades["F"])
    tradeable = a_count + b_count
    questionable = c_count + d_count

    print(f"\n  {'═' * 56}")
    print(f"  SUMMARY: {total_new} new symbols tested")
    print(f"    ✓ Grade A+B (trade):    {tradeable:>4} ({tradeable/total_new*100:.0f}%)")
    print(f"    ? Grade C+D (caution):  {questionable:>4} ({questionable/total_new*100:.0f}%)")
    print(f"    ✗ Grade F (block):      {f_count:>4} ({f_count/total_new*100:.0f}%)")
    print(f"  {'═' * 56}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest new stock candidates")
    parser.add_argument("--bot", choices=["alpacabot", "putseller", "callbuyer", "all"],
                        default="all", help="Which bot to test (default: all)")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    args = parser.parse_args()

    config = Config()
    client = StockHistoricalDataClient(
        api_key=config.API_KEY,
        secret_key=config.API_SECRET,
    )

    all_results = {}
    bots = ["alpacabot", "putseller", "callbuyer"] if args.bot == "all" else [args.bot]

    for bot in bots:
        all_results[bot] = run_bot_backtest(client, bot, args.days)

    # Save full results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "state", "new_stock_backtest_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "days": args.days,
        "results": {bot: {sym: m for sym, m in res.items()} for bot, res in all_results.items()},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nFull results saved: {output_path}")

    # Grand total
    if len(bots) > 1:
        print(f"\n{'═' * 70}")
        print(f"  GRAND TOTAL ACROSS ALL BOTS")
        print(f"{'═' * 70}")
        for bot in bots:
            res = all_results[bot]
            label = BOT_CONFIGS[bot]["label"]
            total = len(res)
            if total == 0:
                continue
            a_b = sum(1 for m in res.values() if m.get("grade") in ("A", "B"))
            f_count = sum(1 for m in res.values() if m.get("grade") == "F")
            print(f"  {label:<30} {total:>4} tested | {a_b:>4} tradeable | {f_count:>4} blocked")


if __name__ == "__main__":
    main()
