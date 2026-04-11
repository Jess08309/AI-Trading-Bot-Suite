"""
Stock Qualification Backtester — Multi-Timeframe + Options-Relevant Metrics
=============================================================================
Replaces the single-period backtest with a robust multi-timeframe grading
system that avoids overfitting to a single lookback window.

Features:
  1. Multi-timeframe: 3m, 6m, 12m with weighted composite scoring
  2. Options-relevant metrics: ATR%, volume consistency, realized vol regime
  3. Produces a grades JSON that scanners load dynamically
  4. Monthly auto-re-evaluation: scanners detect stale grades and warn

Grading methodology:
  - Each timeframe gets its own sub-grade
  - Final grade = weighted composite (12m: 40%, 6m: 35%, 3m: 25%)
  - Stocks that are only good in one timeframe get penalized (regime-dependent)
  - Options-relevant penalties: low volume consistency, extreme vol, low ATR

Usage:
  python tools/backtest_qualified.py               # All 3 bots
  python tools/backtest_qualified.py --bot alpacabot
  python tools/backtest_qualified.py --bot putseller
  python tools/backtest_qualified.py --bot callbuyer
"""
import sys, os, json, time, argparse, statistics, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional
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

# Full SEED_UNIVERSE lists (imported from each scanner)
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

# ═══════════════════════════════════════════════════════════
#  TIMEFRAME WEIGHTS — multi-period composite prevents overfitting
# ═══════════════════════════════════════════════════════════

TIMEFRAME_WINDOWS = [
    {"label": "3m",  "days": 63,  "weight": 0.25},  # Recent trend
    {"label": "6m",  "days": 126, "weight": 0.35},  # Medium-term
    {"label": "12m", "days": 252, "weight": 0.40},  # Full cycle
]


# ═══════════════════════════════════════════════════════════
#  METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════

def compute_metrics(closes: List[float], highs: List[float],
                    lows: List[float], volumes: List[float]) -> Dict:
    """
    Compute comprehensive metrics including options-relevant ones.
    
    Returns dict with:
      - Sharpe ratio
      - Total return %
      - Max drawdown %
      - Annual volatility %
      - Average daily volume
      - ATR% (average true range as % of price — options-relevant)
      - Volume consistency (coefficient of variation — lower = more reliable)
      - Realized vol regime (current 20d vol vs 60d vol — vol expansion/contraction)
      - Sub-grade for this timeframe
    """
    if len(closes) < 20:
        return {"data_points": len(closes), "grade": "F", "score": 0.0,
                "error": f"insufficient data ({len(closes)} bars)"}

    n = len(closes)
    daily_returns = [(closes[i] / closes[i - 1]) - 1.0 for i in range(1, n)]
    mean_ret = statistics.mean(daily_returns)
    stdev = statistics.pstdev(daily_returns) if len(daily_returns) > 1 else 0.001

    total_return = (closes[-1] / closes[0] - 1.0) * 100.0

    # Max drawdown
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

    # ── Options-Relevant Metrics ──

    # ATR% — Average True Range as % of closing price
    # Higher ATR% = bigger daily moves = more options premium opportunity
    atr_values = []
    for i in range(1, min(n, len(highs), len(lows))):
        tr = max(
            highs[i] - lows[i],                    # High-Low
            abs(highs[i] - closes[i - 1]),          # High-PrevClose
            abs(lows[i] - closes[i - 1]),           # Low-PrevClose
        )
        atr_values.append(tr)
    if atr_values and closes[-1] > 0:
        atr_14 = statistics.mean(atr_values[-14:]) if len(atr_values) >= 14 else statistics.mean(atr_values)
        atr_pct = (atr_14 / closes[-1]) * 100.0
    else:
        atr_pct = 0.0

    # Volume consistency — coefficient of variation
    # Lower CV = more predictable volume = better fills
    if volumes and avg_vol > 0:
        vol_stdev = statistics.pstdev(volumes) if len(volumes) > 1 else 0
        volume_cv = vol_stdev / avg_vol  # 0-1+, lower = more consistent
    else:
        volume_cv = 999.0

    # Realized vol regime — is vol expanding or contracting?
    # Ratio of recent 20d vol to longer 60d vol
    # > 1.0 = vol expanding (riskier), < 1.0 = vol contracting (calmer)
    if len(daily_returns) >= 60:
        vol_20d = statistics.pstdev(daily_returns[-20:]) * (252 ** 0.5) * 100.0
        vol_60d = statistics.pstdev(daily_returns[-60:]) * (252 ** 0.5) * 100.0
        vol_regime = vol_20d / vol_60d if vol_60d > 0 else 1.0
    elif len(daily_returns) >= 20:
        vol_regime = 1.0  # Not enough data for comparison
    else:
        vol_regime = 1.0

    # ── Composite Score (0-100) ──
    # This replaces simple grade thresholds with a continuous score
    score = 0.0
    
    # Sharpe contribution (0-35 points)
    sharpe_score = max(0, min(35, (sharpe + 1.0) * 17.5))
    score += sharpe_score
    
    # Return contribution (0-20 points)
    ret_score = max(0, min(20, (total_return + 20) * 0.4))
    score += ret_score
    
    # Drawdown contribution (0-20 points) — less drawdown = more points
    dd_score = max(0, min(20, (max_dd * 100.0 + 60) * 0.333))
    score += dd_score
    
    # ATR% contribution (0-10 points) — moderate ATR is best for options
    # Sweet spot: 1.5-4.0% ATR — enough movement but not insane
    if 1.5 <= atr_pct <= 4.0:
        atr_score = 10.0
    elif 0.5 <= atr_pct < 1.5 or 4.0 < atr_pct <= 6.0:
        atr_score = 6.0
    elif atr_pct > 6.0:
        atr_score = 3.0  # Too volatile for reliable options plays
    else:
        atr_score = 2.0  # Too flat
    score += atr_score
    
    # Volume consistency (0-10 points) — lower CV = better
    if volume_cv < 0.5:
        vcv_score = 10.0
    elif volume_cv < 1.0:
        vcv_score = 7.0
    elif volume_cv < 1.5:
        vcv_score = 4.0
    else:
        vcv_score = 1.0
    score += vcv_score
    
    # Vol regime penalty (0-5 points) — penalize expanding vol
    if vol_regime < 1.2:
        regime_score = 5.0
    elif vol_regime < 1.5:
        regime_score = 3.0
    else:
        regime_score = 0.0  # Vol is expanding rapidly — dangerous
    score += regime_score

    # Grade based on composite score
    if score >= 70:
        grade = "A"
    elif score >= 55:
        grade = "B"
    elif score >= 40:
        grade = "C"
    elif score >= 25:
        grade = "D"
    else:
        grade = "F"

    return {
        "data_points": n,
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "annual_vol_pct": round(annual_vol, 2),
        "avg_daily_volume": round(avg_vol, 0),
        "atr_pct": round(atr_pct, 3),
        "volume_cv": round(volume_cv, 3),
        "vol_regime": round(vol_regime, 3),
        "composite_score": round(score, 1),
        "grade": grade,
    }


def download_bars(client, symbols: List[str], days: int) -> Dict[str, Dict]:
    """Download daily bars from Alpaca. Returns {symbol: {closes, highs, lows, volumes}}."""
    end = datetime.now()
    start = end - timedelta(days=days + 10)  # Buffer for weekends/holidays
    all_bars = {}
    batch_size = 50

    for batch_start in range(0, len(symbols), batch_size):
        batch = symbols[batch_start:batch_start + batch_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = client.get_stock_bars(req)
            for sym in batch:
                bar_list = bars.data.get(sym, [])
                if bar_list:
                    all_bars[sym] = {
                        "closes": [float(b.close) for b in bar_list],
                        "highs": [float(b.high) for b in bar_list],
                        "lows": [float(b.low) for b in bar_list],
                        "volumes": [float(b.volume) for b in bar_list],
                    }
        except Exception as e:
            print(f"  !! Batch error: {e}")
        time.sleep(0.3)

    return all_bars


def score_symbol_multi_timeframe(bar_data: Dict) -> Dict:
    """
    Score a single symbol across 3m/6m/12m timeframes.
    Returns composite grade + per-timeframe breakdown.
    """
    closes = bar_data["closes"]
    highs = bar_data["highs"]
    lows = bar_data["lows"]
    volumes = bar_data["volumes"]
    
    n = len(closes)
    timeframe_results = {}
    weighted_score = 0.0
    total_weight = 0.0
    grades_seen = []

    for tf in TIMEFRAME_WINDOWS:
        label = tf["label"]
        window = tf["days"]
        weight = tf["weight"]

        if n < window:
            # Not enough data for this window — use what we have
            # but reduce weight proportionally
            actual_days = n
            weight *= (actual_days / window)  # Discount for shorter data
            c = closes
            h = highs
            l = lows
            v = volumes
        else:
            # Use last N bars for this window
            c = closes[-window:]
            h = highs[-window:]
            l = lows[-window:]
            v = volumes[-window:]

        metrics = compute_metrics(c, h, l, v)
        timeframe_results[label] = metrics
        weighted_score += metrics.get("composite_score", 0) * weight
        total_weight += weight
        grades_seen.append(metrics.get("grade", "F"))

    # Normalize weighted score
    if total_weight > 0:
        final_score = weighted_score / total_weight
    else:
        final_score = 0.0

    # ── Cross-timeframe consistency penalty ──
    # If grades differ wildly across timeframes, penalize (regime-dependent)
    unique_grades = set(grades_seen)
    if len(unique_grades) >= 3:
        # Very inconsistent across timeframes — penalize
        final_score *= 0.85
    elif "F" in grades_seen and ("A" in grades_seen or "B" in grades_seen):
        # Mixed signals — penalize
        final_score *= 0.90

    # Final composite grade
    if final_score >= 70:
        final_grade = "A"
    elif final_score >= 55:
        final_grade = "B"
    elif final_score >= 40:
        final_grade = "C"
    elif final_score >= 25:
        final_grade = "D"
    else:
        final_grade = "F"

    # Use the 12m metrics for the primary display numbers (if available)
    primary = timeframe_results.get("12m", timeframe_results.get("6m", timeframe_results.get("3m", {})))

    return {
        "composite_score": round(final_score, 1),
        "grade": final_grade,
        "timeframes": timeframe_results,
        "consistency": "high" if len(unique_grades) <= 1 else ("medium" if len(unique_grades) == 2 else "low"),
        # Surface key metrics from primary window
        "total_return_pct": primary.get("total_return_pct", 0),
        "max_drawdown_pct": primary.get("max_drawdown_pct", 0),
        "sharpe": primary.get("sharpe", 0),
        "annual_vol_pct": primary.get("annual_vol_pct", 0),
        "avg_daily_volume": primary.get("avg_daily_volume", 0),
        "atr_pct": primary.get("atr_pct", 0),
        "volume_cv": primary.get("volume_cv", 0),
        "vol_regime": primary.get("vol_regime", 0),
        "data_points": len(closes),
    }


def run_bot_backtest(client, bot_name: str) -> Dict[str, Dict]:
    """Run multi-timeframe backtest for one bot's symbol universe."""
    cfg = BOT_CONFIGS[bot_name]
    seed = list(dict.fromkeys(cfg["seed"]))
    proven = cfg["proven"]
    label = cfg["label"]

    new_symbols = [s for s in seed if s not in proven]

    print(f"\n{'=' * 70}")
    print(f"  {label} — MULTI-TIMEFRAME BACKTEST ({len(new_symbols)} new symbols)")
    print(f"  Windows: {', '.join(tf['label'] + '(' + str(int(tf['weight']*100)) + '%)' for tf in TIMEFRAME_WINDOWS)}")
    print(f"  Metrics: Sharpe + Return + Drawdown + ATR% + VolCV + VolRegime")
    print(f"{'=' * 70}")

    if not new_symbols:
        print("  No new symbols to test!")
        return {}

    # Download 12 months of bars (covers all windows)
    print(f"\n  Downloading bars for {len(new_symbols)} symbols...")
    bar_data = download_bars(client, new_symbols, days=365)
    print(f"  Got data for {len(bar_data)}/{len(new_symbols)} symbols")

    # Score each symbol across all timeframes
    results = {}
    for sym in new_symbols:
        if sym not in bar_data:
            results[sym] = {"grade": "F", "composite_score": 0, "error": "no data"}
            continue
        results[sym] = score_symbol_multi_timeframe(bar_data[sym])

    # Categorize
    grades = {"A": [], "B": [], "C": [], "D": [], "F": []}
    for sym, m in sorted(results.items()):
        g = m.get("grade", "F")
        grades[g].append((sym, m))

    # Print results
    print(f"\n{'─' * 78}")
    print(f"  RESULTS: {label} (Multi-Timeframe)")
    print(f"{'─' * 78}")
    for grade_label in ["A", "B", "C", "D", "F"]:
        items = grades[grade_label]
        if not items:
            continue

        grade_desc = {
            "A": "QUALIFIED — consistent strong performer",
            "B": "QUALIFIED — solid across timeframes",
            "C": "PROBATION — mediocre, needs time gate",
            "D": "PROBATION — weak, risky",
            "F": "BLOCKED — do NOT trade",
        }[grade_label]

        print(f"\n  Grade {grade_label} ({len(items)}): {grade_desc}")
        print(f"  {'Sym':<6} {'Score':>5} {'Ret%':>7} {'MaxDD':>7} {'Shrpe':>6} {'ATR%':>6} {'VolCV':>5} {'VReg':>5} {'Consist':>8} {'3m':>3} {'6m':>3} {'12m':>3}")
        print(f"  {'─' * 75}")

        items.sort(key=lambda x: x[1].get("composite_score", 0), reverse=True)
        for sym, m in items:
            if "error" in m and "timeframes" not in m:
                print(f"  {sym:<6} ERROR: {m.get('error', 'unknown')}")
                continue
            tfs = m.get("timeframes", {})
            g3 = tfs.get("3m", {}).get("grade", "-")
            g6 = tfs.get("6m", {}).get("grade", "-")
            g12 = tfs.get("12m", {}).get("grade", "-")
            print(f"  {sym:<6} {m.get('composite_score', 0):>5.1f} {m.get('total_return_pct', 0):>+6.1f}% "
                  f"{m.get('max_drawdown_pct', 0):>+6.1f}% {m.get('sharpe', 0):>+5.2f} "
                  f"{m.get('atr_pct', 0):>5.2f} {m.get('volume_cv', 0):>5.2f} "
                  f"{m.get('vol_regime', 0):>5.2f} {m.get('consistency', '?'):>8} "
                  f" {g3:>2}  {g6:>2}  {g12:>2}")

    # Summary
    total = len(results)
    a_ct = len(grades["A"])
    b_ct = len(grades["B"])
    c_ct = len(grades["C"])
    d_ct = len(grades["D"])
    f_ct = len(grades["F"])
    tradeable = a_ct + b_ct

    print(f"\n  {'═' * 60}")
    print(f"  SUMMARY: {total} symbols scored (multi-timeframe composite)")
    print(f"    [A] Strong:     {a_ct:>4}")
    print(f"    [B] Solid:      {b_ct:>4}")
    print(f"    [C] Mediocre:   {c_ct:>4}")
    print(f"    [D] Weak:       {d_ct:>4}")
    print(f"    [F] Blocked:    {f_ct:>4}")
    print(f"    ─────────────────────")
    print(f"    Qualified (A+B): {tradeable:>3} ({tradeable/total*100:.0f}%)")
    print(f"    Blocked (F):     {f_ct:>3} ({f_ct/total*100:.0f}%)")
    print(f"  {'═' * 60}")

    return results


def generate_grades_file(all_results: Dict[str, Dict[str, Dict]], output_dir: str):
    """
    Generate the grades JSON file that scanners load dynamically.
    
    File format:
    {
        "generated": "2026-03-08T...",
        "version": 2,
        "method": "multi_timeframe_composite",
        "windows": ["3m(25%)", "6m(35%)", "12m(40%)"],
        "bots": {
            "alpacabot": {
                "qualified": ["SYM1", "SYM2", ...],
                "blocked": ["SYM3", ...],
                "details": { "SYM1": {...}, ... }
            },
            ...
        }
    }
    """
    output = {
        "generated": datetime.now().isoformat(),
        "version": 2,
        "method": "multi_timeframe_composite",
        "windows": [f"{tf['label']}({int(tf['weight']*100)}%)" for tf in TIMEFRAME_WINDOWS],
        "metrics": ["sharpe", "return", "drawdown", "atr_pct", "volume_cv", "vol_regime"],
        "bots": {},
    }

    for bot_name, results in all_results.items():
        qualified = sorted([s for s, m in results.items() if m.get("grade") in ("A", "B")])
        blocked = sorted([s for s, m in results.items() if m.get("grade") == "F"])

        output["bots"][bot_name] = {
            "qualified": qualified,
            "blocked": blocked,
            "stats": {
                "total_scored": len(results),
                "grade_A": sum(1 for m in results.values() if m.get("grade") == "A"),
                "grade_B": sum(1 for m in results.values() if m.get("grade") == "B"),
                "grade_C": sum(1 for m in results.values() if m.get("grade") == "C"),
                "grade_D": sum(1 for m in results.values() if m.get("grade") == "D"),
                "grade_F": sum(1 for m in results.values() if m.get("grade") == "F"),
            },
            "details": {
                sym: {
                    "grade": m.get("grade", "F"),
                    "score": m.get("composite_score", 0),
                    "consistency": m.get("consistency", "?"),
                    "sharpe": m.get("sharpe", 0),
                    "return_pct": m.get("total_return_pct", 0),
                    "max_dd_pct": m.get("max_drawdown_pct", 0),
                    "atr_pct": m.get("atr_pct", 0),
                    "volume_cv": m.get("volume_cv", 0),
                    "vol_regime": m.get("vol_regime", 0),
                    "avg_volume": m.get("avg_daily_volume", 0),
                }
                for sym, m in sorted(results.items())
            },
        }

    # Write to shared location — all bots read from here
    grades_path = os.path.join(output_dir, "backtest_grades.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(grades_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Grades file written: {grades_path}")
    print(f"  Generated: {output['generated']}")
    print(f"  Method: {output['method']}")
    for bot_name, bot_data in output["bots"].items():
        label = BOT_CONFIGS[bot_name]["label"]
        print(f"  {label}: {len(bot_data['qualified'])} qualified, {len(bot_data['blocked'])} blocked")

    # Also copy to each bot's data/state folder for easy access
    bot_dirs = {
        "alpacabot": r"C:\AlpacaBot\data\state",
        "putseller": r"C:\PutSeller\data\state",
        "callbuyer": r"C:\CallBuyer\data\state",
    }
    for bot_name, bot_dir in bot_dirs.items():
        if bot_name in output["bots"]:
            target = os.path.join(bot_dir, "backtest_grades.json")
            os.makedirs(bot_dir, exist_ok=True)
            with open(target, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  → Copied to {target}")

    return grades_path


def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe stock qualification backtest")
    parser.add_argument("--bot", choices=["alpacabot", "putseller", "callbuyer", "all"],
                        default="all", help="Which bot to test (default: all)")
    args = parser.parse_args()

    config = Config()
    client = StockHistoricalDataClient(
        api_key=config.API_KEY,
        secret_key=config.API_SECRET,
    )

    print("═" * 70)
    print("  STOCK QUALIFICATION BACKTEST v2 — Multi-Timeframe + Options Metrics")
    print(f"  Timeframes: {', '.join(tf['label'] for tf in TIMEFRAME_WINDOWS)}")
    print(f"  Weights: {', '.join(str(int(tf['weight']*100)) + '%' for tf in TIMEFRAME_WINDOWS)}")
    print(f"  Metrics: Sharpe | Return | Drawdown | ATR% | VolCV | VolRegime")
    print("═" * 70)

    bots = ["alpacabot", "putseller", "callbuyer"] if args.bot == "all" else [args.bot]
    all_results = {}

    for bot in bots:
        all_results[bot] = run_bot_backtest(client, bot)

    # Generate grades file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "state")
    generate_grades_file(all_results, output_dir)

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
            f_ct = sum(1 for m in res.values() if m.get("grade") == "F")
            print(f"  {label:<30} {total:>4} scored | {a_b:>4} qualified | {f_ct:>4} blocked")


if __name__ == "__main__":
    main()
