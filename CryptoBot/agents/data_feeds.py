"""External market data feeds for the Multi-Agent Advisor.

Pulls real-time data from free, keyless APIs so GPT agents have fresh
context beyond what CryptoBot's internal indicators provide.

Data sources:
    - Alternative.me  : Fear & Greed Index
    - CoinGecko       : BTC dominance, total market cap, top movers, trending coins
    - Binance Futures  : Funding rates, open interest (CoinGecko fallback for US)
    - Reddit           : r/CryptoCurrency + r/Bitcoin headlines (social sentiment)
    - DeFi Llama      : Total DeFi TVL + protocol flows
    - CoinGecko       : BTC ETF / Grayscale proxy (GBTC premium or BTC vs market)

All responses are cached to respect rate limits (CoinGecko free = ~30 req/min).
Stale data safeguards: every cached entry has a timestamp; summary includes
a data_quality field so agents know when data is fresh vs. stale.
"""
from __future__ import annotations
import time, logging, requests
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("agent_advisor")

# ---------------------------------------------------------------------------
# Cache helper
# ---------------------------------------------------------------------------
_cache: Dict[str, tuple] = {}  # key -> (timestamp, data)

def _cached_get(key: str, url: str, ttl: int = 300, params: dict = None) -> Optional[dict]:
    """GET with in-memory TTL cache.  Returns None on failure."""
    now = time.time()
    if key in _cache and (now - _cache[key][0]) < ttl:
        return _cache[key][1]
    try:
        r = requests.get(url, params=params, timeout=10,
                         headers={"Accept": "application/json",
                                  "User-Agent": "CryptoBotAdvisor/1.0"})
        r.raise_for_status()
        data = r.json()
        _cache[key] = (now, data)
        return data
    except Exception as e:
        logger.warning(f"data_feeds: {key} fetch failed: {e}")
        # Return stale cache if available
        if key in _cache:
            return _cache[key][1]
        return None


# ---------------------------------------------------------------------------
# 1. Fear & Greed Index  (alternative.me — free, no key)
# ---------------------------------------------------------------------------
def get_fear_greed() -> Dict:
    """Returns {'value': int 0-100, 'label': str, 'timestamp': str}"""
    data = _cached_get(
        "fear_greed",
        "https://api.alternative.me/fng/?limit=1",
        ttl=600,  # updates ~daily, cache 10 min
    )
    if not data or "data" not in data:
        return {"value": 50, "label": "Neutral", "timestamp": ""}
    entry = data["data"][0]
    return {
        "value": int(entry.get("value", 50)),
        "label": entry.get("value_classification", "Neutral"),
        "timestamp": entry.get("timestamp", ""),
    }


# ---------------------------------------------------------------------------
# 2. Global crypto market stats  (CoinGecko /global — free, no key)
# ---------------------------------------------------------------------------
def get_global_market() -> Dict:
    """BTC dominance, total market cap, 24h volume, market cap change %."""
    data = _cached_get(
        "global_market",
        "https://api.coingecko.com/api/v3/global",
        ttl=300,
    )
    if not data or "data" not in data:
        return {
            "btc_dominance": 0.0,
            "eth_dominance": 0.0,
            "total_market_cap_usd": 0,
            "total_volume_24h_usd": 0,
            "market_cap_change_24h_pct": 0.0,
            "active_cryptocurrencies": 0,
        }
    d = data["data"]
    return {
        "btc_dominance": round(d.get("market_cap_percentage", {}).get("btc", 0), 1),
        "eth_dominance": round(d.get("market_cap_percentage", {}).get("eth", 0), 1),
        "total_market_cap_usd": int(d.get("total_market_cap", {}).get("usd", 0)),
        "total_volume_24h_usd": int(d.get("total_volume", {}).get("usd", 0)),
        "market_cap_change_24h_pct": round(d.get("market_cap_change_percentage_24h_usd", 0), 2),
        "active_cryptocurrencies": d.get("active_cryptocurrencies", 0),
    }


# ---------------------------------------------------------------------------
# 3. Top movers — biggest gainers/losers  (CoinGecko /coins/markets)
# ---------------------------------------------------------------------------
def get_top_movers(limit: int = 20) -> Dict:
    """Top gainers and losers by 24h change among top coins."""
    data = _cached_get(
        "top_movers",
        "https://api.coingecko.com/api/v3/coins/markets",
        ttl=300,
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": str(limit),
            "page": "1",
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        },
    )
    if not data or not isinstance(data, list):
        return {"gainers": [], "losers": [], "breadth_pct_positive": 50.0}

    coins = []
    for c in data:
        coins.append({
            "symbol": c.get("symbol", "").upper(),
            "name": c.get("name", ""),
            "price": c.get("current_price", 0),
            "change_1h": c.get("price_change_percentage_1h_in_currency", 0) or 0,
            "change_24h": c.get("price_change_percentage_24h", 0) or 0,
            "change_7d": c.get("price_change_percentage_7d_in_currency", 0) or 0,
            "volume_24h": c.get("total_volume", 0) or 0,
            "market_cap": c.get("market_cap", 0) or 0,
        })

    positive = sum(1 for c in coins if c["change_24h"] > 0)
    breadth = round((positive / len(coins)) * 100, 1) if coins else 50.0

    sorted_by_change = sorted(coins, key=lambda c: c["change_24h"], reverse=True)
    return {
        "gainers": sorted_by_change[:5],
        "losers": sorted_by_change[-5:][::-1],  # worst first
        "breadth_pct_positive": breadth,
    }


# ---------------------------------------------------------------------------
# 4. Binance Futures funding rates  (free, no key)
#    Note: Uses global endpoint. Falls back to CoinGlass if Binance geo-blocks.
# ---------------------------------------------------------------------------
_FUNDING_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT",
                    "MATICUSDT", "LTCUSDT", "AVAXUSDT", "LINKUSDT", "XRPUSDT"]

def get_funding_rates() -> Dict[str, float]:
    """Current funding rates from Binance Futures.  Positive = longs pay shorts."""
    # Try Binance first
    data = _cached_get(
        "funding_rates",
        "https://fapi.binance.com/fapi/v1/premiumIndex",
        ttl=300,
    )
    if data and isinstance(data, list):
        rates = {}
        for item in data:
            sym = item.get("symbol", "")
            if sym in _FUNDING_SYMBOLS:
                rate = float(item.get("lastFundingRate", 0))
                rates[sym] = round(rate * 100, 4)  # convert to percentage
        if rates:
            return rates

    # Fallback: CoinGecko derivatives for BTC/ETH funding
    data = _cached_get(
        "funding_rates_fallback",
        "https://api.coingecko.com/api/v3/derivatives",
        ttl=600,
        params={"per_page": "20"},
    )
    if data and isinstance(data, list):
        rates = {}
        for item in data:
            sym = item.get("symbol", "").upper()
            fr = item.get("funding_rate")
            if fr is not None:
                for target in _FUNDING_SYMBOLS:
                    base = target.replace("USDT", "")
                    if base in sym:
                        rates[target] = round(float(fr), 4)
                        break
        return rates

    return {}


# ---------------------------------------------------------------------------
# 5. Open interest — skip if Binance geo-blocked (451), use CoinGecko fallback
# ---------------------------------------------------------------------------
def get_open_interest() -> Dict[str, Dict]:
    """Open interest for major pairs — shows leveraged positioning."""
    results = {}
    # Try Binance first for each symbol
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        data = _cached_get(
            f"oi_{sym}",
            "https://fapi.binance.com/fapi/v1/openInterest",
            ttl=300,
            params={"symbol": sym},
        )
        if data and "openInterest" in data:
            results[sym] = {
                "open_interest": float(data.get("openInterest", 0)),
                "symbol": sym,
            }
    
    # If Binance failed (geo-block), try CoinGecko derivatives
    if not results:
        data = _cached_get(
            "oi_coingecko",
            "https://api.coingecko.com/api/v3/derivatives/exchanges",
            ttl=600,
            params={"per_page": "5"},
        )
        if data and isinstance(data, list):
            total_oi = sum(float(ex.get("open_interest_btc", 0) or 0) for ex in data[:5])
            if total_oi > 0:
                results["TOTAL"] = {
                    "open_interest_btc": round(total_oi, 2),
                    "symbol": "ALL_EXCHANGES_TOP5",
                }
    
    return results


# ---------------------------------------------------------------------------
# 6. Trending searches on CoinGecko  (free, no key)
# ---------------------------------------------------------------------------
def get_trending() -> List[str]:
    """Top trending coin searches — indicates retail attention."""
    data = _cached_get(
        "trending",
        "https://api.coingecko.com/api/v3/search/trending",
        ttl=600,
    )
    if not data or "coins" not in data:
        return []
    return [c["item"]["symbol"].upper() for c in data["coins"][:7]]


# ---------------------------------------------------------------------------
# 7. Reddit crypto headlines  (free, no key — public JSON endpoint)
# ---------------------------------------------------------------------------
_REDDIT_HEADERS = {
    "User-Agent": "CryptoBotAdvisor/1.0 (trading bot research)",
    "Accept": "application/json",
}

def get_reddit_sentiment() -> Dict:
    """Scrape top headlines from r/CryptoCurrency and r/Bitcoin.
    
    Returns headline titles, upvote ratios, and a simple bull/bear/neutral tally.
    This gives agents real news context — what retail traders are seeing and talking about.
    """
    headlines = []
    
    for sub in ["CryptoCurrency", "Bitcoin"]:
        data = _cached_get(
            f"reddit_{sub}",
            f"https://www.reddit.com/r/{sub}/hot.json",
            ttl=600,  # 10 min cache — Reddit rate limits ~60 req/min for no-auth
            params={"limit": "10"},
        )
        if not data or "data" not in data:
            continue
        
        for post in data["data"].get("children", []):
            pdata = post.get("data", {})
            # Skip stickied/pinned posts (daily threads, rules, etc.)
            if pdata.get("stickied"):
                continue
            title = pdata.get("title", "")
            if not title:
                continue
            score = pdata.get("score", 0)
            upvote_ratio = pdata.get("upvote_ratio", 0.5)
            num_comments = pdata.get("num_comments", 0)
            flair = pdata.get("link_flair_text", "")
            
            headlines.append({
                "title": title[:120],  # truncate long titles
                "score": score,
                "upvote_ratio": upvote_ratio,
                "comments": num_comments,
                "flair": flair or "",
                "subreddit": sub,
            })
    
    # Sort by engagement (score * comment count)
    headlines.sort(key=lambda h: h["score"] * max(h["comments"], 1), reverse=True)
    top_headlines = headlines[:8]  # top 8 most engaged
    
    # Simple sentiment tally based on keywords in titles
    bullish_words = {"bull", "surge", "rally", "soar", "pump", "moon", "ath", "buy",
                     "breakout", "launch", "etf", "adoption", "record", "gains", "bullish"}
    bearish_words = {"bear", "crash", "dump", "plunge", "fear", "sell", "collapse",
                     "recession", "war", "ban", "hack", "scam", "fraud", "die", "pressure",
                     "drop", "bearish", "tank", "liquidat"}
    
    bull_count = 0
    bear_count = 0
    for h in headlines:
        title_lower = h["title"].lower()
        if any(w in title_lower for w in bullish_words):
            bull_count += 1
        if any(w in title_lower for w in bearish_words):
            bear_count += 1
    
    total = max(bull_count + bear_count, 1)
    sentiment_ratio = (bull_count - bear_count) / total  # -1 to +1
    
    return {
        "headlines": top_headlines,
        "bull_count": bull_count,
        "bear_count": bear_count,
        "neutral_count": len(headlines) - bull_count - bear_count,
        "sentiment_ratio": round(sentiment_ratio, 2),
        "total_posts_scanned": len(headlines),
    }


# ---------------------------------------------------------------------------
# 8. DeFi Llama — Total Value Locked  (free, no key)
# ---------------------------------------------------------------------------
def get_defi_tvl() -> Dict:
    """Total DeFi TVL and recent change — shows capital flows in/out of DeFi."""
    # Current TVL
    data = _cached_get(
        "defi_tvl",
        "https://api.llama.fi/v2/historicalChainTvl",
        ttl=1800,  # 30 min — TVL doesn't change fast
    )
    
    current_tvl = 0.0
    tvl_change_1d_pct = 0.0
    
    if data and isinstance(data, list) and len(data) >= 2:
        # Last two data points for change calculation
        current_tvl = data[-1].get("tvl", 0)
        prev_tvl = data[-2].get("tvl", 0)
        if prev_tvl > 0:
            tvl_change_1d_pct = ((current_tvl - prev_tvl) / prev_tvl) * 100
    
    # Top chains by TVL
    chains_data = _cached_get(
        "defi_chains",
        "https://api.llama.fi/v2/chains",
        ttl=1800,
    )
    
    top_chains = []
    if chains_data and isinstance(chains_data, list):
        sorted_chains = sorted(chains_data, key=lambda c: float(c.get("tvl", 0) or 0), reverse=True)
        for c in sorted_chains[:5]:
            top_chains.append({
                "name": c.get("name", ""),
                "tvl_billion": round(float(c.get("tvl", 0) or 0) / 1e9, 2),
            })
    
    return {
        "total_tvl_billion": round(current_tvl / 1e9, 2) if current_tvl else 0,
        "tvl_change_1d_pct": round(tvl_change_1d_pct, 2),
        "top_chains": top_chains,
    }


# ---------------------------------------------------------------------------
# 9. DeFi Llama — Major protocol yields (stablecoin yields = risk appetite proxy)
# ---------------------------------------------------------------------------
def get_stablecoin_yields() -> Dict:
    """Top stablecoin yields — rising yields = risk-on, falling = risk-off."""
    data = _cached_get(
        "defi_yields",
        "https://yields.llama.fi/pools",
        ttl=1800,
        params={"stablecoin": "true"},
    )
    
    if not data or "data" not in data:
        return {"avg_yield": 0.0, "top_pools": []}
    
    pools = data["data"]
    # Filter for meaningful pools (>$10M TVL, known stables)
    stable_symbols = {"USDC", "USDT", "DAI", "FRAX", "BUSD", "TUSD"}
    big_pools = []
    for p in pools:
        sym = (p.get("symbol") or "").upper()
        tvl = float(p.get("tvlUsd", 0) or 0)
        apy = float(p.get("apy", 0) or 0)
        if tvl > 10_000_000 and any(s in sym for s in stable_symbols) and 0 < apy < 100:
            big_pools.append({
                "pool": f"{p.get('project','?')}/{sym}",
                "chain": p.get("chain", "?"),
                "apy": round(apy, 2),
                "tvl_m": round(tvl / 1e6, 1),
            })
    
    big_pools.sort(key=lambda p: p["tvl_m"], reverse=True)
    top_pools = big_pools[:5]
    avg_yield = sum(p["apy"] for p in top_pools) / max(len(top_pools), 1)
    
    return {
        "avg_yield": round(avg_yield, 2),
        "top_pools": top_pools,
    }


# ---------------------------------------------------------------------------
# 10. BTC long-term holder proxy — CoinGecko BTC market data
# ---------------------------------------------------------------------------
def get_btc_metrics() -> Dict:
    """BTC-specific metrics: ATH distance, 24h vol/mcap ratio, circulating supply %."""
    data = _cached_get(
        "btc_detail",
        "https://api.coingecko.com/api/v3/coins/bitcoin",
        ttl=600,
        params={"localization": "false", "tickers": "false",
                "community_data": "false", "developer_data": "false"},
    )
    if not data or "market_data" not in data:
        return {}
    
    md = data["market_data"]
    current = float(md.get("current_price", {}).get("usd", 0) or 0)
    ath = float(md.get("ath", {}).get("usd", 0) or 0)
    ath_change = float(md.get("ath_change_percentage", {}).get("usd", 0) or 0)
    high_24h = float(md.get("high_24h", {}).get("usd", 0) or 0)
    low_24h = float(md.get("low_24h", {}).get("usd", 0) or 0)
    mcap = float(md.get("market_cap", {}).get("usd", 0) or 0)
    vol_24h = float(md.get("total_volume", {}).get("usd", 0) or 0)
    
    vol_mcap_ratio = (vol_24h / mcap * 100) if mcap > 0 else 0
    range_24h = ((high_24h - low_24h) / low_24h * 100) if low_24h > 0 else 0
    
    return {
        "price": current,
        "ath": ath,
        "ath_change_pct": round(ath_change, 1),
        "high_24h": high_24h,
        "low_24h": low_24h,
        "range_24h_pct": round(range_24h, 2),
        "vol_mcap_ratio_pct": round(vol_mcap_ratio, 2),
    }


# ---------------------------------------------------------------------------
# SAFEGUARD: Data quality tracker
# ---------------------------------------------------------------------------
_feed_status: Dict[str, Dict] = {}

def _track_feed(name: str, success: bool, data_age_seconds: float = 0):
    """Track whether each feed is returning fresh data."""
    _feed_status[name] = {
        "last_check": time.time(),
        "success": success,
        "data_age_seconds": data_age_seconds,
    }

def get_data_quality_report() -> Dict:
    """Generate a quality report for all feeds.
    
    Returns a dict with:
    - feeds_ok: count of feeds returning fresh data
    - feeds_stale: count of feeds returning cached/old data  
    - feeds_failed: count of feeds that errored
    - overall: "GOOD", "DEGRADED", or "POOR"
    - warning: human-readable warning if data quality is bad
    """
    now = time.time()
    ok = stale = failed = 0
    stale_feeds = []
    failed_feeds = []
    
    for feed_name, key in [
        ("Fear & Greed", "fear_greed"),
        ("Global Market", "global_market"),
        ("Top Movers", "top_movers"),
        ("Funding Rates", "funding_rates"),
        ("Trending", "trending"),
        ("Reddit", "reddit_CryptoCurrency"),
        ("DeFi TVL", "defi_tvl"),
        ("BTC Metrics", "btc_detail"),
    ]:
        if key in _cache:
            cache_age = now - _cache[key][0]
            if _cache[key][1] is None:
                failed += 1
                failed_feeds.append(feed_name)
            elif cache_age > 1800:  # >30 min old = stale
                stale += 1
                stale_feeds.append(f"{feed_name} ({int(cache_age/60)}m old)")
            else:
                ok += 1
        else:
            failed += 1
            failed_feeds.append(feed_name)
    
    total = ok + stale + failed
    if total == 0:
        return {"overall": "UNKNOWN", "feeds_ok": 0, "warning": "No feeds checked yet"}
    
    if failed >= 3 or (ok / total) < 0.5:
        quality = "POOR"
    elif stale >= 2 or failed >= 1:
        quality = "DEGRADED"
    else:
        quality = "GOOD"
    
    warning = ""
    if quality != "GOOD":
        parts = []
        if failed_feeds:
            parts.append(f"Failed: {', '.join(failed_feeds)}")
        if stale_feeds:
            parts.append(f"Stale: {', '.join(stale_feeds)}")
        warning = "; ".join(parts)
    
    return {
        "overall": quality,
        "feeds_ok": ok,
        "feeds_stale": stale,
        "feeds_failed": failed,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Master fetch — pulls everything in one call
# ---------------------------------------------------------------------------
def fetch_all_market_context() -> Dict:
    """Fetch all external data feeds and return a unified market context dict.
    
    This is the main entry point called by the advisor before each agent run.
    Individual feed failures are silently handled (stale cache or defaults used).
    
    SAFEGUARDS:
    1. Data quality report included — agents see which feeds are fresh vs stale
    2. Over-reliance guard — if data quality is POOR, summary includes a WARNING
       telling agents to weight external data lower
    3. All data is cached with TTLs to avoid hitting rate limits
    4. Every feed has a try/except — one failure doesn't break the whole system
    """
    fear_greed = get_fear_greed()
    global_market = get_global_market()
    top_movers = get_top_movers()
    funding = get_funding_rates()
    oi = get_open_interest()
    trending = get_trending()
    reddit = get_reddit_sentiment()
    defi = get_defi_tvl()
    stablecoin_yields = get_stablecoin_yields()
    btc_metrics = get_btc_metrics()
    data_quality = get_data_quality_report()

    # Build readable summary strings for agent prompts
    fg_val = fear_greed["value"]
    fg_label = fear_greed["label"]

    gainers_str = ", ".join(
        f"{c['symbol']} +{c['change_24h']:.1f}%" for c in top_movers.get("gainers", [])[:3]
    )
    losers_str = ", ".join(
        f"{c['symbol']} {c['change_24h']:.1f}%" for c in top_movers.get("losers", [])[:3]
    )

    funding_str = ", ".join(
        f"{sym.replace('USDT','')}:{rate:+.4f}%" for sym, rate in sorted(funding.items())
    )

    oi_str = ", ".join(
        f"{sym.replace('USDT','')}:{v.get('open_interest', v.get('open_interest_btc', 0)):,.0f}"
        for sym, v in oi.items()
    )

    trending_str = ", ".join(trending[:5]) if trending else "none"

    mcap_b = global_market["total_market_cap_usd"] / 1e9 if global_market["total_market_cap_usd"] else 0
    vol_b = global_market["total_volume_24h_usd"] / 1e9 if global_market["total_volume_24h_usd"] else 0

    # Reddit headlines summary
    reddit_headlines_str = ""
    for h in reddit.get("headlines", [])[:5]:
        reddit_headlines_str += f"  [{h['score']}pts, {h['comments']}cmt] {h['title']}\n"
    reddit_sentiment_str = (
        f"Bull={reddit['bull_count']} Bear={reddit['bear_count']} "
        f"Neutral={reddit['neutral_count']} (ratio={reddit['sentiment_ratio']:+.2f})"
    )

    # DeFi summary
    defi_str = f"TVL=${defi['total_tvl_billion']}B ({defi['tvl_change_1d_pct']:+.2f}% 1d)"
    chains_str = ", ".join(f"{c['name']}=${c['tvl_billion']}B" for c in defi.get("top_chains", [])[:3])
    
    # Stablecoin yields summary
    yield_str = f"Avg stable yield={stablecoin_yields['avg_yield']:.1f}%"
    
    # BTC metrics
    btc_str = ""
    if btc_metrics:
        btc_str = (
            f"BTC: ${btc_metrics.get('price', 0):,.0f} "
            f"(ATH {btc_metrics.get('ath_change_pct', 0):+.1f}%, "
            f"24h range {btc_metrics.get('range_24h_pct', 0):.1f}%, "
            f"vol/mcap {btc_metrics.get('vol_mcap_ratio_pct', 0):.1f}%)"
        )

    # OVER-RELIANCE SAFEGUARD: Add quality warnings directly into summary
    quality_warning = ""
    if data_quality["overall"] == "POOR":
        quality_warning = (
            "⚠️ DATA QUALITY: POOR — Multiple feeds failed or are stale. "
            "Weight external data LOWER than normal. Rely more on the bot's "
            "internal indicators for this cycle. " + data_quality.get("warning", "")
        )
    elif data_quality["overall"] == "DEGRADED":
        quality_warning = (
            "⚠️ DATA QUALITY: DEGRADED — Some feeds stale/failed. "
            "Exercise caution with affected data points. " + data_quality.get("warning", "")
        )

    return {
        # Raw data for potential programmatic use
        "fear_greed": fear_greed,
        "global_market": global_market,
        "top_movers": top_movers,
        "funding_rates": funding,
        "open_interest": oi,
        "trending": trending,
        "reddit": reddit,
        "defi": defi,
        "stablecoin_yields": stablecoin_yields,
        "btc_metrics": btc_metrics,
        "data_quality": data_quality,

        # Pre-formatted strings for agent prompts
        "summary": {
            "fear_greed_value": fg_val,
            "fear_greed_label": fg_label,
            "btc_dominance": global_market["btc_dominance"],
            "total_market_cap_billion": round(mcap_b, 1),
            "total_volume_24h_billion": round(vol_b, 1),
            "market_cap_change_24h_pct": global_market["market_cap_change_24h_pct"],
            "breadth_pct_positive": top_movers.get("breadth_pct_positive", 50),
            "top_gainers": gainers_str,
            "top_losers": losers_str,
            "funding_rates": funding_str,
            "open_interest": oi_str,
            "trending_coins": trending_str,
            # New feeds
            "reddit_headlines": reddit_headlines_str.strip(),
            "reddit_sentiment": reddit_sentiment_str,
            "defi_tvl": defi_str,
            "defi_top_chains": chains_str,
            "stablecoin_yields": yield_str,
            "btc_metrics": btc_str,
            # Safeguards
            "data_quality": data_quality["overall"],
            "data_quality_warning": quality_warning,
        },
    }
