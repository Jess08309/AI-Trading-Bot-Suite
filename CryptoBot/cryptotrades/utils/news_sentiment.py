"""
Comprehensive FREE crypto news sentiment analysis.
Uses RSS feeds, Reddit, CoinGecko, and Twitter (all FREE).
No paid subscriptions required.
"""

from __future__ import annotations
import os
import time
import requests
from datetime import datetime, timezone
from typing import List, Dict

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False


def log(msg: str):
    """Log message with timestamp."""
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


_sentiment_pipeline = None


def _get_sentiment_pipeline():
    """Get or initialize transformer-based sentiment pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        if TRANSFORMERS_AVAILABLE:
            try:
                log("[SENTIMENT] Loading DistilBERT model...")
                _sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,
                )
                log("[SENTIMENT] Model loaded!")
            except Exception as e:
                log(f"[SENTIMENT] Failed to load model: {e}")
                _sentiment_pipeline = "fallback"
        else:
            _sentiment_pipeline = "fallback"
    return _sentiment_pipeline


def _analyze_sentiment_transformer(text: str) -> float:
    """Use DistilBERT for sentiment. Returns [-1.0, 1.0]."""
    pipe = _get_sentiment_pipeline()
    if pipe == "fallback" or pipe is None:
        return 0.0

    try:
        text = text[:512]
        result = pipe(text)[0]
        label = result['label']
        score = result['score']

        if label == 'POSITIVE':
            return float(score)
        return -float(score)
    except Exception:
        return 0.0


def _analyze_sentiment_keyword(text: str) -> float:
    """Keyword-based sentiment (fallback). Returns [-1.0, 1.0]."""
    positive_keywords = {
        "bull", "surge", "rally", "pump", "moon", "gain", "profit",
        "bullish", "strong", "approved", "success", "growth", "partnership",
        "integration", "adoption", "record", "breakthrough", "secured",
        "epic", "incredible", "amazing", "excellent", "spike", "soar",
        "boom", "breakout", "institutional", "accumulating", "bullrun",
    }

    negative_keywords = {
        "bear", "crash", "plunge", "dump", "loss", "fell", "decline",
        "bearish", "weak", "reject", "fail", "drop", "risk", "concern",
        "warning", "hack", "fraud", "scam", "ban", "regulation", "prison",
        "collapse", "catastrophe", "bearish", "selloff", "downtrend",
        "liquidation", "bankrupt", "lawsuit", "exploit", "vulnerability",
    }

    text_lower = text.lower()

    positive_count = sum(text_lower.count(kw) for kw in positive_keywords)
    negative_count = sum(text_lower.count(kw) for kw in negative_keywords)

    if positive_count + negative_count == 0:
        return 0.0

    net = (positive_count - negative_count) / (positive_count + negative_count)
    return float(max(-1.0, min(1.0, net)))


def analyze_sentiment(text: str) -> float:
    """Analyze sentiment. Returns [-1.0, 1.0]."""
    if not text:
        return 0.0

    pipe = _get_sentiment_pipeline()
    if pipe and pipe != "fallback":
        score = _analyze_sentiment_transformer(text)
        if score != 0.0:
            return score

    return _analyze_sentiment_keyword(text)


def fetch_rss_news() -> List[Dict[str, str]]:
    """Fetch crypto news from multiple FREE RSS feeds."""
    if not FEEDPARSER_AVAILABLE:
        log("[RSS] feedparser not installed: pip install feedparser")
        return []

    rss_feeds = {
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "Cointelegraph": "https://cointelegraph.com/feed",
        "The Block": "https://www.theblock.co/feed",
        "Bitcoin Magazine": "https://bitcoinmagazine.com/feed",
        "Crypto Briefing": "https://cryptobriefing.com/feed/",
    }

    all_news = []

    for source_name, feed_url in rss_feeds.items():
        try:
            log(f"[RSS] Fetching {source_name}...")
            # Use requests with timeout instead of feedparser
            import xml.etree.ElementTree as ET
            response = requests.get(feed_url, timeout=15)
            if response.status_code != 200:
                log(f"[RSS] {source_name}: Failed to fetch (status {response.status_code})")
                continue
            
            try:
                root = ET.fromstring(response.content)
                items = root.findall('.//item')[:10]
                
                for item in items:
                    title_elem = item.find('title')
                    desc_elem = item.find('description')
                    link_elem = item.find('link')
                    pub_elem = item.find('pubDate')
                    
                    all_news.append({
                        "title": title_elem.text if title_elem is not None else "",
                        "body": desc_elem.text if desc_elem is not None else "",
                        "source": source_name,
                        "timestamp": pub_elem.text if pub_elem is not None else "",
                        "url": link_elem.text if link_elem is not None else "",
                    })
            except ET.ParseError:
                log(f"[RSS] {source_name}: XML parse error")
                continue

            log(f"[RSS] {source_name}: {len(items)} articles")
            try:
                time.sleep(0.3)
            except KeyboardInterrupt:
                pass
        except KeyboardInterrupt:
            pass
        except Exception as e:
            log(f"[RSS] {source_name} failed: {e}")

    return all_news


def fetch_reddit_news() -> List[Dict[str, str]]:
    """Fetch crypto sentiment from Reddit (FREE - no auth needed for public)."""
    subreddits = [
        "cryptocurrency",
        "bitcoin",
        "ethereum",
        "CryptoMarkets",
        "defi",
    ]

    all_news = []

    for subreddit in subreddits:
        try:
            log(f"[REDDIT] Fetching r/{subreddit}...")
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            headers = {'User-Agent': 'CryptoTradingBot/1.0'}

            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for post in data.get("data", {}).get("children", [])[:15]:
                post_data = post.get("data", {})
                all_news.append({
                    "title": post_data.get("title", ""),
                    "body": post_data.get("selftext", post_data.get("title", "")),
                    "source": f"Reddit r/{subreddit}",
                    "timestamp": datetime.fromtimestamp(
                        post_data.get("created_utc", 0), tz=timezone.utc
                    ).isoformat(),
                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                })

            log(f"[REDDIT] r/{subreddit}: {min(15, len(data.get('data', {}).get('children', [])))} posts")
            time.sleep(0.5)
        except Exception as e:
            log(f"[REDDIT] r/{subreddit} failed: {e}")

    return all_news


def fetch_coingecko_news() -> List[Dict[str, str]]:
    """Fetch crypto news and trending data from CoinGecko API (completely FREE)."""
    all_news = []

    try:
        log("[COINGECKO] Fetching trending coins...")

        resp = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        trending = data.get("coins", [])
        for item in trending[:10]:
            coin = item.get("item", {})
            all_news.append({
                "title": f"Trending: {coin.get('name', '')} (#{coin.get('market_cap_rank', '?')})",
                "body": f"{coin.get('name')} is trending with {coin.get('data', {}).get('market_cap_usd', 'N/A')} market cap",
                "source": "CoinGecko Trending",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url": coin.get("item", {}).get("id", ""),
            })

        log(f"[COINGECKO] Fetched {len(trending)} trending coins")

        resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=10,
        )
        resp.raise_for_status()
        global_data = resp.json().get("data", {})

        btc_dominance = global_data.get("btc_market_cap_percentage", 0)
        eth_dominance = global_data.get("eth_market_cap_percentage", 0)

        sentiment_text = f"BTC Dominance: {btc_dominance:.1f}%, ETH: {eth_dominance:.1f}%"

        all_news.append({
            "title": f"Global Market Sentiment: {sentiment_text}",
            "body": sentiment_text,
            "source": "CoinGecko Global",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": "https://coingecko.com",
        })

        log("[COINGECKO] Global market data fetched")

    except Exception as e:
        log(f"[COINGECKO] Failed: {e}")

    return all_news


def fetch_twitter_free() -> List[Dict[str, str]]:
    """
    Fetch crypto tweets using Twitter API v2 FREE tier.
    Requires: https://developer.twitter.com/en/portal/dashboard

    FREE tier includes:
    - 300,000 tweets/month
    - 30-day historical search
    - Great for sentiment analysis
    """
    api_key = os.getenv("TWITTER_API_KEY", "").strip()
    if not api_key:
        log("[TWITTER] API key not set, skipping Twitter source")
        return []

    all_news = []
    queries = ["#crypto", "#bitcoin", "#ethereum", "cryptocurrency"]

    try:
        for query in queries:
            log(f"[TWITTER] Searching tweets with '{query}'...")

            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                "Authorization": f"Bearer {api_key}",
            }
            params = {
                "query": f"{query} -is:retweet lang:en",
                "max_results": 100,
                "tweet.fields": "author_id,created_at,public_metrics",
            }

            resp = requests.get(url, headers=headers, params=params, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                for tweet in data.get("data", [])[:20]:
                    all_news.append({
                        "title": f"Tweet: {tweet.get('text', '')[:100]}",
                        "body": tweet.get('text', ''),
                        "source": "Twitter",
                        "timestamp": tweet.get('created_at', ''),
                        "url": f"https://twitter.com/i/web/status/{tweet.get('id', '')}",
                    })

                log(f"[TWITTER] Fetched {len(data.get('data', []))} tweets")
                time.sleep(1)
            else:
                log(f"[TWITTER] API error: {resp.status_code}")

    except Exception as e:
        log(f"[TWITTER] Failed: {e}")

    return all_news


def fetch_all_free_news() -> List[Dict[str, str]]:
    """Fetch from ALL free sources simultaneously."""
    all_news = []
    all_news.extend(fetch_rss_news())
    all_news.extend(fetch_reddit_news())
    all_news.extend(fetch_coingecko_news())
    all_news.extend(fetch_twitter_free())

    log(f"[NEWS] Total articles collected: {len(all_news)}")
    return all_news


def get_news_sentiment(news_list: List[Dict[str, str]]) -> float:
    """
    Calculate aggregate sentiment from all news sources.
    Returns [-1.0, 1.0] where:
    - -1.0 = very bearish
    - 0.0 = neutral
    - 1.0 = very bullish
    """
    if not news_list:
        return 0.0

    sentiments = []
    for article in news_list:
        text = f"{article.get('title', '')} {article.get('body', '')}"
        sentiment = analyze_sentiment(text)
        sentiments.append(sentiment)

    if not sentiments:
        return 0.0

    weighted_sum = sum(s * (i + 1) for i, s in enumerate(sentiments))
    weight_total = sum(i + 1 for i in range(len(sentiments)))

    avg_sentiment = weighted_sum / weight_total if weight_total > 0 else 0.0
    return float(max(-1.0, min(1.0, avg_sentiment)))


def fetch_crypto_news():
    """Alias for fetch_all_free_news()."""
    return fetch_all_free_news()


def fetch_crypto_news_rss():
    """Alias for RSS only."""
    return fetch_rss_news()



# ============================================================
# COIN-SPECIFIC SENTIMENT
# ============================================================

# Map of coin symbols to keywords for article matching
COIN_KEYWORDS = {
    "BTC": ["bitcoin", "btc", "xbt", "satoshi"],
    "ETH": ["ethereum", "eth", "ether", "vitalik"],
    "SOL": ["solana", "sol"],
    "ADA": ["cardano", "ada"],
    "AVAX": ["avalanche", "avax"],
    "DOGE": ["dogecoin", "doge", "shiba"],
    "MATIC": ["polygon", "matic"],
    "LTC": ["litecoin", "ltc"],
    "LINK": ["chainlink", "link"],
    "SHIB": ["shiba", "shib"],
    "XRP": ["ripple", "xrp"],
    "DOT": ["polkadot", "dot"],
    "UNI": ["uniswap", "uni"],
    "ATOM": ["cosmos", "atom"],
    "XLM": ["stellar", "xlm"],
    "AAVE": ["aave"],
}


def _match_coin(text: str) -> List[str]:
    """Match article text to specific coins. Returns list of matched coin symbols."""
    text_lower = text.lower()
    matched = []
    for coin, keywords in COIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                matched.append(coin)
                break
    return matched


def get_coin_sentiment(news_list: List[Dict[str, str]]) -> Dict[str, float]:
    """Calculate per-coin sentiment from news articles.

    Returns dict like {"BTC": 0.3, "ETH": -0.1, ...}
    Only includes coins that have specific mentions.
    """
    coin_scores: Dict[str, List[float]] = {}

    for article in news_list:
        text = f"{article.get('title', '')} {article.get('body', '')}"
        sentiment = analyze_sentiment(text)

        # Match to specific coins
        matched_coins = _match_coin(text)

        for coin in matched_coins:
            if coin not in coin_scores:
                coin_scores[coin] = []
            coin_scores[coin].append(sentiment)

    # Average per coin
    result = {}
    for coin, scores in coin_scores.items():
        if scores:
            result[coin] = float(sum(scores) / len(scores))

    return result


def get_pair_sentiment(pair: str, coin_sentiments: Dict[str, float],
                       global_sentiment: float) -> float:
    """Get sentiment for a specific trading pair.
    Falls back to global sentiment if no coin-specific data.

    Args:
        pair: Trading pair like "BTC/USD" or "PI_XBTUSD"
        coin_sentiments: Per-coin sentiment dict from get_coin_sentiment()
        global_sentiment: Overall market sentiment

    Returns:
        Sentiment score [-1.0, 1.0]
    """
    # Extract coin symbol from pair name
    pair_upper = pair.upper()

    # Map futures symbols back to coins
    futures_map = {
        "PI_XBTUSD": "BTC", "PI_ETHUSD": "ETH", "PI_SOLUSD": "SOL",
        "PI_ADAUSD": "ADA", "PI_AVAXUSD": "AVAX", "PI_DOGEUSD": "DOGE",
        "PI_MATICUSD": "MATIC", "PI_LTCUSD": "LTC", "PI_LINKUSD": "LINK",
        "PI_SHIBUSD": "SHIB", "PI_XRPUSD": "XRP", "PI_DOTUSD": "DOT",
        "PI_UNIUSD": "UNI", "PI_ATOMUSD": "ATOM", "PI_XLMUSD": "XLM",
    }

    coin = futures_map.get(pair_upper, pair_upper.split("/")[0])

    if coin in coin_sentiments:
        # Blend coin-specific (70%) with global (30%)
        return coin_sentiments[coin] * 0.7 + global_sentiment * 0.3

    return global_sentiment
