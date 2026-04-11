"""Patch utils/__init__.py to guard torch-dependent imports."""
import pathlib

p = pathlib.Path("/home/botuser/CryptoBot/cryptotrades/utils/__init__.py")
t = p.read_text()
old = "from .rl_agent import RLTradingAgent"
new = "try:\n    from .rl_agent import RLTradingAgent\nexcept ImportError:\n    RLTradingAgent = None"
t = t.replace(old, new)

# Also guard news_sentiment (needs feedparser)
old2 = """from .news_sentiment import (
    fetch_crypto_news, fetch_crypto_news_rss, get_news_sentiment,
    get_coin_sentiment, get_pair_sentiment,
)"""
new2 = """try:
    from .news_sentiment import (
        fetch_crypto_news, fetch_crypto_news_rss, get_news_sentiment,
        get_coin_sentiment, get_pair_sentiment,
    )
except ImportError:
    fetch_crypto_news = fetch_crypto_news_rss = get_news_sentiment = None
    get_coin_sentiment = get_pair_sentiment = None"""
t = t.replace(old2, new2)

p.write_text(t)
print("PATCHED_OK")
