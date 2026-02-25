"""
Enhanced FREE sentiment + market data sources.
ALL sources are completely free — no API keys or accounts needed.

Sources:
  1. Fear & Greed Index (alternative.me) — crypto-specific sentiment gauge
  2. CoinGecko Global Market — dominance, total market cap, volume
  3. Bitcoin funding rates (CoinGlass free endpoint)
  4. Bitcoin liquidations signal (CoinGlass free)
  5. Open Interest change (Kraken public)
  6. DXY / Dollar strength proxy (free exchange rate API)
  7. Gold price proxy via PAXG (already traded by the bot)
  8. On-chain: mempool congestion proxy (blockchain.info free)
  9. Google Trends proxy (via pytrends — no key needed)
  10. Stablecoin market cap ratio (CoinGecko — risk-on/risk-off signal)

All data cached to avoid hitting rate limits.
"""

from __future__ import annotations
import os
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

# -----------------------------------------------------------
# Cache layer
# -----------------------------------------------------------
_cache: Dict[str, Tuple[float, any]] = {}  # key -> (expiry_ts, data)
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "state")


def _get_cached(key: str, ttl_seconds: int = 300) -> Optional[any]:
    """Get from in-memory cache if not expired."""
    if key in _cache:
        expiry, data = _cache[key]
        if time.time() < expiry:
            return data
    return None


def _set_cached(key: str, data: any, ttl_seconds: int = 300):
    """Store in cache with TTL."""
    _cache[key] = (time.time() + ttl_seconds, data)


def _get_disk_cache(key: str) -> Optional[dict]:
    """Fallback: read last known value from disk."""
    try:
        path = os.path.join(CACHE_DIR, f"enhanced_sentiment_{key}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            # Accept disk cache up to 2 hours old
            cached_at = data.get("cached_at", 0)
            if time.time() - cached_at < 7200:
                return data.get("value")
    except Exception:
        pass
    return None


def _save_disk_cache(key: str, value):
    """Persist to disk for crash recovery."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"enhanced_sentiment_{key}.json")
        with open(path, 'w') as f:
            json.dump({"cached_at": time.time(), "value": value}, f)
    except Exception:
        pass


# -----------------------------------------------------------
# 1. Fear & Greed Index (alternative.me)
# -----------------------------------------------------------
def fetch_fear_greed() -> Dict:
    """
    Crypto Fear & Greed Index.
    Returns: {value: 0-100, label: str, normalized: -1.0 to 1.0}
    0 = Extreme Fear, 100 = Extreme Greed
    FREE — no key needed.
    """
    cached = _get_cached("fear_greed", ttl_seconds=600)
    if cached:
        return cached

    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        entry = data["data"][0]

        result = {
            "value": int(entry["value"]),
            "label": entry["value_classification"],
            "normalized": (int(entry["value"]) - 50) / 50,  # -1.0 to 1.0
            "timestamp": entry.get("timestamp", ""),
            "source": "alternative.me",
        }

        _set_cached("fear_greed", result, ttl_seconds=600)
        _save_disk_cache("fear_greed", result)
        return result

    except Exception as e:
        disk = _get_disk_cache("fear_greed")
        if disk:
            return disk
        return {"value": 50, "label": "Neutral", "normalized": 0.0, "source": "fallback"}


# -----------------------------------------------------------
# 2. CoinGecko Global Market Data
# -----------------------------------------------------------
def fetch_global_market() -> Dict:
    """
    Global crypto market metrics.
    Returns: {btc_dominance, eth_dominance, total_market_cap, market_cap_change_24h,
              total_volume, volume_change_24h, active_cryptos}
    FREE — no key needed.
    """
    cached = _get_cached("global_market", ttl_seconds=300)
    if cached:
        return cached

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()["data"]

        result = {
            "btc_dominance": data.get("market_cap_percentage", {}).get("btc", 0),
            "eth_dominance": data.get("market_cap_percentage", {}).get("eth", 0),
            "total_market_cap_usd": data.get("total_market_cap", {}).get("usd", 0),
            "market_cap_change_24h": data.get("market_cap_change_percentage_24h_usd", 0),
            "total_volume_usd": data.get("total_volume", {}).get("usd", 0),
            "active_cryptos": data.get("active_cryptocurrencies", 0),
            "source": "coingecko",
        }

        _set_cached("global_market", result, ttl_seconds=300)
        _save_disk_cache("global_market", result)
        return result

    except Exception:
        disk = _get_disk_cache("global_market")
        return disk or {"btc_dominance": 50, "market_cap_change_24h": 0, "source": "fallback"}


# -----------------------------------------------------------
# 3. Bitcoin Funding Rate (Kraken Futures — free)
# -----------------------------------------------------------
def fetch_funding_rates() -> Dict:
    """
    Perpetual funding rates from Kraken Futures.
    Positive = longs pay shorts (bullish crowding) → contrarian bearish
    Negative = shorts pay longs (bearish crowding) → contrarian bullish
    FREE — no key needed.
    """
    cached = _get_cached("funding_rates", ttl_seconds=300)
    if cached:
        return cached

    symbols = {
        "PI_XBTUSD": "BTC",
        "PI_ETHUSD": "ETH",
        "PI_SOLUSD": "SOL",
    }

    result = {"rates": {}, "avg_rate": 0.0, "signal": 0.0, "source": "kraken"}

    try:
        resp = requests.get(
            "https://futures.kraken.com/derivatives/api/v3/tickers",
            timeout=10
        )
        resp.raise_for_status()
        tickers = resp.json().get("tickers", [])

        rates = []
        for ticker in tickers:
            symbol = ticker.get("symbol", "").upper()
            if symbol in symbols:
                rate = ticker.get("fundingRate", 0)
                if rate is not None:
                    result["rates"][symbols[symbol]] = float(rate)
                    rates.append(float(rate))

        if rates:
            avg = sum(rates) / len(rates)
            result["avg_rate"] = avg
            # Contrarian signal: high funding = bearish, low = bullish
            # Typical range: -0.01% to +0.03%
            result["signal"] = max(-1.0, min(1.0, -avg * 1000))

        _set_cached("funding_rates", result, ttl_seconds=300)
        _save_disk_cache("funding_rates", result)
        return result

    except Exception:
        disk = _get_disk_cache("funding_rates")
        return disk or {"rates": {}, "avg_rate": 0, "signal": 0, "source": "fallback"}


# -----------------------------------------------------------
# 4. Open Interest (Kraken Futures — free)
# -----------------------------------------------------------
def fetch_open_interest() -> Dict:
    """
    Open interest data from Kraken Futures.
    Rising OI + rising price = strong trend.
    Rising OI + falling price = potential squeeze.
    FREE — no key needed.
    """
    cached = _get_cached("open_interest", ttl_seconds=300)
    if cached:
        return cached

    try:
        resp = requests.get(
            "https://futures.kraken.com/derivatives/api/v3/tickers",
            timeout=10
        )
        resp.raise_for_status()
        tickers = resp.json().get("tickers", [])

        result = {"oi": {}, "source": "kraken"}

        target_symbols = {
            "PI_XBTUSD": "BTC", "PI_ETHUSD": "ETH", "PI_SOLUSD": "SOL",
            "PI_ADAUSD": "ADA", "PI_DOGEUSD": "DOGE", "PI_LINKUSD": "LINK",
        }

        for ticker in tickers:
            symbol = ticker.get("symbol", "").upper()
            if symbol in target_symbols:
                oi = ticker.get("openInterest", 0)
                mark = ticker.get("markPrice", 0)
                result["oi"][target_symbols[symbol]] = {
                    "open_interest": float(oi) if oi else 0,
                    "mark_price": float(mark) if mark else 0,
                }

        _set_cached("open_interest", result, ttl_seconds=300)
        _save_disk_cache("open_interest", result)
        return result

    except Exception:
        disk = _get_disk_cache("open_interest")
        return disk or {"oi": {}, "source": "fallback"}


# -----------------------------------------------------------
# 5. DXY / Dollar Strength Proxy
# -----------------------------------------------------------
def fetch_dollar_strength() -> Dict:
    """
    Dollar strength proxy using free exchange rate API.
    Strong dollar = typically bearish for crypto.
    FREE — no key needed (exchangerate.host or floatrates).
    """
    cached = _get_cached("dollar_strength", ttl_seconds=1800)
    if cached:
        return cached

    try:
        # Use ECB reference rates (completely free, no key)
        resp = requests.get(
            "https://www.floatrates.com/daily/usd.json",
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        # Build a simple dollar index from major currencies
        currencies = {"eur": 0.576, "jpy": 0.136, "gbp": 0.119, "cad": 0.091, "chf": 0.036, "sek": 0.042}
        weighted_sum = 0
        total_weight = 0

        for curr, weight in currencies.items():
            if curr in data:
                rate = data[curr]["rate"]
                weighted_sum += rate * weight
                total_weight += weight

        dxy_proxy = weighted_sum / total_weight if total_weight > 0 else 1.0

        result = {
            "dxy_proxy": dxy_proxy,
            # Normalize: 1.0 = average, >1.0 = stronger dollar (bearish crypto)
            "signal": max(-1.0, min(1.0, -(dxy_proxy - 1.0) * 10)),
            "source": "floatrates",
        }

        _set_cached("dollar_strength", result, ttl_seconds=1800)
        _save_disk_cache("dollar_strength", result)
        return result

    except Exception:
        disk = _get_disk_cache("dollar_strength")
        return disk or {"dxy_proxy": 1.0, "signal": 0.0, "source": "fallback"}


# -----------------------------------------------------------
# 6. Bitcoin Mempool / Network Congestion
# -----------------------------------------------------------
def fetch_mempool_data() -> Dict:
    """
    Bitcoin mempool congestion from blockchain.info.
    High congestion = high activity = often at market extremes.
    FREE — no key needed.
    """
    cached = _get_cached("mempool", ttl_seconds=600)
    if cached:
        return cached

    try:
        # Unconfirmed transactions count
        resp = requests.get(
            "https://blockchain.info/q/unconfirmedcount",
            timeout=10
        )
        resp.raise_for_status()
        unconfirmed = int(resp.text.strip())

        # Hash rate (network health)
        resp2 = requests.get(
            "https://blockchain.info/q/hashrate",
            timeout=10
        )
        hashrate = float(resp2.text.strip()) if resp2.status_code == 200 else 0

        result = {
            "unconfirmed_txs": unconfirmed,
            "hashrate": hashrate,
            # Normalize: typical range 5k-50k unconfirmed txs
            # Very high means congestion (potential extreme moves)
            "congestion_normalized": min(1.0, unconfirmed / 50000),
            "source": "blockchain.info",
        }

        _set_cached("mempool", result, ttl_seconds=600)
        _save_disk_cache("mempool", result)
        return result

    except Exception:
        disk = _get_disk_cache("mempool")
        return disk or {"unconfirmed_txs": 0, "congestion_normalized": 0.5, "source": "fallback"}


# -----------------------------------------------------------
# 7. Stablecoin Supply Ratio (CoinGecko free)
# -----------------------------------------------------------
def fetch_stablecoin_ratio() -> Dict:
    """
    Stablecoin market cap vs total crypto market cap.
    High ratio = lots of dry powder on sidelines = potential buy pressure
    Low ratio = money fully deployed = potential sell pressure
    FREE — CoinGecko, no key needed.
    """
    cached = _get_cached("stablecoin_ratio", ttl_seconds=1800)
    if cached:
        return cached

    try:
        # Get stablecoin market caps
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": "tether,usd-coin,dai,first-digital-usd",
                "order": "market_cap_desc",
            },
            timeout=10
        )
        resp.raise_for_status()
        stables = resp.json()
        stable_cap = sum(c.get("market_cap", 0) for c in stables if c.get("market_cap"))

        # Get total market
        global_data = fetch_global_market()
        total_cap = global_data.get("total_market_cap_usd", 1)

        ratio = stable_cap / max(total_cap, 1) * 100

        result = {
            "stablecoin_cap_usd": stable_cap,
            "total_market_cap_usd": total_cap,
            "ratio_pct": ratio,
            # Higher ratio (>10%) = bearish (money on sidelines but could mean fear)
            # Lower ratio (<5%) = bullish (money deployed)
            # Normalize around 7%
            "signal": max(-1.0, min(1.0, (ratio - 7) / 5)),
            "source": "coingecko",
        }

        _set_cached("stablecoin_ratio", result, ttl_seconds=1800)
        _save_disk_cache("stablecoin_ratio", result)
        return result

    except Exception:
        disk = _get_disk_cache("stablecoin_ratio")
        return disk or {"ratio_pct": 7, "signal": 0, "source": "fallback"}


# -----------------------------------------------------------
# 8. Top Coin Momentum (CoinGecko 24h changes — free)
# -----------------------------------------------------------
def fetch_top_coin_momentum() -> Dict:
    """
    24h price changes for top coins — broad market momentum.
    If everything is green → risk-on.
    If everything is red → risk-off.
    FREE — no key needed.
    """
    cached = _get_cached("coin_momentum", ttl_seconds=300)
    if cached:
        return cached

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": "bitcoin,ethereum,solana,cardano,avalanche-2,dogecoin,chainlink,ripple,litecoin",
                "order": "market_cap_desc",
                "sparkline": "false",
            },
            timeout=10
        )
        resp.raise_for_status()
        coins = resp.json()

        changes = {}
        for coin in coins:
            sym = coin.get("symbol", "").upper()
            change_24h = coin.get("price_change_percentage_24h", 0)
            changes[sym] = float(change_24h) if change_24h else 0.0

        # Aggregate momentum
        avg_change = sum(changes.values()) / max(len(changes), 1)

        result = {
            "changes_24h": changes,
            "avg_change_24h": avg_change,
            # Normalize: ±5% maps to ±1.0
            "signal": max(-1.0, min(1.0, avg_change / 5)),
            "source": "coingecko",
        }

        _set_cached("coin_momentum", result, ttl_seconds=300)
        _save_disk_cache("coin_momentum", result)
        return result

    except Exception:
        disk = _get_disk_cache("coin_momentum")
        return disk or {"avg_change_24h": 0, "signal": 0, "source": "fallback"}


# -----------------------------------------------------------
# 9. Bitcoin Long/Short Ratio (Kraken tickers — free)
# -----------------------------------------------------------
def fetch_long_short_ratio() -> Dict:
    """
    Derive long/short bias from Kraken perpetual spread.
    Mark price > index = longs dominant (potential squeeze down).
    Mark price < index = shorts dominant (potential squeeze up).
    FREE — no key needed.
    """
    cached = _get_cached("long_short", ttl_seconds=300)
    if cached:
        return cached

    try:
        resp = requests.get(
            "https://futures.kraken.com/derivatives/api/v3/tickers",
            timeout=10
        )
        resp.raise_for_status()
        tickers = resp.json().get("tickers", [])

        result = {"spreads": {}, "avg_bias": 0, "signal": 0, "source": "kraken"}
        biases = []

        targets = {"PI_XBTUSD": "BTC", "PI_ETHUSD": "ETH", "PI_SOLUSD": "SOL"}

        for ticker in tickers:
            symbol = ticker.get("symbol", "").upper()
            if symbol in targets:
                mark = ticker.get("markPrice", 0)
                index = ticker.get("indexPrice", 0)
                if mark and index and index > 0:
                    basis = (mark - index) / index * 100  # basis in %
                    result["spreads"][targets[symbol]] = {
                        "mark": float(mark),
                        "index": float(index),
                        "basis_pct": basis,
                    }
                    biases.append(basis)

        if biases:
            avg_bias = sum(biases) / len(biases)
            result["avg_bias"] = avg_bias
            # Positive basis = long-heavy → contrarian bearish
            result["signal"] = max(-1.0, min(1.0, -avg_bias * 20))

        _set_cached("long_short", result, ttl_seconds=300)
        _save_disk_cache("long_short", result)
        return result

    except Exception:
        disk = _get_disk_cache("long_short")
        return disk or {"avg_bias": 0, "signal": 0, "source": "fallback"}


# -----------------------------------------------------------
# UNIFIED: Aggregate all enhanced signals
# -----------------------------------------------------------
def fetch_all_enhanced_signals() -> Dict:
    """
    Fetch ALL free enhanced signals in one call.
    Returns a dict with all signals + a composite score.
    
    This can be called every ~5 minutes without hitting any rate limits.
    """
    signals = {}

    # Fear & Greed
    fg = fetch_fear_greed()
    signals["fear_greed"] = fg

    # Global market
    gm = fetch_global_market()
    signals["global_market"] = gm

    # Funding rates
    fr = fetch_funding_rates()
    signals["funding_rates"] = fr

    # Open Interest
    oi = fetch_open_interest()
    signals["open_interest"] = oi

    # Dollar strength
    ds = fetch_dollar_strength()
    signals["dollar_strength"] = ds

    # Mempool
    mp = fetch_mempool_data()
    signals["mempool"] = mp

    # Stablecoin ratio
    sr = fetch_stablecoin_ratio()
    signals["stablecoin_ratio"] = sr

    # Top coin momentum
    cm = fetch_top_coin_momentum()
    signals["coin_momentum"] = cm

    # Long/short ratio
    ls = fetch_long_short_ratio()
    signals["long_short"] = ls

    # Composite sentiment: weighted blend of all signals
    weights = {
        "fear_greed": 0.25,       # Strongest single indicator
        "coin_momentum": 0.20,    # Broad market direction
        "funding_rates": 0.15,    # Crowding indicator
        "long_short": 0.10,       # Positioning
        "dollar_strength": 0.10,  # Macro
        "stablecoin_ratio": 0.10, # Dry powder
        "mempool": 0.05,          # On-chain activity
        "global_market": 0.05,    # Market cap trend
    }

    composite = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        signal = signals.get(key, {})
        val = signal.get("signal", signal.get("normalized", 0))
        if val is not None and val != 0:
            composite += float(val) * weight
            total_weight += weight

    if total_weight > 0:
        composite = composite / total_weight
    composite = max(-1.0, min(1.0, composite))

    signals["composite"] = {
        "value": composite,
        "components_used": int(total_weight / 0.05),
    }

    return signals


def get_enhanced_sentiment_score() -> float:
    """
    Simple interface: just return a -1.0 to 1.0 sentiment score.
    Blends Fear/Greed + market momentum + funding rates + more.
    """
    try:
        signals = fetch_all_enhanced_signals()
        return signals.get("composite", {}).get("value", 0.0)
    except Exception:
        return 0.0


# -----------------------------------------------------------
# Quick test
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED SENTIMENT - ALL FREE SOURCES")
    print("=" * 60)
    print()

    signals = fetch_all_enhanced_signals()

    for key, data in signals.items():
        if key == "composite":
            continue
        print(f"  [{key}]")
        if isinstance(data, dict):
            for k, v in data.items():
                if k != "source" and not isinstance(v, dict):
                    print(f"    {k}: {v}")
        print()

    comp = signals["composite"]
    print(f"  COMPOSITE SENTIMENT: {comp['value']:+.3f}")
    label = "BULLISH" if comp["value"] > 0.2 else "BEARISH" if comp["value"] < -0.2 else "NEUTRAL"
    print(f"  Market Read: {label}")
    print(f"  Sources used: {comp['components_used']}")
