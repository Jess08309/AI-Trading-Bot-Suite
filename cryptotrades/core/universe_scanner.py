"""
CryptoBot Universe Scanner
===========================
Dynamically discovers and ranks crypto coins for trading.
Expands beyond the static 18-coin watchlist to scan 200+ coins.

Data sources (all FREE, no API key required):
  - Alpaca: available crypto trading pairs (requires API key)
  - CoinGecko: top coins by market cap, trending coins
  - Kraken Futures: available perpetuals (public endpoint)

Flow:
  1. Pull top 250 coins by market cap from CoinGecko
  2. Get trending coins (momentum factor)
  3. Cross-reference with Alpaca available crypto pairs
  4. Score each coin: volume, momentum, market cap, trend
  5. Return ranked list of tradeable symbols

The existing engine quality gates (ML, regime, indicators, health gates)
still apply — this scanner just casts a wider net.
"""
import json
import logging
import os
import time
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

log = logging.getLogger("cryptobot.universe_scanner")

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class ScannerConfig:
    """Scanner tuning parameters."""
    # How often to re-scan (minutes)
    SCAN_INTERVAL_MIN: int = 30

    # CoinGecko limits
    TOP_N_BY_MCAP: int = 250          # Pull top N coins by market cap
    MIN_MARKET_CAP_USD: float = 50e6  # $50M minimum market cap
    MIN_24H_VOLUME_USD: float = 5e6   # $5M minimum 24h volume

    # How many coins to return to the engine
    MAX_SPOT_SYMBOLS: int = 100       # Top N spot coins to trade
    MAX_FUTURES_SYMBOLS: int = 30     # Top N futures to trade

    # Scoring weights (sum to 1.0)
    WEIGHT_VOLUME: float = 0.30       # 24h volume rank
    WEIGHT_MCAP: float = 0.20         # Market cap rank (larger = safer)
    WEIGHT_MOMENTUM: float = 0.25     # 24h price change magnitude
    WEIGHT_TRENDING: float = 0.15     # CoinGecko trending bonus
    WEIGHT_VOLATILITY: float = 0.10   # Higher vol = more opportunity

    # Probation: new symbols must be tracked this long before trading
    PROBATION_HOURS: int = 72           # 3 days observation before eligible

    # Stablecoins and junk to always exclude
    EXCLUDE_SYMBOLS: Set[str] = field(default_factory=lambda: {
        # Stablecoins
        'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD', 'USDE', 'USDD',
        'PYUSD', 'GUSD', 'PAX', 'USDP', 'FRAX', 'LUSD', 'CRVUSD',
        'USD0', 'GHO', 'USDS', 'USDG', 'RLUSD', 'BFUSD', 'USDTB',
        'USDF', 'USD1', 'USYC', 'USDY', 'YLDS', 'OUSG', 'USTB',
        # Wrapped/bridged tokens
        'WBTC', 'WETH', 'STETH', 'RETH', 'CBETH', 'WSTETH',
        'WBETH', 'WSOL',
        # Exchange tokens (not typically on Alpaca crypto)
        'BNB', 'LEO', 'OKB', 'CRO', 'KCS', 'GT', 'MX', 'HT',
        'BGB', 'WBT', 'HTX', 'NEXO',
        # Tokenized assets / RWA with low crypto utility
        'XAUT', 'BUIDL', 'EUTBL', 'JTRSY', 'JAAA', 'FIGR_HELOC',
        'A7A5', 'HASH',
        # Known low-quality / pump-and-dump
        'BSB', 'QUQ', 'BTW', 'FF', 'CC', 'USAD', 'RESOLV', 'WLFI',
    })


# ═══════════════════════════════════════════════════════════
#  ALPACA CRYPTO PAIR DISCOVERY
# ═══════════════════════════════════════════════════════════

class AlpacaDiscovery:
    """Discover which coins are tradeable on Alpaca."""

    ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"
    _cache: Optional[Set[str]] = None
    _cache_time: Optional[datetime] = None
    CACHE_TTL_HOURS = 6

    @classmethod
    def get_available_symbols(cls) -> Set[str]:
        """Get set of base symbols available on Alpaca (e.g., {'BTC', 'ETH', ...})."""
        if cls._cache and cls._cache_time:
            age = (datetime.now() - cls._cache_time).total_seconds() / 3600
            if age < cls.CACHE_TTL_HOURS:
                return cls._cache

        try:
            api_key = os.getenv("ALPACA_API_KEY", "")
            api_secret = os.getenv("ALPACA_API_SECRET", "")
            headers = {
                'User-Agent': 'CryptoBot/1.0',
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': api_secret,
            }
            resp = requests.get(
                cls.ASSETS_URL,
                params={"asset_class": "crypto", "status": "active"},
                timeout=15,
                headers=headers,
            )
            resp.raise_for_status()
            assets = resp.json()

            symbols = set()
            for asset in assets:
                # Alpaca crypto symbols are like "BTC/USD"
                sym = asset.get("symbol", "")
                if "/" in sym:
                    base = sym.split("/")[0].upper()
                    if base:
                        symbols.add(base)

            cls._cache = symbols
            cls._cache_time = datetime.now()
            log.info(f"Alpaca discovery: {len(symbols)} tradeable crypto symbols")
            return symbols
        except Exception as e:
            log.warning(f"Alpaca discovery failed: {e}")
            return cls._cache or set()


# ═══════════════════════════════════════════════════════════
#  KRAKEN FUTURES DISCOVERY
# ═══════════════════════════════════════════════════════════

class KrakenFuturesDiscovery:
    """Discover available Kraken perpetual futures."""

    TICKERS_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"
    _cache: Optional[Dict[str, str]] = None
    _cache_time: Optional[datetime] = None
    CACHE_TTL_HOURS = 6

    @classmethod
    def get_available_perpetuals(cls) -> Dict[str, str]:
        """Get dict of {base_symbol: futures_symbol} for available perpetuals.
        e.g., {'BTC': 'PI_XBTUSD', 'ETH': 'PI_ETHUSD'}
        """
        if cls._cache and cls._cache_time:
            age = (datetime.now() - cls._cache_time).total_seconds() / 3600
            if age < cls.CACHE_TTL_HOURS:
                return cls._cache

        try:
            resp = requests.get(cls.TICKERS_URL, timeout=15, headers={
                'User-Agent': 'CryptoBot/1.0'
            })
            resp.raise_for_status()
            data = resp.json()

            perpetuals = {}
            # Map common names to Kraken's XBT convention
            symbol_map = {
                'XBT': 'BTC', 'ETH': 'ETH', 'SOL': 'SOL', 'ADA': 'ADA',
                'DOGE': 'DOGE', 'LINK': 'LINK', 'AVAX': 'AVAX', 'DOT': 'DOT',
                'BCH': 'BCH', 'LTC': 'LTC', 'XRP': 'XRP', 'ATOM': 'ATOM',
                'UNI': 'UNI', 'AAVE': 'AAVE', 'MATIC': 'MATIC', 'NEAR': 'NEAR',
                'FIL': 'FIL', 'APE': 'APE', 'SAND': 'SAND', 'MANA': 'MANA',
                'CRV': 'CRV', 'COMP': 'COMP', 'SUSHI': 'SUSHI', 'YFI': 'YFI',
                'SNX': 'SNX', 'GRT': 'GRT', 'ALGO': 'ALGO', 'XLM': 'XLM',
                'EOS': 'EOS', 'XTZ': 'XTZ', 'FLOW': 'FLOW', 'AXS': 'AXS',
                'SHIB': 'SHIB', 'ARB': 'ARB', 'OP': 'OP', 'APT': 'APT',
                'SUI': 'SUI', 'SEI': 'SEI', 'TIA': 'TIA', 'PEPE': 'PEPE',
            }

            for ticker in data.get("tickers", []):
                sym = ticker.get("symbol", "")
                # Only perpetual instruments (PI_ prefix)
                if sym.startswith("PI_") and sym.endswith("USD"):
                    # Extract base: PI_XBTUSD → XBT
                    base_raw = sym[3:-3]  # Remove PI_ and USD
                    base_mapped = symbol_map.get(base_raw, base_raw)
                    perpetuals[base_mapped] = sym

            cls._cache = perpetuals
            cls._cache_time = datetime.now()
            log.info(f"Kraken futures discovery: {len(perpetuals)} perpetuals available")
            return perpetuals
        except Exception as e:
            log.warning(f"Kraken futures discovery failed: {e}")
            return cls._cache or {}


# ═══════════════════════════════════════════════════════════
#  COINGECKO MARKET DATA
# ═══════════════════════════════════════════════════════════

class CoinGeckoData:
    """Pull market data from CoinGecko free API."""

    BASE_URL = "https://api.coingecko.com/api/v3"
    _rate_limit_delay = 2.0  # seconds between API calls (free tier: ~30/min)

    @classmethod
    def get_top_coins(cls, n: int = 250) -> List[Dict]:
        """Get top N coins by market cap with price/volume/change data."""
        all_coins = []
        per_page = min(n, 250)
        pages = (n + per_page - 1) // per_page

        for page in range(1, pages + 1):
            try:
                resp = requests.get(
                    f"{cls.BASE_URL}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": per_page,
                        "page": page,
                        "sparkline": "false",
                        "price_change_percentage": "1h,24h,7d",
                    },
                    timeout=20,
                    headers={'User-Agent': 'CryptoBot/1.0'},
                )
                resp.raise_for_status()
                coins = resp.json()
                if isinstance(coins, list):
                    all_coins.extend(coins)
                else:
                    log.warning(f"CoinGecko unexpected response: {type(coins)}")
                    break

                if page < pages:
                    time.sleep(cls._rate_limit_delay)
            except Exception as e:
                log.warning(f"CoinGecko page {page} failed: {e}")
                break

        return all_coins

    @classmethod
    def get_trending(cls) -> Set[str]:
        """Get currently trending coin symbols."""
        try:
            resp = requests.get(
                f"{cls.BASE_URL}/search/trending",
                timeout=15,
                headers={'User-Agent': 'CryptoBot/1.0'},
            )
            resp.raise_for_status()
            data = resp.json()
            trending = set()
            for c in data.get("coins", []):
                sym = c.get("item", {}).get("symbol", "").upper()
                if sym:
                    trending.add(sym)
            return trending
        except Exception as e:
            log.warning(f"CoinGecko trending failed: {e}")
            return set()


# ═══════════════════════════════════════════════════════════
#  UNIVERSE SCANNER
# ═══════════════════════════════════════════════════════════

class CryptoUniverseScanner:
    """
    Dynamically discovers and ranks crypto coins for CryptoBot.

    Replaces the static 18-coin SPOT_SYMBOLS with a ranked list
    of ~100 coins based on volume, momentum, and market cap.

    The engine's existing quality gates (ML confidence, regime detection,
    symbol health, correlation, etc.) still apply downstream.
    """

    def __init__(self, config: ScannerConfig = None, core_symbols: List[str] = None):
        self.config = config or ScannerConfig()

        # Core symbols that ALWAYS get included (the proven ones)
        self.core_spot: List[str] = core_symbols or [
            "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "AVAX/USD",
            "DOGE/USD", "LINK/USD", "XRP/USD", "LTC/USD", "UNI/USD",
            "XLM/USD", "BCH/USD", "DOT/USD", "MATIC/USD", "ATOM/USD",
            "NEAR/USD", "AAVE/USD", "PAXG/USD",
        ]

        # Scan state
        self._last_scan_time: Optional[datetime] = None
        self._cached_spot: List[str] = list(self.core_spot)
        self._cached_futures: List[str] = []
        self._cached_scores: Dict[str, Dict] = {}
        self._scan_lock = threading.Lock()

        # Probation tracking — new symbols observe before trading
        self._first_seen: Dict[str, str] = {}  # symbol -> ISO datetime
        self._probation_dir = os.path.join("data", "state")
        self._probation_file = os.path.join(self._probation_dir, "scanner_probation.json")
        self._load_probation()

        # Stats
        self.total_scans = 0
        self.last_scan_duration = 0.0
        self.coins_discovered = 0
        self.coins_on_alpaca = 0
        self.futures_discovered = 0

        graduated = sum(
            1 for s, t in self._first_seen.items()
            if (datetime.now() - datetime.fromisoformat(t)).total_seconds() / 3600
               >= self.config.PROBATION_HOURS
        )
        log.info(
            f"CryptoUniverseScanner initialized | core={len(self.core_spot)} "
            f"| probation={len(self._first_seen) - graduated} "
            f"| graduated={graduated}"
        )

    # ── Probation persistence ────────────────────────────

    def _load_probation(self):
        """Load probation state from disk."""
        try:
            if os.path.exists(self._probation_file):
                with open(self._probation_file, "r") as f:
                    self._first_seen = json.load(f)
                log.info(f"Loaded probation state: {len(self._first_seen)} tracked symbols")
        except Exception as e:
            log.warning(f"Probation load failed: {e}")
            self._first_seen = {}

    def _save_probation(self):
        """Save probation state to disk."""
        try:
            os.makedirs(self._probation_dir, exist_ok=True)
            with open(self._probation_file, "w") as f:
                json.dump(self._first_seen, f, indent=2)
        except Exception as e:
            log.warning(f"Probation save failed: {e}")

    def should_scan(self) -> bool:
        """Check if it's time for a new scan."""
        if self._last_scan_time is None:
            return True
        elapsed = (datetime.now() - self._last_scan_time).total_seconds() / 60
        return elapsed >= self.config.SCAN_INTERVAL_MIN

    def get_spot_symbols(self) -> List[str]:
        """Get current ranked spot symbol list."""
        return list(self._cached_spot)

    def get_futures_symbols(self) -> List[str]:
        """Get current ranked futures symbol list."""
        return list(self._cached_futures)

    def scan(self) -> Tuple[List[str], List[str]]:
        """
        Run a full universe scan.

        Returns:
            (spot_symbols, futures_symbols) — ranked lists ready for the engine.
        """
        with self._scan_lock:
            return self._do_scan()

    def _do_scan(self) -> Tuple[List[str], List[str]]:
        """Internal scan implementation."""
        scan_start = time.time()
        self.total_scans += 1
        self._last_scan_time = datetime.now()

        log.info("═══ Universe Scan Starting ═══")

        # ── Step 1: Get Alpaca available crypto pairs ──
        alpaca_symbols = AlpacaDiscovery.get_available_symbols()
        self.coins_on_alpaca = len(alpaca_symbols)
        log.info(f"  Alpaca: {len(alpaca_symbols)} tradeable crypto symbols")

        # ── Step 2: Get top coins from CoinGecko ──
        top_coins = CoinGeckoData.get_top_coins(self.config.TOP_N_BY_MCAP)
        self.coins_discovered = len(top_coins)
        log.info(f"  CoinGecko: {len(top_coins)} coins by market cap")

        # ── Step 3: Get trending coins ──
        time.sleep(CoinGeckoData._rate_limit_delay)
        trending = CoinGeckoData.get_trending()
        log.info(f"  Trending: {len(trending)} coins")

        # ── Step 4: Filter and score ──
        scored_coins = []
        for coin in top_coins:
            sym = coin.get("symbol", "").upper()

            # Skip excluded
            if sym in self.config.EXCLUDE_SYMBOLS:
                continue

            # Must be on Alpaca
            if sym not in alpaca_symbols:
                continue

            # Minimum market cap
            mcap = coin.get("market_cap") or 0
            if mcap < self.config.MIN_MARKET_CAP_USD:
                continue

            # Minimum volume
            vol = coin.get("total_volume") or 0
            if vol < self.config.MIN_24H_VOLUME_USD:
                continue

            # Compute score
            score_data = self._score_coin(coin, trending, len(scored_coins))
            scored_coins.append((sym, score_data))

        # Sort by composite score
        scored_coins.sort(key=lambda x: x[1]["composite"], reverse=True)

        # ── Step 5: Build spot symbols list ──
        # Core symbols always first
        spot_symbols = list(self.core_spot)
        core_bases = {s.replace("/USD", "") for s in self.core_spot}

        # Add top scored coins (candidates — probation applied below)
        all_candidates = []
        for sym, score_data in scored_coins:
            coin_symbol = f"{sym}/USD"
            if coin_symbol not in spot_symbols and sym not in core_bases:
                all_candidates.append(coin_symbol)

        # ── Step 5b: Probation filter ──
        # New symbols enter observation; only graduated ones trade
        now = datetime.now()
        in_probation = []
        newly_discovered = []

        for sym in all_candidates:
            if sym not in self._first_seen:
                self._first_seen[sym] = now.isoformat()
                newly_discovered.append(sym)
                in_probation.append(sym)
            else:
                first = datetime.fromisoformat(self._first_seen[sym])
                hours = (now - first).total_seconds() / 3600
                if hours >= self.config.PROBATION_HOURS:
                    if len(spot_symbols) < self.config.MAX_SPOT_SYMBOLS:
                        spot_symbols.append(sym)  # Graduated!
                else:
                    in_probation.append(sym)

        self._save_probation()

        if newly_discovered:
            log.info(
                f"  NEW: {len(newly_discovered)} symbols discovered → "
                f"entering {self.config.PROBATION_HOURS}h probation"
            )
        if in_probation:
            log.info(
                f"  PROBATION: {len(in_probation)} symbols observing "
                f"(not yet tradeable)"
            )
        graduated_count = len(spot_symbols) - len(self.core_spot)
        if graduated_count > 0:
            log.info(f"  GRADUATED: {graduated_count} symbols passed probation")

        # ── Step 6: Discover futures ──
        kraken_futures = KrakenFuturesDiscovery.get_available_perpetuals()
        self.futures_discovered = len(kraken_futures)

        # Build futures list — only for coins in our proven spot list
        # (futures for probation symbols NOT included)
        spot_bases = {s.replace("/USD", "") for s in spot_symbols}
        futures_symbols = []
        for base, futures_sym in kraken_futures.items():
            if base in spot_bases:
                futures_symbols.append(futures_sym)
                if len(futures_symbols) >= self.config.MAX_FUTURES_SYMBOLS:
                    break

        # ── Cache results ──
        self._cached_spot = spot_symbols
        self._cached_futures = futures_symbols
        self._cached_scores = {sym: data for sym, data in scored_coins}

        self.last_scan_duration = time.time() - scan_start

        log.info(
            f"═══ Universe Scan Complete ({self.last_scan_duration:.1f}s) ═══\n"
            f"  Spot: {len(spot_symbols)} symbols "
            f"({len(self.core_spot)} core + {len(spot_symbols) - len(self.core_spot)} discovered)\n"
            f"  Futures: {len(futures_symbols)} perpetuals\n"
            f"  Filtered: {self.coins_discovered} coins → {len(scored_coins)} scored → "
            f"{len(spot_symbols)} selected"
        )

        # Log top discoveries
        new_discoveries = [
            (sym, data) for sym, data in scored_coins[:20]
            if sym not in core_bases
        ]
        if new_discoveries:
            log.info("  Top new discoveries:")
            for sym, data in new_discoveries[:10]:
                log.info(
                    f"    {sym:<8} score={data['composite']:.3f} | "
                    f"vol=${data['volume']/1e6:>8,.1f}M | "
                    f"mcap=${data['mcap']/1e9:>6,.1f}B | "
                    f"24h={data['change_24h']:>+6.1f}% | "
                    f"{'TRENDING' if data['is_trending'] else ''}"
                )

        return spot_symbols, futures_symbols

    def _score_coin(self, coin: Dict, trending: Set[str], rank_idx: int) -> Dict:
        """
        Score a coin for inclusion in the trading universe.

        Factors:
          - Volume rank (higher volume = better liquidity)
          - Market cap rank (larger = more stable/safer)
          - Momentum (24h price change magnitude — we want movers)
          - Trending bonus (CoinGecko trending = market attention)
          - Volatility (higher = more trading opportunity)
        """
        sym = coin.get("symbol", "").upper()
        mcap = coin.get("market_cap") or 0
        vol = coin.get("total_volume") or 0
        chg_24h = coin.get("price_change_percentage_24h") or 0
        chg_7d = coin.get("price_change_percentage_7d_in_currency") or 0
        price = coin.get("current_price") or 0

        # Volume score: log-scaled, normalized
        vol_score = min(1.0, max(0.0, (vol / 1e9)))  # $1B = max score

        # Market cap score: log-scaled
        mcap_score = min(1.0, max(0.0, (mcap / 100e9)))  # $100B = max score

        # Momentum score: absolute 24h change (we want movers in either direction)
        # Sweet spot: 2-15% daily moves
        abs_chg = abs(chg_24h)
        if abs_chg < 1.0:
            momentum_score = abs_chg * 0.3  # Low momentum, penalize
        elif abs_chg < 15.0:
            momentum_score = min(1.0, abs_chg / 15.0)  # Linear 1-15%
        else:
            momentum_score = max(0.3, 1.0 - (abs_chg - 15) / 50)  # Too volatile, discount

        # Trending bonus
        is_trending = sym in trending
        trend_score = 1.0 if is_trending else 0.0

        # Volatility proxy: 7d change magnitude
        vol_proxy = min(1.0, abs(chg_7d) / 30.0) if chg_7d else 0.3

        # Composite score
        composite = (
            self.config.WEIGHT_VOLUME * vol_score +
            self.config.WEIGHT_MCAP * mcap_score +
            self.config.WEIGHT_MOMENTUM * momentum_score +
            self.config.WEIGHT_TRENDING * trend_score +
            self.config.WEIGHT_VOLATILITY * vol_proxy
        )

        return {
            "composite": composite,
            "volume": vol,
            "mcap": mcap,
            "price": price,
            "change_24h": chg_24h,
            "change_7d": chg_7d,
            "vol_score": vol_score,
            "mcap_score": mcap_score,
            "momentum_score": momentum_score,
            "trend_score": trend_score,
            "volatility_score": vol_proxy,
            "is_trending": is_trending,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return scanner state for dashboard/logging."""
        return {
            "enabled": True,
            "total_scans": self.total_scans,
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "last_scan_duration_s": round(self.last_scan_duration, 1),
            "spot_symbols": len(self._cached_spot),
            "futures_symbols": len(self._cached_futures),
            "core_symbols": len(self.core_spot),
            "discovered_symbols": len(self._cached_spot) - len(self.core_spot),
            "alpaca_available": self.coins_on_alpaca,
            "coingecko_coins": self.coins_discovered,
            "kraken_futures": self.futures_discovered,
            "scan_interval_min": self.config.SCAN_INTERVAL_MIN,
            "top_discoveries": [
                {
                    "symbol": sym,
                    "score": round(data["composite"], 3),
                    "volume_m": round(data["volume"] / 1e6, 1),
                    "mcap_b": round(data["mcap"] / 1e9, 1),
                    "change_24h": round(data["change_24h"], 1),
                    "trending": data["is_trending"],
                }
                for sym, data in list(self._cached_scores.items())[:15]
            ],
        }
