"""
AlpacaBot Universe Scanner
===========================
Dynamically discovers and ranks stocks for options scalping.
Expands beyond the static 33-symbol universe to scan 500+ stocks.

Data source: Alpaca API (already authenticated, no extra keys needed)
  - get_all_assets() → all tradeable US equities
  - get_stock_snapshot() → current price/volume for batch scoring

Flow:
  1. Fetch all active, tradeable US equities from Alpaca
  2. Batch-fetch snapshots for price/volume filtering
  3. Score each stock for scalping suitability
  4. Return ranked list to the existing MarketScanner

The existing scanner's 14-indicator signal generation still applies
downstream — this just expands what gets scanned.
"""
import json
import logging
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

log = logging.getLogger("alpacabot.universe_scanner")

# ═══════════════════════════════════════════════════════════
#  SEED UNIVERSE — large curated list of optionable stocks
#  Used as starting point; Alpaca asset discovery adds more
# ═══════════════════════════════════════════════════════════

# S&P 500 + Nasdaq 100 + popular options stocks (deduped)
SEED_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN",
    "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SNPS", "CDNS", "NXPI",
    "ADI", "ON", "MCHP", "FTNT", "PANW", "CRWD", "ZS", "NET",
    "DDOG", "SNOW", "PLTR", "UBER", "ABNB", "DASH", "COIN", "SQ",
    "PYPL", "SHOP", "MELI", "SE", "RBLX", "U", "TTWO", "EA",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC",
    "SCHW", "BLK", "AXP", "COF", "DFS", "V", "MA", "FIS", "FISV",
    "ICE", "CME", "NDAQ", "SPGI", "MCO", "BRK.B",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    "DHR", "BMY", "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT",
    "SYK", "BSX", "EW", "ZTS", "DXCM", "ILMN", "MRNA", "CVS",
    # Consumer
    "WMT", "COST", "HD", "LOW", "TGT", "TJX", "ROST",
    "DG", "DLTR", "NKE", "LULU", "SBUX", "MCD", "YUM", "CMG",
    "DPZ", "WYNN", "MGM", "LVS", "MAR", "HLT", "BKNG", "ABNB",
    "PG", "KO", "PEP", "MDLZ", "CL", "KMB", "GIS", "HSY",
    # Industrials
    "CAT", "DE", "GE", "HON", "MMM", "BA", "LMT", "RTX", "NOC",
    "GD", "TDG", "ITW", "EMR", "ROK", "CARR", "OTIS", "UPS",
    "FDX", "CSX", "UNP", "NSC", "DAL", "UAL", "AAL", "LUV",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
    "OXY", "DVN", "HAL", "BKR", "FANG", "HES", "PXD",
    # Communications
    "NFLX", "DIS", "CMCSA", "CHTR", "TMUS", "VZ", "T",
    "PARA", "WBD", "SPOT", "ROKU",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "GOLD",
    # REITs / Utilities
    "AMT", "PLD", "EQIX", "SPG", "O", "DLR", "NEE", "DUK", "SO",
    # ETFs (highly liquid options)
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
    "SMH", "ARKK", "GLD", "SLV", "TLT", "HYG", "EEM", "EFA",
    "XBI", "KWEB", "SOXX",
    # High-vol momentum names
    "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM", "SOFI",
    "HOOD", "MARA", "RIOT", "BITF", "CLSK", "HUT",
    "SMCI", "ARM", "IONQ", "RGTI", "QUBT",
    "CELH", "ENPH", "SEDG", "FSLR", "RUN",
    # Additional options-active names
    "AI", "AFRM", "UPST", "OPEN", "RDFN", "Z", "ZG",
    "DKNG", "PENN", "GNOG", "RSI", "CHWY", "W", "ETSY",
    "WISH", "CLOV", "PLBY", "SPCE", "ASTR",
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

# Deduplicate
SEED_UNIVERSE = list(dict.fromkeys(SEED_UNIVERSE))


# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class UniverseScannerConfig:
    """Configuration for the universe scanner."""
    # Scan frequency
    SCAN_INTERVAL_MIN: int = 30

    # Asset filters
    MIN_PRICE: float = 5.0          # Skip penny stocks
    MAX_PRICE: float = 2000.0       # Skip BRK.A etc.
    MIN_DAILY_VOLUME: int = 500_000 # Minimum daily volume
    MIN_DAILY_DOLLAR_VOL: float = 10e6  # $10M min dollar volume

    # How many symbols to return
    MAX_SCAN_SYMBOLS: int = 200     # Top N stocks for scanning

    # Scoring weights
    WEIGHT_VOLUME: float = 0.30     # Higher volume = better options liquidity
    WEIGHT_MOMENTUM: float = 0.25   # Price movement (scalping needs movement)
    WEIGHT_VOLATILITY: float = 0.25 # Higher intraday range = more scalp opportunities
    WEIGHT_SPREAD: float = 0.10     # Tighter spread = better fills
    WEIGHT_TREND: float = 0.10      # Clear trend = directional scalps

    # Probation: only for symbols NOT in any backtest result
    PROBATION_HOURS: int = 48          # 2 days observation before eligible

    # Always include these (proven performers from backtest)
    ALWAYS_INCLUDE: Set[str] = field(default_factory=lambda: {
        "AVGO", "PYPL", "NFLX", "NKE", "SBUX", "F", "COST", "SHOP",
        "IWM", "META", "JPM", "AMZN", "DIS", "SMH", "RIVN", "XLK",
        "GM", "SOFI", "QCOM", "MARA", "MA", "AAPL",
    })

    # ── Backtest Qualification Gate ──
    # Loaded dynamically from data/state/backtest_grades.json
    # Run: python tools/backtest_qualified.py to regenerate
    # Grade A+B pass immediately. Grade F permanently blocked.
    # Grade C+D and unknown symbols go to time-based probation.
    BACKTEST_QUALIFIED: Set[str] = field(default_factory=set)
    BACKTEST_BLOCKED: Set[str] = field(default_factory=set)
    GRADES_MAX_AGE_DAYS: int = 30  # Warn if grades older than this


# ═══════════════════════════════════════════════════════════
#  UNIVERSE SCANNER
# ═══════════════════════════════════════════════════════════

class StockUniverseScanner:
    """
    Discovers and ranks stocks for AlpacaBot's options scalping strategy.

    Uses Alpaca's own API to:
      1. Discover all tradeable US equities
      2. Batch-fetch current snapshots
      3. Score by scalping suitability (volume, movement, spread)
      4. Return ranked list for MarketScanner to evaluate
    """

    def __init__(self, api, config: UniverseScannerConfig = None):
        """
        Args:
            api: AlpacaAPI instance (already connected)
            config: Scanner config (defaults if None)
        """
        self.api = api
        self.config = config or UniverseScannerConfig()

        # Load dynamic grades from backtest_grades.json
        self._load_backtest_grades("alpacabot")

        # Cached results
        self._cached_universe: List[str] = list(SEED_UNIVERSE)
        self._cached_scores: Dict[str, Dict] = {}
        self._last_scan_time: Optional[datetime] = None
        self._scan_lock = threading.Lock()

        # Probation tracking — only for symbols NOT in any backtest result
        self._first_seen: Dict[str, str] = {}  # symbol -> ISO datetime
        self._probation_dir = os.path.join("data", "state")
        self._probation_file = os.path.join(self._probation_dir, "scanner_probation.json")
        self._load_probation()

        # Stats
        self.total_scans = 0
        self.last_scan_duration = 0.0
        self.assets_discovered = 0
        self.assets_passed_filter = 0

        log.info(
            f"StockUniverseScanner initialized | seed={len(SEED_UNIVERSE)} "
            f"| qualified={len(self.config.BACKTEST_QUALIFIED)} "
            f"| blocked={len(self.config.BACKTEST_BLOCKED)} "
            f"| probation_tracked={len(self._first_seen)}"
        )

    # ── Dynamic grades loading ────────────────────────────────

    def _load_backtest_grades(self, bot_name: str):
        """Load qualified/blocked sets from backtest_grades.json (dynamic, not hardcoded)."""
        grades_path = os.path.join("data", "state", "backtest_grades.json")
        try:
            if os.path.exists(grades_path):
                with open(grades_path, "r") as f:
                    data = json.load(f)
                bot_data = data.get("bots", {}).get(bot_name, {})
                self.config.BACKTEST_QUALIFIED = set(bot_data.get("qualified", []))
                self.config.BACKTEST_BLOCKED = set(bot_data.get("blocked", []))

                # Check staleness
                generated = data.get("generated", "")
                if generated:
                    gen_dt = datetime.fromisoformat(generated)
                    age_days = (datetime.now() - gen_dt).days
                    if age_days > self.config.GRADES_MAX_AGE_DAYS:
                        log.warning(
                            f"⚠ Backtest grades are {age_days} days old! "
                            f"Run: python tools/backtest_qualified.py to refresh"
                        )
                    else:
                        log.info(
                            f"Loaded backtest grades (v{data.get('version', '?')}, "
                            f"{age_days}d old, method={data.get('method', '?')})"
                        )
                log.info(
                    f"  Grades: {len(self.config.BACKTEST_QUALIFIED)} qualified, "
                    f"{len(self.config.BACKTEST_BLOCKED)} blocked"
                )
            else:
                log.warning(
                    f"No backtest_grades.json found at {grades_path} — "
                    f"all unknown symbols will enter probation. "
                    f"Run: python tools/backtest_qualified.py"
                )
        except Exception as e:
            log.warning(f"Failed to load backtest grades: {e}")

    # ── Probation persistence ────────────────────────────────

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

    def get_universe(self) -> List[str]:
        """Get current ranked universe of stock symbols."""
        return list(self._cached_universe)

    def scan(self) -> List[str]:
        """
        Run a full universe scan. Returns ranked list of symbols.
        """
        with self._scan_lock:
            return self._do_scan()

    def _do_scan(self) -> List[str]:
        """Internal scan implementation."""
        scan_start = time.time()
        self.total_scans += 1
        self._last_scan_time = datetime.now()

        log.info("═══ Stock Universe Scan Starting ═══")

        # ── Step 1: Get all tradeable assets from Alpaca ──
        try:
            all_symbols = self._discover_assets()
            log.info(f"  Discovered {len(all_symbols)} tradeable equities")
        except Exception as e:
            log.warning(f"Asset discovery failed: {e}, using seed universe")
            all_symbols = list(SEED_UNIVERSE)

        # ── Step 2: Combine with seed universe (dedup) ──
        combined = list(dict.fromkeys(list(SEED_UNIVERSE) + all_symbols))
        log.info(f"  Combined universe: {len(combined)} unique symbols")

        # ── Step 3: Batch-fetch snapshots ──
        snapshots = self._batch_snapshots(combined)
        log.info(f"  Got snapshots for {len(snapshots)} symbols")

        # ── Step 4: Filter and score ──
        scored = []
        for sym, snap in snapshots.items():
            if not self._passes_filter(sym, snap):
                continue
            score = self._score_stock(sym, snap)
            scored.append((sym, score))

        self.assets_passed_filter = len(scored)

        # Sort by composite score
        scored.sort(key=lambda x: x[1]["composite"], reverse=True)

        # ── Step 5: Build final universe (backtest qualification gate) ──
        # Always-include symbols first (proven — always tradeable)
        universe = list(self.config.ALWAYS_INCLUDE)
        seen = set(universe)

        now = datetime.now()
        qualified_added = []
        blocked_count = 0
        in_probation = []
        newly_discovered = []

        for sym, score in scored:
            if len(universe) >= self.config.MAX_SCAN_SYMBOLS:
                break
            if sym in seen:
                continue

            # Gate 1: Permanently blocked (Grade F backtest failures)
            if sym in self.config.BACKTEST_BLOCKED:
                blocked_count += 1
                continue

            # Gate 2: Pre-qualified (Grade A+B backtest winners)
            if sym in self.config.BACKTEST_QUALIFIED:
                universe.append(sym)
                seen.add(sym)
                qualified_added.append(sym)
                continue

            # Gate 3: Unknown symbol (not in any backtest) → time probation
            if sym not in self._first_seen:
                self._first_seen[sym] = now.isoformat()
                newly_discovered.append(sym)
                in_probation.append(sym)
            else:
                first = datetime.fromisoformat(self._first_seen[sym])
                hours = (now - first).total_seconds() / 3600
                if hours >= self.config.PROBATION_HOURS:
                    universe.append(sym)  # Graduated from probation
                    seen.add(sym)
                else:
                    in_probation.append(sym)

        self._save_probation()

        if qualified_added:
            log.info(f"  QUALIFIED: {len(qualified_added)} symbols passed backtest gate (A+B)")
        if blocked_count:
            log.info(f"  BLOCKED: {blocked_count} symbols failed backtest (Grade F)")
        if newly_discovered:
            log.info(
                f"  NEW: {len(newly_discovered)} unknown symbols → "
                f"{self.config.PROBATION_HOURS}h probation"
            )
        if in_probation:
            log.info(
                f"  PROBATION: {len(in_probation)} symbols observing "
                f"(not yet tradeable)"
            )

        # Cache
        self._cached_universe = universe
        self._cached_scores = {sym: data for sym, data in scored}
        self.last_scan_duration = time.time() - scan_start

        log.info(
            f"═══ Stock Universe Scan Complete ({self.last_scan_duration:.1f}s) ═══\n"
            f"  Universe: {len(universe)} symbols "
            f"({len(self.config.ALWAYS_INCLUDE)} locked + "
            f"{len(universe) - len(self.config.ALWAYS_INCLUDE)} discovered)\n"
            f"  Filtered: {len(combined)} candidates → {len(scored)} scored → "
            f"{len(universe)} selected"
        )

        # Log top discoveries
        top_new = [
            (sym, data) for sym, data in scored[:30]
            if sym not in self.config.ALWAYS_INCLUDE
        ][:10]
        if top_new:
            log.info("  Top new discoveries:")
            for sym, data in top_new:
                log.info(
                    f"    {sym:<6} score={data['composite']:.3f} | "
                    f"price=${data['price']:>8,.2f} | "
                    f"vol={data['volume']/1e6:>6,.1f}M | "
                    f"chg={data['change_pct']:>+6.2f}% | "
                    f"range={data['intraday_range_pct']:.2f}%"
                )

        return universe

    def _discover_assets(self) -> List[str]:
        """Get all tradeable US equities from Alpaca."""
        try:
            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetClass, AssetStatus

            request = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY,
            )
            assets = self.api._trading.get_all_assets(request)
            self.assets_discovered = len(assets)

            # Filter for real equities on major exchanges
            symbols = []
            skip_patterns = (
                '.', '-', 'TEST', 'ZZZZ',  # warrants, units, test symbols
            )
            valid_exchanges = {'NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS', 'NYSE ARCA'}

            for a in assets:
                if not a.tradable:
                    continue
                sym = a.symbol
                # Skip symbols with dots/dashes (warrants, preferred shares, units)
                if any(p in sym for p in skip_patterns[:2]):
                    continue
                if len(sym) > 5:  # Skip long symbols (usually warrants, units)
                    continue
                # Only major exchanges
                exchange = getattr(a, 'exchange', '') or ''
                if exchange and exchange not in valid_exchanges:
                    continue
                symbols.append(sym)

            return symbols
        except Exception as e:
            log.warning(f"Alpaca asset discovery error: {e}")
            return []

    def _batch_snapshots(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch snapshots for all symbols in batches."""
        from alpaca.data.requests import StockSnapshotRequest

        results = {}
        batch_size = 500
        batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]

        for i, batch in enumerate(batches):
            try:
                request = StockSnapshotRequest(symbol_or_symbols=batch)
                snapshots = self.api._data.get_stock_snapshot(request)

                for sym, snap in snapshots.items():
                    try:
                        daily = snap.daily_bar
                        prev = snap.previous_daily_bar
                        minute = snap.minute_bar
                        quote = snap.latest_quote

                        if not daily or not prev:
                            continue

                        results[sym] = {
                            "price": float(daily.close),
                            "volume": float(daily.volume),
                            "open": float(daily.open),
                            "high": float(daily.high),
                            "low": float(daily.low),
                            "close": float(daily.close),
                            "prev_close": float(prev.close),
                            "prev_volume": float(prev.volume),
                            "change_pct": (float(daily.close) / float(prev.close) - 1) * 100 if float(prev.close) > 0 else 0,
                            "intraday_range_pct": ((float(daily.high) - float(daily.low)) / float(daily.close) * 100) if float(daily.close) > 0 else 0,
                            "bid": float(quote.bid_price) if quote and quote.bid_price else 0,
                            "ask": float(quote.ask_price) if quote and quote.ask_price else 0,
                            "dollar_volume": float(daily.close) * float(daily.volume),
                        }
                    except Exception:
                        continue

                log.debug(f"  Batch {i+1}/{len(batches)}: {len(batch)} requested, {len(results)} total")
                time.sleep(0.5)  # Rate limit between batches
            except Exception as e:
                log.warning(f"Snapshot batch {i+1} failed: {e}")
                time.sleep(1.0)

        return results

    def _passes_filter(self, symbol: str, snap: Dict) -> bool:
        """Check if a stock passes basic filters."""
        price = snap.get("price", 0)
        volume = snap.get("volume", 0)
        dollar_vol = snap.get("dollar_volume", 0)

        if price < self.config.MIN_PRICE or price > self.config.MAX_PRICE:
            return False
        if volume < self.config.MIN_DAILY_VOLUME:
            return False
        if dollar_vol < self.config.MIN_DAILY_DOLLAR_VOL:
            return False
        return True

    def _score_stock(self, symbol: str, snap: Dict) -> Dict:
        """Score a stock for scalping suitability."""
        price = snap.get("price", 0)
        volume = snap.get("volume", 0)
        change_pct = snap.get("change_pct", 0)
        intraday_range = snap.get("intraday_range_pct", 0)
        bid = snap.get("bid", 0)
        ask = snap.get("ask", 0)
        dollar_vol = snap.get("dollar_volume", 0)

        # Volume score: log-scaled, cap at $1B daily
        vol_score = min(1.0, dollar_vol / 1e9)

        # Momentum score: we want movement (absolute change)
        abs_chg = abs(change_pct)
        if abs_chg < 0.5:
            momentum_score = abs_chg * 0.5  # Low movement, less interesting
        elif abs_chg < 5.0:
            momentum_score = min(1.0, abs_chg / 5.0)
        else:
            momentum_score = max(0.3, 1.0 - (abs_chg - 5) / 20)  # Too crazy, less reliable

        # Volatility score: intraday range
        if intraday_range < 0.5:
            vol_proxy = intraday_range * 0.5
        elif intraday_range < 4.0:
            vol_proxy = min(1.0, intraday_range / 4.0)
        else:
            vol_proxy = max(0.4, 1.0 - (intraday_range - 4) / 10)

        # Spread score: tighter is better
        spread_score = 0.5  # default
        if bid > 0 and ask > 0 and price > 0:
            spread_pct = (ask - bid) / price * 100
            if spread_pct < 0.02:
                spread_score = 1.0
            elif spread_pct < 0.1:
                spread_score = 0.8
            elif spread_pct < 0.5:
                spread_score = 0.5
            else:
                spread_score = 0.2

        # Trend score: price relative to previous close
        trend_score = min(1.0, abs(change_pct) / 3.0)

        # Composite
        composite = (
            self.config.WEIGHT_VOLUME * vol_score +
            self.config.WEIGHT_MOMENTUM * momentum_score +
            self.config.WEIGHT_VOLATILITY * vol_proxy +
            self.config.WEIGHT_SPREAD * spread_score +
            self.config.WEIGHT_TREND * trend_score
        )

        return {
            "composite": composite,
            "price": price,
            "volume": volume,
            "dollar_volume": dollar_vol,
            "change_pct": change_pct,
            "intraday_range_pct": intraday_range,
            "vol_score": vol_score,
            "momentum_score": momentum_score,
            "volatility_score": vol_proxy,
            "spread_score": spread_score,
            "trend_score": trend_score,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return scanner state for dashboard/logging."""
        return {
            "enabled": True,
            "total_scans": self.total_scans,
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "last_scan_duration_s": round(self.last_scan_duration, 1),
            "universe_size": len(self._cached_universe),
            "always_include": len(self.config.ALWAYS_INCLUDE),
            "discovered": self.assets_discovered,
            "passed_filter": self.assets_passed_filter,
            "scan_interval_min": self.config.SCAN_INTERVAL_MIN,
            "top_stocks": [
                {
                    "symbol": sym,
                    "score": round(data["composite"], 3),
                    "price": data["price"],
                    "volume_m": round(data["volume"] / 1e6, 1),
                    "change_pct": round(data["change_pct"], 2),
                    "range_pct": round(data["intraday_range_pct"], 2),
                }
                for sym, data in list(self._cached_scores.items())[:20]
            ],
        }
