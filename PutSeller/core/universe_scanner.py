"""
PutSeller Universe Scanner
===========================
Dynamically discovers and ranks stocks for credit put spread selling.
Expands beyond the static 21-symbol watchlist to scan 500+ stocks.

Strategy-specific scoring:
  - Favors STABLE uptrending stocks (put sellers want no crashes)
  - Favors HIGH IV / HIGH VOLUME (better premiums, better fills)  
  - Penalizes high-momentum / volatile names (crash risk)
  - Favors large-cap (more liquid options, tighter spreads)

Data source: Alpaca API (already authenticated)
"""
import json
import logging
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

log = logging.getLogger("putseller.universe_scanner")

# ═══════════════════════════════════════════════════════════
#  SEED UNIVERSE — large-cap stocks ideal for put selling
# ═══════════════════════════════════════════════════════════

SEED_UNIVERSE = [
    # Blue-chip tech (stable, liquid options)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN",
    "MU", "AMAT", "LRCX", "MRVL", "SNPS", "CDNS", "NOW", "WDAY",
    "PANW", "CRWD", "FTNT", "NFLX",
    # Financials (steady, high OI)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC",
    "SCHW", "BLK", "AXP", "COF", "DFS", "V", "MA", "BRK.B",
    "FIS", "FISV", "ICE", "CME", "SPGI", "MCO",
    # Healthcare (defensive, steady)
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    "DHR", "BMY", "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT",
    "SYK", "BSX", "EW", "ZTS", "CVS", "CI", "HUM", "MCK",
    # Consumer staples (defensive, low vol)
    "WMT", "COST", "PG", "KO", "PEP", "MDLZ", "CL", "KMB",
    "GIS", "K", "HSY", "MO", "PM", "STZ", "TAP", "SJM",
    # Consumer discretionary (growth + options liquidity)
    "HD", "LOW", "TGT", "TJX", "ROST", "NKE", "LULU", "SBUX",
    "MCD", "YUM", "CMG", "DPZ", "MAR", "HLT", "BKNG",
    # Industrials (cyclical but stable)
    "CAT", "DE", "GE", "HON", "BA", "LMT", "RTX", "NOC",
    "GD", "ITW", "EMR", "UPS", "FDX", "CSX", "UNP", "NSC",
    "WM", "RSG", "CARR", "OTIS",
    # Energy (high IV, good premiums)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
    "OXY", "DVN", "HAL", "BKR", "FANG",
    # Communications
    "DIS", "CMCSA", "CHTR", "TMUS", "VZ", "T",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX",
    # REITs
    "AMT", "PLD", "EQIX", "SPG", "O", "DLR",
    # Utilities (defensive, boring = perfect for put selling)
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL",
    # ETFs (most liquid options in the world)
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
    "XLU", "XLP", "XLI", "XLB", "XLRE", "XLC", "XLY",
    "SMH", "GLD", "SLV", "TLT", "HYG", "EEM", "EFA",
    "XBI", "IBB", "SOXX",
    # Additional high-OI names frequently used for credit spreads
    "F", "GM", "RIVN", "LCID", "PLTR", "COIN", "SQ", "PYPL",
    "SHOP", "UBER", "ABNB", "DASH", "SOFI", "HOOD",
    "AI", "SMCI", "ARM", "MARA", "RIOT", "DKNG",
    "SNAP", "PINS", "ROKU", "SPOT", "RBLX",
    "ENPH", "FSLR", "RUN", "SEDG",
    "MRNA", "BNTX",
    "CVNA", "W", "CHWY", "ETSY",
]

SEED_UNIVERSE = list(dict.fromkeys(SEED_UNIVERSE))


# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class PutScannerConfig:
    """Configuration tuned for put selling strategy."""
    SCAN_INTERVAL_MIN: int = 60         # Scan every hour (put selling is slower)

    # Asset filters
    MIN_PRICE: float = 10.0             # Need decent premium
    MAX_PRICE: float = 2000.0
    MIN_DAILY_VOLUME: int = 500_000
    MIN_DAILY_DOLLAR_VOL: float = 20e6  # $20M min (need liquid options)

    # Universe size
    MAX_SCAN_SYMBOLS: int = 150         # Top N stocks

    # Scoring weights — DIFFERENT from scalping!
    # Put sellers want: stability + volume + IV, NOT momentum
    WEIGHT_STABILITY: float = 0.30      # Stable uptrend (low drawdown risk)
    WEIGHT_OPTION_LIQUIDITY: float = 0.25  # High volume = liquid options
    WEIGHT_IV_PROXY: float = 0.25       # Higher IV = better premiums
    WEIGHT_MARKET_CAP: float = 0.15     # Larger = safer
    WEIGHT_SECTOR_SAFETY: float = 0.05  # Defensive sectors get a bonus

    # Probation: only for symbols NOT in any backtest result
    PROBATION_HOURS: int = 72           # 3 days observation (conservative for puts)

    # Always include (existing watchlist)
    ALWAYS_INCLUDE: Set[str] = field(default_factory=lambda: {
        "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "AMD", "CRM", "AVGO", "NFLX", "JPM", "V",
        "MA", "HD", "UNH", "PG", "COST",
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

class PutSellerUniverseScanner:
    """
    Discovers and ranks stocks for credit put spread selling.

    Scoring is OPPOSITE of momentum scanners:
      - Stability > Momentum (we DON'T want crashes)
      - Volume = options liquidity
      - IV proxy = better premiums  
      - Large cap = safer
    """

    def __init__(self, api, config: PutScannerConfig = None):
        self.api = api
        self.config = config or PutScannerConfig()

        # Load dynamic grades from backtest_grades.json
        self._load_backtest_grades("putseller")

        self._cached_universe: List[str] = list(self.config.ALWAYS_INCLUDE)
        self._cached_scores: Dict[str, Dict] = {}
        self._last_scan_time: Optional[datetime] = None
        self._scan_lock = threading.Lock()

        # Probation tracking — only for symbols NOT in any backtest result
        self._first_seen: Dict[str, str] = {}  # symbol -> ISO datetime
        self._probation_dir = os.path.join("data", "state")
        self._probation_file = os.path.join(self._probation_dir, "scanner_probation.json")
        self._load_probation()

        self.total_scans = 0
        self.last_scan_duration = 0.0
        self.assets_discovered = 0
        self.assets_passed_filter = 0

        log.info(
            f"PutSellerUniverseScanner initialized | seed={len(SEED_UNIVERSE)} "
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
        if self._last_scan_time is None:
            return True
        elapsed = (datetime.now() - self._last_scan_time).total_seconds() / 60
        return elapsed >= self.config.SCAN_INTERVAL_MIN

    def get_universe(self) -> List[str]:
        return list(self._cached_universe)

    def scan(self) -> List[str]:
        with self._scan_lock:
            return self._do_scan()

    def _do_scan(self) -> List[str]:
        scan_start = time.time()
        self.total_scans += 1
        self._last_scan_time = datetime.now()

        log.info("═══ PutSeller Universe Scan Starting ═══")

        # ── Step 1: Discover assets ──
        try:
            all_symbols = self._discover_assets()
            log.info(f"  Discovered {len(all_symbols)} tradeable equities")
        except Exception as e:
            log.warning(f"Asset discovery failed: {e}, using seed")
            all_symbols = list(SEED_UNIVERSE)

        # ── Step 2: Combine with seed ──
        combined = list(dict.fromkeys(list(SEED_UNIVERSE) + all_symbols))
        log.info(f"  Combined universe: {len(combined)} unique symbols")

        # ── Step 3: Batch snapshots ──
        snapshots = self._batch_snapshots(combined)
        log.info(f"  Got snapshots for {len(snapshots)} symbols")

        # ── Step 4: Filter and score for put selling ──
        scored = []
        for sym, snap in snapshots.items():
            if not self._passes_filter(sym, snap):
                continue
            score = self._score_for_put_selling(sym, snap)
            scored.append((sym, score))

        self.assets_passed_filter = len(scored)
        scored.sort(key=lambda x: x[1]["composite"], reverse=True)

        # ── Step 5: Build universe (backtest qualification gate) ──
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

        self._cached_universe = universe
        self._cached_scores = {sym: data for sym, data in scored}
        self.last_scan_duration = time.time() - scan_start

        log.info(
            f"═══ PutSeller Universe Scan Complete ({self.last_scan_duration:.1f}s) ═══\n"
            f"  Universe: {len(universe)} symbols\n"
            f"  Filtered: {len(combined)} → {len(scored)} scored → {len(universe)} selected"
        )

        top_new = [
            (sym, data) for sym, data in scored[:30]
            if sym not in self.config.ALWAYS_INCLUDE
        ][:10]
        if top_new:
            log.info("  Top new put-selling candidates:")
            for sym, data in top_new:
                log.info(
                    f"    {sym:<6} score={data['composite']:.3f} | "
                    f"price=${data['price']:>8,.2f} | "
                    f"vol={data['volume']/1e6:>6,.1f}M | "
                    f"stability={data['stability_score']:.2f} | "
                    f"iv_proxy={data['iv_proxy_score']:.2f}"
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

            symbols = []
            for a in assets:
                if not a.tradable:
                    continue
                sym = a.symbol
                if '.' in sym or '-' in sym:
                    continue
                if len(sym) > 5:
                    continue
                symbols.append(sym)

            return symbols
        except Exception as e:
            log.warning(f"Asset discovery error: {e}")
            return []

    def _batch_snapshots(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch snapshots in batches."""
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
                        quote = snap.latest_quote

                        if not daily or not prev:
                            continue

                        price = float(daily.close)
                        prev_close = float(prev.close)

                        results[sym] = {
                            "price": price,
                            "volume": float(daily.volume),
                            "open": float(daily.open),
                            "high": float(daily.high),
                            "low": float(daily.low),
                            "close": price,
                            "prev_close": prev_close,
                            "prev_volume": float(prev.volume),
                            "change_pct": (price / prev_close - 1) * 100 if prev_close > 0 else 0,
                            "intraday_range_pct": ((float(daily.high) - float(daily.low)) / price * 100) if price > 0 else 0,
                            "bid": float(quote.bid_price) if quote and quote.bid_price else 0,
                            "ask": float(quote.ask_price) if quote and quote.ask_price else 0,
                            "dollar_volume": price * float(daily.volume),
                        }
                    except Exception:
                        continue

                time.sleep(0.5)
            except Exception as e:
                log.warning(f"Snapshot batch {i+1} failed: {e}")
                time.sleep(1.0)

        return results

    def _passes_filter(self, symbol: str, snap: Dict) -> bool:
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

    def _score_for_put_selling(self, symbol: str, snap: Dict) -> Dict:
        """Score a stock specifically for PUT SELLING suitability.

        Key differences from scalping/momentum scoring:
          - STABILITY is king (we want boring stocks that won't crash)
          - IV proxy matters (higher IV = better premiums)
          - Volume = options liquidity
          - We PENALIZE high momentum (crash risk)
        """
        price = snap.get("price", 0)
        volume = snap.get("volume", 0)
        change_pct = snap.get("change_pct", 0)
        intraday_range = snap.get("intraday_range_pct", 0)
        dollar_vol = snap.get("dollar_volume", 0)

        # STABILITY score: small change + small range = stable
        # Put sellers love boring. Penalize big moves.
        abs_chg = abs(change_pct)
        if abs_chg < 0.5:
            stability_score = 1.0  # Beautiful — barely moved
        elif abs_chg < 1.5:
            stability_score = 0.8  # Normal
        elif abs_chg < 3.0:
            stability_score = 0.5  # Getting volatile
        elif abs_chg < 5.0:
            stability_score = 0.3  # Risky
        else:
            stability_score = 0.1  # Too volatile for put selling

        # Bonus if stock is UP (we're selling puts = bullish)
        if change_pct > 0.5:
            stability_score = min(1.0, stability_score + 0.1)
        # Penalty if stock is DOWN significantly (could keep falling)
        if change_pct < -2.0:
            stability_score *= 0.7

        # OPTION LIQUIDITY score
        liquidity_score = min(1.0, dollar_vol / 500e6)  # $500M = perfect

        # IV PROXY score: higher intraday range = higher implied vol = better premiums
        # But not too high (that means the stock is unstable)
        if intraday_range < 0.5:
            iv_score = 0.3  # Low IV, low premiums
        elif intraday_range < 2.0:
            iv_score = min(1.0, intraday_range / 2.0)  # Sweet spot
        elif intraday_range < 4.0:
            iv_score = 0.8  # Decent IV but getting risky
        else:
            iv_score = 0.4  # Too volatile, premiums good but risk high

        # MARKET CAP proxy: higher price * volume ≈ larger company
        mcap_proxy = min(1.0, dollar_vol / 1e9)

        # SECTOR SAFETY: ETFs and defensive names get a bonus
        defensive_etfs = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
                         "XLU", "XLP", "XLI", "GLD", "TLT", "SMH"}
        defensive_stocks = {"WMT", "PG", "KO", "PEP", "JNJ", "COST", "UNH",
                           "MCD", "DUK", "SO", "NEE", "V", "MA"}
        sector_score = 0.5
        if symbol in defensive_etfs:
            sector_score = 1.0
        elif symbol in defensive_stocks:
            sector_score = 0.9

        composite = (
            self.config.WEIGHT_STABILITY * stability_score +
            self.config.WEIGHT_OPTION_LIQUIDITY * liquidity_score +
            self.config.WEIGHT_IV_PROXY * iv_score +
            self.config.WEIGHT_MARKET_CAP * mcap_proxy +
            self.config.WEIGHT_SECTOR_SAFETY * sector_score
        )

        return {
            "composite": composite,
            "price": price,
            "volume": volume,
            "dollar_volume": dollar_vol,
            "change_pct": change_pct,
            "intraday_range_pct": intraday_range,
            "stability_score": stability_score,
            "liquidity_score": liquidity_score,
            "iv_proxy_score": iv_score,
            "mcap_proxy_score": mcap_proxy,
            "sector_score": sector_score,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "total_scans": self.total_scans,
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "last_scan_duration_s": round(self.last_scan_duration, 1),
            "universe_size": len(self._cached_universe),
            "always_include": len(self.config.ALWAYS_INCLUDE),
            "passed_filter": self.assets_passed_filter,
            "scan_interval_min": self.config.SCAN_INTERVAL_MIN,
            "top_stocks": [
                {
                    "symbol": sym,
                    "score": round(data["composite"], 3),
                    "price": data["price"],
                    "stability": round(data["stability_score"], 2),
                    "iv_proxy": round(data["iv_proxy_score"], 2),
                    "liquidity": round(data["liquidity_score"], 2),
                }
                for sym, data in list(self._cached_scores.items())[:20]
            ],
        }
