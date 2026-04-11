"""
CallBuyer Universe Scanner
============================
Dynamically discovers and ranks stocks for momentum call buying.
Expands beyond the static 18-symbol watchlist to scan 500+ stocks.

Strategy-specific scoring:
  - Favors HIGH MOMENTUM (breakout candidates, strong uptrends)
  - Favors VOLUME SURGES (institutional interest)
  - Favors VOLATILITY (bigger moves = bigger call profits)
  - Penalizes stable/flat stocks (no edge for call buying)

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

log = logging.getLogger("callbuyer.universe_scanner")

# ═══════════════════════════════════════════════════════════
#  SEED UNIVERSE — momentum/growth stocks ideal for call buying
# ═══════════════════════════════════════════════════════════

SEED_UNIVERSE = [
    # High-beta tech (big movers, liquid options)
    "NVDA", "TSLA", "AMD", "META", "AMZN", "GOOGL", "NFLX", "AVGO",
    "CRM", "SHOP", "SQ", "COIN", "SMCI", "ARM", "MRVL", "MU",
    "AMAT", "LRCX", "QCOM", "INTC", "ADBE", "ORCL", "NOW", "WDAY",
    # AI / Growth plays
    "PLTR", "AI", "IONQ", "RGTI", "QUBT", "SNOW", "DDOG", "CRWD",
    "PANW", "ZS", "NET", "CFLT", "MDB", "ESTC", "S", "OKTA",
    # Momentum movers
    "UBER", "ABNB", "DASH", "SOFI", "HOOD", "AFRM", "UPST",
    "CVNA", "CPNG", "GRAB", "DUOL", "APP", "TTD", "RBLX",
    # EV / Clean energy (high beta)
    "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM",
    "ENPH", "FSLR", "RUN", "SEDG", "PLUG", "BE",
    # Crypto-adjacent (volatile, great for calls during bull runs)
    "MARA", "RIOT", "BITF", "CLSK", "HUT",
    # eVTOL / Speculative growth
    "ACHR", "JOBY", "LILM", "EVTL",
    # Biotech (catalyst-driven, explosive moves)
    "MRNA", "BNTX", "CRSP", "BEAM", "EDIT", "NTLA",
    "XBI",  # Biotech ETF
    # Social media / Consumer digital
    "SNAP", "PINS", "MTCH", "BMBL", "ROKU", "SPOT",
    # E-commerce / Consumer
    "W", "CHWY", "ETSY", "DKNG", "PENN", "CPRT", "MNST",
    # Chinese ADRs (volatile, big moves)
    "BABA", "JD", "PDD", "BIDU", "NTES", "KWEB",
    # Semis
    "TSM", "ASML", "SOXX",
    # Financials (when they run, they RUN)
    "JPM", "GS", "SCHW", "MS", "COF",
    # Energy (high beta in commodities)
    "XOM", "CVX", "COP", "OXY", "SLB", "DVN", "FANG",
    "XLE",  # Energy ETF
    # ETFs (for sector momentum)
    "QQQ", "SPY", "IWM", "SMH", "ARKK", "XLK", "XBI",
    # Additional high-vol names
    "LYFT", "OPEN", "RDFN", "CELH",
    "SE", "MELI",
    "BA", "CAT", "DE",
    # Options-active large caps
    "AAPL", "MSFT", "V", "MA", "HD", "UNH", "LLY",
    "COST", "WMT", "PG", "DIS", "CMCSA",
]

SEED_UNIVERSE = list(dict.fromkeys(SEED_UNIVERSE))


# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class CallScannerConfig:
    """Configuration tuned for call buying / momentum strategy."""
    SCAN_INTERVAL_MIN: int = 30

    # Asset filters
    MIN_PRICE: float = 5.0
    MAX_PRICE: float = 2000.0
    MIN_DAILY_VOLUME: int = 750_000     # Need good volume for options
    MIN_DAILY_DOLLAR_VOL: float = 15e6  # $15M minimum

    # Universe size
    MAX_SCAN_SYMBOLS: int = 150         # Top N stocks

    # Scoring weights — MOMENTUM FOCUSED
    WEIGHT_MOMENTUM: float = 0.35       # Price movement is #1 priority
    WEIGHT_VOLUME_SURGE: float = 0.25   # Volume surge = institutional buying
    WEIGHT_VOLATILITY: float = 0.20     # Higher vol = bigger moves
    WEIGHT_TREND: float = 0.15          # Uptrend confirmation
    WEIGHT_OPTION_LIQUID: float = 0.05  # Minimum liquidity

    # Probation: only for symbols NOT in any backtest result
    PROBATION_HOURS: int = 48          # 2 days observation before eligible

    # Always include (existing watchlist)
    ALWAYS_INCLUDE: Set[str] = field(default_factory=lambda: {
        "NVDA", "TSLA", "AMD", "META", "AMZN", "GOOGL", "NFLX", "AVGO",
        "CRM", "SHOP", "SQ", "COIN", "QQQ", "SPY", "IWM", "JPM", "GS", "XOM",
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

class CallBuyerUniverseScanner:
    """
    Discovers and ranks stocks for momentum call buying.

    Scoring is MOMENTUM-FIRST:
      - Big movers > Small movers
      - Volume surges > Normal volume
      - Volatile > Stable
      - Uptrend > Downtrend
    """

    def __init__(self, api, config: CallScannerConfig = None):
        self.api = api
        self.config = config or CallScannerConfig()

        # Load dynamic grades from backtest_grades.json
        self._load_backtest_grades("callbuyer")

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
            f"CallBuyerUniverseScanner initialized | seed={len(SEED_UNIVERSE)} "
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

        log.info("═══ CallBuyer Universe Scan Starting ═══")

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

        # ── Step 4: Filter and score for momentum ──
        scored = []
        for sym, snap in snapshots.items():
            if not self._passes_filter(sym, snap):
                continue
            score = self._score_for_call_buying(sym, snap)
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
        graduated_count = len(universe) - len(self.config.ALWAYS_INCLUDE)
        if graduated_count > 0:
            log.info(f"  GRADUATED: {graduated_count} symbols passed probation")

        self._cached_universe = universe
        self._cached_scores = {sym: data for sym, data in scored}
        self.last_scan_duration = time.time() - scan_start

        log.info(
            f"═══ CallBuyer Universe Scan Complete ({self.last_scan_duration:.1f}s) ═══\n"
            f"  Universe: {len(universe)} symbols\n"
            f"  Filtered: {len(combined)} → {len(scored)} scored → {len(universe)} selected"
        )

        top_new = [
            (sym, data) for sym, data in scored[:30]
            if sym not in self.config.ALWAYS_INCLUDE
        ][:10]
        if top_new:
            log.info("  Top momentum candidates:")
            for sym, data in top_new:
                log.info(
                    f"    {sym:<6} score={data['composite']:.3f} | "
                    f"price=${data['price']:>8,.2f} | "
                    f"chg={data['change_pct']:>+6.2f}% | "
                    f"vol_surge={data['volume_surge']:.1f}x | "
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
                        prev_vol = float(prev.volume)

                        results[sym] = {
                            "price": price,
                            "volume": float(daily.volume),
                            "open": float(daily.open),
                            "high": float(daily.high),
                            "low": float(daily.low),
                            "close": price,
                            "prev_close": prev_close,
                            "prev_volume": prev_vol,
                            "change_pct": (price / prev_close - 1) * 100 if prev_close > 0 else 0,
                            "intraday_range_pct": ((float(daily.high) - float(daily.low)) / price * 100) if price > 0 else 0,
                            "volume_surge": float(daily.volume) / prev_vol if prev_vol > 0 else 1.0,
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

    def _score_for_call_buying(self, symbol: str, snap: Dict) -> Dict:
        """Score a stock specifically for CALL BUYING / MOMENTUM suitability.

        Key differences from put selling:
          - MOMENTUM is king (we want stocks that MOVE)
          - VOLUME SURGE = institutional buying signal
          - Higher volatility = bigger potential gains
          - UPTREND preferred (we're buying calls = bullish)
        """
        price = snap.get("price", 0)
        volume = snap.get("volume", 0)
        change_pct = snap.get("change_pct", 0)
        intraday_range = snap.get("intraday_range_pct", 0)
        volume_surge = snap.get("volume_surge", 1.0)
        dollar_vol = snap.get("dollar_volume", 0)

        # MOMENTUM score: bigger positive moves = better
        # We WANT stocks going UP (buying calls)
        if change_pct > 5.0:
            momentum_score = 1.0  # Ripping — prime call territory
        elif change_pct > 3.0:
            momentum_score = 0.9
        elif change_pct > 1.5:
            momentum_score = 0.7
        elif change_pct > 0.5:
            momentum_score = 0.5
        elif change_pct > 0.0:
            momentum_score = 0.3  # Barely up
        elif change_pct > -1.0:
            momentum_score = 0.2  # Slightly down — maybe bounce
        elif change_pct > -3.0:
            momentum_score = 0.1  # Down, less interesting
        else:
            momentum_score = 0.05  # Dumping — not for calls

        # VOLUME SURGE score: more volume than usual = big interest
        if volume_surge > 3.0:
            vol_surge_score = 1.0  # 3x normal volume = major event
        elif volume_surge > 2.0:
            vol_surge_score = 0.8
        elif volume_surge > 1.5:
            vol_surge_score = 0.6
        elif volume_surge > 1.0:
            vol_surge_score = 0.4
        else:
            vol_surge_score = 0.2  # Below average volume

        # VOLATILITY score: higher range = bigger potential gains
        if intraday_range > 5.0:
            volatility_score = 1.0
        elif intraday_range > 3.0:
            volatility_score = 0.8
        elif intraday_range > 1.5:
            volatility_score = 0.6
        elif intraday_range > 0.5:
            volatility_score = 0.3
        else:
            volatility_score = 0.1  # Dead flat — no edge

        # TREND score: uptrend confirmation
        if change_pct > 2.0:
            trend_score = 1.0
        elif change_pct > 0.5:
            trend_score = 0.7
        elif change_pct > -0.5:
            trend_score = 0.4  # Flat
        else:
            trend_score = 0.1  # Downtrend

        # OPTION LIQUIDITY baseline
        liquidity_score = min(1.0, dollar_vol / 200e6)

        composite = (
            self.config.WEIGHT_MOMENTUM * momentum_score +
            self.config.WEIGHT_VOLUME_SURGE * vol_surge_score +
            self.config.WEIGHT_VOLATILITY * volatility_score +
            self.config.WEIGHT_TREND * trend_score +
            self.config.WEIGHT_OPTION_LIQUID * liquidity_score
        )

        return {
            "composite": composite,
            "price": price,
            "volume": volume,
            "dollar_volume": dollar_vol,
            "change_pct": change_pct,
            "intraday_range_pct": intraday_range,
            "volume_surge": volume_surge,
            "momentum_score": momentum_score,
            "vol_surge_score": vol_surge_score,
            "volatility_score": volatility_score,
            "trend_score": trend_score,
            "liquidity_score": liquidity_score,
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
                    "change_pct": round(data["change_pct"], 2),
                    "volume_surge": round(data["volume_surge"], 1),
                    "momentum": round(data["momentum_score"], 2),
                }
                for sym, data in list(self._cached_scores.items())[:20]
            ],
        }
