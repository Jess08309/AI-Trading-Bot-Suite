"""
AlpacaBot Market Scanner
========================
Scans a universe of liquid, optionable stocks for scalp signals.
Instead of being locked to a fixed watchlist, the scanner finds
the best opportunities dynamically each session.

Flow:
  1. Pre-market: load scanner universe (top liquid optionable stocks)
  2. Each scan cycle: fetch 10-min bars for all candidates
  3. Run 14-indicator signal generation on each
  4. Rank by signal strength
  5. Return top N signals for the engine to trade

Universe criteria:
  - Market cap > $10B (liquid options)
  - Average daily volume > 5M shares
  - Options available on Alpaca
  - Tight bid-ask spreads
"""
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from core.config import Config
from core.api_client import AlpacaAPI
from core.indicators import compute_all_indicators

log = logging.getLogger("alpacabot.scanner")


# ═══════════════════════════════════════════════════════════
#  SCANNER UNIVERSE — data-driven from DTE sweep backtest
#  Each symbol tested at every available expiration (1-30 DTE)
#  Validated: 2025-02-17  |  365d of 10-min bars | 150 tests
# ═══════════════════════════════════════════════════════════

# KEEP tier: profit factor >= 1.2, proven over 1 year at optimal DTE
# AVGO  +$29,066 (+116%) PF 1.34 | 184 trades | 2DTE
# PYPL  +$26,185 (+105%) PF 2.13 |  50 trades | 2DTE
# NFLX  +$25,598 (+102%) PF 1.57 | 122 trades | 2DTE
# NKE   +$23,410 (+94%)  PF 1.27 | 243 trades | 2DTE
# SBUX  +$19,509 (+78%)  PF 1.24 | 349 trades | 9DTE
# F     +$12,067 (+48%)  PF 1.39 | 199 trades | 30DTE
# COST  +$6,947  (+28%)  PF 1.28 |  82 trades | 2DTE
# SHOP  +$6,318  (+25%)  PF 1.27 |  93 trades | 2DTE
# IWM   +$4,679  (+19%)  PF 1.38 |  29 trades | 1DTE (daily exp)
# META  +$4,051  (+16%)  PF 1.20 |  70 trades | 2DTE
TIER1 = ["AVGO", "PYPL", "NFLX", "NKE", "SBUX", "F", "COST", "SHOP", "IWM", "META"]

# MAYBE tier: marginally profitable (PF 1.0-1.2), included for diversification
# JPM +$3,872 PF 1.17 | AMZN +$3,807 PF 1.14 | DIS +$3,786 PF 1.07
# SMH +$3,609 PF 1.20 | RIVN +$3,143 PF 1.10 | XLK +$3,018 PF 1.09
# GM +$2,953 PF 1.10  | SOFI +$2,623 PF 1.08 | QCOM +$2,549 PF 1.06
# MARA +$2,400 PF 1.07 | MA +$2,332 PF 1.16  | AAPL +$2,040 PF 1.16
TIER2 = ["JPM", "AMZN", "DIS", "SMH", "RIVN", "XLK", "GM", "SOFI", "QCOM", "MARA", "MA", "AAPL"]

# WATCH tier: barely positive, monitor for improvement
# WMT +$1,923 PF 1.12 | V +$1,828 PF 1.10 | LLY +$1,778 PF 1.16
# INTC +$1,732 PF 1.12 | ABBV +$1,148 PF 1.11 | COIN +$1,136 PF 1.05
# JNJ +$1,010 PF 1.04  | AMD +$1,003 PF 1.18  | MU +$381 PF 1.05
# UBER +$370 PF 1.02   | NVDA +$175 PF 1.02
TIER3 = ["WMT", "V", "LLY", "INTC", "ABBV", "COIN", "JNJ", "AMD", "MU", "UBER", "NVDA"]

# Eliminated: SNOW(-$2,499), CVX(-$836), ARKK(-$3,398), MSFT(-$4,180),
# TSLA(-$3,870), GOOGL(-$3,910), SPY(-$3,885), QQQ(-$4,198), etc.
TIER4 = []  # all DROP symbols removed from universe

# Full universe — all tiers combined
FULL_UNIVERSE = TIER1 + TIER2 + TIER3 + TIER4
# Remove duplicates while preserving order
SCANNER_UNIVERSE = list(dict.fromkeys(FULL_UNIVERSE))


class MarketScanner:
    """
    Scans a large universe of stocks for scalp signals.
    Returns ranked opportunities sorted by signal strength.
    """

    def __init__(self, config: Config, api: AlpacaAPI):
        self.config = config
        self.api = api

        # Scanner settings
        self.universe = list(SCANNER_UNIVERSE)  # copy
        self.enabled = getattr(config, 'SCANNER_ENABLED', True)
        self.max_scan_symbols = getattr(config, 'SCANNER_MAX_SYMBOLS', 50)
        self.min_signal_score = config.MIN_SIGNAL_SCORE
        self.scan_interval_min = getattr(config, 'SCANNER_INTERVAL_MIN', 10)

        # Price cache for scanned symbols
        self.price_cache: Dict[str, List[float]] = {}
        self.last_scan_time: Optional[datetime] = None
        self.last_scan_results: List[Dict[str, Any]] = []

        # Track which symbols have options available (cache discovery)
        self.has_options: Dict[str, bool] = {}

        # Stats
        self.total_scans = 0
        self.symbols_scanned = 0
        self.signals_found = 0

        # Rate limiting
        self._fetch_delay = 0.4  # seconds between API calls to avoid rate limits

        log.info(f"Scanner initialized: {len(self.universe)} symbols in universe, "
                 f"scanning top {self.max_scan_symbols}")

    def should_scan(self) -> bool:
        """Check if enough time has passed since last scan."""
        if not self.enabled:
            return False
        if self.last_scan_time is None:
            return True
        elapsed = (datetime.now() - self.last_scan_time).total_seconds() / 60
        return elapsed >= self.scan_interval_min

    def scan(self, existing_positions: Dict[str, Any] = None,
             fixed_watchlist: List[str] = None) -> List[Dict[str, Any]]:
        """
        Scan the universe for scalp signals.

        Args:
            existing_positions: current open positions (to avoid duplicates)
            fixed_watchlist: always include these symbols (e.g., MSFT, NVDA)

        Returns:
            List of signal dicts sorted by score (best first), each containing:
              - symbol, direction, score, bull, bear, price, rank
        """
        self.total_scans += 1
        self.last_scan_time = datetime.now()
        scan_start = time.time()

        # Build scan list: fixed watchlist first, then universe
        scan_symbols = []

        # Always include fixed watchlist at the front
        if fixed_watchlist:
            for sym in fixed_watchlist:
                if sym not in scan_symbols:
                    scan_symbols.append(sym)

        # Add universe symbols up to max
        for sym in self.universe:
            if len(scan_symbols) >= self.max_scan_symbols:
                break
            if sym not in scan_symbols:
                scan_symbols.append(sym)

        # Skip symbols we already have positions on
        held_underlyings = set()
        if existing_positions:
            for pos in existing_positions.values():
                held_underlyings.add(pos.get("underlying", ""))

        # Fetch bars and generate signals
        signals = []
        errors = 0

        for i, symbol in enumerate(scan_symbols):
            if symbol in held_underlyings:
                continue

            try:
                # Fetch 10-min bars
                bars = self._fetch_bars(symbol)
                if bars is None or len(bars) < self.config.LOOKBACK_BARS:
                    continue

                # Generate signal
                signal = self._generate_signal(symbol, bars)
                if signal is not None:
                    signals.append(signal)
                    self.signals_found += 1

                self.symbols_scanned += 1

            except Exception as e:
                errors += 1
                log.debug(f"Scanner error on {symbol}: {e}")

            # Rate limiting — sleep after every symbol
            time.sleep(self._fetch_delay)

        # Sort by score (highest first)
        signals.sort(key=lambda s: s["score"], reverse=True)

        # Add rank
        for i, sig in enumerate(signals):
            sig["rank"] = i + 1

        self.last_scan_results = signals
        scan_time = time.time() - scan_start

        log.info(
            f"Scanner: scanned {len(scan_symbols)} symbols in {scan_time:.1f}s | "
            f"Found {len(signals)} signals | Errors: {errors}"
        )

        return signals

    def _fetch_bars(self, symbol: str) -> Optional[np.ndarray]:
        """Fetch 10-min bars for a symbol."""
        try:
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            tf = TimeFrame(10, TimeFrameUnit.Minute)
            bars = self.api.get_bars(symbol, tf, days=3)

            if bars and len(bars) > 0:
                prices = [float(bar.close) for bar in bars]
                self.price_cache[symbol] = prices
                return np.array(prices)
            return None
        except Exception as e:
            log.debug(f"Bar fetch failed for {symbol}: {e}")
            return None

    def _generate_signal(self, symbol: str, prices: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Generate a scalp signal using all 14 indicators.
        Exact same logic as the trading engine.
        """
        chunk = prices[-self.config.LOOKBACK_BARS - 1:]
        indicators = compute_all_indicators(chunk)

        bull, bear = 0, 0

        # 1. RSI
        rsi = indicators.get("rsi", 50)
        if rsi < 25:
            bull += 2
        elif 35 < rsi < 55:
            bull += 1
        elif rsi > 75:
            bear += 2
        elif 50 < rsi < 65:
            bear += 1

        # 2. MACD histogram
        macd_h = indicators.get("macd_hist", 0)
        if macd_h > 0:
            bull += 1
            if macd_h > 0.1:
                bull += 1
        elif macd_h < 0:
            bear += 1
            if macd_h < -0.1:
                bear += 1

        # 3. Stochastic
        stoch = indicators.get("stochastic", 50)
        if stoch < 20:
            bull += 1
        elif stoch > 80:
            bear += 1

        # 4. Bollinger Band position
        bb = indicators.get("bb_position", 0.5)
        if bb < 0.10:
            bull += 2
        elif bb > 0.90:
            bear += 2
        elif bb < 0.30:
            bull += 1
        elif bb > 0.70:
            bear += 1

        # 5. ATR normalized
        atr_n = indicators.get("atr_normalized", 0)
        if atr_n > 0.005:
            if bull > bear:
                bull += 1
            elif bear > bull:
                bear += 1

        # 6. CCI
        cci_val = indicators.get("cci", 0)
        if cci_val < -100:
            bull += 1
        elif cci_val > 100:
            bear += 1

        # 7. ROC
        roc_val = indicators.get("roc", 0)
        if roc_val > 0.3:
            bull += 1
        elif roc_val < -0.3:
            bear += 1

        # 8. Williams %R
        wr = indicators.get("williams_r", -50)
        if wr > -20:
            bear += 1
        elif wr < -80:
            bull += 1

        # 9. Volatility Ratio
        vol_r = indicators.get("volatility_ratio", 1.0)
        if vol_r > 1.3:
            if bull > bear:
                bull += 1
            elif bear > bull:
                bear += 1

        # 10. Z-Score
        zs = indicators.get("zscore", 0)
        if zs < -2.0:
            bull += 1
        elif zs > 2.0:
            bear += 1

        # 11. Trend Strength
        ts = indicators.get("trend_strength", 0)
        if ts > 25:
            if bull > bear:
                bull += 1
            elif bear > bull:
                bear += 1

        # 12-14. Price momentum
        pc1 = indicators.get("price_change_1", 0)
        pc5 = indicators.get("price_change_5", 0)
        if pc1 > 0.001:
            bull += 1
        elif pc1 < -0.001:
            bear += 1
        if pc5 > 0.003:
            bull += 1
        elif pc5 < -0.003:
            bear += 1

        # Decision
        direction = None
        score = 0
        if bull >= self.min_signal_score and bull > bear + 1:
            direction = "call"
            score = bull
        elif bear >= self.min_signal_score and bear > bull + 1:
            direction = "put"
            score = bear

        if direction is None:
            return None

        current_price = float(chunk[-1])

        # ── Momentum sanity check ──
        # Don't buy puts on stocks ripping UP, or calls on stocks tanking.
        # Uses 20-bar (~3.3hr) and 5-bar (~50min) price change.
        pc20 = indicators.get("price_change_20", 0) if len(chunk) > 20 else 0
        pc5_val = indicators.get("price_change_5", 0)
        if direction == "put" and pc20 > 0.015 and pc5_val > 0.005:
            # Stock is up >1.5% over 20 bars AND still rising — don't short
            return None
        if direction == "call" and pc20 < -0.015 and pc5_val < -0.005:
            # Stock is down >1.5% over 20 bars AND still falling — don't go long
            return None

        return {
            "symbol": symbol,
            "direction": direction,
            "score": score,
            "bull": bull,
            "bear": bear,
            "price": current_price,
            "rsi": indicators.get("rsi", 0),
            "macd_h": indicators.get("macd_hist", 0),
            "bb_pos": indicators.get("bb_position", 0.5),
            "trend": indicators.get("trend_strength", 0),
            "timestamp": datetime.now().isoformat(),
        }

    def get_state(self) -> Dict[str, Any]:
        """Return scanner state for dashboard."""
        return {
            "enabled": self.enabled,
            "universe_size": len(self.universe),
            "max_scan": self.max_scan_symbols,
            "total_scans": self.total_scans,
            "symbols_scanned": self.symbols_scanned,
            "signals_found": self.signals_found,
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "scan_interval_min": self.scan_interval_min,
            "latest_signals": [
                {
                    "rank": s.get("rank", 0),
                    "symbol": s["symbol"],
                    "direction": s["direction"],
                    "score": s["score"],
                    "price": s["price"],
                    "rsi": round(s.get("rsi", 0), 1),
                    "trend": round(s.get("trend", 0), 1),
                }
                for s in self.last_scan_results[:15]  # top 15 for dashboard
            ],
        }
