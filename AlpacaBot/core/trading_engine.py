"""
AlpacaBot SCALP Trading Engine v3.0 — AI-Assisted
====================================================
Per-symbol DTE options scalp strategy with ML ensemble.

AI Stack (same architecture as the crypto bot):
  1. ML Model: GradientBoosting trained on price history (20 features)
  2. Market Sentiment: Fear & Greed, VIX, SPY trend, sector momentum
  3. RL Shadow Agent: Q-learning agent observing trades (shadow mode)
  4. Meta-Learner: dynamically weights ML + Sentiment + Rule-based scores

Trade Decision Flow:
  - 14 indicators → bull/bear score (rule-based, same as v2.4)
  - ML model predicts direction + confidence
  - Sentiment provides market mood context
  - Meta-learner ensemble combines all three → final score
  - Quality gates: MIN_ML_CONFIDENCE + ensemble threshold + rule score
  - RL shadow agent observes and learns (does NOT affect live trades yet)

Safety (prevents blowup after good streak):
  - Model quality gate: rejects retrains below 54% accuracy
  - Model rollback: keeps last 5 versions, can revert if new model degrades
  - Dynamic thresholds: meta-learner tightens entry requirements on losing streaks
  - If ML fails, rule_score weight automatically increases (graceful degradation)
  - RL only graduates to live influence after 50-trade outperformance proof
"""
import csv
import hashlib
import json
import logging
import os
import time
import threading
import numpy as np
from dataclasses import asdict
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from collections import defaultdict
from zoneinfo import ZoneInfo

from core.config import Config
from core.api_client import AlpacaAPI
from core.indicators import compute_all_indicators
from core.options_handler import OptionsHandler
from core.risk_manager import RiskManager
from core.scanner import MarketScanner

# AI/ML components
from utils.ml_model import OptionsMLModel
from utils.sentiment import MarketSentimentAnalyzer
from utils.rl_agent import RLShadowAgent
from utils.meta_learner import MetaLearner

log = logging.getLogger("alpacabot.engine")


class ScalpTradingEngine:
    """
    Scalp options trading engine.
    Fetches 5-min bars, generates scalp signals, trades 1DTE options.
    Exposes state for the web dashboard.
    """

    def __init__(self, config: Config):
        self.config = config
        self.api = AlpacaAPI(config)
        self.options = OptionsHandler(config, self.api)
        self.risk = RiskManager(config)
        self.scanner = MarketScanner(config, self.api)

        # Universe Scanner — dynamically discovers stocks beyond static tiers
        try:
            from core.universe_scanner import StockUniverseScanner
            self.universe_scanner = StockUniverseScanner(api=self.api)
            log.info("Universe Scanner loaded — dynamic stock discovery enabled")
        except Exception as e:
            self.universe_scanner = None
            log.warning(f"Universe Scanner not available: {e}")

        # Active positions: key = option_symbol
        self.positions: Dict[str, Dict[str, Any]] = {}

        # Price history per symbol (5-min closes)
        self.price_bars: Dict[str, List[float]] = {}
        self.bar_timestamps: Dict[str, List[str]] = {}
        self.last_bar_fetch: Dict[str, datetime] = {}

        # Signal cooldowns per symbol
        self.cooldowns: Dict[str, datetime] = {}

        # Activity log for dashboard
        self.activity_log: List[Dict[str, Any]] = []
        self.max_activity_log = 200

        # Trade history
        self.trade_history: List[Dict[str, Any]] = []

        # Stats
        self.cycle = 0
        self.signals_today = 0
        self.trades_today = 0
        self.start_time = datetime.now()
        self.running = False
        self.status_text = "Initializing..."
        self.last_signal_check = None

        # Thread lock for dashboard reads
        self._lock = threading.Lock()

        # ── AI/ML Components ──
        self.ml_model = OptionsMLModel(model_dir="data/models", min_accuracy=0.54)
        import os
        self.sentiment = MarketSentimentAnalyzer(
            api_client=self.api,
            finnhub_key=os.getenv("FINNHUB_API_KEY", ""),
        )
        self.rl_agent = RLShadowAgent(
            state_file="data/state/rl_agent.json",
            shadow_report_file="data/state/rl_shadow_report.json",
            shadow_events_file="data/state/rl_shadow_events.jsonl",
        )
        self.meta_learner = MetaLearner(state_file="data/state/meta_learner.json")
        self.ml_ready = False
        self.last_model_retrain = datetime.now()
        self.MODEL_RETRAIN_HOURS = 8  # Retrain every 8 hours
        self.MIN_ML_CONFIDENCE = 0.55  # Minimum ML confidence to trade (lowered from 0.65 — was blocking all signals)
        self.RL_SHADOW_MODE = True     # RL observes but doesn't influence

        # SPY regime cache (refreshed every 10 min) — used to block counter-trend trades
        self._spy_regime: str = "neutral"   # "bull", "bear", or "neutral"
        self._spy_regime_last: float = 0.0  # epoch seconds of last fetch
        self._spy_sma20: float = 0.0
        self._spy_sma50: float = 0.0
        self._spy_price: float = 0.0

        # Pre-market warmup state
        self._warmup_done_date: Optional[date] = None  # date of last warmup
        self._warmup_data: Dict[str, Any] = {}          # warmup results
        self._WARMUP_MINUTES_BEFORE_OPEN = 30            # start warmup N min before open

        # Regime flip detection state
        self._regime_history: List[tuple] = []     # [(epoch, regime_str)]
        self._last_flip_time: float = 0.0
        self._last_flip_from: str = ""
        self._last_flip_to: str = ""
        self._regime_flip_cooldown_min: float = 45  # cooldown after regime flip
        self._regime_flip_severity: float = 0.0
        self._regime_flip_mult: float = 1.0         # position size multiplier during flip
        self._regime_flip_block: bool = False        # block entries during severe flip

        # Put vs Call win-rate tracking — auto-disable puts when WR < 30%
        self._put_wins: int = 0
        self._put_losses: int = 0
        self._call_wins: int = 0
        self._call_losses: int = 0
        self._recent_put_outcomes: List[bool] = []   # last 20 put outcomes (True=win)
        self._puts_auto_disabled: bool = False

        # Ensure directories
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.TRADE_LOG) or "data", exist_ok=True)
        os.makedirs(os.path.dirname(config.STATE_FILE) or "data/state", exist_ok=True)
        os.makedirs("data/models", exist_ok=True)

    # ==========================================================
    #  PUBLIC STATE (for dashboard)
    # ==========================================================

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Thread-safe snapshot of bot state for the web dashboard."""
        with self._lock:
            # Account info
            try:
                acct = self.api.get_account()
            except Exception:
                acct = {"equity": 0, "cash": 0, "buying_power": 0, "portfolio_value": 0}

            positions_list = []
            for sym, pos in self.positions.items():
                pnl = 0
                pnl_pct = 0
                if pos.get("entry_price", 0) > 0:
                    pnl = (pos.get("current_price", 0) - pos["entry_price"]) * 100 * pos.get("qty", 1)
                    pnl_pct = (pos.get("current_price", 0) - pos["entry_price"]) / pos["entry_price"]
                positions_list.append({
                    "symbol": sym,
                    "underlying": pos.get("underlying", ""),
                    "type": pos.get("option_type", ""),
                    "direction": pos.get("direction", ""),
                    "strike": pos.get("strike", 0),
                    "expiration": pos.get("expiration", ""),
                    "entry_price": pos.get("entry_price", 0),
                    "current_price": pos.get("current_price", 0),
                    "peak_price": pos.get("peak_price", 0),
                    "qty": pos.get("qty", 0),
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "entry_time": pos.get("entry_time", ""),
                    "score": pos.get("score", 0),
                })

            # Compute session P&L
            closed_pnl = sum(t.get("pnl", 0) for t in self.trade_history)
            open_pnl = sum(p["pnl"] for p in positions_list)

            return {
                "running": self.running,
                "status": self.status_text,
                "uptime": str(datetime.now() - self.start_time).split(".")[0],
                "cycle": self.cycle,
                "account": acct,
                "positions": positions_list,
                "num_positions": len(positions_list),
                "max_positions": self.config.MAX_POSITIONS,
                "trade_history": list(reversed(self.trade_history[-50:])),
                "activity_log": list(reversed(self.activity_log[-30:])),
                "signals_today": self.signals_today,
                "trades_today": self.trades_today,
                "closed_pnl": closed_pnl,
                "open_pnl": open_pnl,
                "total_pnl": closed_pnl + open_pnl,
                "risk": {
                    "daily_pnl": self.risk.daily_pnl,
                    "consecutive_losses": self.risk.consecutive_losses,
                    "breaker_active": self.risk.breaker_active,
                    "breaker_reason": self.risk.breaker_reason,
                    "drawdown": (self.risk.current_balance - self.risk.peak_balance) / max(1, self.risk.peak_balance),
                    **self.risk.get_throttle(),  # graduated response state
                },
                "config": {
                    "watchlist": self.config.WATCHLIST,
                    "scanner": self.config.SCANNER_ENABLED,
                    "dte": "per-symbol",
                    "stop_loss": self.config.STOP_LOSS_PCT,
                    "take_profit": self.config.TAKE_PROFIT_PCT,
                    "trailing_stop": self.config.TRAILING_STOP_PCT,
                    "max_positions": self.config.MAX_POSITIONS,
                    "paper": self.config.PAPER,
                },
                "scanner": self.scanner.get_state(),
                "prices": {sym: bars[-1] if bars else 0 for sym, bars in self.price_bars.items()},
                "last_signal_check": self.last_signal_check.isoformat() if self.last_signal_check else None,
                "timestamp": datetime.now().isoformat(),
                # AI/ML status
                "ai": {
                    "ml_model": self.ml_model.status(),
                    "sentiment": self.sentiment.status(),
                    "rl_agent": self.rl_agent.get_shadow_report(),
                    "meta_learner": self.meta_learner.status(),
                    "ml_ready": self.ml_ready,
                    "rl_shadow_mode": self.RL_SHADOW_MODE,
                },
            }

    def _log_activity(self, msg: str, level: str = "info"):
        """Add to activity log (shown on dashboard)."""
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "msg": msg,
            "level": level,
        }
        with self._lock:
            self.activity_log.append(entry)
            if len(self.activity_log) > self.max_activity_log:
                self.activity_log = self.activity_log[-self.max_activity_log:]
        getattr(log, level, log.info)(msg)

    # ==========================================================
    #  MAIN LOOP
    # ==========================================================

    def run(self):
        """Main scalp trading loop."""
        log.info("=" * 60)
        log.info("AlpacaBot SCALP Engine v3.0 — AI-Assisted Starting")
        log.info("=" * 60)
        log.info(self.config.summary())

        if not self.api.connect():
            log.error("Failed to connect to Alpaca. Check API keys.")
            return

        # Load saved state
        self.risk.load_state()
        self._load_positions()
        self._load_trade_history()
        self._load_put_call_wr()

        # Initialize ML model
        if self.ml_model.load_model():
            self.ml_ready = True
            self._log_activity("ML Model loaded successfully (ml_ready=yes)")
        else:
            self._log_activity("ML Model not found — attempting immediate bootstrap train...")
            self.ml_ready = False
            # Try immediate training with historical data from Alpaca
            self._bootstrap_ml_model()

        self._log_activity(
            f"AI Stack: ML={self.ml_ready} | Sentiment=ON | "
            f"RL={'SHADOW' if self.RL_SHADOW_MODE else 'LIVE'} | "
            f"MetaLearner=ON | MIN_ML_CONF={self.MIN_ML_CONFIDENCE}"
        )

        # Sync balance — capped to allocation % (PutSeller gets the rest)
        try:
            acct = self.api.get_account()
            allocated = acct["portfolio_value"] * self.config.ALLOCATION_PCT
            self.risk.current_balance = allocated
            if allocated > self.risk.peak_balance:
                self.risk.peak_balance = allocated
            self._log_activity(f"Account: ${acct['portfolio_value']:,.2f} equity, ${acct['buying_power']:,.2f} buying power")
            self._log_activity(f"AlpacaBot allocation ({self.config.ALLOCATION_PCT:.0%}): ${allocated:,.2f}")
        except Exception as e:
            self._log_activity(f"Failed to sync account: {e}", "warning")

        self._save_runtime_fingerprint()

        self.running = True
        self.status_text = "Running - waiting for market"
        self._log_activity(f"Bot started. Watchlist: {', '.join(self.config.WATCHLIST)}")
        self._log_activity(f"Strategy: per-symbol DTE scalp, SL {self.config.STOP_LOSS_PCT:.0%}, TP +{self.config.TAKE_PROFIT_PCT:.0%}")

        while self.running:
            try:
                self.cycle += 1

                # Market open check
                if not self._is_market_open():
                    # Pre-market warmup: run once, ~30 min before open
                    if self._should_warmup():
                        self.status_text = "Pre-market warmup"
                        self._pre_market_warmup()
                    else:
                        self.status_text = "Market closed - waiting"
                    self._sleep(30)
                    continue

                self.status_text = "Active - monitoring"

                # Phase 0: Universe scan (every 30 min) — expand scanner universe
                if (self.universe_scanner
                        and self.universe_scanner.should_scan()
                        and self._is_market_open()):
                    try:
                        expanded = self.universe_scanner.scan()
                        self.scanner.universe = list(expanded)
                        self.scanner.max_scan_symbols = max(
                            self.config.SCANNER_MAX_SYMBOLS,
                            min(len(expanded), 200)
                        )
                        self._log_activity(
                            f"UNIVERSE: Expanded to {len(expanded)} symbols "
                            f"({self.universe_scanner.assets_passed_filter} passed filters)"
                        )
                    except Exception as e:
                        self._log_activity(f"Universe scan failed: {e}", "warning")

                # Phase 1: Update prices (5-min bars)
                self._fetch_bars()

                # Phase 2: Check exits on open positions
                self._check_exits()

                # Phase 3: Generate signals and open new positions
                if self.cycle % self.config.SIGNAL_CHECK_BARS == 0:
                    self._scan_and_trade()

                # Phase 4: Housekeeping
                if self.cycle % 6 == 0:
                    self._log_status()
                if self.cycle % 12 == 0:
                    self._save_state()

                # Phase 5: ML model retrain check (every ~30 cycles = 7.5 min)
                if self.cycle % 30 == 0:
                    self._maybe_retrain_model()
                    # Update meta-learner thresholds based on performance
                    win_rate = self._recent_win_rate()
                    self.meta_learner.update_thresholds(
                        win_rate,
                        self.risk.consecutive_losses,
                        self.risk.rolling_loss_rate(),
                    )

                # Sleep for bar interval (but check exits more often)
                self._sleep(self.config.RISK_CHECK_INTERVAL)

            except KeyboardInterrupt:
                self._log_activity("Shutdown requested (Ctrl+C)")
                self.running = False
            except Exception as e:
                self._log_activity(f"Error in cycle {self.cycle}: {e}", "error")
                log.error(f"Cycle error details:", exc_info=True)
                self._sleep(30)

        # Clean shutdown
        self.status_text = "Shutting down..."
        self._save_state()
        self._log_activity("Bot stopped")

    # ==========================================================
    #  DATA FETCHING (10-min bars)
    # ==========================================================

    def _fetch_bars(self):
        """Fetch latest 10-min bars for all watchlist symbols."""
        for symbol in self.config.WATCHLIST:
            last = self.last_bar_fetch.get(symbol)
            # Only fetch every 60 seconds minimum
            if last and (datetime.now() - last).total_seconds() < 55:
                continue

            try:
                from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                tf = TimeFrame(10, TimeFrameUnit.Minute)
                bars = self.api.get_bars(symbol, tf, days=3)

                if bars:
                    prices = [float(bar.close) for bar in bars]
                    timestamps = [str(bar.timestamp) for bar in bars]
                    with self._lock:
                        self.price_bars[symbol] = prices
                        self.bar_timestamps[symbol] = timestamps
                    self.last_bar_fetch[symbol] = datetime.now()

                    if len(prices) > 0:
                        log.debug(f"{symbol}: {len(prices)} bars, latest ${prices[-1]:.2f}")
            except Exception as e:
                log.warning(f"Bar fetch failed for {symbol}: {e}")

    def _get_prices(self, symbol: str) -> Optional[np.ndarray]:
        """Get price array for a symbol from cache."""
        with self._lock:
            bars = self.price_bars.get(symbol)
        if bars and len(bars) >= self.config.LOOKBACK_BARS:
            return np.array(bars)
        return None

    def _get_market_regime(self) -> str:
        """Return SPY market regime: 'bull', 'bear', or 'neutral'.

        Uses SPY's 20-day AND 50-day SMA.
        SPY above MA20 AND MA20 > MA50 = strong bull; below both = bear.
        Cached for 10 minutes to avoid extra API calls.
        Also stores SMA20/SMA50/price for the put regime gate.

        Returns:
            str: 'bull', 'bear', or 'neutral' (neutral = data unavailable)
        """
        if time.time() - self._spy_regime_last < 600:
            return self._spy_regime   # use cache

        try:
            from alpaca.data.timeframe import TimeFrame
            bars = self.api.get_bars("SPY", TimeFrame.Day, days=60)
            if bars and len(bars) >= 50:
                closes = [float(b.close) for b in bars]
                ma20 = float(np.mean(closes[-20:]))
                ma50 = float(np.mean(closes[-50:]))
                current = closes[-1]
                prev = closes[-2] if len(closes) >= 2 else current

                # Store for put regime gate
                self._spy_sma20 = ma20
                self._spy_sma50 = ma50
                self._spy_price = current

                # Require both current AND previous bar above/below MA to avoid whipsaws
                if current > ma20 and prev > ma20:
                    regime = "bull"
                elif current < ma20 and prev < ma20:
                    regime = "bear"
                else:
                    regime = "neutral"
                log.info(
                    f"[Regime] SPY ${current:.2f} vs MA20 ${ma20:.2f} / "
                    f"MA50 ${ma50:.2f} → {regime.upper()}"
                )
                self._detect_regime_flip(regime)
                self._spy_regime = regime
            elif bars and len(bars) >= 20:
                # Fallback: only 20-day data available
                closes = [float(b.close) for b in bars]
                ma20 = float(np.mean(closes[-20:]))
                current = closes[-1]
                prev = closes[-2] if len(closes) >= 2 else current
                self._spy_sma20 = ma20
                self._spy_price = current
                if current > ma20 and prev > ma20:
                    regime = "bull"
                elif current < ma20 and prev < ma20:
                    regime = "bear"
                else:
                    regime = "neutral"
                log.info(f"[Regime] SPY ${current:.2f} vs MA20 ${ma20:.2f} → {regime.upper()}")
                self._detect_regime_flip(regime)
                self._spy_regime = regime
        except Exception as e:
            log.debug(f"Regime filter: SPY fetch failed ({e}) — using cached '{self._spy_regime}'")

        self._spy_regime_last = time.time()
        return self._spy_regime

    # ── Regime Flip Detection ────────────────────────────
    _FLIP_SEVERITY_MAP = {
        ("bull", "bear"): 1.0,
        ("bear", "bull"): 0.9,
        ("bull", "neutral"): 0.4,
        ("bear", "neutral"): 0.35,
        ("neutral", "bear"): 0.6,
        ("neutral", "bull"): 0.4,
    }

    def _detect_regime_flip(self, new_regime: str):
        """Track regime transitions and set flip cooldown/blocking."""
        now = time.time()
        old_regime = self._spy_regime

        # Record history (keep last 20)
        self._regime_history.append((now, new_regime))
        if len(self._regime_history) > 20:
            self._regime_history = self._regime_history[-20:]

        if old_regime and new_regime != old_regime:
            severity = self._FLIP_SEVERITY_MAP.get(
                (old_regime, new_regime), 0.5
            )
            self._last_flip_time = now
            self._last_flip_from = old_regime
            self._last_flip_to = new_regime
            self._regime_flip_severity = severity

            # Count flips in last 4 hours for whipsaw detection
            cutoff = now - 4 * 3600
            flips = 0
            for i in range(1, len(self._regime_history)):
                ts, reg = self._regime_history[i]
                if ts < cutoff:
                    continue
                prev_reg = self._regime_history[i - 1][1]
                if reg != prev_reg:
                    flips += 1

            whipsaw = flips >= 3

            # Set multiplier and blocking
            if whipsaw:
                self._regime_flip_mult = 0.30
                self._regime_flip_block = True
            elif severity >= 0.8:
                self._regime_flip_mult = 0.40
                self._regime_flip_block = True
            else:
                self._regime_flip_mult = max(0.50, 1.0 - severity * 0.5)
                self._regime_flip_block = False

            log.warning(
                f"[Regime FLIP] {old_regime.upper()} → {new_regime.upper()} | "
                f"severity={severity:.2f}, mult={self._regime_flip_mult:.2f}, "
                f"block={self._regime_flip_block}, "
                f"flips_4h={flips}{' WHIPSAW' if whipsaw else ''}"
            )
        elif self._last_flip_time > 0:
            # Check if cooldown has expired
            elapsed_min = (now - self._last_flip_time) / 60
            if elapsed_min >= self._regime_flip_cooldown_min:
                if self._regime_flip_mult < 1.0:
                    log.info(
                        f"[Regime] Flip cooldown expired — "
                        f"restoring normal sizing (was {self._regime_flip_mult:.2f}x)"
                    )
                self._regime_flip_mult = 1.0
                self._regime_flip_block = False
                self._regime_flip_severity = 0.0
            else:
                # Fade the multiplier toward 1.0 as cooldown expires
                remaining_pct = 1.0 - (elapsed_min / self._regime_flip_cooldown_min)
                base_mult = max(0.30, 1.0 - self._regime_flip_severity * 0.5)
                self._regime_flip_mult = base_mult + (1.0 - base_mult) * (1.0 - remaining_pct)
                # Unblock entries after first half of cooldown
                if remaining_pct < 0.5:
                    self._regime_flip_block = False

    def _spy_trend_is_up(self) -> bool:
        """Return True if SPY's 20-bar trend is UP.
        Checks: SMA20 > SMA50  OR  price > SMA20.
        Used by the strict put regime gate.
        """
        # Ensure regime data is fresh
        self._get_market_regime()
        if self._spy_sma20 > 0 and self._spy_sma50 > 0:
            return self._spy_sma20 > self._spy_sma50 or self._spy_price > self._spy_sma20
        if self._spy_sma20 > 0:
            return self._spy_price > self._spy_sma20
        return self._spy_regime == "bull"

    # ==========================================================
    #  TIME-OF-DAY WINDOW
    # ==========================================================

    def _get_time_window(self) -> str:
        """Return 'morning', 'afternoon', or 'closed' based on current ET time.

        Morning (9:30-11:00 ET): aggressive — lower entry thresholds.
        Afternoon (11:00-16:00 ET): picky — only strong signals.
        Closed: outside market hours.
        """
        now_et = datetime.now(ZoneInfo("America/New_York"))
        minutes_since_open = (now_et.hour - 9) * 60 + (now_et.minute - 30)

        if minutes_since_open < 0 or minutes_since_open >= 390:  # before 9:30 or after 16:00
            return "closed"
        elif minutes_since_open < getattr(self.config, 'MORNING_WINDOW_END_MIN', 90):
            return "morning"
        else:
            return "afternoon"

    # ==========================================================
    #  SCALP SIGNAL GENERATION (all 14 indicators)
    # ==========================================================

    def _generate_scalp_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Generate a scalp signal using all 14 indicators.
        This is the EXACT same logic that produced +66.7% in backtest.
        Returns signal dict or None.
        """
        prices = self._get_prices(symbol)
        if prices is None:
            return None

        # Use last LOOKBACK bars
        chunk = prices[-self.config.LOOKBACK_BARS - 1:]
        indicators = compute_all_indicators(chunk)

        bull, bear = 0, 0

        # 1. RSI: oversold bounce or momentum
        rsi = indicators.get("rsi", 50)
        if rsi < 25:
            bull += 2
        elif 35 < rsi < 55:
            bull += 1
        elif rsi > 75:
            bear += 2
        elif 50 < rsi < 65:
            bear += 1

        # 2. MACD histogram: momentum direction
        macd_h = indicators.get("macd_hist", 0)
        if macd_h > 0:
            bull += 1
            if macd_h > 0.1:
                bull += 1
        elif macd_h < 0:
            bear += 1
            if macd_h < -0.1:
                bear += 1

        # 3. Stochastic: timing entry
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

        # 5. ATR (normalized): volatility check
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

        # 9. Volatility Ratio: breakout detection
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

        # 12-14. Short-term price momentum
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

        # ── Time-of-Day Threshold Adjustment ──
        time_window = self._get_time_window()
        effective_min_score = self.config.MIN_SIGNAL_SCORE
        if getattr(self.config, 'MORNING_WINDOW_ENABLED', False):
            if time_window == "morning":
                effective_min_score = max(2, self.config.MIN_SIGNAL_SCORE
                                         - self.config.MORNING_SCORE_REDUCTION)
            elif time_window == "afternoon":
                effective_min_score = self.config.MIN_SIGNAL_SCORE \
                                     + self.config.AFTERNOON_SCORE_INCREASE

        # Decision
        direction = None
        score = 0
        if bull >= effective_min_score and bull > bear + 1:
            direction = "call"
            score = bull
        elif bear >= effective_min_score and bear > bull + 1:
            direction = "put"
            score = bear

        if direction is None:
            return None

        # ── Graduated Response: direction lock ──
        can_dir, dir_reason = self.risk.can_trade_direction(direction)
        if not can_dir:
            log.debug(f"{symbol}: {dir_reason} — skipped")
            return None

        # ── Graduated Response: dynamic minimum score ──
        throttle = self.risk.get_throttle()
        if score < throttle["min_score"]:
            log.debug(f"{symbol}: score {score} < {throttle['min_score']} "
                      f"(tier={throttle['tier_name']}) — skipped")
            return None

        current_price = float(chunk[-1])
        self.signals_today += 1

        # ── ML Prediction ──
        ml_pred = {"direction": 0.5, "confidence": 0.5, "up_prob": 0.5, "down_prob": 0.5}
        if self.ml_ready and self.ml_model.model is not None:
            ml_pred = self.ml_model.predict(chunk)

        ml_conf = ml_pred["confidence"]
        ml_direction = ml_pred["direction"]  # >0.5 = bullish

        # ML alignment check: does ML agree with rule-based direction?
        ml_agrees = (
            (direction == "call" and ml_direction > 0.5) or
            (direction == "put" and ml_direction < 0.5)
        )

        # ── Sentiment ──
        global_sent = self.sentiment.get_sentiment()
        symbol_adj = self.sentiment.get_per_symbol_adjustment(symbol, prices)
        combined_sent = np.clip(global_sent + symbol_adj, -1.0, 1.0)

        # ── Meta-Learner Ensemble ──
        # Normalize rule score to 0-1 range (score of 3-16 → 0.19 to 1.0)
        rule_normalized = min(1.0, score / 16.0)
        # For direction alignment: if bearish signal, flip ML direction
        ml_for_ensemble = ml_direction if direction == "call" else (1.0 - ml_direction)
        sent_for_ensemble = (combined_sent + 1.0) / 2.0  # Map -1..+1 → 0..1

        predictions = {
            "ml_model": ml_for_ensemble,
            "sentiment": sent_for_ensemble,
            "rule_score": rule_normalized,
        }
        ensemble_score = self.meta_learner.get_ensemble_score(predictions)

        # ── Quality Gates ──

        # 0a. Direction enable flags (PUTS_ENABLED / CALLS_ENABLED in config)
        if direction == "put" and not self.config.PUTS_ENABLED:
            log.debug(f"{symbol}: PUT signals disabled (PUTS_ENABLED=False) — skipped")
            return None
        if direction == "call" and not self.config.CALLS_ENABLED:
            log.debug(f"{symbol}: CALL signals disabled (CALLS_ENABLED=False) — skipped")
            return None

        # 0a2. Put win-rate auto-disable — if put WR < 30% over last 20 put trades, block
        if direction == "put" and self._puts_auto_disabled:
            log.debug(f"{symbol}: PUT auto-disabled — put WR < 30% over last 20 trades")
            return None

        # 0b. SPY regime filter — don't trade against the macro trend
        #     CALLs blocked in bear regime (SPY below 20d MA)
        #     PUTs: in bull regime, require extra-strong bearish evidence
        regime = self._get_market_regime()

        # Regime flip blocking — halt entries during severe transitions
        if self._regime_flip_block:
            log.debug(
                f"{symbol}: BLOCKED by regime flip cooldown "
                f"({self._last_flip_from}→{self._last_flip_to}, "
                f"severity={self._regime_flip_severity:.2f})"
            )
            return None

        if direction == "call" and regime == "bear":
            log.debug(f"{symbol}: CALL blocked — SPY in BEAR regime")
            return None
        if direction == "put" and regime == "bull":
            # ── Strict put gate: SPY is bullish, demand overwhelming bearish evidence ──
            log.debug(f"{symbol}: PUT in BULL regime — applying strict put gates")
            put_blocked_reason = None
            if ensemble_score < 0.75:
                put_blocked_reason = f"ensemble {ensemble_score:.2f} < 0.75"
            elif ml_direction >= 0.40:
                put_blocked_reason = f"ML direction {ml_direction:.2f} >= 0.40 (need < 0.40)"
            elif combined_sent >= -0.15:
                put_blocked_reason = f"sentiment {combined_sent:+.2f} >= -0.15 (need bearish)"
            if put_blocked_reason:
                log.debug(f"{symbol}: PUT blocked in BULL regime — {put_blocked_reason}")
                return None
            log.info(f"{symbol}: PUT ALLOWED in bull regime — strong bearish evidence "
                     f"(ens={ensemble_score:.2f}, ml_dir={ml_direction:.2f}, sent={combined_sent:+.2f})")

        # 0b2. Even in neutral regime, if SPY 20-bar trend is up, apply strict gates
        if direction == "put" and regime == "neutral" and self._spy_trend_is_up():
            put_blocked_reason = None
            if ensemble_score < 0.75:
                put_blocked_reason = f"ensemble {ensemble_score:.2f} < 0.75"
            elif ml_direction >= 0.40:
                put_blocked_reason = f"ML direction {ml_direction:.2f} >= 0.40 (need < 0.40)"
            elif combined_sent >= -0.15:
                put_blocked_reason = f"sentiment {combined_sent:+.2f} >= -0.15 (need bearish)"
            if put_blocked_reason:
                log.debug(f"{symbol}: PUT blocked — SPY trend UP (neutral regime) — {put_blocked_reason}")
                return None
            log.info(f"{symbol}: PUT ALLOWED despite SPY uptrend — strong bearish evidence "
                     f"(ens={ensemble_score:.2f}, ml_dir={ml_direction:.2f}, sent={combined_sent:+.2f})")

        # 0c. Sentiment direction filter — don't fight the market mood
        #    Bearish sentiment (-0.15+) blocks new calls;
        #    Bullish sentiment (+0.10+) blocks new puts (strict: was 0.15).
        if direction == "call" and combined_sent < -0.15:
            log.debug(f"{symbol}: Sentiment too bearish ({combined_sent:+.2f}) for CALL — skipped")
            return None
        if direction == "put" and combined_sent > 0.10:
            log.debug(f"{symbol}: Sentiment too bullish ({combined_sent:+.2f}) for PUT — skipped (threshold 0.10)")
            return None

        # 1. If ML is ready, it MUST agree with direction (hard gate)
        if self.ml_ready and ml_conf >= self.MIN_ML_CONFIDENCE and not ml_agrees:
            log.debug(f"{symbol}: ML DISAGREES with {direction} (dir={ml_direction:.2f}, conf={ml_conf:.2f}) — blocked")
            return None

        # 2. If ML is ready, it must meet minimum confidence
        if self.ml_ready and ml_conf < self.MIN_ML_CONFIDENCE:
            log.debug(f"{symbol}: ML confidence {ml_conf:.2f} < {self.MIN_ML_CONFIDENCE} — skipped")
            return None

        # 3. Meta-learner ensemble gate (with morning/afternoon adjustment)
        meta_conf_adj = 0.0
        meta_rule_adj = 0
        if getattr(self.config, 'MORNING_WINDOW_ENABLED', False):
            if time_window == "morning":
                meta_conf_adj = -self.config.MORNING_META_CONF_REDUCTION   # negative = looser
                meta_rule_adj = -self.config.MORNING_META_RULE_REDUCTION
                log.debug(f"{symbol}: Morning window — relaxed meta thresholds "
                          f"(conf adj={meta_conf_adj:+.2f}, rule adj={meta_rule_adj:+d})")
            elif time_window == "afternoon":
                meta_conf_adj = +self.config.AFTERNOON_META_CONF_INCREASE  # positive = stricter
                meta_rule_adj = +self.config.AFTERNOON_META_RULE_INCREASE
                log.debug(f"{symbol}: Afternoon window — tightened meta thresholds "
                          f"(conf adj={meta_conf_adj:+.2f}, rule adj={meta_rule_adj:+d})")

        if not self.meta_learner.should_trade(ensemble_score, score,
                                              conf_adjust=meta_conf_adj,
                                              rule_adjust=meta_rule_adj):
            log.debug(f"{symbol}: Ensemble gate failed [{time_window}] "
                      f"(score={ensemble_score:.2f}, rule={score})")
            return None

        # Build enriched signal
        reason = (
            f"ML:{ml_direction:.2f}|Conf:{ml_conf:.2f}|"
            f"Score:{score}({bull}b/{bear}b)|"
            f"Sent:{combined_sent:+.2f}|Ens:{ensemble_score:.2f}|Win:{time_window}"
        )

        return {
            "direction": direction,
            "score": score,
            "bull": bull,
            "bear": bear,
            "price": current_price,
            "indicators": indicators,
            # AI enrichments
            "ml_confidence": ml_conf,
            "ml_direction": ml_direction,
            "ml_agrees": ml_agrees,
            "sentiment": combined_sent,
            "ensemble_score": ensemble_score,
            "reason": reason,
        }

    # ==========================================================
    #  SHADOW SCANNER (runs during circuit breaker)
    # ==========================================================

    def _shadow_scan(self, breaker_reason: str):
        """Run scanner during circuit breaker pauses — log signals but don't trade.
        
        This lets us evaluate whether the breaker is costing us good opportunities.
        Shadow trades are logged to data/shadow_trades.csv for post-analysis.
        """
        if not self.config.SCANNER_ENABLED or not self.scanner.should_scan():
            return

        scanner_signals = self.scanner.scan(
            existing_positions=self.positions,
            fixed_watchlist=self.config.WATCHLIST,
        )

        if not scanner_signals:
            return

        # Log the top signals we WOULD have traded
        top = scanner_signals[0]
        log.info(
            f"SHADOW: Breaker active ({breaker_reason}) — "
            f"WOULD trade {top['symbol']} {top['direction'].upper()} "
            f"score={top['score']} @ ${top['price']:.2f}"
        )

        # Write shadow trades to CSV for post-analysis
        import csv, os
        shadow_path = os.path.join("data", "shadow_trades.csv")
        file_exists = os.path.exists(shadow_path)
        try:
            with open(shadow_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "symbol", "direction", "score",
                        "price", "rsi", "trend", "breaker_reason",
                    ])
                for sig in scanner_signals[:3]:  # log top 3 signals
                    writer.writerow([
                        datetime.now().isoformat(),
                        sig["symbol"],
                        sig["direction"],
                        sig["score"],
                        f"{sig['price']:.2f}",
                        f"{sig.get('rsi', 0):.1f}",
                        f"{sig.get('trend', 0):.2f}",
                        breaker_reason,
                    ])
        except Exception as e:
            log.debug(f"Shadow trade log error: {e}")

    # ==========================================================
    #  TRADE EXECUTION
    # ==========================================================


    def _check_portfolio_exposure(self) -> tuple:
        """Check aggregate worst-case loss across all bots sharing this Alpaca account.
        Returns (can_trade, exposure_pct).
        """
        PORTFOLIO_MAX_PCT = 0.50  # 50% of equity (combined allocation: PS 35% + CB 15% + AB 0%)
        POSITION_FILES = {
            r"C:\PutSeller\data\state\positions.json": "max_loss_total",
            r"C:\CallBuyer\data\state\positions.json": "entry_total",
            r"C:\AlpacaBot\data\state\positions.json": "cost",
        }
        try:
            acct = self.api.get_account()
            equity = acct.get("equity", 0)
            if not equity or equity < 1000:
                return True, 0.0
            total_risk = 0.0
            for path, risk_field in POSITION_FILES.items():
                try:
                    if os.path.exists(path):
                        with open(path, encoding="utf-8-sig") as f:
                            positions = json.load(f)
                        total_risk += sum(
                            abs(p.get(risk_field, 0)) for p in positions.values()
                        )
                except (json.JSONDecodeError, OSError):
                    pass
            exposure_pct = total_risk / equity
            if exposure_pct > PORTFOLIO_MAX_PCT:
                log.warning(
                    f"PORTFOLIO CAP: aggregate risk ${total_risk:,.0f} = "
                    f"{exposure_pct:.1%} of ${equity:,.0f} equity "
                    f"(cap={PORTFOLIO_MAX_PCT:.0%}) \u2014 blocking new entries"
                )
                return False, exposure_pct
            return True, exposure_pct
        except Exception as e:
            log.debug(f"Portfolio exposure check failed (proceeding): {e}")
            return True, 0.0

    def _scan_and_trade(self):
        """Scan for scalp signals — fixed watchlist + scanner universe — and open positions."""
        self.last_signal_check = datetime.now()

        can_trade, reason = self.risk.can_open_position(len(self.positions))
        if not can_trade:
            # ── Shadow Scanner: keep scanning during cooldown/hard stop, log what we'd trade ──
            if "Cooldown" in reason or "HARD STOP" in reason:
                self._shadow_scan(reason)
            else:
                log.debug(f"Skipping scan: {reason}")
            return

        # ── Portfolio-Level Aggregate Exposure Cap ─────────
        can_trade_portfolio, portfolio_exposure = self._check_portfolio_exposure()
        if not can_trade_portfolio:
            return

        opens_this_cycle = 0  # throttle: max new opens per scan cycle
        pos_before_count = len(self.positions)

        # ── Phase 1: Fixed watchlist (always scanned, highest priority) ──
        for symbol in self.config.WATCHLIST:
            if len(self.positions) >= self.config.MAX_POSITIONS:
                break
            if opens_this_cycle >= self.config.MAX_OPENS_PER_CYCLE:
                break
            if any(p.get("underlying") == symbol for p in self.positions.values()):
                continue
            cd = self.cooldowns.get(symbol)
            if cd and datetime.now() < cd:
                continue
            can_sym, sym_reason = self.risk.can_trade_symbol(symbol)
            if not can_sym:
                continue
            signal = self._generate_scalp_signal(symbol)
            if signal is None:
                continue

            # RL Shadow: let the RL agent observe this signal
            rl_features = {**signal.get("indicators", {}),
                           "sentiment": signal.get("sentiment", 0),
                           "ml_confidence": signal.get("ml_confidence", 0.5)}
            rl_result = self.rl_agent.shadow_evaluate(
                {"underlying": symbol, **signal}, rl_features
            )

            reason_str = signal.get('reason', 'score=' + str(signal['score']))
            self._log_activity(
                f"SIGNAL: {symbol} {signal['direction'].upper()} "
                f"{reason_str} "
                f"@ ${signal['price']:.2f} | RL:{rl_result['action']}"
            )
            self._execute_scalp(symbol, signal)
            if len(self.positions) > pos_before_count:
                opens_this_cycle += 1
                pos_before_count = len(self.positions)

        # ── Phase 2: Scanner universe (find new opportunities) ──
        if (self.config.SCANNER_ENABLED
                and len(self.positions) < self.config.MAX_POSITIONS
                and opens_this_cycle < self.config.MAX_OPENS_PER_CYCLE
                and self.scanner.should_scan()):

            scanner_signals = self.scanner.scan(
                existing_positions=self.positions,
                fixed_watchlist=self.config.WATCHLIST,
            )

            if scanner_signals:
                self._log_activity(
                    f"SCANNER: {len(scanner_signals)} signals found | "
                    f"Top: {scanner_signals[0]['symbol']} {scanner_signals[0]['direction'].upper()} "
                    f"score={scanner_signals[0]['score']}"
                )

            # Trade the best scanner signals (skip fixed watchlist, already handled)
            for sig in scanner_signals:
                if len(self.positions) >= self.config.MAX_POSITIONS:
                    break
                if opens_this_cycle >= self.config.MAX_OPENS_PER_CYCLE:
                    break
                sym = sig["symbol"]
                if sym in self.config.WATCHLIST:
                    continue  # already handled above
                if any(p.get("underlying") == sym for p in self.positions.values()):
                    continue
                cd = self.cooldowns.get(sym)
                if cd and datetime.now() < cd:
                    continue

                # ── Graduated Response: direction lock + min score for scanner signals ──
                can_dir, dir_reason = self.risk.can_trade_direction(sig["direction"])
                if not can_dir:
                    log.debug(f"Scanner {sym}: {dir_reason} — skipped")
                    continue
                throttle = self.risk.get_throttle()
                if sig["score"] < throttle["min_score"]:
                    log.debug(f"Scanner {sym}: score {sig['score']} < {throttle['min_score']} "
                              f"(tier={throttle['tier_name']}) — skipped")
                    continue

                # Fetch bars into engine cache for this symbol
                try:
                    cached = self.scanner.price_cache.get(sym)
                    if cached:
                        with self._lock:
                            self.price_bars[sym] = cached
                except Exception:
                    pass

                # ── AI Stack enrichment for scanner signals ──
                # Same ML + sentiment + ensemble logic as watchlist path
                prices = self._get_prices(sym)
                chunk = prices if prices is not None else None
                if chunk is not None:
                    chunk = chunk[-self.config.LOOKBACK_BARS - 1:]

                ml_pred = {"direction": 0.5, "confidence": 0.5, "up_prob": 0.5, "down_prob": 0.5}
                if chunk is not None and self.ml_ready and self.ml_model.model is not None:
                    ml_pred = self.ml_model.predict(chunk)

                ml_conf = ml_pred["confidence"]
                ml_direction = ml_pred["direction"]
                ml_agrees = (
                    (sig["direction"] == "call" and ml_direction > 0.5) or
                    (sig["direction"] == "put" and ml_direction < 0.5)
                )

                # Sentiment
                global_sent = self.sentiment.get_sentiment()
                symbol_adj = self.sentiment.get_per_symbol_adjustment(
                    sym, prices if prices is not None else np.array([])
                )
                combined_sent = np.clip(global_sent + symbol_adj, -1.0, 1.0)

                # Ensemble
                rule_normalized = min(1.0, sig["score"] / 16.0)
                ml_for_ensemble = ml_direction if sig["direction"] == "call" else (1.0 - ml_direction)
                sent_for_ensemble = (combined_sent + 1.0) / 2.0
                predictions = {
                    "ml_model": ml_for_ensemble,
                    "sentiment": sent_for_ensemble,
                    "rule_score": rule_normalized,
                }
                ensemble_score = self.meta_learner.get_ensemble_score(predictions)

                # Fetch regime for scanner put gates
                regime = self._get_market_regime()

                # ── Quality Gates (same as watchlist path) ──

                # Put auto-disable check
                if sig["direction"] == "put" and self._puts_auto_disabled:
                    log.debug(f"Scanner {sym}: PUT auto-disabled — put WR < 30%")
                    continue

                # SPY regime gate for scanner puts
                if sig["direction"] == "put" and regime in ("bull", "neutral"):
                    spy_up = regime == "bull" or self._spy_trend_is_up()
                    if spy_up:
                        put_blocked = None
                        if ensemble_score < 0.75:
                            put_blocked = f"ensemble {ensemble_score:.2f} < 0.75"
                        elif ml_direction >= 0.40:
                            put_blocked = f"ML dir {ml_direction:.2f} >= 0.40"
                        elif combined_sent >= -0.15:
                            put_blocked = f"sent {combined_sent:+.2f} >= -0.15"
                        if put_blocked:
                            log.debug(f"Scanner {sym}: PUT blocked (SPY up) — {put_blocked}")
                            continue

                # Sentiment filter for scanner puts (stricter: 0.10)
                if sig["direction"] == "put" and combined_sent > 0.10:
                    log.debug(f"Scanner {sym}: Sentiment too bullish ({combined_sent:+.2f}) "
                              f"for PUT — skipped (threshold 0.10)")
                    continue
                if sig["direction"] == "call" and combined_sent < -0.15:
                    log.debug(f"Scanner {sym}: Sentiment too bearish ({combined_sent:+.2f}) "
                              f"for CALL — skipped")
                    continue

                # ML agreement: if ML is confident and disagrees, block
                if self.ml_ready and ml_conf >= self.MIN_ML_CONFIDENCE and not ml_agrees:
                    log.debug(f"Scanner {sym}: ML DISAGREES with {sig['direction']} "
                              f"(dir={ml_direction:.2f}, conf={ml_conf:.2f}) — blocked")
                    continue
                # ML confidence minimum
                if self.ml_ready and ml_conf < self.MIN_ML_CONFIDENCE:
                    log.debug(f"Scanner {sym}: ML confidence {ml_conf:.2f} < "
                              f"{self.MIN_ML_CONFIDENCE} — skipped")
                    continue
                # Ensemble gate
                if not self.meta_learner.should_trade(ensemble_score, sig["score"]):
                    log.debug(f"Scanner {sym}: Ensemble gate failed "
                              f"(score={ensemble_score:.2f}, rule={sig['score']})")
                    continue

                self._log_activity(
                    f"SCANNER SIGNAL: {sym} {sig['direction'].upper()} "
                    f"score={sig['score']} rank=#{sig.get('rank', '?')} "
                    f"@ ${sig['price']:.2f} | "
                    f"ML:{ml_direction:.2f}|Conf:{ml_conf:.2f}|Ens:{ensemble_score:.2f}"
                )
                # Build signal dict compatible with _execute_scalp
                engine_signal = {
                    "direction": sig["direction"],
                    "score": sig["score"],
                    "bull": sig["bull"],
                    "bear": sig["bear"],
                    "price": sig["price"],
                    "indicators": {},
                    # AI enrichment (was missing — caused 0.5 defaults)
                    "ml_confidence": ml_conf,
                    "ml_direction": ml_direction,
                    "ml_agrees": ml_agrees,
                    "sentiment": combined_sent,
                    "ensemble_score": ensemble_score,
                }
                self._execute_scalp(sym, engine_signal)
                if len(self.positions) > pos_before_count:
                    opens_this_cycle += 1
                    pos_before_count = len(self.positions)

    def _execute_scalp(self, underlying: str, signal: Dict[str, Any]) -> bool:
        """Execute a scalp trade. Level 3: tries vertical spread first,
        falls back to single-leg if no spread available.
        Returns True if a position was opened."""
        direction = signal["direction"]
        current_price = signal["price"]

        # Level 3: Try vertical spread first (defined risk, better capital efficiency)
        if self.config.SPREADS_ENABLED:
            spread = self.options.find_spread_contracts(underlying, direction, current_price)
            if spread:
                return self._execute_spread(underlying, signal, spread)
            log.debug(f"{underlying}: No spread available, trying single-leg")

        # Fall back to single-leg trade
        opt_direction = "bullish" if direction == "call" else "bearish"

        # Find best contract
        contract = self.options.find_best_contract(underlying, opt_direction, current_price)
        if not contract:
            self._log_activity(f"No contract found for {underlying} {direction}", "warning")
            return False

        # Get premium (ask price)
        premium = contract.get("ask", contract.get("mid", 0))
        if premium <= 0:
            self._log_activity(f"No valid premium for {contract['symbol']}", "warning")
            return False

        # Position sizing — reduce size when ML not ready or throttled
        balance = self.risk.current_balance
        pos_pct = self.config.MAX_POSITION_PCT
        if not self.ml_ready:
            pos_pct *= 0.5  # half-size when trading without ML

        # ── Graduated Response: apply size multiplier ──
        throttle = self.risk.get_throttle()
        pos_pct *= throttle["size_multiplier"]
        if throttle["size_multiplier"] < 1.0:
            log.info(f"Size throttled: {throttle['tier_name']} tier → "
                     f"{throttle['size_multiplier']:.0%} size ({pos_pct:.1%} of balance)")

        max_spend = balance * pos_pct
        cost_per = premium * 100

        if cost_per > max_spend:
            self._log_activity(f"Contract too expensive: ${cost_per:.0f} > ${max_spend:.0f} max", "warning")
            return False

        qty = max(1, int(max_spend / cost_per))

        # Regime flip position sizing reduction
        if self._regime_flip_mult < 1.0:
            adjusted_qty = max(1, int(qty * self._regime_flip_mult))
            if adjusted_qty < qty:
                log.info(
                    f"Flip sizing: {qty} → {adjusted_qty} contracts "
                    f"(mult={self._regime_flip_mult:.2f}, "
                    f"flip={self._last_flip_from}→{self._last_flip_to})"
                )
                qty = adjusted_qty

        total_cost = cost_per * qty

        # Execute order
        self._log_activity(
            f"OPENING: {qty}x {contract['symbol']} ({direction.upper()}) "
            f"@ ${premium:.2f} = ${total_cost:.2f}"
        )

        order_id = self.api.buy_option(
            symbol=contract["symbol"],
            qty=qty,
            # Market order — always fills immediately
        )

        if order_id:
            with self._lock:
                self.positions[contract["symbol"]] = {
                    "underlying": underlying,
                    "symbol": contract["symbol"],
                    "direction": direction,
                    "option_type": contract.get("type", direction),
                    "strike": contract.get("strike", 0),
                    "expiration": contract.get("expiration", ""),
                    "entry_price": premium,
                    "current_price": premium,
                    "peak_price": premium,
                    "qty": qty,
                    "cost": total_cost,
                    "order_id": order_id,
                    "entry_time": datetime.now().isoformat(),
                    "score": signal["score"],
                    # Store AI context for learning on exit
                    "ml_confidence": signal.get("ml_confidence", 0.5),
                    "ml_direction": signal.get("ml_direction", 0.5),
                    "sentiment": signal.get("sentiment", 0.0),
                    "ensemble_score": signal.get("ensemble_score", 0.5),
                    "reason_entry": signal.get("reason", ""),
                }
            self.trades_today += 1
            self._log_activity(
                f"OPENED: {contract['symbol']} | {direction.upper()} "
                f"K=${contract.get('strike', 0)} exp={contract.get('expiration', '')} "
                f"| ${total_cost:.2f}"
            )
            return True
        else:
            self._log_activity(f"ORDER FAILED for {contract['symbol']}", "error")
            return False

    def _execute_spread(self, underlying: str, signal: Dict[str, Any],
                        spread: Dict[str, Any]) -> bool:
        """Execute a vertical spread trade (Level 3).
        Bull call spread when bullish, bear put spread when bearish.
        Returns True if spread was opened."""
        direction = signal["direction"]
        long_leg = spread["long_leg"]
        short_leg = spread["short_leg"]
        net_debit = spread["net_debit"]

        # Position sizing based on max loss (net debit x 100)
        balance = self.risk.current_balance
        pos_pct = self.config.MAX_POSITION_PCT
        if not self.ml_ready:
            pos_pct *= 0.5

        # Graduated Response: apply size multiplier
        throttle = self.risk.get_throttle()
        pos_pct *= throttle["size_multiplier"]
        if throttle["size_multiplier"] < 1.0:
            log.info(f"Spread size throttled: {throttle['tier_name']} tier "
                     f"-> {throttle['size_multiplier']:.0%} size")

        max_spend = balance * pos_pct
        cost_per_set = net_debit * 100

        if cost_per_set <= 0:
            return False

        qty = max(1, int(max_spend / cost_per_set))

        # Regime flip position sizing reduction (spreads)
        if self._regime_flip_mult < 1.0:
            adjusted_qty = max(1, int(qty * self._regime_flip_mult))
            if adjusted_qty < qty:
                log.info(
                    f"Spread flip sizing: {qty} → {adjusted_qty} sets "
                    f"(mult={self._regime_flip_mult:.2f})"
                )
                qty = adjusted_qty

        total_cost = cost_per_set * qty

        # Limit price at mid of the spread for better fills
        long_mid = long_leg.get("mid", long_leg.get("ask", 0))
        short_mid = short_leg.get("mid", short_leg.get("bid", 0))
        limit_debit = round(long_mid - short_mid, 2)
        if limit_debit <= 0:
            limit_debit = round(net_debit * 0.95, 2)

        strategy_name = "bull_call_spread" if direction == "call" else "bear_put_spread"

        self._log_activity(
            f"OPENING SPREAD: {qty}x {underlying} {strategy_name} "
            f"${long_leg['strike']}/{short_leg['strike']} exp {spread['expiration']} "
            f"| debit ${limit_debit:.2f} = ${limit_debit * 100 * qty:.0f}"
        )

        legs = [
            {
                "symbol": long_leg["symbol"],
                "side": "buy",
                "position_intent": "buy_to_open",
                "ratio_qty": 1,
            },
            {
                "symbol": short_leg["symbol"],
                "side": "sell",
                "position_intent": "sell_to_open",
                "ratio_qty": 1,
            },
        ]

        order_id = self.api.submit_mleg_order(legs, qty=qty, limit_price=limit_debit)

        if order_id:
            with self._lock:
                self.positions[long_leg["symbol"]] = {
                    "underlying": underlying,
                    "symbol": long_leg["symbol"],
                    "direction": direction,
                    "option_type": long_leg.get("type", direction),
                    "strike": long_leg.get("strike", 0),
                    "expiration": spread["expiration"],
                    "entry_price": long_leg.get("ask", 0),
                    "current_price": long_mid,
                    "peak_price": long_mid,
                    "qty": qty,
                    "cost": total_cost,
                    "order_id": order_id,
                    "entry_time": datetime.now().isoformat(),
                    "score": signal["score"],
                    # Spread-specific
                    "strategy": "spread",
                    "strategy_name": strategy_name,
                    "short_leg_symbol": short_leg["symbol"],
                    "short_leg_strike": short_leg.get("strike", 0),
                    "short_leg_entry_price": short_leg.get("bid", 0),
                    "net_debit": limit_debit,
                    "spread_width": spread["spread_width"],
                    "max_profit": spread["max_profit_per_contract"],
                    "max_loss": spread["max_loss_per_contract"],
                    "peak_spread_value": limit_debit,
                    # AI context
                    "ml_confidence": signal.get("ml_confidence", 0.5),
                    "ml_direction": signal.get("ml_direction", 0.5),
                    "sentiment": signal.get("sentiment", 0.0),
                    "ensemble_score": signal.get("ensemble_score", 0.5),
                    "reason_entry": signal.get("reason", ""),
                }
            self.trades_today += 1
            self._log_activity(
                f"OPENED SPREAD: {long_leg['symbol']}/{short_leg['symbol']} "
                f"| {strategy_name} K=${long_leg['strike']}/{short_leg['strike']} "
                f"| debit ${limit_debit:.2f} x {qty} = ${total_cost:.0f}"
            )
            return True
        else:
            self._log_activity(f"SPREAD ORDER FAILED for {underlying}", "error")
            return False

    # ==========================================================
    #  EXIT MANAGEMENT
    # ==========================================================

    def _check_exits(self):
        """Check all open positions for scalp exit conditions."""
        if not self.positions:
            return

        exits = []

        for sym, pos in list(self.positions.items()):
            is_spread = pos.get("strategy") == "spread"

            if is_spread:
                # Spread: update both legs, PnL based on net spread value
                try:
                    lq = self.api.get_option_quote(sym)
                    sq = self.api.get_option_quote(pos["short_leg_symbol"])
                    if lq and sq and lq.get("mid", 0) > 0:
                        net_val = lq["mid"] - sq["mid"]
                        with self._lock:
                            pos["current_price"] = lq["mid"]
                            pos["current_spread_value"] = net_val
                            if net_val > pos.get("peak_spread_value", pos.get("net_debit", 0)):
                                pos["peak_spread_value"] = net_val
                except Exception as e:
                    log.debug(f"Spread quote update failed: {e}")

                entry = pos.get("net_debit", 0)
                current = pos.get("current_spread_value", entry)
                peak = pos.get("peak_spread_value", entry)
            else:
                # Single leg: update quote
                try:
                    quote = self.api.get_option_quote(sym)
                    if quote and quote.get("mid", 0) > 0:
                        with self._lock:
                            pos["current_price"] = quote["mid"]
                            if quote["mid"] > pos.get("peak_price", 0):
                                pos["peak_price"] = quote["mid"]
                except Exception as e:
                    log.debug(f"Quote update failed for {sym}: {e}")

                entry = pos.get("entry_price", 0)
                current = pos.get("current_price", entry)
                peak = pos.get("peak_price", entry)

            if entry <= 0:
                continue

            reason = None

            # ── SPREAD EXIT LOGIC (% of max profit, ride winners) ──
            if is_spread:
                spread_width = pos.get("spread_width", 0)
                max_profit = pos.get("max_profit", spread_width - entry)
                if max_profit <= 0:
                    max_profit = spread_width - entry  # fallback calc

                profit_captured = current - entry
                pct_of_max = profit_captured / max_profit if max_profit > 0 else 0
                pnl_pct = (current - entry) / entry  # for logging

                # Check DTE for near-expiry trailing tightening
                dte = 999
                exp = pos.get("expiration", "")
                if exp:
                    try:
                        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                        dte = (exp_date - datetime.now().date()).days
                    except ValueError:
                        pass

                # 1) Stop loss: debit lost > threshold
                if pnl_pct <= self.config.SPREAD_STOP_LOSS_PCT:
                    reason = f"SPREAD_STOP_LOSS ({pnl_pct:.0%} of debit)"

                # 2) Take profit: captured >= X% of max profit
                elif pct_of_max >= self.config.SPREAD_TP_PCT_OF_MAX:
                    reason = f"SPREAD_TAKE_PROFIT ({pct_of_max:.0%} of max, PnL {pnl_pct:.0%})"

                # 3) Trailing stop (% of max profit used as trigger)
                elif pct_of_max >= self.config.SPREAD_TRAILING_TRIGGER:
                    # Peak tracking uses spread value
                    drop_from_peak = (current - peak) / peak if peak > 0 else 0
                    # Near expiry: tighten trailing stop
                    trail_pct = (self.config.SPREAD_NEAREXP_TRAIL_PCT
                                 if dte <= 1
                                 else self.config.SPREAD_TRAILING_STOP_PCT)
                    if drop_from_peak <= -trail_pct:
                        reason = (f"SPREAD_TRAILING_STOP ({drop_from_peak:.0%} from peak, "
                                  f"{pct_of_max:.0%} of max{'[near-exp]' if dte <= 1 else ''})")

                # 4) DTE exit (still applies for spreads on expiry day)
                if not reason and dte <= self.config.MIN_DTE_EXIT:
                    reason = f"SPREAD_DTE_EXIT ({dte}d, captured {pct_of_max:.0%} of max, PnL {pnl_pct:.0%})"

                # 5) Max hold (same as single leg)
                if not reason:
                    entry_time = pos.get("entry_time")
                    underlying = pos.get("underlying", sym)
                    if entry_time:
                        try:
                            et = datetime.fromisoformat(entry_time) if isinstance(entry_time, str) else entry_time
                            hold_hours = (datetime.now() - et).total_seconds() / 3600
                            max_hold_days = self.config.get_max_hold_days(underlying)
                            if hold_hours >= max_hold_days * 24:
                                reason = f"SPREAD_MAX_HOLD ({hold_hours:.1f}h, {pct_of_max:.0%} of max)"
                        except Exception:
                            pass

            # ── SINGLE-LEG EXIT LOGIC (scalp: tight TP/SL) ──
            else:
                pnl_pct = (current - entry) / entry
                reason = None

                # Stop loss
                if pnl_pct <= self.config.STOP_LOSS_PCT:
                    reason = f"STOP_LOSS ({pnl_pct:.0%})"

                # Take profit
                elif pnl_pct >= self.config.TAKE_PROFIT_PCT:
                    reason = f"TAKE_PROFIT ({pnl_pct:.0%})"

                # Trailing stop
                elif peak > entry * (1 + self.config.TRAILING_TRIGGER):
                    drop_from_peak = (current - peak) / peak
                    if drop_from_peak <= -self.config.TRAILING_STOP_PCT:
                        reason = f"TRAILING_STOP ({drop_from_peak:.0%} from peak)"

                # DTE exit - close before expiry day (calendar-day based)
                if not reason:
                    exp = pos.get("expiration", "")
                    if exp:
                        try:
                            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                            dte = (exp_date - datetime.now().date()).days
                            if dte <= self.config.MIN_DTE_EXIT:
                                reason = f"DTE_EXIT ({dte}d to expiry)"
                        except ValueError:
                            pass

                # Max hold time (per-symbol DTE-dependent)
                if not reason:
                    entry_time = pos.get("entry_time")
                    underlying = pos.get("underlying", sym)
                    if entry_time:
                        try:
                            et = datetime.fromisoformat(entry_time) if isinstance(entry_time, str) else entry_time
                            hold_hours = (datetime.now() - et).total_seconds() / 3600
                            max_hold_days = self.config.get_max_hold_days(underlying)
                            if hold_hours >= max_hold_days * 24:
                                reason = f"MAX_HOLD ({hold_hours:.1f}h, max {max_hold_days}d for {underlying})"
                        except Exception:
                            pass

            if reason:
                exits.append((sym, pos, reason))

        for sym, pos, reason in exits:
            self._close_position(sym, pos, reason)

    def _close_position(self, symbol: str, pos: Dict[str, Any], reason: str):
        """Close a position and record the trade.
        Uses close_position API first, falls back to sell_option,
        and removes phantom positions that don't exist on Alpaca."""
        entry_price = pos.get("entry_price", 0)
        current_price = pos.get("current_price", entry_price)
        qty = pos.get("qty", 1)

        # Track sell failures with exponential backoff (retry every 2^n cycles)
        sell_fails = pos.get("_sell_fails", 0)
        if sell_fails >= 3:
            # Already failed 3+ times — verify position still exists on Alpaca
            if not self.api.position_exists(symbol):
                self._log_activity(
                    f"PHANTOM REMOVED: {symbol} not on Alpaca after {sell_fails} sell failures — "
                    f"removing from local state",
                    "warning"
                )
                with self._lock:
                    del self.positions[symbol]
                return

            # Exponential backoff: retry every 2^(fails-3) cycles (min 1, max 60)
            backoff_cycles = min(60, 2 ** (sell_fails - 3))
            retry_counter = pos.get("_backoff_counter", 0) + 1
            with self._lock:
                pos["_backoff_counter"] = retry_counter
            if retry_counter < backoff_cycles:
                # Only log once every 30 cycles to avoid spam
                if retry_counter == 1 or retry_counter % 30 == 0:
                    self._log_activity(
                        f"SELL BACKOFF: {symbol} failed {sell_fails}x — "
                        f"retrying in {backoff_cycles - retry_counter} cycles",
                        "warning"
                    )
                return
            # Reset backoff counter for this retry attempt
            with self._lock:
                pos["_backoff_counter"] = 0

            log.info(f"SELL RETRY: {symbol} — attempt {sell_fails + 1} after backoff")

        is_spread = pos.get("strategy") == "spread"

        if is_spread:
            # Close spread via mleg order (close both legs simultaneously)
            short_sym = pos.get("short_leg_symbol", "")
            legs = [
                {"symbol": symbol, "side": "sell",
                 "position_intent": "sell_to_close", "ratio_qty": 1},
                {"symbol": short_sym, "side": "buy",
                 "position_intent": "buy_to_close", "ratio_qty": 1},
            ]
            order_id = self.api.submit_mleg_order(legs, qty=qty)
            closed = order_id is not None

            if not closed:
                # Fall back: close each leg individually
                closed_long = self.api.close_position(symbol)
                closed_short = self.api.close_position(short_sym) if short_sym else True
                closed = closed_long or closed_short

                if not closed:
                    # Last resort: use Alpaca's qty from the live position
                    try:
                        live_pos = self.api.trading.get_open_position(symbol)
                        live_qty = int(live_pos.qty) if live_pos else qty
                        if live_qty != qty:
                            log.warning(f"Qty mismatch for {symbol}: local={qty} alpaca={live_qty}")
                        order_id = self.api.sell_option(symbol=symbol, qty=live_qty)
                        closed = order_id is not None
                    except Exception as e:
                        log.error(f"Last-resort sell failed for {symbol}: {e}")
        else:
            # Single leg: original close logic
            closed = self.api.close_position(symbol)
            order_id = None
            if not closed:
                order_id = self.api.sell_option(symbol=symbol, qty=qty)
                closed = order_id is not None

        if not closed:
            # Increment failure counter
            with self._lock:
                pos["_sell_fails"] = sell_fails + 1
            self._log_activity(
                f"Failed to close {symbol} (attempt {sell_fails + 1}/3)", "error"
            )
            return

        # Success — calculate PnL
        if is_spread:
            entry_debit = pos.get("net_debit", 0)
            exit_value = pos.get("current_spread_value", entry_debit)
            pnl = (exit_value - entry_debit) * 100 * qty
            pnl_pct = (exit_value - entry_debit) / entry_debit if entry_debit > 0 else 0
            max_prof = pos.get("max_profit", 0)
            pct_max = ((exit_value - entry_debit) / max_prof * 100) if max_prof > 0 else 0
            self._log_activity(
                f"CLOSED [{pos.get('strategy_name', 'spread')}]: {symbol} "
                f"| PnL ${pnl:+,.0f} ({pnl_pct:+.0%}) "
                f"| {pct_max:.0f}% of max profit "
                f"| {reason}"
            )
        else:
            pnl = (current_price - entry_price) * 100 * qty
            pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            self._log_activity(
                f"CLOSED: {symbol} | PnL ${pnl:+,.0f} ({pnl_pct:+.0%}) | {reason}"
            )

        trade_record = {
            "symbol": symbol,
            "underlying": pos.get("underlying", ""),
            "direction": pos.get("direction", ""),
            "option_type": pos.get("option_type", ""),
            "strike": pos.get("strike", 0),
            "expiration": pos.get("expiration", ""),
            "entry_price": entry_price,
            "exit_price": current_price,
            "qty": qty,
            "cost": pos.get("cost", 0),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
            "entry_time": pos.get("entry_time", ""),
            "exit_time": datetime.now().isoformat(),
            "score": pos.get("score", 0),
            # Spread context (for analytics)
            "strategy": pos.get("strategy", "single_leg"),
            "spread_width": pos.get("spread_width", 0),
            "net_debit": pos.get("net_debit", 0),
            "max_profit": pos.get("max_profit", 0),
            "pct_of_max_profit": (
                (pos.get("current_spread_value", 0) - pos.get("net_debit", 0))
                / pos.get("max_profit", 1) if pos.get("max_profit", 0) > 0 else 0
            ) if is_spread else 0,
            # AI context (stored at entry, used for learning)
            "ml_confidence": pos.get("ml_confidence", 0.5),
            "ml_direction": pos.get("ml_direction", 0.5),
            "sentiment": pos.get("sentiment", 0.0),
            "ensemble_score": pos.get("ensemble_score", 0.5),
        }

        # Record with risk manager
        self.risk.record_trade(trade_record)

        # ── AI Learning: feed outcome to RL agent + meta-learner ──
        underlying = pos.get("underlying", "")
        direction_closed = pos.get("direction", "")
        try:
            # RL Shadow: record the outcome
            self.rl_agent.shadow_record_outcome(underlying, pnl, pnl_pct)

            # Meta-learner: record prediction accuracy
            # Reconstruct what each source predicted at entry time
            actual_outcome = 1.0 if pnl > 0 else 0.0
            meta_predictions = {
                "ml_model": pos.get("ml_direction", 0.5),
                "sentiment": (pos.get("sentiment", 0) + 1.0) / 2.0,
                "rule_score": min(1.0, pos.get("score", 3) / 16.0),
            }
            self.meta_learner.record_prediction(meta_predictions, actual_outcome)
        except Exception as e:
            log.debug(f"AI learning record failed: {e}")

        # ── Track put vs call win rates ──
        is_win = pnl > 0
        if direction_closed == "put":
            if is_win:
                self._put_wins += 1
            else:
                self._put_losses += 1
            self._recent_put_outcomes.append(is_win)
            if len(self._recent_put_outcomes) > 20:
                self._recent_put_outcomes = self._recent_put_outcomes[-20:]
            # Auto-disable/re-enable puts based on rolling 20-trade win rate
            if len(self._recent_put_outcomes) >= 20:
                put_wr = sum(self._recent_put_outcomes) / len(self._recent_put_outcomes)
                if put_wr < 0.30 and not self._puts_auto_disabled:
                    self._puts_auto_disabled = True
                    log.warning(f"[PUT AUTO-DISABLE] Put WR {put_wr:.0%} < 30% over last "
                                f"{len(self._recent_put_outcomes)} puts — puts disabled")
                    self._log_activity(
                        f"PUT AUTO-DISABLED: WR {put_wr:.0%} < 30% over last 20 put trades",
                        "warning"
                    )
                elif put_wr >= 0.30 and self._puts_auto_disabled:
                    self._puts_auto_disabled = False
                    log.info(f"[PUT RE-ENABLED] Put WR recovered to {put_wr:.0%} — puts re-enabled")
                    self._log_activity(
                        f"PUT RE-ENABLED: WR recovered to {put_wr:.0%}", "info"
                    )
        elif direction_closed == "call":
            if is_win:
                self._call_wins += 1
            else:
                self._call_losses += 1

        # Log to CSV + history
        self._log_trade(trade_record)
        with self._lock:
            self.trade_history.append(trade_record)

        # Set cooldown
        cooldown_sec = self.config.COOLDOWN_BARS * self.config.BAR_INTERVAL_SEC
        self.cooldowns[pos.get("underlying", "")] = datetime.now() + timedelta(seconds=cooldown_sec)

        # Remove from active
        with self._lock:
            del self.positions[symbol]

        sign = "+" if pnl >= 0 else ""
        strat_tag = f" [{pos.get('strategy_name', 'SPREAD')}]" if is_spread else ""
        self._log_activity(
            f"CLOSED{strat_tag}: {pos.get('underlying', '')} {pos.get('direction', '').upper()} "
            f"| PnL ${sign}{pnl:.2f} ({pnl_pct:+.0%}) | {reason}",
            "info" if pnl >= 0 else "warning"
        )

    # ==========================================================
    #  AI/ML MODEL MANAGEMENT
    # ==========================================================

    def _bootstrap_ml_model(self):
        """Fetch historical bars and train ML model on startup."""
        try:
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from core.scanner import SCANNER_UNIVERSE
            tf = TimeFrame(10, TimeFrameUnit.Minute)
            symbols = list(SCANNER_UNIVERSE)[:20]  # top 20 for speed
            self._log_activity(f"Bootstrapping ML: fetching 30d bars for {len(symbols)} symbols...")

            price_dict = {}
            for sym in symbols:
                try:
                    bars = self.api.get_bars(sym, tf, days=30)
                    if bars and len(bars) >= 60:
                        prices = [float(bar.close) for bar in bars]
                        price_dict[sym] = np.array(prices)
                        with self._lock:
                            self.price_bars[sym] = prices  # cache for future retrains
                except Exception:
                    pass

            if len(price_dict) >= 2:
                self._log_activity(f"Bootstrap data: {len(price_dict)} symbols, training...")
                success = self.ml_model.train(price_dict, min_samples=200)
                if success:
                    self.ml_ready = True
                    self._log_activity(
                        f"ML BOOTSTRAP SUCCESS: {self.ml_model.test_accuracy:.1%} accuracy | "
                        f"{len(price_dict)} symbols | ML ready for trading"
                    )
                else:
                    self._log_activity("ML bootstrap: model rejected (below accuracy threshold)")
            else:
                self._log_activity("ML bootstrap: not enough data from API")
        except Exception as e:
            self._log_activity(f"ML bootstrap failed: {e}", "warning")

    def _maybe_retrain_model(self):
        """Retrain ML model periodically on collected price history."""
        elapsed = (datetime.now() - self.last_model_retrain).total_seconds() / 3600
        if elapsed < self.MODEL_RETRAIN_HOURS:
            return

        # Need enough data for meaningful training
        ready = sum(1 for prices in self.price_bars.values() if len(prices) >= 60)
        total_points = sum(len(prices) for prices in self.price_bars.values())
        if ready < 2 or total_points < 200:
            log.info(f"ML retrain skipped: insufficient data (series={ready}, points={total_points})")
            self.last_model_retrain = datetime.now()
            return

        log.info("Scheduled ML model retrain starting...")
        try:
            # Build price dict for training
            price_dict = {}
            for sym, prices in self.price_bars.items():
                if len(prices) >= 60:
                    price_dict[sym] = np.array(prices)

            success = self.ml_model.train(price_dict, min_samples=200)
            if success:
                self.ml_ready = True
                self._log_activity(f"ML model retrained: {self.ml_model.test_accuracy:.1%} accuracy")
            else:
                self._log_activity("ML retrain: model rejected (below threshold) or insufficient data")
        except Exception as e:
            log.warning(f"ML retrain failed: {e}")
        finally:
            self.last_model_retrain = datetime.now()

    def _recent_win_rate(self) -> float:
        """Calculate recent win rate from last 20 trades."""
        recent = self.trade_history[-20:]
        if not recent:
            return 50.0
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        return (wins / len(recent)) * 100.0

    # ==========================================================
    #  PRE-MARKET WARMUP
    # ==========================================================

    def _should_warmup(self) -> bool:
        """Return True if we're within 30 min of market open and haven't warmed up today."""
        if self._warmup_done_date == date.today():
            return False
        try:
            next_open = self.api.next_open()
            if next_open is None:
                return False
            # Make next_open naive for comparison if needed
            if hasattr(next_open, 'tzinfo') and next_open.tzinfo:
                from datetime import timezone
                now = datetime.now(timezone.utc)
            else:
                now = datetime.now()
            minutes_until_open = (next_open - now).total_seconds() / 60
            return 0 < minutes_until_open <= self._WARMUP_MINUTES_BEFORE_OPEN
        except Exception as e:
            log.debug(f"Warmup check failed: {e}")
            return False

    def _pre_market_warmup(self):
        """Run pre-market analysis ~30 min before market opens.

        1. Fetch daily bars for all watchlist symbols (feature pre-computation)
        2. Compute overnight gaps (last close vs pre-market if available)
        3. Pre-warm SPY regime detection
        4. Run universe scan to refresh watchlist
        5. Sync account balances
        """
        self._log_activity("PRE-MARKET WARMUP starting...", "info")
        warmup_start = time.time()
        results = {"gaps": {}, "regime": "unknown", "symbols_warmed": 0}

        # ── 1. Fetch daily bars & compute overnight gaps ──
        all_symbols = list(set(self.config.WATCHLIST))
        if hasattr(self, 'scanner') and hasattr(self.scanner, 'universe'):
            all_symbols = list(set(all_symbols + list(self.scanner.universe)[:50]))

        gaps = {}
        warmed = 0
        for symbol in all_symbols:
            try:
                from alpaca.data.timeframe import TimeFrame
                bars = self.api.get_bars(symbol, TimeFrame.Day, days=5)
                if bars and len(bars) >= 2:
                    prev_close = float(bars[-2].close)
                    last_close = float(bars[-1].close)
                    gap_pct = ((last_close - prev_close) / prev_close) * 100
                    gaps[symbol] = round(gap_pct, 2)
                    warmed += 1
                time.sleep(0.3)  # rate limit
            except Exception as e:
                log.debug(f"Warmup bar fetch failed for {symbol}: {e}")

        results["gaps"] = gaps
        results["symbols_warmed"] = warmed

        # Log notable gaps (>2%)
        big_gaps = {s: g for s, g in gaps.items() if abs(g) > 2.0}
        if big_gaps:
            sorted_gaps = sorted(big_gaps.items(), key=lambda x: abs(x[1]), reverse=True)
            gap_str = ", ".join(f"{s}:{g:+.1f}%" for s, g in sorted_gaps[:10])
            self._log_activity(f"WARMUP gaps >2%: {gap_str}")

        # ── 2. Pre-warm SPY regime ──
        try:
            regime = self._get_market_regime()
            results["regime"] = regime
            self._log_activity(f"WARMUP regime: {regime} (SPY ${self._spy_price:.2f}, SMA20=${self._spy_sma20:.2f}, SMA50=${self._spy_sma50:.2f})")
        except Exception as e:
            log.warning(f"Warmup regime check failed: {e}")

        # ── 3. Universe scan ──
        if self.universe_scanner:
            try:
                expanded = self.universe_scanner.scan()
                self.scanner.universe = list(expanded)
                self.scanner.max_scan_symbols = max(
                    self.config.SCANNER_MAX_SYMBOLS,
                    min(len(expanded), 200)
                )
                results["universe_size"] = len(expanded)
                self._log_activity(f"WARMUP universe: {len(expanded)} symbols")
            except Exception as e:
                log.warning(f"Warmup universe scan failed: {e}")

        # ── 4. Sync account ──
        try:
            acct = self.api.get_account()
            allocated = acct["portfolio_value"] * self.config.ALLOCATION_PCT
            self.risk.current_balance = allocated
            results["equity"] = acct["portfolio_value"]
            results["allocation"] = allocated
            self._log_activity(f"WARMUP account: ${acct['portfolio_value']:,.2f} equity, allocation ${allocated:,.2f}")
        except Exception as e:
            log.warning(f"Warmup account sync failed: {e}")

        # ── 5. Pre-warm sentiment ──
        try:
            sent = self.sentiment.get_composite_score()
            results["sentiment"] = sent
            self._log_activity(f"WARMUP sentiment: {sent:.3f}")
        except Exception as e:
            log.debug(f"Warmup sentiment failed: {e}")

        # ── Done ──
        elapsed = time.time() - warmup_start
        self._warmup_done_date = date.today()
        self._warmup_data = results
        self.status_text = f"Pre-market warmup complete ({warmed} symbols)"
        self._log_activity(
            f"PRE-MARKET WARMUP complete: {warmed} symbols warmed, "
            f"regime={results.get('regime', '?')}, "
            f"{len(big_gaps)} big gaps, "
            f"{elapsed:.1f}s elapsed"
        )

    # ==========================================================
    #  MARKET STATUS
    # ==========================================================

    def _is_market_open(self) -> bool:
        """Check if market is open, with caching."""
        try:
            return self.api.is_market_open()
        except Exception:
            # If we can't check, assume closed during off-hours
            now = datetime.now()
            if now.weekday() >= 5:
                return False
            hour = now.hour
            return 9 <= hour < 16

    # ==========================================================
    #  LOGGING & STATE
    # ==========================================================

    def _log_trade(self, trade: Dict[str, Any]):
        """Append trade to CSV log."""
        filepath = self.config.TRADE_LOG
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        header = [
            "timestamp", "symbol", "underlying", "direction", "option_type",
            "strike", "expiration", "entry_price", "exit_price", "qty",
            "cost", "pnl", "pnl_pct", "exit_reason", "score",
            "entry_time", "exit_time",
            "ml_confidence", "ml_direction", "sentiment", "ensemble_score",
        ]

        write_header = not os.path.exists(filepath)

        # If file exists but header is stale (fewer columns), rewrite it
        if not write_header:
            try:
                with open(filepath, "r") as f:
                    existing_header = f.readline().strip().split(",")
                if len(existing_header) < len(header):
                    # Read all existing data, rewrite with updated header
                    import csv as csv_mod
                    with open(filepath, "r") as f:
                        reader = csv_mod.reader(f)
                        next(reader)  # skip old header
                        existing_rows = list(reader)
                    with open(filepath, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        for row in existing_rows:
                            writer.writerow(row)
                    log.info(f"Updated trade CSV header: {len(existing_header)} → {len(header)} columns")
            except Exception as e:
                log.debug(f"Header check failed: {e}")

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([
                datetime.now().isoformat(),
                trade.get("symbol", ""),
                trade.get("underlying", ""),
                trade.get("direction", ""),
                trade.get("option_type", ""),
                trade.get("strike", ""),
                trade.get("expiration", ""),
                f"{trade.get('entry_price', 0):.4f}",
                f"{trade.get('exit_price', 0):.4f}",
                trade.get("qty", 0),
                f"{trade.get('cost', 0):.2f}",
                f"{trade.get('pnl', 0):.2f}",
                f"{trade.get('pnl_pct', 0):.4f}",
                trade.get("exit_reason", ""),
                trade.get("score", 0),
                trade.get("entry_time", ""),
                trade.get("exit_time", ""),
                f"{trade.get('ml_confidence', 0):.4f}",
                f"{trade.get('ml_direction', 0.5):.4f}",
                f"{trade.get('sentiment', 0):.4f}",
                f"{trade.get('ensemble_score', 0):.4f}",
            ])

    def _log_status(self):
        """Log current bot status."""
        num_pos = len(self.positions)
        total_cost = sum(p.get("cost", 0) for p in self.positions.values())
        total_unrealized = sum(
            (p.get("current_price", 0) - p.get("entry_price", 0)) * 100 * p.get("qty", 1)
            for p in self.positions.values()
        )
        log.info(
            f"[Cycle {self.cycle}] Pos: {num_pos}/{self.config.MAX_POSITIONS} "
            f"| Invested: ${total_cost:.2f} | Unrealized: ${total_unrealized:+.2f} "
            f"| Daily PnL: ${self.risk.daily_pnl:+.2f} | Signals: {self.signals_today}"
        )
        # AI status line
        meta_stats = self.meta_learner.get_source_stats()
        ml_w = meta_stats.get("ml_model", {}).get("weight", 0)
        rule_w = meta_stats.get("rule_score", {}).get("weight", 0)
        sent_w = meta_stats.get("sentiment", {}).get("weight", 0)
        put_total = self._put_wins + self._put_losses
        call_total = self._call_wins + self._call_losses
        put_wr = (self._put_wins / put_total * 100) if put_total > 0 else 0
        call_wr = (self._call_wins / call_total * 100) if call_total > 0 else 0
        log.info(
            f"[AI] ML={self.ml_ready}(w={ml_w:.2f}) | "
            f"Rules(w={rule_w:.2f}) | Sent(w={sent_w:.2f}) | "
            f"RL={self.rl_agent.total_episodes}ep | "
            f"WinRate={self._recent_win_rate():.0f}% | "
            f"CallWR={call_wr:.0f}%({self._call_wins}W/{self._call_losses}L) | "
            f"PutWR={put_wr:.0f}%({self._put_wins}W/{self._put_losses}L)"
            f"{' [PUTS AUTO-DISABLED]' if self._puts_auto_disabled else ''}"
        )

    def _file_meta(self, path: str) -> Dict[str, Any]:
        if not path:
            return {"exists": False}

        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return {"exists": False}

        try:
            stat = os.stat(abs_path)
            return {
                "exists": True,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except Exception:
            return {"exists": True}

    def _sha256_file(self, path: str) -> Optional[str]:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return None

        try:
            digest = hashlib.sha256()
            with open(abs_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except Exception:
            return None

    def _save_runtime_fingerprint(self):
        """Persist startup fingerprint for deployment verification."""
        try:
            state_dir = os.path.dirname(self.config.STATE_FILE) or "data/state"
            os.makedirs(state_dir, exist_ok=True)

            engine_path = os.path.abspath(__file__)
            model_path = getattr(self.ml_model, "model_path", os.path.join("data", "models", "options_model.joblib"))
            fingerprint = {
                "timestamp": datetime.now().isoformat(),
                "engine_path": engine_path,
                "engine_sha256": self._sha256_file(engine_path),
                "config": asdict(self.config),
                "flags": {
                    "paper": self.config.PAPER,
                    "scanner_enabled": self.config.SCANNER_ENABLED,
                    "puts_enabled": self.config.PUTS_ENABLED,
                    "calls_enabled": self.config.CALLS_ENABLED,
                    "spreads_enabled": self.config.SPREADS_ENABLED,
                    "dashboard_host": self.config.DASHBOARD_HOST,
                    "dashboard_port": self.config.DASHBOARD_PORT,
                    "ml_ready": self.ml_ready,
                    "rl_shadow_mode": self.RL_SHADOW_MODE,
                },
                "model_files": {
                    "ml_model": self._file_meta(model_path),
                    "rl_agent_json": self._file_meta("data/state/rl_agent.json"),
                    "rl_shadow_report": self._file_meta("data/state/rl_shadow_report.json"),
                    "meta_learner": self._file_meta("data/state/meta_learner.json"),
                },
            }

            latest_path = os.path.join(state_dir, "runtime_fingerprint_latest.json")
            with open(latest_path, "w", encoding="utf-8") as handle:
                json.dump(fingerprint, handle, indent=2)

            history_path = os.path.join(state_dir, "runtime_fingerprint_history.jsonl")
            with open(history_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(fingerprint) + "\n")

            log.info(f"Runtime fingerprint saved: {latest_path}")
        except Exception as exc:
            log.warning(f"Runtime fingerprint save failed: {exc}")

    def _save_state(self):
        """Save all state to disk."""
        self.risk.save_state()

        # Save positions
        pos_file = os.path.join(os.path.dirname(self.config.STATE_FILE), "positions.json")
        os.makedirs(os.path.dirname(pos_file), exist_ok=True)
        with self._lock:
            with open(pos_file, "w") as f:
                json.dump(self.positions, f, indent=2, default=str)

        # Save put/call win-rate counters
        wr_file = os.path.join(os.path.dirname(self.config.STATE_FILE), "put_call_wr.json")
        try:
            with open(wr_file, "w") as f:
                json.dump({
                    "put_wins": self._put_wins,
                    "put_losses": self._put_losses,
                    "call_wins": self._call_wins,
                    "call_losses": self._call_losses,
                    "recent_put_outcomes": self._recent_put_outcomes,
                    "puts_auto_disabled": self._puts_auto_disabled,
                }, f, indent=2)
        except Exception as e:
            log.debug(f"Failed to save put/call WR state: {e}")

    def _load_positions(self):
        """Load positions from disk and verify they still exist on Alpaca."""
        pos_file = os.path.join(os.path.dirname(self.config.STATE_FILE), "positions.json")
        if os.path.exists(pos_file):
            try:
                with open(pos_file) as f:
                    saved = json.load(f)
                if saved:
                    self._log_activity(f"Loaded {len(saved)} positions from disk — verifying with Alpaca...")
                    # Verify each position actually exists on Alpaca
                    try:
                        live_positions = self.api.get_positions()
                        live_symbols = {p.get("symbol") for p in live_positions}
                    except Exception:
                        live_symbols = None  # can't verify, keep all

                    if live_symbols is not None:
                        verified = {}
                        stale = []
                        for sym, pos in saved.items():
                            if sym in live_symbols:
                                verified[sym] = pos
                            else:
                                stale.append(sym)
                        if stale:
                            self._log_activity(
                                f"STALE positions removed (not on Alpaca): {', '.join(stale)}",
                                "warning"
                            )
                            # Save cleaned state
                            with open(pos_file, "w") as f:
                                json.dump(verified, f, indent=2)
                        self.positions = verified
                        self._log_activity(f"Verified {len(verified)} live positions")
                    else:
                        self.positions = saved
            except Exception as e:
                log.warning(f"Failed to load positions: {e}")

    def _load_trade_history(self):
        """Load recent trade history from CSV."""
        filepath = self.config.TRADE_LOG
        if not os.path.exists(filepath):
            return
        try:
            import csv as csv_mod
            with open(filepath, "r") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    # Strip None keys — DictReader uses None for extra
                    # fields when rows have more columns than the header.
                    # This causes json.dumps(sort_keys=True) to crash
                    # comparing None with str keys.
                    row.pop(None, None)
                    try:
                        row["pnl"] = float(row.get("pnl", 0))
                        row["pnl_pct"] = float(row.get("pnl_pct", 0))
                        row["strike"] = float(row.get("strike", 0))
                        row["qty"] = int(float(row.get("qty", 0)))
                    except (ValueError, TypeError):
                        pass
                    self.trade_history.append(row)
            # Keep last 200
            self.trade_history = self.trade_history[-200:]
            if self.trade_history:
                self._log_activity(f"Loaded {len(self.trade_history)} historical trades")
        except Exception as e:
            log.warning(f"Failed to load trade history: {e}")

    def _load_put_call_wr(self):
        """Load put/call win-rate counters from disk."""
        wr_file = os.path.join(os.path.dirname(self.config.STATE_FILE), "put_call_wr.json")
        if not os.path.exists(wr_file):
            # Bootstrap from trade history if available
            for t in self.trade_history:
                d = t.get("direction", "")
                win = float(t.get("pnl", 0)) > 0
                if d == "put":
                    if win:
                        self._put_wins += 1
                    else:
                        self._put_losses += 1
                    self._recent_put_outcomes.append(win)
                elif d == "call":
                    if win:
                        self._call_wins += 1
                    else:
                        self._call_losses += 1
            self._recent_put_outcomes = self._recent_put_outcomes[-20:]
            # Check auto-disable on bootstrap
            if len(self._recent_put_outcomes) >= 20:
                put_wr = sum(self._recent_put_outcomes) / len(self._recent_put_outcomes)
                if put_wr < 0.30:
                    self._puts_auto_disabled = True
                    self._log_activity(
                        f"PUT AUTO-DISABLED on startup: WR {put_wr:.0%} < 30% (bootstrapped from history)",
                        "warning"
                    )
            if self._put_wins + self._put_losses > 0:
                self._log_activity(
                    f"Bootstrapped put/call WR from trade history: "
                    f"puts {self._put_wins}W/{self._put_losses}L, "
                    f"calls {self._call_wins}W/{self._call_losses}L"
                )
            return
        try:
            with open(wr_file) as f:
                data = json.load(f)
            self._put_wins = data.get("put_wins", 0)
            self._put_losses = data.get("put_losses", 0)
            self._call_wins = data.get("call_wins", 0)
            self._call_losses = data.get("call_losses", 0)
            self._recent_put_outcomes = data.get("recent_put_outcomes", [])
            self._puts_auto_disabled = data.get("puts_auto_disabled", False)
            pt = self._put_wins + self._put_losses
            ct = self._call_wins + self._call_losses
            self._log_activity(
                f"Loaded put/call WR: puts {self._put_wins}W/{self._put_losses}L "
                f"({self._put_wins/pt*100:.0f}% WR), calls {self._call_wins}W/{self._call_losses}L "
                f"({self._call_wins/ct*100:.0f}% WR)"
                if pt > 0 and ct > 0 else
                f"Loaded put/call WR state (puts={pt} trades, calls={ct} trades)"
            )
            if self._puts_auto_disabled:
                self._log_activity("PUT AUTO-DISABLED (loaded from saved state)", "warning")
        except Exception as e:
            log.warning(f"Failed to load put/call WR state: {e}")

    def _sleep(self, seconds: int):
        """Interruptible sleep."""
        for _ in range(seconds):
            if not self.running:
                break
            time.sleep(1)
