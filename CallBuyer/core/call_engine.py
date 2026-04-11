"""
CallBuyer Engine — Momentum Call Buying Strategy (Left Leg)

Main trading loop:
  1. Every 10 min: scan watchlist for momentum breakouts
  2. Compute features → ML score → meta-learner decision
  3. Find best ITM call → size position → buy
  4. Every 3 min: check exits (TP / SL / DTE / trailing stop)
  5. Record outcomes → retrain ML when enough data
"""
import csv
import json
import logging
import os
import time as _time
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from core.config import CallBuyerConfig
from core.api_client import CallBuyerAPI
from core.risk_manager import RiskManager
from core.feature_engine import CallBuyerFeatureEngine
from core.ml_model import CallBuyerMLModel
from core.meta_learner import MetaLearner
from core.regime_detector import RegimeDetector
from core.earnings_check import has_earnings_within

log = logging.getLogger("callbuyer.engine")


class CallBuyerEngine:
    """Main engine for momentum-based call buying strategy."""

    def __init__(self, config: CallBuyerConfig):
        self.config = config
        self.api = CallBuyerAPI(config)
        self.risk = RiskManager(config)
        self.features = CallBuyerFeatureEngine()
        self.ml = CallBuyerMLModel(
            models_dir=config.MODEL_DIR,
            state_dir="data/state",
        )
        self.meta = MetaLearner(state_dir="data/state")

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self._last_scan = 0.0
        self._last_check = 0.0
        self._cycle = 0
        self._spy_bars = None
        self._regime_detector = RegimeDetector(bot_name="CallBuyer")
        self._current_regime: Optional[Dict[str, Any]] = None
        self._regime_flip_state: Dict[str, Any] = {}  # Flip detection state

        # Universe Scanner — dynamically discovers stocks beyond static watchlist
        try:
            from core.universe_scanner import CallBuyerUniverseScanner
            self.universe_scanner = CallBuyerUniverseScanner(api=self.api)
            log.info("Universe Scanner loaded — dynamic stock discovery enabled")
        except Exception as e:
            self.universe_scanner = None
            log.warning(f"Universe Scanner not available: {e}")
        self._dynamic_watchlist = list(self.config.WATCHLIST)

        # Pre-market warmup state
        self._warmup_done_date: Optional[date] = None
        self._warmup_data: Dict[str, Any] = {}
        self._WARMUP_MINUTES_BEFORE_OPEN = 30
        self._last_adoption: float = 0.0
        self._recently_closed_contracts: set = set()  # prevent re-adoption after sell

        self._load_positions()

    # ── Persistence ──────────────────────────────────────

    def _load_positions(self):
        try:
            if os.path.exists(self.config.POSITIONS_FILE):
                with open(self.config.POSITIONS_FILE) as f:
                    self.positions = json.load(f)
                if self.positions:
                    log.info(f"Loaded {len(self.positions)} positions from disk")
        except Exception as e:
            log.warning(f"Could not load positions: {e}")

    def _save_positions(self):
        try:
            os.makedirs(os.path.dirname(self.config.POSITIONS_FILE), exist_ok=True)
            with open(self.config.POSITIONS_FILE, "w") as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save positions: {e}")

    def _save_state(self):
        try:
            state = {
                "cycle": self._cycle,
                "positions": len(self.positions),
                "ml_status": self.ml.get_status(),
                "meta_status": self.meta.get_status(),
                "updated": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self.config.STATE_FILE), exist_ok=True)
            with open(self.config.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save state: {e}")

    # ── Position Adoption — Protect Orphaned Positions ──

    def _adopt_orphaned_positions(self):
        """Scan Alpaca for long call positions not tracked in positions.json.

        Adopts orphaned positions so exit rules (trailing stop, take profit,
        stop loss, DTE exit) protect them. Skips positions that are spread
        legs managed by PutSeller.
        """
        try:
            import requests as req

            url = f"{self.config.BASE_URL}/v2/positions"
            headers = {
                "APCA-API-KEY-ID": self.config.API_KEY,
                "APCA-API-SECRET-KEY": self.config.API_SECRET,
                "accept": "application/json",
            }
            resp = req.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                log.warning(f"Position adoption: API returned {resp.status_code}")
                return

            all_positions = resp.json()
            option_positions = [
                p for p in all_positions
                if p.get("asset_class") == "us_option"
            ]

            if not option_positions:
                return

            # Build map of short calls with their quantities
            short_call_map = {}  # symbol -> qty
            short_call_syms = set()
            short_put_syms = set()

            for p in option_positions:
                sym = p["symbol"]
                q = int(p["qty"])
                if q < 0:
                    opt_type = self._parse_option_type(sym)
                    if opt_type == "C":
                        short_call_syms.add(sym)
                        short_call_map[sym] = abs(q)
                    elif opt_type == "P":
                        short_put_syms.add(sym)

            # Already-tracked contracts in CallBuyer
            tracked_contracts = set()
            for pos in self.positions.values():
                tracked_contracts.add(pos["contract"])

            # PutSeller-tracked contracts
            putseller_contracts = set()
            try:
                # Cross-platform: check both Windows and Linux paths
                ps_path = r"C:\PutSeller\data\state\positions.json"
                if not os.path.exists(ps_path):
                    ps_path = os.path.expanduser("~/PutSeller/data/state/positions.json")
                if os.path.exists(ps_path):
                    with open(ps_path, encoding="utf-8-sig") as f:
                        ps_positions = json.load(f)
                    for ps_pos in ps_positions.values():
                        putseller_contracts.add(ps_pos.get("short_symbol", ""))
                        putseller_contracts.add(ps_pos.get("long_symbol", ""))
            except Exception:
                pass

            adopted = 0
            for p in option_positions:
                sym = p["symbol"]
                qty = int(p["qty"])
                opt_type = self._parse_option_type(sym)

                # Only adopt LONG CALLS (qty > 0, type = C)
                if qty <= 0 or opt_type != "C":
                    continue

                # Skip if already tracked by CallBuyer
                if sym in tracked_contracts:
                    continue

                # Skip if recently closed (Alpaca API may lag)
                if sym in self._recently_closed_contracts:
                    log.debug(f"ADOPT: Skipping {sym} — recently closed")
                    continue

                # Skip if tracked by PutSeller (it's a spread leg)
                if sym in putseller_contracts:
                    continue

                # Skip if there's a matching short call with same
                # underlying+expiry+qty (= this is a spread leg)
                underlying = self._parse_underlying(sym)
                expiry_str = self._parse_exp_str(sym)
                is_spread_leg = False
                for short_sym in short_call_syms:
                    if (self._parse_underlying(short_sym) == underlying
                            and self._parse_exp_str(short_sym) == expiry_str
                            and short_call_map.get(short_sym, 0) == qty):
                        is_spread_leg = True
                        break

                if is_spread_leg:
                    log.info(f"ADOPT: Skipping {sym} — paired with short call (spread leg)")
                    continue

                # This is a standalone long call — adopt it
                avg_entry = float(p.get("avg_entry_price", 0))
                market_value = float(p.get("market_value", 0))
                current_price = market_value / (qty * 100) if qty > 0 else 0

                exp_date = self._parse_exp_from_contract(sym)
                dte = (exp_date - date.today()).days if exp_date else 0
                strike = self._parse_strike(sym)

                pos_id = f"adopted_{underlying}_{sym[-15:-9]}"
                # Ensure uniqueness
                if pos_id in self.positions:
                    pos_id = f"{pos_id}_{strike:.0f}"

                high_water = max(current_price, avg_entry)
                trailing_active = current_price >= avg_entry * 1.30

                self.positions[pos_id] = {
                    "symbol": underlying,
                    "contract": sym,
                    "strike": strike,
                    "dte_at_entry": dte,
                    "qty": qty,
                    "entry_price": avg_entry,
                    "entry_total": avg_entry * qty * 100,
                    "entry_time": datetime.now().isoformat(),
                    "confidence": 0,
                    "rule_score": 0,
                    "ml_proba": 0,
                    "features": None,
                    "high_water": high_water,
                    "trailing_active": trailing_active,
                    "order_id": "ADOPTED",
                }

                pnl_pct = (
                    (current_price - avg_entry) / avg_entry * 100
                    if avg_entry > 0 else 0
                )
                trailing_str = (
                    "TRAILING ACTIVE" if trailing_active else "watching"
                )
                log.info(
                    f"ADOPTED orphan: {pos_id} | {qty}x {sym} | "
                    f"entry=${avg_entry:.2f} now=${current_price:.2f} "
                    f"pnl={pnl_pct:+.1f}% hw=${high_water:.2f} | {trailing_str}"
                )
                adopted += 1

            if adopted:
                self._save_positions()
                log.info(
                    f"Position adoption complete: {adopted} orphaned "
                    f"positions now managed by exit rules"
                )

        except Exception as e:
            log.error(f"Position adoption failed: {e}", exc_info=True)

    @staticmethod
    def _parse_option_type(occ_symbol: str) -> Optional[str]:
        """Extract 'C' or 'P' from OCC option symbol."""
        for i in range(len(occ_symbol) - 9, 0, -1):
            if occ_symbol[i + 6] in ("C", "P"):
                return occ_symbol[i + 6]
        return None

    @staticmethod
    def _parse_underlying(occ_symbol: str) -> str:
        """Extract underlying ticker from OCC symbol (e.g., GOOGL260501C00305000 → GOOGL)."""
        for i in range(len(occ_symbol) - 9, 0, -1):
            if occ_symbol[i + 6] in ("C", "P"):
                return occ_symbol[:i]
        return occ_symbol

    @staticmethod
    def _parse_exp_str(occ_symbol: str) -> str:
        """Extract YYMMDD string from OCC symbol."""
        for i in range(len(occ_symbol) - 9, 0, -1):
            if occ_symbol[i + 6] in ("C", "P"):
                return occ_symbol[i:i + 6]
        return ""

    @staticmethod
    def _parse_strike(occ_symbol: str) -> float:
        """Extract strike price from OCC symbol (last 8 digits / 1000)."""
        try:
            return int(occ_symbol[-8:]) / 1000.0
        except (ValueError, IndexError):
            return 0.0

    # ── Main Loop ────────────────────────────────────────


    def _check_portfolio_exposure(self) -> tuple:
        """Check aggregate worst-case loss across all bots sharing this Alpaca account.
        Returns (can_trade, exposure_pct).
        """
        PORTFOLIO_MAX_PCT = 0.50  # 50% of equity (combined allocation: PS 35% + CB 15% + AB 0%)
        # Cross-platform path resolution (works on both Windows C:\ and Linux /home/botuser/)
        _base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        POSITION_FILES = {
            os.path.join(_base, "PutSeller", "data", "state", "positions.json"): "max_loss_total",
            os.path.join(_base, "CallBuyer", "data", "state", "positions.json"): "entry_total",
            os.path.join(_base, "AlpacaBot", "data", "state", "positions.json"): "cost",
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

    def run(self):
        """Main trading loop — scan → trade → manage → repeat."""
        self.running = True
        log.info("=" * 60)
        log.info("CallBuyer Engine starting...")
        log.info(self.config.summary())
        log.info("=" * 60)

        if not self.api.connect():
            log.error("Failed to connect to Alpaca API — exiting")
            return

        # Adopt orphaned positions from Alpaca that aren't tracked
        self._adopt_orphaned_positions()

        while self.running:
            try:
                if not self.api.is_market_open():
                    # Pre-market warmup: run once, ~30 min before open
                    if self._should_warmup():
                        self._pre_market_warmup()
                    else:
                        next_open = self.api.next_open()
                        if next_open:
                            log.info(f"Market closed. Next open: {next_open}")
                    self._save_state()
                    _time.sleep(60)
                    continue

                now = _time.time()
                self._cycle += 1

                # Universe scan — expand watchlist (every 30 min)
                if (self.universe_scanner
                        and self.universe_scanner.should_scan()):
                    try:
                        expanded = self.universe_scanner.scan()
                        self._dynamic_watchlist = expanded
                        log.info(
                            f"Universe scan: expanded watchlist to "
                            f"{len(expanded)} symbols "
                            f"({self.universe_scanner.assets_passed_filter} passed filters)"
                        )
                    except Exception as e:
                        log.warning(f"Universe scan failed: {e}")

                # Check positions every 3 minutes
                if now - self._last_check >= self.config.CHECK_INTERVAL_SEC:
                    self._check_all_positions()
                    self._last_check = now

                # Re-adopt orphaned positions every 30 minutes
                if now - self._last_adoption >= 1800:
                    self._adopt_orphaned_positions()
                    self._last_adoption = now

                # Scan for new opportunities every 10 minutes
                if now - self._last_scan >= self.config.SCAN_INTERVAL_SEC:
                    self._scan_for_opportunities()
                    self._last_scan = now

                    # Check if ML model needs retraining
                    if self.config.ML_ENABLED and self.ml.should_retrain():
                        log.info("Triggering ML model retrain...")
                        success = self.ml.train()
                        log.info(f"ML retrain {'succeeded' if success else 'below quality gate'}")

                self._save_state()
                _time.sleep(30)

            except KeyboardInterrupt:
                log.info("Shutdown requested")
                self.running = False
            except Exception as e:
                log.error(f"Loop error: {e}", exc_info=True)
                _time.sleep(60)

        log.info("CallBuyer Engine stopped")
        self._save_positions()
        self._save_state()

    def stop(self):
        self.running = False

    # ── Scanning ─────────────────────────────────────────

    def _scan_for_opportunities(self):
        """Scan watchlist for momentum breakout opportunities."""
        if len(self.positions) >= self.config.MAX_POSITIONS:
            log.info(f"At max positions ({len(self.positions)}), skipping scan")
            return

        if not self.risk.can_trade():
            log.info("Risk manager blocking trades")
            return

        # Check timing (use Eastern time for market hours)
        now = datetime.now(ZoneInfo("America/New_York"))
        market_open_min = now.hour * 60 + now.minute - (self.config.MARKET_OPEN_HOUR * 60 + self.config.MARKET_OPEN_MIN)
        market_close_min = (self.config.MARKET_CLOSE_HOUR * 60 + self.config.MARKET_CLOSE_MIN) - (now.hour * 60 + now.minute)

        if market_open_min < self.config.NO_OPEN_FIRST_MIN:
            log.info(f"Waiting {self.config.NO_OPEN_FIRST_MIN - market_open_min}min after open")
            return
        if market_close_min < self.config.NO_OPEN_LAST_MIN:
            log.info(f"Too close to close ({market_close_min}min), skipping scan")
            return

        log.info(f"─── Scan Cycle {self._cycle} | Positions: {len(self.positions)}/{self.config.MAX_POSITIONS} ───")

        # Get SPY bars for sector momentum feature
        try:
            self._spy_bars = self.api.get_bars("SPY", days=30)
        except Exception:
            self._spy_bars = None

        # ── Regime Detection ─────────────────────────────
        try:
            regime_bars = self._spy_bars or self.api.get_bars("SPY", days=90)
            if regime_bars and len(regime_bars) >= 30:
                self._current_regime = self._regime_detector.detect(regime_bars)
                r = self._current_regime

                # ── Regime Flip Detection ────────────
                regime_name = r["regime"]
                flip_state = self._regime_detector.record_regime(
                    regime_name, r["confidence"]
                )
                self._regime_flip_state = flip_state

                if flip_state.get("is_cooldown"):
                    log.warning(
                        f"Market Regime: {regime_name} | "
                        f"FLIP COOLDOWN {flip_state['cooldown_remaining_min']:.0f}m "
                        f"(severity={flip_state['flip_severity']:.2f}, "
                        f"whipsaw={flip_state['whipsaw_score']:.2f}, "
                        f"mult={flip_state['adjustment_multiplier']:.2f})"
                    )
                else:
                    log.info(
                        f"Market Regime: {regime_name} "
                        f"(conf={r['confidence']:.2f}, "
                        f"trend={r['trend_strength']:.2f}, "
                        f"vol_ratio={r['volatility_ratio']:.2f}, "
                        f"stability={flip_state.get('regime_stability', 0):.2f})"
                    )
            else:
                self._current_regime = None
        except Exception as e:
            log.warning(f"Regime detection failed: {e}")
            self._current_regime = None

        # ── Portfolio-Level Aggregate Exposure Cap ─────────
        can_trade_portfolio, portfolio_exposure = self._check_portfolio_exposure()
        if not can_trade_portfolio:
            return

        candidates = []
        skipped = {"rsi": 0, "no_bars": 0, "features": 0, "low_conf": 0}
        for symbol in self._dynamic_watchlist:
            # Skip if already at max per underlying
            pos_count = sum(1 for p in self.positions.values() if p.get("symbol") == symbol)
            if pos_count >= self.config.MAX_PER_UNDERLYING:
                continue

            # Earnings guard — don't buy calls into earnings volatility
            if has_earnings_within(symbol, self.config.EARNINGS_BUFFER_DAYS):
                log.info(f"  {symbol}: earnings within {self.config.EARNINGS_BUFFER_DAYS}d — skipping")
                continue

            score = self._evaluate_symbol(symbol)
            if score is not None:
                if score.get("should_trade"):
                    candidates.append(score)
                else:
                    skipped["low_conf"] += 1
                    log.debug(f"  {symbol}: rejected — {score.get('reason', '?')} (conf={score.get('confidence', 0):.3f})")
            else:
                # Will be counted below from debug logs
                pass

        if not candidates:
            log.info(f"No momentum candidates found this scan (evaluated {len(self._dynamic_watchlist)})")
            return

        # Sort by confidence descending
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        log.info(f"Found {len(candidates)} candidates:")
        for c in candidates[:5]:
            log.info(f"  {c['symbol']}: conf={c['confidence']:.3f} rules={c['rule_score']:.1f} "
                     f"ml={c['ml_proba']:.2f} ({c['reason']})")

        # Try to open trades for top candidates
        opened = 0
        max_new = self.config.MAX_POSITIONS - len(self.positions)
        for candidate in candidates[:min(3, max_new)]:
            if not candidate["should_trade"]:
                continue
            if self._open_call(candidate):
                opened += 1

        if opened:
            log.info(f"Opened {opened} new call position(s)")

    # ── Time-of-Day Window ───────────────────────────────

    def _get_time_window(self) -> str:
        """Return 'morning', 'afternoon', or 'closed' based on current ET time.

        Morning (9:30-11:00 ET): momentum breakouts strongest at open.
        Afternoon (11:00-16:00 ET): only take strong setups.
        """
        now_et = datetime.now(ZoneInfo("America/New_York"))
        minutes_since_open = (now_et.hour - 9) * 60 + (now_et.minute - 30)

        if minutes_since_open < 0 or minutes_since_open >= 390:
            return "closed"
        elif minutes_since_open < getattr(self.config, 'MORNING_WINDOW_END_MIN', 90):
            return "morning"
        else:
            return "afternoon"

    def _evaluate_symbol(self, symbol: str) -> Optional[Dict]:
        """Evaluate a symbol for call-buying opportunity.

        Returns candidate dict or None if not interesting.
        """
        try:
            # Get daily bars for feature computation
            bars = self.api.get_bars(symbol, days=90)
            if not bars or len(bars) < 50:
                log.info(f"  {symbol}: bars={len(bars) if bars else 0}/50 — skipping (need 50+ daily bars)")
                return None

            # Get current IV from ATM options if possible
            price = float(bars[-1].close)
            iv_current = None
            hv20 = None
            try:
                hv20 = self.api.calculate_hv20(symbol)
            except Exception:
                pass

            # Build features
            feature_vec = self.features.build_features(
                daily_bars=bars,
                iv_current=iv_current,
                hv20=hv20,
                spy_bars=self._spy_bars,
            )
            if feature_vec is None:
                log.info(f"  {symbol}: build_features returned None — skipping")
                return None

            # Quick rules filter before ML
            rsi = feature_vec[0] * 100
            if rsi < self.config.MIN_RSI or rsi > self.config.MAX_RSI:
                log.info(f"  {symbol}: RSI {rsi:.0f} outside [{self.config.MIN_RSI}-{self.config.MAX_RSI}] — skipping")
                return None  # RSI outside bullish range

            # Rules-based score
            rule_score = self.features.compute_rule_score(feature_vec)

            # ML prediction
            ml_proba, ml_active = self.ml.predict(feature_vec)

            # Meta-learner decision (with morning/afternoon adjustment)
            time_window = self._get_time_window()
            conf_adj = 0.0
            rule_adj = 0.0
            if getattr(self.config, 'MORNING_WINDOW_ENABLED', False):
                if time_window == "morning":
                    conf_adj = -self.config.MORNING_CONF_BOOST    # negative = lower threshold = easier to pass
                    rule_adj = -self.config.MORNING_RULE_REDUCTION
                    log.debug(f"  {symbol}: Morning window — mild boost "
                              f"(conf adj={conf_adj:+.2f}, rule adj={rule_adj:+.1f})")
                elif time_window == "afternoon":
                    conf_adj = +self.config.AFTERNOON_CONF_PENALTY  # positive = raise threshold = harder
                    rule_adj = +self.config.AFTERNOON_RULE_INCREASE

            confidence, should_trade, reason = self.meta.evaluate(
                rule_score=rule_score,
                ml_proba=ml_proba,
                ml_active=ml_active,
                conf_adjust=conf_adj,
                rule_adjust=rule_adj,
            )

            # ── Regime-based confidence adjustment ───────
            if self._current_regime:
                regime_name = self._current_regime["regime"]
                adj = self._current_regime.get("suggested_adjustments", {})
                conf_offset = adj.get("confidence_offset", 0.0)
                if conf_offset != 0.0:
                    original_conf = confidence
                    # Offset is applied to the *threshold*, so we subtract
                    # from confidence to simulate raising/lowering bar
                    confidence = confidence - conf_offset
                    confidence = max(0.0, min(1.0, confidence))
                    log.debug(
                        f"  {symbol}: regime={regime_name} "
                        f"conf {original_conf:.3f} -> {confidence:.3f} "
                        f"(offset {conf_offset:+.2f})"
                    )
                    # Re-evaluate should_trade with adjusted confidence
                    should_trade, reason = self.meta.evaluate(
                        rule_score=rule_score,
                        ml_proba=ml_proba,
                        ml_active=ml_active,
                        conf_adjust=conf_adj,
                        rule_adjust=rule_adj,
                    )[1:]
                    # Override with regime-adjusted confidence
                    if confidence < 0.30:
                        should_trade = False
                        reason = f"regime_{regime_name}_low_conf"

                # Regime flip: block entries during severe transitions
                if should_trade and self._regime_flip_state.get("should_block_entries", False):
                    should_trade = False
                    reason = (
                        f"regime_flip_blocked "
                        f"({self._regime_flip_state.get('last_flip_from')}→"
                        f"{self._regime_flip_state.get('last_flip_to')})"
                    )

            log.info(f"  {symbol}: RSI={rsi:.0f} rules={rule_score:.1f} "
                     f"ml={ml_proba:.2f} conf={confidence:.3f} "
                     f"{'PASS' if should_trade else 'SKIP'} ({reason})")

            return {
                "symbol": symbol,
                "price": price,
                "features": feature_vec,
                "rule_score": rule_score,
                "ml_proba": ml_proba,
                "ml_active": ml_active,
                "confidence": confidence,
                "should_trade": should_trade,
                "reason": reason,
            }

        except Exception as e:
            log.warning(f"Error evaluating {symbol}: {e}")
            return None

    # ── Opening Positions ────────────────────────────────

    def _open_call(self, candidate: Dict) -> bool:
        """Find the best ITM call and buy it."""
        symbol = candidate["symbol"]
        price = candidate["price"]

        try:
            # Find options chain (calls only)
            exp_after = (date.today() + timedelta(days=self.config.MIN_DTE)).isoformat()
            exp_before = (date.today() + timedelta(days=self.config.MAX_DTE)).isoformat()
            chain = self.api.get_options_chain(
                underlying=symbol,
                option_type="call",
                expiration_after=exp_after,
                expiration_before=exp_before,
                strike_price_gte=round(price * (1 + self.config.MIN_OTM_PCT - 0.01), 2),  # ITM range
                strike_price_lte=round(price * (1 + self.config.MAX_OTM_PCT + 0.01), 2),  # slight OTM buffer
            )
            if not chain:
                log.info(f"{symbol}: No suitable call options found")
                return False

            # Filter and rank strikes
            best = self._select_strike(symbol, price, chain)
            if not best:
                log.info(f"{symbol}: No strike passed filters")
                return False

            contract_symbol, strike, dte, ask_price = best

            # Size the position
            acct = self.api.get_account()
            account_equity = acct.get("equity", 0) if acct else 0
            if not account_equity:
                return False

            allocation = account_equity * self.config.ALLOCATION_PCT
            max_risk = self.risk.size_position(
                allocation=allocation,
                option_price=ask_price,
            )
            if max_risk is None or max_risk < 1:
                log.info(f"{symbol}: Position too small or risk too high")
                return False

            qty = max_risk  # number of contracts

            # Regime flip position sizing reduction
            flip_mult = self._regime_flip_state.get("adjustment_multiplier", 1.0)
            if flip_mult < 1.0:
                adjusted_qty = max(1, int(qty * flip_mult))
                if adjusted_qty < qty:
                    log.info(
                        f"{symbol}: flip sizing {qty} → {adjusted_qty} contracts "
                        f"(mult={flip_mult:.2f})"
                    )
                    qty = adjusted_qty

            # Execute buy
            log.info(f"BUYING {qty}x {contract_symbol} @ ${ask_price:.2f} "
                     f"(strike={strike}, DTE={dte}, total=${ask_price * qty * 100:.0f})")

            order = self.api.buy_call(
                option_symbol=contract_symbol,
                qty=qty,
                limit_price=ask_price,
            )
            if not order:
                log.warning(f"Order failed for {contract_symbol}")
                return False

            # Record position
            pos_id = f"cb_{symbol}_{datetime.now().strftime('%H%M%S')}"
            self.positions[pos_id] = {
                "symbol": symbol,
                "contract": contract_symbol,
                "strike": strike,
                "dte_at_entry": dte,
                "qty": qty,
                "entry_price": ask_price,
                "entry_total": ask_price * qty * 100,
                "entry_time": datetime.now().isoformat(),
                "confidence": candidate["confidence"],
                "rule_score": candidate["rule_score"],
                "ml_proba": candidate["ml_proba"],
                "features": candidate["features"].tolist(),
                "high_water": ask_price,
                "trailing_active": False,
                "order_id": str(order.id) if hasattr(order, "id") else str(order),
            }
            self._save_positions()

            # Log features for ML training
            self.features.log_features(
                symbol=symbol,
                features=candidate["features"],
                outcome=None,  # filled in when trade closes
            )

            self.risk.record_trade(won=None)  # outcome pending
            log.info(f"✓ Opened {pos_id}: {qty}x {contract_symbol}")
            return True

        except Exception as e:
            log.error(f"Error opening call for {symbol}: {e}", exc_info=True)
            return False

    def _select_strike(self, symbol: str, price: float,
                       chain: list) -> Optional[Tuple[str, float, int, float]]:
        """Select optimal ITM call strike from chain.

        Returns (contract_symbol, strike, dte, ask_price) or None.
        """
        scored = []
        for opt in chain:
            try:
                strike = float(opt.get("strike", 0))
                ask = float(opt.get("ask", 0))
                bid = float(opt.get("bid", 0))
                oi = int(opt.get("open_interest", 0))
                contract_sym = opt.get("symbol", "")
                exp_str = opt.get("expiration", "")

                if not contract_sym or ask <= 0:
                    continue

                # Calculate moneyness (negative = ITM, positive = OTM)
                otm_pct = (strike - price) / price if price > 0 else 0

                # Must be within ITM/OTM range (ITM up to 5%, OTM up to 2%)
                if otm_pct < self.config.MIN_OTM_PCT or otm_pct > self.config.MAX_OTM_PCT:
                    continue

                # Calculate DTE
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = (exp_date - date.today()).days
                except Exception:
                    continue

                if dte < self.config.MIN_DTE or dte > self.config.MAX_DTE:
                    continue

                # Must have reasonable spread
                spread = (ask - bid) / ask if ask > 0 else 1.0
                if spread > 0.30:
                    continue  # too wide

                # Must have some open interest (liquidity)
                if oi < 50:
                    continue

                # Score the option
                # Prefer: close to target DTE, close to target OTM%, tight spread, high OI
                dte_score = 1.0 - abs(dte - self.config.TARGET_DTE) / self.config.MAX_DTE
                otm_score = max(0.0, 1.0 - abs(otm_pct - self.config.TARGET_OTM_PCT) / 0.07)
                spread_score = 1.0 - spread
                oi_score = min(oi / 500.0, 1.0)

                total_score = (
                    otm_score * 0.35 +
                    dte_score * 0.30 +
                    spread_score * 0.20 +
                    oi_score * 0.15
                )

                scored.append((contract_sym, strike, dte, ask, total_score))

            except Exception:
                continue

        if not scored:
            return None

        # Sort by score descending, pick best
        scored.sort(key=lambda x: x[4], reverse=True)
        best = scored[0]
        log.info(f"  {symbol}: Best call {best[0]} strike={best[1]} DTE={best[2]} "
                 f"ask=${best[3]:.2f} score={best[4]:.3f}")
        return (best[0], best[1], best[2], best[3])

    # ── Position Management ──────────────────────────────

    def _check_all_positions(self):
        """Check all open positions for exit conditions."""
        if not self.positions:
            return

        to_close = []
        for pos_id, pos in self.positions.items():
            exit_reason = self._check_exit(pos_id, pos)
            if exit_reason:
                to_close.append((pos_id, exit_reason))

        for pos_id, reason in to_close:
            self._close_position(pos_id, reason)

    def _check_exit(self, pos_id: str, pos: Dict) -> Optional[str]:
        """Check if a position should be closed.

        Returns exit reason string or None to hold.
        """
        try:
            contract = pos["contract"]
            entry_price = pos["entry_price"]

            # Get current option price
            quote = self.api.get_option_quote(contract)
            if not quote:
                log.debug(f"No quote for {contract}")
                return None

            current_bid = quote.get("bid", 0)
            if current_bid <= 0:
                # Fallback to mid price when bid unavailable
                current_bid = quote.get("mid", 0)
            if current_bid <= 0:
                log.debug(f"No usable price for {contract}, skipping exit check")
                return None

            # P&L calculation
            pnl_pct = (current_bid - entry_price) / entry_price if entry_price > 0 else 0

            # Update high water mark for trailing stop
            if current_bid > pos.get("high_water", entry_price):
                pos["high_water"] = current_bid

            # 1. Take Profit
            if pnl_pct >= self.config.TAKE_PROFIT_PCT:
                return f"TAKE_PROFIT ({pnl_pct:+.1%})"

            # 2. Stop Loss
            if pnl_pct <= self.config.STOP_LOSS_PCT:
                return f"STOP_LOSS ({pnl_pct:+.1%})"

            # 3. DTE Exit — avoid theta crush
            try:
                contract_str = contract
                # Extract expiration from OCC symbol (last 15 chars format)
                # e.g., NVDA260321C00150000 → 260321 → 2026-03-21 → DTE
                exp_date = self._parse_exp_from_contract(contract_str)
                if exp_date:
                    dte = (exp_date - date.today()).days
                    if dte <= self.config.MIN_DTE_EXIT:
                        return f"DTE_EXIT ({dte} DTE remaining, pnl={pnl_pct:+.1%})"
            except Exception:
                pass

            # 4. Trailing Stop — after 30%+ gain, trail at 20%
            high_water = pos.get("high_water", entry_price)
            if high_water > entry_price * 1.30:  # 30%+ gain reached
                pos["trailing_active"] = True
                drawdown = (current_bid - high_water) / high_water if high_water > 0 else 0
                if drawdown <= -self.config.TRAILING_STOP_PCT:
                    return f"TRAILING_STOP (peak=${high_water:.2f}, now=${current_bid:.2f}, dd={drawdown:+.1%})"

            # Log position status
            if self._cycle % 10 == 0:
                status = "trailing" if pos.get("trailing_active") else "holding"
                log.info(f"  {pos_id}: {contract} pnl={pnl_pct:+.1%} bid=${current_bid:.2f} [{status}]")

        except Exception as e:
            log.warning(f"Error checking {pos_id}: {e}")

        return None

    def _close_position(self, pos_id: str, reason: str):
        """Close a call position."""
        if pos_id not in self.positions:
            return

        pos = self.positions[pos_id]
        contract = pos["contract"]
        qty = pos["qty"]
        entry_price = pos["entry_price"]

        log.info(f"CLOSING {pos_id}: {reason}")

        try:
            # Get current price for P&L
            quote = self.api.get_option_quote(contract)
            current_price = quote.get("bid", 0) if quote else 0

            # Sell to close
            order = self.api.sell_call(
                option_symbol=contract,
                qty=qty,
                limit_price=current_price if current_price > 0.05 else None,
            )

            if not order:
                log.warning(f"Close order failed for {contract}, trying market order")
                order = self.api.sell_call(
                    option_symbol=contract,
                    qty=qty,
                    limit_price=None,  # market order
                )

            if not order:
                log.error(f"SELL FAILED for {pos_id} ({contract}) — position kept, will retry next cycle")
                return

            # Calculate P&L
            pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            pnl_dollar = (current_price - entry_price) * qty * 100
            won = pnl_pct > 0

            log.info(f"{'WIN' if won else 'LOSS'} {pos_id}: "
                     f"entry=${entry_price:.2f} exit=${current_price:.2f} "
                     f"pnl={pnl_pct:+.1%} (${pnl_dollar:+.0f}) reason={reason}")

            # Record outcome for ML training
            if self.config.ML_ENABLED and pos.get("features"):
                features = np.array(pos["features"])
                self.ml.record_outcome(
                    symbol=pos["symbol"],
                    features=features,
                    won=won,
                    pnl_pct=pnl_pct * 100,
                )

            # Update meta-learner
            self.meta.record_result(won=won, pnl_pct=pnl_pct * 100)

            # Record in risk manager
            self.risk.record_trade(won=won)

            # Log to CSV
            self._log_trade(pos, current_price, pnl_pct, pnl_dollar, reason, won)

            # Remove position and track as recently closed (cap at 200)
            self._recently_closed_contracts.add(contract)
            if len(self._recently_closed_contracts) > 200:
                # Trim to most recent 100 (sets are unordered, but this prevents unbounded growth)
                self._recently_closed_contracts = set(list(self._recently_closed_contracts)[-100:])
            del self.positions[pos_id]
            self._save_positions()

        except Exception as e:
            log.error(f"Error closing {pos_id}: {e}", exc_info=True)

    def _log_trade(self, pos: Dict, exit_price: float, pnl_pct: float,
                   pnl_dollar: float, reason: str, won: bool):
        """Append trade to CSV log."""
        try:
            log_path = self.config.TRADE_LOG
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            header_needed = not os.path.exists(log_path)

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow([
                        "timestamp", "symbol", "contract", "qty",
                        "entry_price", "exit_price", "pnl_pct", "pnl_dollar",
                        "reason", "won", "confidence", "rule_score", "ml_proba",
                        "dte_at_entry", "hold_time_hours",
                    ])

                entry_time = datetime.fromisoformat(pos.get("entry_time", datetime.now().isoformat()))
                hold_hours = (datetime.now() - entry_time).total_seconds() / 3600

                writer.writerow([
                    datetime.now().isoformat(),
                    pos.get("symbol", ""),
                    pos.get("contract", ""),
                    pos.get("qty", 0),
                    pos.get("entry_price", 0),
                    exit_price,
                    f"{pnl_pct:.4f}",
                    f"{pnl_dollar:.2f}",
                    reason,
                    "WIN" if won else "LOSS",
                    pos.get("confidence", 0),
                    pos.get("rule_score", 0),
                    pos.get("ml_proba", 0),
                    pos.get("dte_at_entry", 0),
                    f"{hold_hours:.1f}",
                ])
        except Exception as e:
            log.warning(f"Could not log trade: {e}")

    @staticmethod
    def _parse_exp_from_contract(contract: str) -> Optional[date]:
        """Parse expiration date from OCC option symbol.

        Format: SYMBOL + YYMMDD + C/P + 8-digit strike
        e.g., NVDA260321C00150000 → 2026-03-21
        """
        try:
            # Find the date portion — 6 digits before C or P
            for i in range(len(contract) - 9, 0, -1):
                if contract[i + 6] in ("C", "P"):
                    yymmdd = contract[i:i + 6]
                    yy = int(yymmdd[:2])
                    mm = int(yymmdd[2:4])
                    dd = int(yymmdd[4:6])
                    return date(2000 + yy, mm, dd)
        except Exception:
            pass
        return None

    # ── Status API ───────────────────────────────────────

    def get_status(self) -> Dict:
        """Return engine status for dashboard."""
        return {
            "running": self.running,
            "cycle": self._cycle,
            "positions": len(self.positions),
            "max_positions": self.config.MAX_POSITIONS,
            "ml": self.ml.get_status(),
            "meta": self.meta.get_status(),
            "risk": {
                "can_trade": self.risk.can_trade(),
                "daily_trades": self.risk.daily_trades,
                "daily_pnl_pct": round(self.risk.daily_pnl_pct, 2),
            },
            "position_details": {
                pid: {
                    "symbol": p["symbol"],
                    "contract": p["contract"],
                    "qty": p["qty"],
                    "entry_price": p["entry_price"],
                    "entry_time": p["entry_time"],
                    "confidence": p.get("confidence", 0),
                }
                for pid, p in self.positions.items()
            },
        }

    # ── Pre-Market Warmup ────────────────────────────────────

    def _should_warmup(self) -> bool:
        """Return True if within 30 min of market open and not yet warmed up today."""
        if self._warmup_done_date == date.today():
            return False
        try:
            next_open = self.api.next_open()
            if next_open is None:
                return False
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
        """Pre-market analysis ~30 min before open.

        1. Fetch daily bars (60 days) for feature engine pre-computation
        2. Detect overnight gaps
        3. Pre-compute RSI / momentum features so first scan is warm
        4. Run universe scan to refresh watchlist
        5. Sync account balances
        """
        log.info("=" * 50)
        log.info("PRE-MARKET WARMUP starting...")
        warmup_start = _time.time()
        results = {"gaps": {}, "features": {}, "symbols_warmed": 0}

        # ── 1. Fetch daily bars & compute features for all watchlist symbols ──
        all_symbols = list(set(self._dynamic_watchlist))
        gaps = {}
        features_precomputed = 0
        warmed = 0
        for symbol in all_symbols:
            try:
                bars = self.api.get_bars(symbol, days=90)  # daily bars (default)
                if bars and len(bars) >= 2:
                    prev_close = float(bars[-2].close)
                    last_close = float(bars[-1].close)
                    gap_pct = ((last_close - prev_close) / prev_close) * 100
                    gaps[symbol] = round(gap_pct, 2)
                    warmed += 1

                    # Pre-compute features if feature engine available
                    if self.features and len(bars) >= 50:
                        try:
                            feat = self.features.build_features(bars)
                            if feat is not None:
                                features_precomputed += 1
                        except Exception:
                            pass
                _time.sleep(0.3)
            except Exception as e:
                log.debug(f"Warmup bar fetch failed for {symbol}: {e}")

        results["gaps"] = gaps
        results["symbols_warmed"] = warmed
        results["features_precomputed"] = features_precomputed

        # Log notable gaps
        big_gaps = {s: g for s, g in gaps.items() if abs(g) > 2.0}
        if big_gaps:
            sorted_gaps = sorted(big_gaps.items(), key=lambda x: abs(x[1]), reverse=True)
            gap_str = ", ".join(f"{s}:{g:+.1f}%" for s, g in sorted_gaps[:10])
            log.info(f"WARMUP gaps >2%: {gap_str}")

        # ── 2. Universe scan ──
        if self.universe_scanner:
            try:
                expanded = self.universe_scanner.scan()
                self._dynamic_watchlist = expanded
                results["universe_size"] = len(expanded)
                log.info(f"WARMUP universe: {len(expanded)} symbols")
            except Exception as e:
                log.warning(f"Warmup universe scan failed: {e}")

        # ── 3. Sync account ──
        try:
            acct = self.api.get_account()
            allocation = acct["equity"] * self.config.ALLOCATION_PCT
            self.risk.update_allocation(acct["equity"])
            results["equity"] = acct["equity"]
            results["allocation"] = allocation
            log.info(f"WARMUP account: ${acct['equity']:,.2f} equity, allocation ${allocation:,.2f}")
        except Exception as e:
            log.warning(f"Warmup account sync failed: {e}")

        # ── 4. Check held positions for gap risk ──
        if self.positions:
            for pos_id, pos in self.positions.items():
                symbol = pos.get("symbol", "")
                gap = gaps.get(symbol, 0)
                if gap < -3.0:
                    log.warning(
                        f"WARMUP ALERT: {symbol} gapped {gap:+.1f}% — "
                        f"held call may open at a loss"
                    )
            log.info(f"WARMUP: {len(self.positions)} positions checked for gap exposure")

        # ── Done ──
        elapsed = _time.time() - warmup_start
        self._warmup_done_date = date.today()
        self._warmup_data = results
        log.info(
            f"PRE-MARKET WARMUP complete: {warmed} symbols, "
            f"{features_precomputed} features pre-computed, "
            f"{len(big_gaps)} big gaps, "
            f"{elapsed:.1f}s elapsed"
        )
        log.info("=" * 50)
