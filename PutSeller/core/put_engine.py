"""
IronCondor Engine — Credit Put & Call Spread Strategy.

Sells OTM put spreads AND OTM call spreads on high-quality large-cap stocks.
Combines both sides for iron condor risk profile.

Put Side (Bull Put Spread):
- Sell OTM put (short leg) + buy further OTM put (long leg)
- Win when stock stays above short put strike

Call Side (Bear Call Spread):
- Sell OTM call (short leg) + buy further OTM call (long leg)
- Win when stock stays below short call strike

Both sides:
- Collect net credit upfront
- Max profit = credit received (if stock stays between strikes)
- Target 30-45 DTE, close at 50% profit or 14 DTE
"""
import atexit
import json
import logging
import os
import time as _time
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List

from core.config import PutSellerConfig
from core.api_client import PutSellerAPI
from core.risk_manager import RiskManager
from core.earnings_check import has_earnings_within

# Sentiment module
try:
    from utils.sentiment import IronCondorSentiment
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Regime detection
try:
    from core.regime_detector import RegimeDetector
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False

# ML Stack imports
try:
    import numpy as np
    from core.feature_engine import PutSellerFeatureEngine
    from core.ml_model import PutSellerMLModel
    from core.meta_learner import PutSellerMetaLearner
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

log = logging.getLogger("ironcondor.engine")


class PutSellerEngine:
    """Main engine for credit put + call spread (iron condor) strategy."""

    def __init__(self, config: PutSellerConfig):
        self.config = config
        self.api = PutSellerAPI(config)
        self.risk = RiskManager(config)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self._last_scan = 0.0
        self._last_check = 0.0
        self._cycle = 0
        self._spy_bars = None
        self._regime_detector = RegimeDetector(bot_name="IronCondor") if REGIME_AVAILABLE else None
        self._current_regime: Optional[Dict[str, Any]] = None
        self._regime_flip_state: Dict[str, Any] = {}  # Flip detection state

        # ML stack (graceful fallback if deps missing)
        if ML_AVAILABLE:
            self.features = PutSellerFeatureEngine()
            self.ml = PutSellerMLModel(models_dir="models", state_dir="data/state")
            self.meta = PutSellerMetaLearner(state_dir="data/state")
            log.info("ML stack loaded (feature engine + model + meta-learner)")
        else:
            self.features = None
            self.ml = None
            self.meta = None
            log.warning("ML stack not available — running rules-only mode")

        # Sentiment module
        if SENTIMENT_AVAILABLE:
            import os
            finnhub_key = os.getenv("FINNHUB_API_KEY", "")
            self.sentiment = IronCondorSentiment(
                api_client=self.api, finnhub_key=finnhub_key
            )
            log.info(f"Sentiment module loaded (finnhub={'enabled' if finnhub_key else 'disabled'})")
        else:
            self.sentiment = None
            log.warning("Sentiment module not available — running without sentiment")

        # Load positions from disk
        self._load_positions()

        # Graceful shutdown: save positions on exit
        atexit.register(self._shutdown_save)

        # Universe Scanner — dynamically discovers stocks beyond static watchlist
        try:
            from core.universe_scanner import PutSellerUniverseScanner
            self.universe_scanner = PutSellerUniverseScanner(api=self.api)
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

    # ── Persistence ──────────────────────────────────────
    def _load_positions(self):
        """Load active positions from disk."""
        try:
            path = self.config.POSITIONS_FILE
            if os.path.exists(path):
                with open(path) as f:
                    self.positions = json.load(f)
                if self.positions:
                    log.info(f"Loaded {len(self.positions)} positions from disk")
        except Exception as e:
            log.warning(f"Could not load positions: {e}")

    def _save_positions(self):
        """Save active positions to disk."""
        try:
            os.makedirs(os.path.dirname(self.config.POSITIONS_FILE), exist_ok=True)
            with open(self.config.POSITIONS_FILE, "w") as f:
                json.dump(self.positions, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Failed to save positions: {e}")

    def _shutdown_save(self):
        """Called on exit — save positions so they survive restarts."""
        if self.positions:
            log.info(f"Shutdown: saving {len(self.positions)} positions to disk")
            self._save_positions()

    # ── Position Adoption — Protect Orphaned Spreads ─────

    def _adopt_orphaned_spreads(self):
        """Scan Alpaca for option spreads not tracked in positions.json.

        Pairs short+long legs into spreads and adopts orphaned ones so
        exit rules (take profit, stop loss, DTE, emergency, delta breach)
        protect them.
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

            # Already-tracked symbols in PutSeller
            tracked_symbols = set()
            for pos in self.positions.values():
                tracked_symbols.add(pos.get("short_symbol", ""))
                tracked_symbols.add(pos.get("long_symbol", ""))

            # CallBuyer-tracked contracts (don't steal their positions)
            callbuyer_contracts = set()
            try:
                cb_path = r"C:\CallBuyer\data\state\positions.json"
                if os.path.exists(cb_path):
                    with open(cb_path, encoding="utf-8-sig") as f:
                        cb_positions = json.load(f)
                    for cb_pos in cb_positions.values():
                        callbuyer_contracts.add(cb_pos.get("contract", ""))
            except Exception:
                pass

            # Group by underlying + expiry + option_type to find spread pairs
            short_legs = {}  # key: (underlying, expiry, type) -> list of positions
            long_legs = {}

            for p in option_positions:
                sym = p["symbol"]
                qty = int(p["qty"])
                opt_type = self._parse_option_type(sym)
                underlying = self._parse_underlying(sym)
                expiry_str = self._parse_exp_str(sym)
                strike = self._parse_strike_from_occ(sym)

                if not opt_type or not underlying or not expiry_str:
                    continue

                key = (underlying, expiry_str, opt_type)
                entry = {
                    "symbol": sym, "qty": abs(qty), "strike": strike,
                    "avg_entry": float(p.get("avg_entry_price", 0)),
                    "market_value": float(p.get("market_value", 0)),
                }

                if qty < 0:
                    short_legs.setdefault(key, []).append(entry)
                elif qty > 0:
                    long_legs.setdefault(key, []).append(entry)

            # Match short+long into spreads
            adopted = 0
            for key, shorts in short_legs.items():
                underlying, expiry_str, opt_type = key
                longs = long_legs.get(key, [])
                if not longs:
                    continue

                for short in shorts:
                    # Skip if already tracked
                    if short["symbol"] in tracked_symbols:
                        continue

                    # Find matching long (same qty, different strike)
                    best_long = None
                    for lng in longs:
                        if lng["qty"] == short["qty"] and lng["symbol"] not in tracked_symbols:
                            # Any paired short+long with different strikes = spread
                            if lng["strike"] != short["strike"]:
                                best_long = lng
                                break

                    if not best_long:
                        continue

                    # Skip if long leg is managed by CallBuyer
                    if best_long["symbol"] in callbuyer_contracts:
                        continue

                    # Parse expiration date
                    try:
                        yy = int(expiry_str[:2])
                        mm = int(expiry_str[2:4])
                        dd = int(expiry_str[4:6])
                        exp_date = date(2000 + yy, mm, dd)
                        exp_iso = exp_date.isoformat()
                        dte = (exp_date - date.today()).days
                    except (ValueError, IndexError):
                        continue

                    spread_width = abs(short["strike"] - best_long["strike"])
                    spread_type = "put" if opt_type == "P" else "call"

                    # Estimate credit from Alpaca avg_entry_prices
                    credit = short["avg_entry"] - best_long["avg_entry"]
                    if credit <= 0:
                        credit = spread_width * 0.15  # fallback estimate

                    pos_id = (
                        f"{underlying}_{exp_iso}_{opt_type}"
                        f"{short['strike']:.0f}"
                    )
                    if pos_id in self.positions:
                        continue  # already exists

                    max_loss_per = (spread_width - credit) * 100
                    qty = short["qty"]

                    self.positions[pos_id] = {
                        "underlying": underlying,
                        "spread_type": spread_type,
                        "short_symbol": short["symbol"],
                        "long_symbol": best_long["symbol"],
                        "short_strike": short["strike"],
                        "long_strike": best_long["strike"],
                        "spread_width": spread_width,
                        "expiration": exp_iso,
                        "dte_at_open": dte,
                        "qty": qty,
                        "credit_per_share": round(credit, 3),
                        "total_credit": round(credit * qty * 100, 2),
                        "max_loss_per_contract": round(max_loss_per, 2),
                        "max_loss_total": round(max_loss_per * qty, 2),
                        "max_profit_total": round(credit * qty * 100, 2),
                        "order_id": "ADOPTED",
                        "open_date": date.today().isoformat(),
                        "open_time": "ADOPTED",
                        "current_debit": 0,
                        "current_pnl_total": 0,
                        "current_pnl_pct": 0,
                        "roc_annual": 0,
                        "short_delta": None,
                        "entry_iv": None,
                        "iv_premium": None,
                        "features": None,
                        "ml_confidence": None,
                        "ml_rule_score": None,
                        "current_pnl_per_share": 0,
                    }

                    # Remove long from available pool
                    longs.remove(best_long)

                    log.info(
                        f"ADOPTED spread: {pos_id} | {qty}x "
                        f"{spread_type} spread | "
                        f"short={short['symbol']} long={best_long['symbol']} | "
                        f"credit=${credit:.2f} width=${spread_width:.0f} "
                        f"dte={dte}"
                    )
                    adopted += 1

            if adopted:
                self._save_positions()
                log.info(
                    f"Spread adoption complete: {adopted} orphaned "
                    f"spreads now managed by exit rules"
                )

        except Exception as e:
            log.error(f"Spread adoption failed: {e}", exc_info=True)

    @staticmethod
    def _parse_option_type(occ_symbol: str) -> Optional[str]:
        """Extract 'C' or 'P' from OCC option symbol."""
        for i in range(len(occ_symbol) - 9, 0, -1):
            if occ_symbol[i + 6] in ("C", "P"):
                return occ_symbol[i + 6]
        return None

    @staticmethod
    def _parse_underlying(occ_symbol: str) -> str:
        """Extract underlying ticker from OCC symbol."""
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
    def _parse_strike_from_occ(occ_symbol: str) -> float:
        """Extract strike price from OCC symbol (last 8 digits / 1000)."""
        try:
            return int(occ_symbol[-8:]) / 1000.0
        except (ValueError, IndexError):
            return 0.0


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

    def _log_trade(self, trade_data: Dict):
        """Append trade to CSV log."""
        try:
            os.makedirs(os.path.dirname(self.config.TRADE_LOG), exist_ok=True)
            file_exists = os.path.exists(self.config.TRADE_LOG)
            with open(self.config.TRADE_LOG, "a") as f:
                if not file_exists:
                    f.write("timestamp,underlying,spread_type,short_strike,long_strike,expiration,"
                            "qty,credit,close_debit,pnl,pnl_pct,hold_days,exit_reason,open_date\n")
                f.write(
                    f"{trade_data['timestamp']},{trade_data['underlying']},"
                    f"{trade_data.get('spread_type', 'put')},"
                    f"{trade_data['short_strike']},{trade_data['long_strike']},"
                    f"{trade_data['expiration']},{trade_data['qty']},"
                    f"{trade_data['credit']:.2f},{trade_data.get('close_debit', 0):.2f},"
                    f"{trade_data['pnl']:.2f},{trade_data['pnl_pct']:.1f}%,"
                    f"{trade_data.get('hold_days', 0)},{trade_data['exit_reason']},"
                    f"{trade_data.get('open_date', '')}\n"
                )
        except Exception as e:
            log.error(f"Failed to log trade: {e}")

    # ── Main Loop ────────────────────────────────────────
    def run(self):
        """Main event loop."""
        log.info("=" * 60)
        log.info("IronCondor Engine v2.0 — Credit Put + Call Spreads Starting")
        log.info("=" * 60)
        log.info(self.config.summary())

        if not self.api.connect():
            log.error("Failed to connect to Alpaca. Exiting.")
            return

        # Update allocation from actual equity
        acct = self.api.get_account()
        self.risk.update_allocation(acct["equity"])
        allocation = acct["equity"] * self.config.ALLOCATION_PCT
        log.info(f"Account: ${acct['equity']:,.2f} equity | "
                 f"IronCondor allocation: ${allocation:,.2f}")

        # Adopt orphaned spreads from Alpaca that aren't tracked
        self._adopt_orphaned_spreads()

        self.running = True
        while self.running:
            try:
                now = _time.time()

                # Check market hours
                if not self.api.is_market_open():
                    # Pre-market warmup: run once, ~30 min before open
                    if self._should_warmup():
                        self._pre_market_warmup()
                    elif self._cycle == 0:
                        next_open = self.api.next_open()
                        log.info(f"Market closed. Next open: {next_open}")
                    _time.sleep(60)
                    continue

                self._cycle += 1

                # Universe scan — expand watchlist (every 60 min)
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

                # Position management — every 5 minutes
                if now - self._last_check >= self.config.CHECK_INTERVAL_SEC:
                    self._manage_positions()
                    self._last_check = now

                # Re-adopt orphaned spreads every 30 minutes
                if now - self._last_adoption >= 1800:
                    self._adopt_orphaned_spreads()
                    self._last_adoption = now

                # New opportunity scan — every 15 minutes
                if now - self._last_scan >= self.config.SCAN_INTERVAL_SEC:
                    self._scan_opportunities()
                    self._last_scan = now

                # Status update every ~10 cycles
                if self._cycle % 10 == 0:
                    self._print_status()

                _time.sleep(30)

            except KeyboardInterrupt:
                log.info("Shutdown requested")
                self.running = False
            except Exception as e:
                log.error(f"Engine error: {e}", exc_info=True)
                _time.sleep(30)

        log.info("IronCondor engine stopped")

    # ── Position Management (Exits) ──────────────────────
    def _manage_positions(self):
        """Check all open positions for exit conditions."""
        if not self.positions:
            return

        to_close = []
        for pos_id, pos in self.positions.items():
            try:
                exit_reason = self._check_exit(pos)
                if exit_reason:
                    to_close.append((pos_id, exit_reason))
            except Exception as e:
                log.error(f"Exit check failed for {pos_id}: {e}")

        for pos_id, reason in to_close:
            self._close_position(pos_id, reason)

    def _check_exit(self, pos: Dict[str, Any]) -> Optional[str]:
        """Check if a position should be closed. Returns exit reason or None."""
        # Manual force-close flag (set in positions.json)
        if pos.get("force_close"):
            log.info(f"FORCE_CLOSE triggered for {pos.get('underlying', '?')}")
            return "FORCE_CLOSE (manual)"

        underlying = pos["underlying"]
        short_sym = pos["short_symbol"]
        long_sym = pos["long_symbol"]
        credit = pos["credit_per_share"]
        spread_width = pos["spread_width"]

        # Get current quotes
        short_quote = self.api.get_option_quote(short_sym)
        long_quote = self.api.get_option_quote(long_sym)

        if not short_quote or not long_quote:
            log.debug(f"Could not get quotes for {underlying} spread")
            return None

        # Current spread value = short_mid - long_mid (what it costs to close)
        short_mid = short_quote["mid"]
        long_mid = long_quote["mid"]
        current_debit = short_mid - long_mid  # cost to buy back spread

        # Update position tracking
        pos["current_debit"] = current_debit
        pos["current_pnl_per_share"] = credit - current_debit
        pos["current_pnl_total"] = pos["current_pnl_per_share"] * pos["qty"] * 100
        pos["current_pnl_pct"] = (
            (credit - current_debit) / credit * 100 if credit > 0 else 0
        )

        self._save_positions()

        # ── Exit Rule 1: Take profit at 50% ─────────────
        profit_pct = pos["current_pnl_pct"]
        if profit_pct >= self.config.TAKE_PROFIT_PCT * 100:
            return f"TAKE_PROFIT ({profit_pct:.0f}%)"

        # ── Exit Rule 2: Stop loss (2x credit) ──────────
        if current_debit >= credit * self.config.STOP_LOSS_MULT:
            return f"STOP_LOSS (debit ${current_debit:.2f} > {self.config.STOP_LOSS_MULT}x credit ${credit:.2f})"

        # ── Exit Rule 3: DTE exit (14 days) ─────────────
        expiration = datetime.strptime(pos["expiration"], "%Y-%m-%d").date()
        days_to_exp = (expiration - date.today()).days
        if days_to_exp <= self.config.MIN_DTE_EXIT:
            return f"DTE_EXIT ({days_to_exp}d remaining)"

        # ── Exit Rule 4: Emergency — price near short strike ──
        # Regime-aware: widen buffer in bearish regime (stock more likely to keep falling)
        current_price = self.api.get_latest_price(underlying)
        if current_price:
            short_strike = pos["short_strike"]
            emergency_pct = self.config.EMERGENCY_BUFFER_PCT
            spread_type = pos.get("spread_type", "put")

            # Tighten (widen buffer) when regime is adverse for this spread side
            if self._current_regime:
                regime_name = self._current_regime.get("regime", "")
                if spread_type == "put" and regime_name in ("TRENDING_DOWN", "HIGH_VOLATILITY"):
                    emergency_pct *= self.config.REGIME_BEAR_BUFFER_MULT
                elif spread_type == "call" and regime_name == "TRENDING_UP":
                    emergency_pct *= self.config.REGIME_BEAR_BUFFER_MULT

            buffer = short_strike * emergency_pct

            if spread_type == "call":
                if current_price >= short_strike - buffer:
                    return (f"EMERGENCY_CALL (${current_price:.2f} within "
                            f"{emergency_pct:.0%} of ${short_strike:.2f} short call)")
            else:
                if current_price <= short_strike + buffer:
                    return (f"EMERGENCY_PUT (${current_price:.2f} within "
                            f"{emergency_pct:.0%} of ${short_strike:.2f} short put)")

        # ── Exit Rule 5: Delta breach — short leg moving ITM ──
        short_snap = self.api.get_option_snapshot(short_sym)
        if short_snap:
            live_delta = abs(short_snap.get("delta", 0))
            if live_delta >= self.config.DELTA_EXIT_THRESHOLD:
                return (f"DELTA_BREACH (|delta| {live_delta:.2f} >= "
                        f"{self.config.DELTA_EXIT_THRESHOLD} — approaching ITM)")

            # ── Exit Rule 6: IV spike — risk expanded beyond entry ──
            entry_iv = pos.get("entry_iv")
            if entry_iv and entry_iv > 0:
                live_iv = short_snap.get("iv", 0)
                if live_iv > 0 and live_iv > entry_iv * self.config.IV_SPIKE_EXIT_MULT:
                    return (f"IV_SPIKE (IV {live_iv:.2f} > "
                            f"{self.config.IV_SPIKE_EXIT_MULT}x entry {entry_iv:.2f} — risk expanded)")

        return None

    def _close_position(self, pos_id: str, reason: str):
        """Close a credit spread position."""
        pos = self.positions.get(pos_id)
        if not pos:
            return

        underlying = pos["underlying"]
        short_sym = pos["short_symbol"]
        long_sym = pos["long_symbol"]
        qty = pos["qty"]
        credit = pos["credit_per_share"]
        current_debit = pos.get("current_debit", credit)
        spread_type = pos.get("spread_type", "put")
        spread_label = "bear call spread" if spread_type == "call" else "bull put spread"

        log.info(f"CLOSING {underlying} {spread_label}: {reason}")

        # Try mleg close first
        order_id = self.api.close_credit_spread(
            short_sym, long_sym, qty,
            debit_limit=round(current_debit * 1.10, 2)  # 10% slippage allowance on close
        )

        close_filled = False
        if order_id:
            # Verify close fill — wait up to 30s
            for _wait in range(6):
                _time.sleep(5)
                order_status = self.api.get_order(order_id)
                if order_status and order_status.get("status") == "filled":
                    close_filled = True
                    actual_debit = abs(order_status.get("filled_avg_price", 0)) or current_debit
                    current_debit = actual_debit  # use actual fill price for PnL
                    log.info(f"{underlying}: close filled @ ${actual_debit:.2f}")
                    break
                elif order_status and order_status["status"] in ("canceled", "expired", "rejected"):
                    log.warning(f"{underlying}: close order {order_status['status']}")
                    break

        if not close_filled:
            # CRITICAL: Cancel the pending MLEG order before trying individual legs
            # to prevent double-execution (both MLEG and individual legs filling)
            if order_id:
                self.api.cancel_order(order_id)
                _time.sleep(1)  # brief pause for cancellation to propagate
            log.warning(f"MLEG close failed or unfilled, trying individual legs")
            leg_success = self.api.close_individual_legs(short_sym, long_sym, qty)
            close_filled = leg_success

        if not close_filled:
            log.error(f"{underlying}: CLOSE FAILED — keeping position tracked for retry")
            return

        # Calculate PnL
        pnl_per_share = credit - current_debit
        total_pnl = pnl_per_share * qty * 100
        pnl_pct = (pnl_per_share / credit * 100) if credit > 0 else 0

        # Record trade
        hold_days = (date.today() - datetime.strptime(
            pos["open_date"], "%Y-%m-%d").date()).days

        self.risk.record_trade(total_pnl, underlying,
                               spread_type=pos.get("spread_type", "put"))

        # ── ML outcome recording ─────────────────────────
        won = total_pnl >= 0
        if ML_AVAILABLE and self.ml and self.meta:
            # Record in meta-learner for threshold adaptation
            self.meta.record_result(won=won, pnl_pct=pnl_pct)

            # Record features + outcome for ML model training
            if pos.get("features"):
                feat_vec = np.array(pos["features"])
                self.ml.record_outcome(
                    symbol=underlying,
                    features=feat_vec,
                    won=won,
                    pnl_pct=pnl_pct,
                )

        strike_suffix = "C" if spread_type == "call" else "P"
        log_level = "INFO" if total_pnl >= 0 else "WARNING"
        getattr(log, log_level.lower())(
            f"CLOSED: {underlying} ${pos['short_strike']:.0f}/${pos['long_strike']:.0f}{strike_suffix} "
            f"| PnL ${total_pnl:+,.2f} ({pnl_pct:+.0f}%) "
            f"| {hold_days}d hold | {reason}"
        )

        self._log_trade({
            "timestamp": datetime.now().isoformat(),
            "underlying": underlying,
            "spread_type": spread_type,
            "short_strike": pos["short_strike"],
            "long_strike": pos["long_strike"],
            "expiration": pos["expiration"],
            "qty": qty,
            "credit": credit,
            "close_debit": current_debit,
            "pnl": total_pnl,
            "pnl_pct": pnl_pct,
            "hold_days": hold_days,
            "exit_reason": reason,
            "open_date": pos.get("open_date", ""),
        })

        # Remove position
        del self.positions[pos_id]
        self._save_positions()

    # ── Opportunity Scanner ──────────────────────────────
    def _scan_opportunities(self):
        """Scan watchlist for credit put AND call spread opportunities."""
        if not self._is_trading_window():
            log.debug("Outside trading window — skipping scan")
            return

        log.info(f"Scan starting — {len(self._dynamic_watchlist)} symbols, "
                 f"{len(self.positions)} open positions")

        # Update allocation from current equity
        try:
            acct = self.api.get_account()
            self.risk.update_allocation(acct["equity"])
            log.info(f"Allocation: ${acct['equity'] * self.config.ALLOCATION_PCT:,.2f}")
        except Exception as e:
            log.warning(f"Could not refresh allocation: {e}")

        # Refresh SPY bars for feature computation
        if ML_AVAILABLE and self.features:
            try:
                self._spy_bars = self.api.get_bars("SPY", days=30)
            except Exception:
                self._spy_bars = None

        # ── VIX / SPY Crash Filter ───────────────────────
        # Block new short-premium entries when market is crashing.
        # This prevents opening spreads right before a gap/vol expansion
        # wipes out weeks of credit income.
        try:
            spy_bars = self._spy_bars or self.api.get_bars("SPY", days=5)
            if spy_bars and len(spy_bars) >= 2:
                spy_today = float(spy_bars[-1].close)
                spy_prev = float(spy_bars[-2].close)
                spy_intraday_pct = (spy_today - spy_prev) / spy_prev if spy_prev > 0 else 0.0

                if spy_intraday_pct < -0.015:
                    log.warning(
                        f"SPY CRASH FILTER: SPY down {spy_intraday_pct:.2%} today — "
                        "blocking ALL new spread openings this cycle"
                    )
                    return

            vixy_bars = self.api.get_bars("VIXY", days=5)
            if vixy_bars and len(vixy_bars) >= 2:
                vixy_today = float(vixy_bars[-1].close)
                vixy_prev = float(vixy_bars[-2].close)
                vixy_spike_pct = (vixy_today - vixy_prev) / vixy_prev if vixy_prev > 0 else 0.0

                if vixy_spike_pct > 0.20:
                    log.warning(
                        f"VIX CRASH FILTER: VIXY up {vixy_spike_pct:.2%} today — "
                        "blocking ALL new spread openings this cycle"
                    )
                    return
        except Exception as e:
            log.debug(f"Crash filter check failed (proceeding): {e}")


        # ── Portfolio-Level Aggregate Exposure Cap ─────────
        can_trade_portfolio, portfolio_exposure = self._check_portfolio_exposure()
        if not can_trade_portfolio:
            return

        # ── Regime Detection ─────────────────────────────
        if self._regime_detector:
            try:
                regime_bars = self._spy_bars or self.api.get_bars("SPY", days=60)
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

                    # HIGH_VOLATILITY → halt new spread openings
                    if regime_name == "HIGH_VOLATILITY" and r["confidence"] >= 0.30:
                        log.warning(
                            "HIGH_VOLATILITY regime detected — "
                            "skipping new spread openings this cycle"
                        )
                        return

                    # Regime flip → halt if severe
                    if flip_state.get("should_block_entries", False):
                        log.warning(
                            f"Regime flip blocking new entries — "
                            f"{flip_state.get('last_flip_from')} → {flip_state.get('last_flip_to')} "
                            f"(severity={flip_state['flip_severity']:.2f}, "
                            f"cooldown={flip_state['cooldown_remaining_min']:.0f}m)"
                        )
                        return
                else:
                    self._current_regime = None
            except Exception as e:
                log.warning(f"Regime detection failed: {e}")
                self._current_regime = None

        # Check ML retraining
        if ML_AVAILABLE and self.ml and self.ml.should_retrain():
            log.info("Triggering ML model retrain...")
            success = self.ml.train()
            log.info(f"ML retrain {'succeeded' if success else 'below quality gate'}")

        # ── Sentiment Check ──────────────────────────────
        market_sentiment = 0.0
        if self.sentiment:
            try:
                market_sentiment = self.sentiment.get_sentiment()
                # Extreme fear — pause new openings (let existing positions work)
                if market_sentiment < -0.60:
                    log.warning(
                        f"Sentiment extremely bearish ({market_sentiment:+.3f}) — "
                        "pausing new spread openings this cycle"
                    )
                    return
            except Exception as e:
                log.debug(f"Sentiment fetch failed: {e}")

        capital_in_use = sum(
            p.get("max_loss_total", 0) for p in self.positions.values()
        )

        # ── Regime-Based Position Caps ───────────────────
        # In non-RANGING regimes, drastically reduce max concurrent spreads
        # to limit tail-risk exposure from correlated losses.
        effective_max_puts = self.config.MAX_POSITIONS
        effective_max_calls = self.config.MAX_CALL_POSITIONS
        if self._current_regime:
            regime_name = self._current_regime["regime"]
            if regime_name in ("TRENDING_DOWN", "HIGH_VOLATILITY"):
                effective_max_puts = min(5, self.config.MAX_POSITIONS)
                effective_max_calls = min(3, self.config.MAX_CALL_POSITIONS)
                log.info(
                    f"Regime cap active ({regime_name}): "
                    f"max puts {effective_max_puts}, max calls {effective_max_calls}"
                )
            elif regime_name == "TRENDING_UP":
                # Trending up is safer for puts but risky for calls
                effective_max_puts = min(15, self.config.MAX_POSITIONS)
                effective_max_calls = min(5, self.config.MAX_CALL_POSITIONS)

        # ── Balanced Put/Call Scanning ───────────────────
        # Per tastylive iron condor mechanics: keep put and call sides
        # balanced and directionally neutral.  Prioritise whichever
        # side is underweight relative to its cap.
        put_count = sum(1 for p in self.positions.values()
                        if p.get("spread_type") != "call")
        call_count = sum(1 for p in self.positions.values()
                        if p.get("spread_type") == "call")
        put_pct = put_count / max(effective_max_puts, 1)
        call_pct = call_count / max(effective_max_calls, 1)

        # Determine sentiment-driven target allocation:
        # Bearish  → favour calls (they profit when stocks drop)
        # Bullish  → favour puts  (they profit when stocks stay/rise)
        # Neutral  → balanced
        if market_sentiment < -0.15:
            put_budget, call_budget = 1, 3   # bearish: open more calls
        elif market_sentiment > 0.15:
            put_budget, call_budget = 3, 1   # bullish: open more puts
        else:
            put_budget, call_budget = 2, 2   # neutral: balanced

        # Decide scan order: underweight side goes FIRST
        calls_first = call_pct < put_pct

        log.info(f"Balance: puts={put_count}/{effective_max_puts} "
                 f"({put_pct:.0%}), calls={call_count}/{effective_max_calls} "
                 f"({call_pct:.0%}) | sent={market_sentiment:+.3f} "
                 f"| budget P{put_budget}/C{call_budget} "
                 f"| {'calls' if calls_first else 'puts'} first")

        put_opened = 0
        call_opened = 0
        opened = 0
        scanned = 0
        skipped = 0
        self._buying_power_exhausted = False
        for symbol in self._dynamic_watchlist:
            if not self.running:
                break
            if self._buying_power_exhausted:
                log.info("Buying power exhausted — stopping scan")
                break

            # ── Check put side ───────────
            if put_opened < put_budget and put_count + put_opened < effective_max_puts:
                can_open, reason = self.risk.can_open_position(
                    len(self.positions), symbol, self.positions,
                    spread_type="put"
                )
                if can_open:
                    try:
                        scanned += 1
                        if has_earnings_within(symbol, self.config.MAX_DTE):
                            log.info(f"{symbol}: earnings within {self.config.MAX_DTE}d — skipping")
                            skipped += 1
                        else:
                            spread = self._find_spread(symbol)
                            if spread:
                                spread["spread_type"] = "put"
                                if self._has_strike_overlap(spread):
                                    skipped += 1
                                elif self._execute_spread(spread, capital_in_use):
                                    put_opened += 1
                                    opened += 1
                                    qty = spread.get("qty", 1)
                                    capital_in_use += spread["max_loss_per_contract"] * qty
                    except Exception as e:
                        log.error(f"Scan error (put) for {symbol}: {e}")
                else:
                    log.debug(f"{symbol}: skip put — {reason}")

            # ── Check call side ──────────
            if (call_opened < call_budget and self.config.CALL_SPREADS_ENABLED
                    and call_count + call_opened < effective_max_calls):
                can_open_call, call_reason = self.risk.can_open_position(
                    len(self.positions), symbol, self.positions,
                    spread_type="call"
                )
                if can_open_call:
                    try:
                        if has_earnings_within(symbol, self.config.MAX_DTE):
                            log.info(f"{symbol}: earnings within {self.config.MAX_DTE}d — skipping call")
                            skipped += 1
                        else:
                            call_spread = self._find_call_spread(symbol)
                            if call_spread:
                                if self._has_strike_overlap(call_spread):
                                    skipped += 1
                                elif self._execute_spread(call_spread, capital_in_use):
                                    call_opened += 1
                                    opened += 1
                                    qty = call_spread.get("qty", 1)
                                    capital_in_use += call_spread["max_loss_per_contract"] * qty
                    except Exception as e:
                        log.error(f"Scan error (call) for {symbol}: {e}")
                else:
                    log.debug(f"{symbol}: skip call — {call_reason}")

            # Stop if both budgets exhausted
            if put_opened >= put_budget and call_opened >= call_budget:
                break

        call_total = sum(1 for p in self.positions.values() if p.get("spread_type") == "call")
        put_total = sum(1 for p in self.positions.values() if p.get("spread_type") != "call")
        log.info(f"Scan complete: scanned={scanned}, opened={opened} "
                 f"(P+{put_opened}/C+{call_opened}), skipped={skipped} "
                 f"| puts={put_total} calls={call_total}")

    def _find_spread(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Find the best bull put spread for a symbol.

        Returns spread info dict or None if nothing qualifies.
        """
        # Get current price
        price = self.api.get_latest_price(symbol)
        if not price:
            return None

        # Check IV premium (only sell when IV is elevated)
        hv20 = self.api.calculate_hv20(symbol)
        if hv20 is None or hv20 < 0.05:
            log.debug(f"{symbol}: skip — HV20 too low or unavailable")
            return None

        # Spread width
        width = self.config.get_spread_width(price)

        # DTE range
        today = date.today()
        exp_after = (today + timedelta(days=self.config.MIN_DTE)).isoformat()
        exp_before = (today + timedelta(days=self.config.MAX_DTE)).isoformat()

        # Get put options in strike range: 15-30% below current price
        strike_max = price * (1 - self.config.SHORT_DELTA_MIN * 0.5)  # rough filter
        strike_min = price * (1 - self.config.SHORT_DELTA_MAX * 2.0)  # wide net

        chain = self.api.get_options_chain(
            underlying=symbol,
            expiration_after=exp_after,
            expiration_before=exp_before,
            option_type="put",
            strike_price_gte=strike_min,
            strike_price_lte=strike_max,
        )

        if not chain:
            log.debug(f"{symbol}: no put contracts found")
            return None

        # Group by expiration
        by_exp: Dict[str, List[Dict]] = {}
        for c in chain:
            exp = c["expiration"]
            by_exp.setdefault(exp, []).append(c)

        # For each expiration, find the best spread
        best_spread = None
        best_score = -1

        for exp_date, contracts in by_exp.items():
            # Sort by strike descending (closest to ATM first)
            contracts.sort(key=lambda c: c["strike"], reverse=True)

            dte = (datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days
            if dte < self.config.MIN_DTE or dte > self.config.MAX_DTE:
                continue

            # Pre-filter: build candidate pairs without API calls
            candidates = []
            for short_c in contracts:
                short_strike = short_c["strike"]
                if short_strike >= price:
                    continue

                long_strike_target = short_strike - width
                long_c = None
                for c in contracts:
                    if abs(c["strike"] - long_strike_target) < 1.0:
                        long_c = c
                        break
                if not long_c:
                    continue

                actual_width = short_strike - long_c["strike"]
                if actual_width <= 0:
                    continue

                # OI check — realistic minimums (50/20 balances data vs quality)
                if short_c.get("open_interest", 0) < 50:
                    continue
                if long_c.get("open_interest", 0) < 20:
                    continue

                # Prefer strikes closer to ATM (higher credit potential)
                candidates.append((short_c, long_c, actual_width))

            # Only check top 3 candidates per expiration (limit API calls)
            for short_c, long_c, actual_width in candidates[:3]:
                short_strike = short_c["strike"]

                # Get quotes + greeks for delta check
                short_snap = self.api.get_option_snapshot(short_c["symbol"])
                if not short_snap:
                    short_quote = self.api.get_option_quote(short_c["symbol"])
                    long_quote = self.api.get_option_quote(long_c["symbol"])
                    if not short_quote or not long_quote:
                        continue
                    # Bid-ask width check — reject illiquid options
                    _skip = False
                    for _q in (short_quote, long_quote):
                        if _q["mid"] > 0 and _q["ask"] > 0 and _q["bid"] > 0:
                            _ba = (_q["ask"] - _q["bid"]) / _q["mid"]
                            if _ba > 0.20:
                                log.debug(f"{symbol}: bid-ask too wide ({_ba:.0%})")
                                _skip = True
                                break
                    if _skip:
                        continue
                    short_mid = short_quote["mid"]
                    long_mid = long_quote["mid"]
                    short_delta = None
                    short_iv = None
                else:
                    short_mid = short_snap["mid"]
                    short_delta = abs(short_snap.get("delta", 0))
                    short_iv = short_snap.get("iv", 0)
                    long_quote = self.api.get_option_quote(long_c["symbol"])
                    if not long_quote:
                        continue
                    long_mid = long_quote["mid"]
                    # Bid-ask check on both legs
                    if short_mid > 0:
                        _s_ba = (short_snap.get("ask", 0) - short_snap.get("bid", 0)) / short_mid
                        if _s_ba > 0.20:
                            log.debug(f"{symbol}: short leg bid-ask too wide ({_s_ba:.0%})")
                            continue
                    if long_quote["mid"] > 0:
                        _l_ba = (long_quote["ask"] - long_quote["bid"]) / long_quote["mid"]
                        if _l_ba > 0.20:
                            log.debug(f"{symbol}: long leg bid-ask too wide ({_l_ba:.0%})")
                            continue

                credit = short_mid - long_mid
                if credit <= 0:
                    continue

                credit_pct = credit / actual_width

                # ── Regime-based credit threshold adjustment ──
                effective_min_credit = self.config.MIN_CREDIT_PCT
                regime_otm_mult = 1.0
                if self._current_regime:
                    r_adj = self._current_regime.get("suggested_adjustments", {})
                    credit_mult = r_adj.get("credit_threshold", 1.0)
                    effective_min_credit = self.config.MIN_CREDIT_PCT * credit_mult
                    regime_otm_mult = r_adj.get("otm_buffer", 1.0)

                if credit_pct < effective_min_credit:
                    log.debug(f"{symbol}: ${short_strike:.0f} exp {exp_date} — "
                              f"credit ${credit:.2f} ({credit_pct:.0%}) < {effective_min_credit:.0%}")
                    continue

                # ── Leveraged ETF: require wider OTM distance ──
                if symbol in self.config.LEVERAGED_ETFS:
                    regime_otm_mult = max(regime_otm_mult, self.config.LEVERAGED_OTM_MULT)

                # ── Regime-based OTM buffer check ──
                if regime_otm_mult > 1.0:
                    otm_pct_check = (price - short_strike) / price if price > 0 else 0
                    required_otm = 0.05 * regime_otm_mult  # base 5% OTM * multiplier
                    if otm_pct_check < required_otm:
                        log.debug(
                            f"{symbol}: ${short_strike:.0f} — "
                            f"OTM {otm_pct_check:.1%} < regime-required {required_otm:.1%}"
                        )
                        continue

                # Delta filter
                if short_delta is not None:
                    if short_delta < self.config.SHORT_DELTA_MIN:
                        log.debug(f"{symbol}: ${short_strike:.0f} — delta {short_delta:.2f} too low")
                        continue
                    if short_delta > self.config.SHORT_DELTA_MAX:
                        log.debug(f"{symbol}: ${short_strike:.0f} — delta {short_delta:.2f} too high")
                        continue

                # IV premium check
                if short_iv and hv20:
                    iv_premium = short_iv / hv20
                    if iv_premium < self.config.MIN_IV_PREMIUM:
                        log.debug(f"{symbol}: ${short_strike:.0f} — IV/HV {iv_premium:.2f} < {self.config.MIN_IV_PREMIUM}")
                        continue
                else:
                    iv_premium = None

                # Annualized ROC
                max_loss_per_share = actual_width - credit
                if max_loss_per_share <= 0:
                    continue
                roc = credit / max_loss_per_share
                roc_annual = roc * (365 / dte) if dte > 0 else 0
                if roc_annual < self.config.MIN_ROC_ANNUAL:
                    log.debug(f"{symbol}: ${short_strike:.0f} — ROC {roc_annual:.0%}/yr < {self.config.MIN_ROC_ANNUAL:.0%}")
                    continue

                otm_pct = (price - short_strike) / price
                dte_score = 1.0 - abs(dte - self.config.TARGET_DTE) / self.config.MAX_DTE
                credit_score = credit_pct
                oi_score = min(short_c["open_interest"] / 500, 1.0)
                score = (credit_score * 0.4) + (dte_score * 0.3) + (oi_score * 0.3)

                if score > best_score:
                    best_score = score
                    best_spread = {
                        "underlying": symbol,
                        "price": price,
                        "short_symbol": short_c["symbol"],
                        "long_symbol": long_c["symbol"],
                        "short_strike": short_strike,
                        "long_strike": long_c["strike"],
                        "spread_width": actual_width,
                        "expiration": exp_date,
                        "dte": dte,
                        "credit": credit,
                        "credit_pct": credit_pct,
                        "max_loss_per_share": max_loss_per_share,
                        "max_loss_per_contract": max_loss_per_share * 100,
                        "max_profit_per_contract": credit * 100,
                        "roc": roc,
                        "roc_annual": roc_annual,
                        "short_delta": short_delta,
                        "entry_iv": short_iv,
                        "iv_premium": iv_premium,
                        "otm_pct": otm_pct,
                        "short_oi": short_c["open_interest"],
                        "score": score,
                    }

        if best_spread:
            # ── Sentiment filter (if available) ──────────
            # Per tastylive: iron condors need BOTH sides to stay neutral.
            # Never fully block puts — just demand higher credit in
            # bearish conditions.  This keeps the portfolio balanced.
            if self.sentiment:
                try:
                    sym_sent = self.sentiment.get_combined_score(symbol)
                    best_spread["sentiment"] = sym_sent
                    # Bull put spreads LOSE when stock drops.
                    # Very bearish: demand extra premium, but don't block entirely
                    if sym_sent < -0.50:
                        best_spread["credit_adj"] = "strong_bearish_premium"
                        log.info(f"{symbol}: sentiment very bearish ({sym_sent:+.3f}), "
                                 "demanding higher credit for put spread")
                    elif sym_sent < -0.15:
                        best_spread["credit_adj"] = "bearish_premium"
                        log.info(f"{symbol}: sentiment cautious ({sym_sent:+.3f}), "
                                 "requiring extra premium margin")
                except Exception as e:
                    log.debug(f"Sentiment check failed for {symbol}: {e}")

            # ── ML scoring (if available) ────────────────
            if ML_AVAILABLE and self.features and self.meta:
                try:
                    bars = self.api.get_bars(symbol, days=60)
                    if bars and len(bars) >= 50:
                        spread_info = {
                            "iv_premium": best_spread.get("iv_premium"),
                            "credit_pct": best_spread["credit_pct"],
                            "dte": best_spread["dte"],
                            "target_dte": self.config.TARGET_DTE,
                            "max_dte": self.config.MAX_DTE,
                            "short_oi": best_spread["short_oi"],
                            "otm_pct": best_spread["otm_pct"],
                            "short_strike": best_spread["short_strike"],
                            "price": best_spread["price"],
                        }
                        feat_vec = self.features.build_features(bars, spread_info, self._spy_bars)
                        if feat_vec is not None:
                            rule_score = self.features.compute_rule_score(feat_vec)
                            ml_proba, ml_active = self.ml.predict(feat_vec) if self.ml else (0.5, False)
                            confidence, should_trade, ml_reason = self.meta.evaluate(
                                rule_score, ml_proba, ml_active
                            )
                            best_spread["features"] = feat_vec.tolist()
                            best_spread["ml_confidence"] = confidence
                            best_spread["ml_rule_score"] = rule_score
                            best_spread["ml_proba"] = ml_proba
                            best_spread["ml_reason"] = ml_reason

                            if not should_trade:
                                log.info(f"{symbol}: ML BLOCKED — {ml_reason}")
                                return None

                            ml_str = f"ml={ml_proba:.2f}" if ml_active else "ml=warmup"
                            log.info(f"{symbol}: ML PASS conf={confidence:.3f} rules={rule_score:.1f} {ml_str}")

                            # Log features for future training
                            self.features.log_features(symbol, feat_vec)
                except Exception as e:
                    log.debug(f"ML scoring error for {symbol}: {e}")

            iv_str = f" | IV/HV {best_spread['iv_premium']:.2f}" if best_spread.get('iv_premium') else ""
            sent_str = f" | sent={best_spread.get('sentiment', 0):+.2f}" if best_spread.get('sentiment') else ""
            log.info(
                f"FOUND: {symbol} put ${best_spread['short_strike']:.0f}/"
                f"${best_spread['long_strike']:.0f} exp {best_spread['expiration']} "
                f"| credit ${best_spread['credit']:.2f} ({best_spread['credit_pct']:.0%}) "
                f"| OTM {best_spread['otm_pct']:.1%} "
                f"| ROC {best_spread['roc_annual']:.0%}/yr{iv_str}{sent_str}"
            )
        else:
            log.info(f"{symbol}: no qualifying put spread found")

        return best_spread

    def _find_call_spread(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Find the best bear call spread (credit) for a symbol.

        Bear call spread:
        - SELL OTM call (lower strike, closer to ATM) → collect premium
        - BUY further OTM call (higher strike) → protection/cap risk
        - Win when stock stays BELOW the short call strike at expiry.

        Returns spread info dict or None if nothing qualifies.
        """
        if not self.config.CALL_SPREADS_ENABLED:
            return None

        # Get current price
        price = self.api.get_latest_price(symbol)
        if not price:
            return None

        # IV check
        hv20 = self.api.calculate_hv20(symbol)
        if hv20 is None or hv20 < 0.05:
            log.debug(f"{symbol}: skip call — HV20 too low or unavailable")
            return None

        # Spread width (same as put side)
        width = self.config.get_spread_width(price)

        # DTE range
        today = date.today()
        exp_after = (today + timedelta(days=self.config.MIN_DTE)).isoformat()
        exp_before = (today + timedelta(days=self.config.MAX_DTE)).isoformat()

        # Get call options above current price (OTM calls)
        strike_min = price * (1 + self.config.CALL_SHORT_DELTA_MIN * 0.5)
        strike_max = price * (1 + self.config.CALL_SHORT_DELTA_MAX * 2.0)

        chain = self.api.get_options_chain(
            underlying=symbol,
            expiration_after=exp_after,
            expiration_before=exp_before,
            option_type="call",
            strike_price_gte=strike_min,
            strike_price_lte=strike_max,
        )

        if not chain:
            log.debug(f"{symbol}: no call contracts found")
            return None

        # Group by expiration
        by_exp: Dict[str, List[Dict]] = {}
        for c in chain:
            exp = c["expiration"]
            by_exp.setdefault(exp, []).append(c)

        best_spread = None
        best_score = -1

        for exp_date, contracts in by_exp.items():
            # Sort by strike ascending (closest to ATM first)
            contracts.sort(key=lambda c: c["strike"])

            dte = (datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days
            if dte < self.config.MIN_DTE or dte > self.config.MAX_DTE:
                continue

            candidates = []
            for short_c in contracts:
                short_strike = short_c["strike"]
                if short_strike <= price:
                    continue  # must be OTM (above current price)

                long_strike_target = short_strike + width
                long_c = None
                for c in contracts:
                    if abs(c["strike"] - long_strike_target) < 1.0:
                        long_c = c
                        break
                if not long_c:
                    continue

                actual_width = long_c["strike"] - short_strike
                if actual_width <= 0:
                    continue

                # OI check — realistic minimums (50/20 balances data vs quality)
                if short_c.get("open_interest", 0) < 50:
                    continue
                if long_c.get("open_interest", 0) < 20:
                    continue

                candidates.append((short_c, long_c, actual_width))

            # Check top 3 candidates per expiration
            for short_c, long_c, actual_width in candidates[:3]:
                short_strike = short_c["strike"]

                short_snap = self.api.get_option_snapshot(short_c["symbol"])
                if not short_snap:
                    short_quote = self.api.get_option_quote(short_c["symbol"])
                    long_quote = self.api.get_option_quote(long_c["symbol"])
                    if not short_quote or not long_quote:
                        continue
                    # Bid-ask width check — reject illiquid options
                    _skip = False
                    for _q in (short_quote, long_quote):
                        if _q["mid"] > 0 and _q["ask"] > 0 and _q["bid"] > 0:
                            _ba = (_q["ask"] - _q["bid"]) / _q["mid"]
                            if _ba > 0.20:
                                log.debug(f"{symbol}: CALL bid-ask too wide ({_ba:.0%})")
                                _skip = True
                                break
                    if _skip:
                        continue
                    short_mid = short_quote["mid"]
                    long_mid = long_quote["mid"]
                    short_delta = None
                    short_iv = None
                else:
                    short_mid = short_snap["mid"]
                    short_delta = abs(short_snap.get("delta", 0))
                    short_iv = short_snap.get("iv", 0)
                    long_quote = self.api.get_option_quote(long_c["symbol"])
                    if not long_quote:
                        continue
                    long_mid = long_quote["mid"]
                    # Bid-ask check on both legs
                    if short_mid > 0:
                        _s_ba = (short_snap.get("ask", 0) - short_snap.get("bid", 0)) / short_mid
                        if _s_ba > 0.20:
                            log.debug(f"{symbol}: CALL short leg bid-ask too wide ({_s_ba:.0%})")
                            continue
                    if long_quote["mid"] > 0:
                        _l_ba = (long_quote["ask"] - long_quote["bid"]) / long_quote["mid"]
                        if _l_ba > 0.20:
                            log.debug(f"{symbol}: CALL long leg bid-ask too wide ({_l_ba:.0%})")
                            continue

                # Credit = short premium - long premium (short is more expensive, closer to ATM)
                credit = short_mid - long_mid
                if credit <= 0:
                    continue

                credit_pct = credit / actual_width

                # Regime-based credit threshold adjustment
                effective_min_credit = self.config.CALL_MIN_CREDIT_PCT
                if self._current_regime:
                    r_adj = self._current_regime.get("suggested_adjustments", {})
                    credit_mult = r_adj.get("credit_threshold", 1.0)
                    effective_min_credit = self.config.CALL_MIN_CREDIT_PCT * credit_mult

                if credit_pct < effective_min_credit:
                    log.debug(f"{symbol}: CALL ${short_strike:.0f} exp {exp_date} — "
                              f"credit ${credit:.2f} ({credit_pct:.0%}) < {effective_min_credit:.0%}")
                    continue

                # ── Leveraged ETF: require wider OTM distance (calls) ──
                if symbol in self.config.LEVERAGED_ETFS:
                    otm_pct_check = (short_strike - price) / price if price > 0 else 0
                    required_otm = 0.05 * self.config.LEVERAGED_OTM_MULT
                    if otm_pct_check < required_otm:
                        log.debug(
                            f"{symbol}: CALL ${short_strike:.0f} — "
                            f"OTM {otm_pct_check:.1%} < leveraged-required {required_otm:.1%}"
                        )
                        continue

                # Delta filter
                if short_delta is not None:
                    if short_delta < self.config.CALL_SHORT_DELTA_MIN:
                        log.debug(f"{symbol}: CALL ${short_strike:.0f} — delta {short_delta:.2f} too low")
                        continue
                    if short_delta > self.config.CALL_SHORT_DELTA_MAX:
                        log.debug(f"{symbol}: CALL ${short_strike:.0f} — delta {short_delta:.2f} too high")
                        continue

                # IV premium check
                if short_iv and hv20:
                    iv_premium = short_iv / hv20
                    if iv_premium < self.config.MIN_IV_PREMIUM:
                        log.debug(f"{symbol}: CALL ${short_strike:.0f} — IV/HV {iv_premium:.2f} < {self.config.MIN_IV_PREMIUM}")
                        continue
                else:
                    iv_premium = None

                # Annualized ROC
                max_loss_per_share = actual_width - credit
                if max_loss_per_share <= 0:
                    continue
                roc = credit / max_loss_per_share
                roc_annual = roc * (365 / dte) if dte > 0 else 0
                if roc_annual < self.config.CALL_MIN_ROC_ANNUAL:
                    log.debug(f"{symbol}: CALL ${short_strike:.0f} — ROC {roc_annual:.0%}/yr < {self.config.CALL_MIN_ROC_ANNUAL:.0%}")
                    continue

                otm_pct = (short_strike - price) / price  # positive = OTM above
                dte_score = 1.0 - abs(dte - self.config.TARGET_DTE) / self.config.MAX_DTE
                credit_score = credit_pct
                oi_score = min(short_c["open_interest"] / 500, 1.0)
                score = (credit_score * 0.4) + (dte_score * 0.3) + (oi_score * 0.3)

                if score > best_score:
                    best_score = score
                    best_spread = {
                        "underlying": symbol,
                        "price": price,
                        "spread_type": "call",
                        "short_symbol": short_c["symbol"],
                        "long_symbol": long_c["symbol"],
                        "short_strike": short_strike,
                        "long_strike": long_c["strike"],
                        "spread_width": actual_width,
                        "expiration": exp_date,
                        "dte": dte,
                        "credit": credit,
                        "credit_pct": credit_pct,
                        "max_loss_per_share": max_loss_per_share,
                        "max_loss_per_contract": max_loss_per_share * 100,
                        "max_profit_per_contract": credit * 100,
                        "roc": roc,
                        "roc_annual": roc_annual,
                        "short_delta": short_delta,
                        "entry_iv": short_iv,
                        "iv_premium": iv_premium,
                        "otm_pct": otm_pct,
                        "short_oi": short_c["open_interest"],
                        "score": score,
                    }

        if best_spread:
            # ── Sentiment filter (if available) ──────────
            # Per tastylive: iron condors need BOTH sides to stay neutral.
            # Never fully block calls — just demand higher credit in
            # bullish conditions.  This keeps the portfolio balanced.
            if self.sentiment:
                try:
                    sym_sent = self.sentiment.get_combined_score(symbol)
                    best_spread["sentiment"] = sym_sent
                    # Bear call spreads LOSE when stock rises.
                    # Very bullish: demand extra premium, but don't block entirely
                    if sym_sent > 0.50:
                        best_spread["credit_adj"] = "strong_bullish_premium"
                        log.info(f"{symbol}: sentiment very bullish ({sym_sent:+.3f}), "
                                 "demanding higher credit for call spread")
                    elif sym_sent > 0.15:
                        best_spread["credit_adj"] = "bullish_premium"
                        log.info(f"{symbol}: sentiment cautious ({sym_sent:+.3f}), "
                                 "requiring extra premium margin for call spread")
                except Exception as e:
                    log.debug(f"Sentiment check failed for {symbol}: {e}")

            iv_str = f" | IV/HV {best_spread['iv_premium']:.2f}" if best_spread.get('iv_premium') else ""
            sent_str = f" | sent={best_spread.get('sentiment', 0):+.2f}" if best_spread.get('sentiment') else ""
            log.info(
                f"FOUND: {symbol} CALL ${best_spread['short_strike']:.0f}/"
                f"${best_spread['long_strike']:.0f} exp {best_spread['expiration']} "
                f"| credit ${best_spread['credit']:.2f} ({best_spread['credit_pct']:.0%}) "
                f"| OTM {best_spread['otm_pct']:.1%} "
                f"| ROC {best_spread['roc_annual']:.0%}/yr{iv_str}{sent_str}"
            )
        else:
            log.debug(f"{symbol}: no qualifying call spread found")

        return best_spread

    def _has_strike_overlap(self, spread: Dict[str, Any]) -> bool:
        """Check if proposed spread shares any option legs with existing positions.
        Alpaca rejects orders with 'position intent mismatch' when a leg symbol
        already exists as part of another open position."""
        new_short = spread["short_symbol"]
        new_long = spread["long_symbol"]
        underlying = spread["underlying"]

        for pos_id, pos in self.positions.items():
            if pos["underlying"] != underlying:
                continue
            existing_legs = {pos["short_symbol"], pos["long_symbol"]}
            if new_short in existing_legs or new_long in existing_legs:
                log.warning(
                    f"{underlying}: leg overlap with {pos_id} — skipping "
                    f"(would cause position intent mismatch)"
                )
                return True
        return False

    def _execute_spread(self, spread: Dict[str, Any],
                        capital_in_use: float) -> bool:
        """Execute a credit spread (put or call) and track the position."""
        underlying = spread["underlying"]
        credit = spread["credit"]
        max_loss_per_contract = spread["max_loss_per_contract"]

        # Size the position
        qty = self.risk.size_position(max_loss_per_contract, capital_in_use)
        if qty <= 0:
            log.debug(f"{underlying}: position too small (max_loss ${max_loss_per_contract:.0f})")
            return False

        # Cap at live-realistic size (tighter for leveraged/inverse ETFs)
        if underlying in self.config.LEVERAGED_ETFS:
            qty = min(qty, self.config.LEVERAGED_QTY_CAP)
        else:
            qty = min(qty, 3)

        # Regime flip position sizing reduction
        flip_mult = self._regime_flip_state.get("adjustment_multiplier", 1.0)
        if flip_mult < 1.0:
            adjusted_qty = max(1, int(qty * flip_mult))
            if adjusted_qty < qty:
                log.info(
                    f"{underlying}: flip sizing {qty} → {adjusted_qty} contracts "
                    f"(mult={flip_mult:.2f}, whipsaw={self._regime_flip_state.get('whipsaw_score', 0):.2f})"
                )
                qty = adjusted_qty

        spread["qty"] = qty  # set qty for capital tracking in scan loop

        total_credit = credit * qty * 100
        total_max_loss = max_loss_per_contract * qty

        spread_type = spread.get("spread_type", "put")
        type_label = "bear_call_spread" if spread_type == "call" else "bull_put_spread"
        strike_suffix = "C" if spread_type == "call" else "P"

        log.info(
            f"OPENING SPREAD: {qty}x {underlying} {type_label} "
            f"${spread['short_strike']:.0f}/${spread['long_strike']:.0f}{strike_suffix} "
            f"exp {spread['expiration']} | "
            f"credit ${credit:.2f}/shr = ${total_credit:.0f} | "
            f"max loss ${total_max_loss:.0f}"
        )

        # Submit credit spread order
        # Use mid credit with realistic slippage for live-like fills
        fill_credit = round(credit * 0.85, 2)  # accept 15% less credit (realistic slippage)

        order_id = self.api.submit_credit_spread(
            short_symbol=spread["short_symbol"],
            long_symbol=spread["long_symbol"],
            qty=qty,
            credit_limit=fill_credit,
        )

        if order_id == "NO_BUYING_POWER":
            log.error(f"SPREAD ORDER FAILED for {underlying} — insufficient buying power")
            self._buying_power_exhausted = True
            return False
        if not order_id:
            log.error(f"SPREAD ORDER FAILED for {underlying}")
            return False

        # Verify fill — wait up to 120s for the order to fill
        filled = False
        for _wait in range(24):
            _time.sleep(5)
            order_status = self.api.get_order(order_id)
            if order_status and order_status.get("status") == "filled":
                filled = True
                actual_fill = order_status.get("filled_avg_price", fill_credit)
                log.info(f"{underlying}: order filled @ ${abs(actual_fill):.2f}")
                break
            elif order_status and order_status["status"] in ("canceled", "expired", "rejected"):
                log.warning(f"{underlying}: order {order_status['status']} — no fill")
                return False

        if not filled:
            log.warning(f"{underlying}: order not filled after 120s — canceling")
            try:
                self.api.trading.cancel_order_by_id(order_id)
            except Exception:
                pass
            return False

        # Track the position — unique key to prevent overwrites
        base_id = f"{underlying}_{spread['expiration']}_{spread_type[0].upper()}{int(spread['short_strike'])}"
        pos_id = base_id
        suffix = 2
        while pos_id in self.positions:
            pos_id = f"{base_id}_{suffix}"
            suffix += 1
        self.positions[pos_id] = {
            "underlying": underlying,
            "spread_type": spread_type,
            "short_symbol": spread["short_symbol"],
            "long_symbol": spread["long_symbol"],
            "short_strike": spread["short_strike"],
            "long_strike": spread["long_strike"],
            "spread_width": spread["spread_width"],
            "expiration": spread["expiration"],
            "dte_at_open": spread["dte"],
            "qty": qty,
            "credit_per_share": credit,
            "total_credit": total_credit,
            "max_loss_per_contract": max_loss_per_contract,
            "max_loss_total": total_max_loss,
            "max_profit_total": total_credit,
            "order_id": order_id,
            "open_date": date.today().isoformat(),
            "open_time": datetime.now().isoformat(),
            "current_debit": credit,  # starts at credit value
            "current_pnl_total": 0,
            "current_pnl_pct": 0,
            "roc_annual": spread["roc_annual"],
            "short_delta": spread.get("short_delta"),
            "entry_iv": spread.get("entry_iv"),
            "iv_premium": spread.get("iv_premium"),
            "features": spread.get("features"),  # ML features for outcome recording
            "ml_confidence": spread.get("ml_confidence"),
            "ml_rule_score": spread.get("ml_rule_score"),
        }

        self._save_positions()

        log.info(
            f"OPENED: {pos_id} | "
            f"credit ${total_credit:.0f} | max loss ${total_max_loss:.0f} | "
            f"ROC {spread['roc_annual']:.0%}/yr"
        )

        return True

    # ── Status ───────────────────────────────────────────
    def _print_status(self):
        """Print periodic status update."""
        stats = self.risk.get_stats()
        total_risk = sum(p.get("max_loss_total", 0) for p in self.positions.values())
        total_credit = sum(p.get("total_credit", 0) for p in self.positions.values())
        unrealized = sum(p.get("current_pnl_total", 0) for p in self.positions.values())

        put_count = sum(1 for p in self.positions.values() if p.get("spread_type") != "call")
        call_count = sum(1 for p in self.positions.values() if p.get("spread_type") == "call")

        log.info(
            f"[Cycle {self._cycle}] "
            f"Pos: {len(self.positions)}/{self.config.MAX_POSITIONS + self.config.MAX_CALL_POSITIONS} "
            f"(P:{put_count} C:{call_count}) | "
            f"Credit: ${total_credit:,.0f} | Risk: ${total_risk:,.0f} | "
            f"Unrealized: ${unrealized:+,.0f} | "
            f"Daily PnL: ${stats['daily_pnl']:+,.2f}"
        )
        if ML_AVAILABLE and self.meta:
            ms = self.meta.get_status()
            mls = self.ml.get_status() if self.ml else {}
            log.info(f"  ML: {ms['mode']} | WR {ms['recent_win_rate']:.0%} | "
                     f"model={'active' if mls.get('active') else 'warmup'} "
                     f"({mls.get('warmup_progress', '?')})")
        if self.sentiment:
            ss = self.sentiment.status()
            sources = " | ".join(f"{k}={v:+.3f}" for k, v in ss.get("sources", {}).items())
            log.info(f"  Sentiment: {ss['composite']:+.3f} | {sources}")

    def _is_trading_window(self) -> bool:
        """Check if we're in the safe trading window (avoid open/close volatility)."""
        now = datetime.now()
        market_open = now.replace(
            hour=self.config.MARKET_OPEN_HOUR,
            minute=self.config.MARKET_OPEN_MIN, second=0
        )
        market_close = now.replace(
            hour=self.config.MARKET_CLOSE_HOUR,
            minute=self.config.MARKET_CLOSE_MIN, second=0
        )

        earliest = market_open + timedelta(minutes=self.config.NO_OPEN_FIRST_MIN)
        latest = market_close - timedelta(minutes=self.config.NO_OPEN_LAST_MIN)

        return earliest <= now <= latest

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

        1. Pre-fetch daily bars + HV20 for all watchlist symbols
        2. Detect overnight gaps (Friday close vs pre-market)
        3. Check held positions for risk (DTE remaining, gap exposure)
        4. Run universe scan to refresh watchlist
        5. Sync account balances
        """
        log.info("=" * 50)
        log.info("PRE-MARKET WARMUP starting...")
        warmup_start = _time.time()
        results = {"gaps": {}, "hv20": {}, "symbols_warmed": 0}

        # ── 1. Fetch daily bars, compute HV20 & gaps for watchlist ──
        all_symbols = list(set(self._dynamic_watchlist))
        gaps = {}
        hv_cache = {}
        warmed = 0
        for symbol in all_symbols:
            try:
                bars = self.api.get_bars(symbol, days=40)  # daily bars (default)
                if bars and len(bars) >= 2:
                    prev_close = float(bars[-2].close)
                    last_close = float(bars[-1].close)
                    gap_pct = ((last_close - prev_close) / prev_close) * 100
                    gaps[symbol] = round(gap_pct, 2)

                    # Pre-compute HV20
                    if len(bars) >= 21:
                        closes = [float(b.close) for b in bars[-21:]]
                        import math
                        log_returns = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
                        if log_returns:
                            import statistics
                            hv = statistics.stdev(log_returns) * math.sqrt(252) * 100
                            hv_cache[symbol] = round(hv, 1)
                    warmed += 1
                _time.sleep(0.3)
            except Exception as e:
                log.debug(f"Warmup bar fetch failed for {symbol}: {e}")

        results["gaps"] = gaps
        results["hv20"] = hv_cache
        results["symbols_warmed"] = warmed

        # Log notable gaps
        big_gaps = {s: g for s, g in gaps.items() if abs(g) > 2.0}
        if big_gaps:
            sorted_gaps = sorted(big_gaps.items(), key=lambda x: abs(x[1]), reverse=True)
            gap_str = ", ".join(f"{s}:{g:+.1f}%" for s, g in sorted_gaps[:10])
            log.info(f"WARMUP gaps >2%: {gap_str}")

        # ── 2. Check held positions for gap risk ──
        if self.positions:
            for pos_id, pos in self.positions.items():
                underlying = pos.get("underlying", "")
                gap = gaps.get(underlying, 0)
                short_strike = pos.get("short_strike", 0)
                if gap < -3.0 and short_strike > 0:
                    log.warning(
                        f"WARMUP ALERT: {underlying} gapped {gap:+.1f}% — "
                        f"held spread ${short_strike:.0f} may be at risk"
                    )
            log.info(f"WARMUP: {len(self.positions)} positions checked for gap exposure")

        # ── 3. Universe scan ──
        if self.universe_scanner:
            try:
                expanded = self.universe_scanner.scan()
                self._dynamic_watchlist = expanded
                results["universe_size"] = len(expanded)
                log.info(f"WARMUP universe: {len(expanded)} symbols")
            except Exception as e:
                log.warning(f"Warmup universe scan failed: {e}")

        # ── 4. Sync account ──
        try:
            acct = self.api.get_account()
            self.risk.update_allocation(acct["equity"])
            allocation = acct["equity"] * self.config.ALLOCATION_PCT
            results["equity"] = acct["equity"]
            results["allocation"] = allocation
            log.info(f"WARMUP account: ${acct['equity']:,.2f} equity, allocation ${allocation:,.2f}")
        except Exception as e:
            log.warning(f"Warmup account sync failed: {e}")

        # ── 5. Pre-warm sentiment ──
        if self.sentiment:
            try:
                sent = self.sentiment.get_sentiment()
                results["sentiment"] = sent
                log.info(f"WARMUP sentiment: {sent:+.3f}")
            except Exception as e:
                log.debug(f"Warmup sentiment failed: {e}")

        # ── Done ──
        elapsed = _time.time() - warmup_start
        self._warmup_done_date = date.today()
        self._warmup_data = results
        log.info(
            f"PRE-MARKET WARMUP complete: {warmed} symbols, "
            f"{len(big_gaps)} big gaps, "
            f"{len(hv_cache)} HV20 computed, "
            f"{elapsed:.1f}s elapsed"
        )
        log.info("=" * 50)
