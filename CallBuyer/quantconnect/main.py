"""
CallBuyer Momentum Call Buying — QuantConnect/LEAN port.

Ports the MECHANICAL rules from CallBuyer/core/config.py + core/feature_engine.py
+ core/call_engine.py + core/risk_manager.py onto QuantConnect's cloud engine.

Ported (faithfully, same thresholds/formulas as the live bot):
  - Fixed watchlist (10 momentum-friendly, high-beta liquid names)
  - RSI pre-filter (40-85) then the rules-based momentum score (0-10 scale):
    RSI zone, multi-timeframe ROC momentum composite, volume surge vs 20-day avg,
    20-day-high breakout, 20/50/200 SMA alignment, realized-vol IV-rank proxy,
    SPY 5-day sector momentum, MACD histogram — computed exactly as
    CallBuyerFeatureEngine.build_features()/compute_rule_score() do
  - The meta-learner's decision gate run in its "ML not active" (100% rules)
    branch: confidence = rule_score/10, requires rule_score >= effective min
    (2.0, adjusted by morning/afternoon) AND confidence >= effective threshold
    (0.30, adjusted) — this is the exact fallback path MetaLearner.evaluate()
    takes whenever the ML model isn't active, so it's a genuine "rules-only"
    slice of the real ensemble logic, not an approximation of it
  - Morning/afternoon confidence & rule-score adjustments (MORNING_WINDOW_END_MIN
    = 90 min after 9:30 ET)
  - ITM call strike selection (5% ITM to 2% OTM, target 2% ITM), scored by
    OTM-proximity/DTE-proximity/spread/OI exactly as _select_strike() does
  - Position sizing (2% of equity risk per trade, capped 1-5 contracts),
    MAX_POSITIONS=4, MAX_PER_UNDERLYING=2
  - Exit rules: +50% take profit, -25% stop loss (2-poll confirmation), 7-DTE
    exit, and the 30%-gain-triggered 15% trailing stop

NOT ported (needs external services identical to the live bot's own stack —
duplicating them here adds no independent research value; this is intentionally
the "rules-only" skeleton of CallBuyer for a clean, independent backtest):
  - ML model win-probability prediction + its 0.40 ensemble weight, regime
    detector + regime-flip position-sizing/entry-blocking, earnings-date buffer
    check (earnings_check.py), universe_scanner.py dynamic discovery.

Run this in QuantConnect's cloud IDE (Algorithm Lab) — compiling/backtesting LEAN
locally is not available in this workspace.
"""
from AlgorithmImports import *
import numpy as np


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def _rsi_simple(closes: np.ndarray, period: int = 14) -> float:
    """Simple mean-based RSI over the trailing window — matches
    CallBuyerFeatureEngine._rsi() exactly (NOT the Wilder-smoothed version
    used by AlpacaBot's indicators.py — the two bots use different RSI
    formulas in real life, so this port keeps them distinct)."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    return float(100 - 100 / (1 + rs))


def compute_features(closes, opens, highs, lows, volumes, spy_closes) -> dict:
    """Ported verbatim from CallBuyerFeatureEngine.build_features() — only the
    features that actually feed compute_rule_score() (indices 0,1,2,3,4,6,7,11)."""
    f = {}
    f["rsi"] = _rsi_simple(closes, 14)

    roc5 = (closes[-1] / closes[-6] - 1) if closes[-6] != 0 else 0
    roc10 = (closes[-1] / closes[-11] - 1) if closes[-11] != 0 else 0
    roc20 = (closes[-1] / closes[-21] - 1) if closes[-21] != 0 else 0
    momentum_raw = 0.5 * roc5 + 0.3 * roc10 + 0.2 * roc20
    f["momentum_composite"] = np.clip(momentum_raw, -0.15, 0.15) / 0.15

    avg_vol = np.mean(volumes[-21:-1]) if len(volumes) > 21 else np.mean(volumes[:-1])
    f["volume_surge"] = np.clip(volumes[-1] / avg_vol if avg_vol > 0 else 1.0, 0.3, 4.0) / 4.0

    high_20 = np.max(highs[-21:-1])
    f["breakout_score"] = np.clip((closes[-1] - high_20) / high_20 if high_20 > 0 else 0, 0, 0.10) / 0.10

    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:])
    sma200 = np.mean(closes[-min(200, len(closes)):]) if len(closes) >= 200 else np.mean(closes)
    alignment = 0.0
    if closes[-1] > sma20:
        alignment += 0.25
    if sma20 > sma50:
        alignment += 0.25
    if sma50 > sma200:
        alignment += 0.25
    if closes[-1] > sma200:
        alignment += 0.25
    f["sma_alignment"] = alignment

    rv5 = np.std(np.diff(np.log(closes[-6:]))) * np.sqrt(252) if len(closes) > 6 else 0
    rv20 = np.std(np.diff(np.log(closes[-21:]))) * np.sqrt(252) if len(closes) > 21 else rv5
    ratio = rv5 / rv20 if rv20 > 0 else 1.0
    f["iv_rank"] = 1.0 - np.clip(ratio, 0.5, 2.0) / 2.0

    if spy_closes is not None and len(spy_closes) >= 6:
        spy_roc = (spy_closes[-1] / spy_closes[-6] - 1) if spy_closes[-6] != 0 else 0
        f["sector_momentum"] = np.clip(spy_roc, -0.05, 0.05) / 0.05 * 0.5 + 0.5
    else:
        f["sector_momentum"] = 0.5

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line[-9:], 9)[-1] if len(macd_line) >= 9 else macd_line[-1]
    histogram = macd_line[-1] - signal_line
    f["macd_histogram"] = np.clip(histogram / closes[-1] * 100 if closes[-1] > 0 else 0, -1, 1) * 0.5 + 0.5

    return f


def compute_rule_score(f: dict) -> float:
    """Ported verbatim from CallBuyerFeatureEngine.compute_rule_score()."""
    score = 0.0
    rsi = f["rsi"]
    if 50 <= rsi <= 75:
        score += 2.0
    elif 75 < rsi <= 85:
        score += 2.0
    elif 45 <= rsi < 50:
        score += 1.0

    if f["momentum_composite"] > 0.3:
        score += 2.0
    elif f["momentum_composite"] > 0.1:
        score += 1.0

    if f["volume_surge"] > 0.4:
        score += 1.5
    elif f["volume_surge"] > 0.3:
        score += 0.5

    if f["breakout_score"] > 0.01:
        score += 1.5

    if f["sma_alignment"] >= 0.75:
        score += 1.5
    elif f["sma_alignment"] >= 0.50:
        score += 0.5

    if f["iv_rank"] > 0.6:
        score += 0.5
    if f["sector_momentum"] > 0.6:
        score += 0.5
    if f["macd_histogram"] > 0.6:
        score += 0.5

    return min(score, 10.0)


class CallBuyerMomentumAlgorithm(QCAlgorithm):

    WATCHLIST = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AMD", "META", "AMZN", "NFLX", "AVGO"]

    MIN_RSI = 40.0
    MAX_RSI = 85.0

    MIN_DTE = 21
    MAX_DTE = 60
    TARGET_DTE = 45

    MIN_OTM_PCT = -0.05   # up to 5% ITM
    MAX_OTM_PCT = 0.02    # up to 2% OTM
    TARGET_OTM_PCT = -0.02

    MIN_OPEN_INTEREST = 50
    MAX_SPREAD_PCT = 0.30

    MAX_POSITION_RISK_PCT = 0.02
    MAX_POSITIONS = 4
    MAX_PER_UNDERLYING = 2

    META_CONFIDENCE_THRESHOLD = 0.30
    META_MIN_RULE_SCORE = 2.0

    MORNING_WINDOW_END_MIN = 90
    MORNING_CONF_BOOST = 0.03
    MORNING_RULE_REDUCTION = 0.5
    AFTERNOON_CONF_PENALTY = 0.02
    AFTERNOON_RULE_INCREASE = 0.5

    TAKE_PROFIT_PCT = 0.50
    STOP_LOSS_PCT = -0.25
    MIN_DTE_EXIT = 7
    TRAILING_STOP_PCT = 0.15
    TRAILING_ARM_GAIN = 0.30

    def initialize(self) -> None:
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2026, 8, 1)
        self.set_cash(50000)
        self.set_warm_up(timedelta(days=15))

        self.equity_symbols = {}
        self.option_symbol_by_ticker = {}

        for ticker in self.WATCHLIST:
            equity = self.add_equity(ticker, Resolution.MINUTE)
            option = self.add_option(ticker, Resolution.MINUTE)
            option.set_filter(self._filter_chain)
            self.equity_symbols[ticker] = equity.symbol
            self.option_symbol_by_ticker[ticker] = option.symbol

        self.positions = {}  # option symbol -> dict

        first_ticker = self.WATCHLIST[0]
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open(self.equity_symbols[first_ticker], 5),
            self._daily_scan,
        )
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(minutes=15)),
            self._check_exits,
        )

    def _filter_chain(self, universe):
        return universe.strikes(-6, 6).expiration(self.MIN_DTE, self.MAX_DTE)

    def _time_window(self) -> str:
        market_open = self.time.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_since_open = (self.time - market_open).total_seconds() / 60
        return "morning" if 0 <= minutes_since_open <= self.MORNING_WINDOW_END_MIN else "afternoon"

    def _get_daily_arrays(self, symbol):
        history = self.history(symbol, 210, Resolution.DAILY)
        if history is None or history.empty:
            return None
        if history.index.nlevels > 1:
            history = history.loc[symbol]
        history = history.sort_index()
        if len(history) < 50:
            return None
        return {
            "close": history["close"].values.astype(float),
            "open": history["open"].values.astype(float),
            "high": history["high"].values.astype(float),
            "low": history["low"].values.astype(float),
            "volume": history["volume"].values.astype(float),
        }

    def _meta_evaluate(self, rule_score: float, conf_adjust: float, rule_adjust: float):
        rule_norm = min(rule_score / 10.0, 1.0)
        confidence = rule_norm  # ML not active — 100% rules (rules-only port)
        effective_min_rule = max(1.5, self.META_MIN_RULE_SCORE + rule_adjust)
        effective_conf = max(0.25, min(0.85, self.META_CONFIDENCE_THRESHOLD + conf_adjust))
        if rule_score < effective_min_rule or confidence < effective_conf:
            return confidence, False
        return confidence, True

    def _evaluate_symbol(self, ticker: str, spy_closes):
        arrays = self._get_daily_arrays(self.equity_symbols[ticker])
        if arrays is None:
            return None
        closes = arrays["close"]

        rsi = _rsi_simple(closes, 14)
        if rsi < self.MIN_RSI or rsi > self.MAX_RSI:
            return None

        features = compute_features(closes, arrays["open"], arrays["high"], arrays["low"], arrays["volume"], spy_closes)
        rule_score = compute_rule_score(features)

        time_window = self._time_window()
        conf_adj, rule_adj = 0.0, 0.0
        if time_window == "morning":
            conf_adj = -self.MORNING_CONF_BOOST
            rule_adj = -self.MORNING_RULE_REDUCTION
        elif time_window == "afternoon":
            conf_adj = self.AFTERNOON_CONF_PENALTY
            rule_adj = self.AFTERNOON_RULE_INCREASE

        confidence, should_trade = self._meta_evaluate(rule_score, conf_adj, rule_adj)
        return {
            "price": float(closes[-1]),
            "rule_score": rule_score,
            "confidence": confidence,
            "should_trade": should_trade,
        }

    def _daily_scan(self) -> None:
        if self.is_warming_up:
            return
        if len(self.positions) >= self.MAX_POSITIONS:
            return

        spy_arrays = self._get_daily_arrays(self.equity_symbols["SPY"])
        spy_closes = spy_arrays["close"] if spy_arrays else None

        for ticker in self.WATCHLIST:
            if len(self.positions) >= self.MAX_POSITIONS:
                break
            underlying_count = sum(1 for p in self.positions.values() if p["underlying"] == ticker)
            if underlying_count >= self.MAX_PER_UNDERLYING:
                continue

            candidate = self._evaluate_symbol(ticker, spy_closes)
            if not candidate or not candidate["should_trade"]:
                continue
            self._open_call(ticker, candidate)

    def _select_strike(self, price: float, chain):
        scored = []
        for c in chain:
            if c.right != OptionRight.CALL:
                continue
            strike = c.strike
            otm_pct = (strike - price) / price if price > 0 else 0
            if otm_pct < self.MIN_OTM_PCT or otm_pct > self.MAX_OTM_PCT:
                continue

            dte = (c.expiry.date() - self.time.date()).days
            if dte < self.MIN_DTE or dte > self.MAX_DTE:
                continue

            ask, bid = c.ask_price, c.bid_price
            if ask <= 0:
                continue
            spread = (ask - bid) / ask if ask > 0 else 1.0
            if spread > self.MAX_SPREAD_PCT:
                continue

            oi = c.open_interest
            if oi < self.MIN_OPEN_INTEREST:
                continue

            dte_score = 1.0 - abs(dte - self.TARGET_DTE) / self.MAX_DTE
            otm_score = max(0.0, 1.0 - abs(otm_pct - self.TARGET_OTM_PCT) / 0.07)
            spread_score = 1.0 - spread
            oi_score = min(oi / 500.0, 1.0)
            total = otm_score * 0.35 + dte_score * 0.30 + spread_score * 0.20 + oi_score * 0.15
            scored.append((c, total))

        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _open_call(self, ticker: str, candidate: dict) -> bool:
        option_symbol = self.option_symbol_by_ticker[ticker]
        chain = self.current_slice.option_chains.get(option_symbol)
        if not chain:
            return False

        contract = self._select_strike(candidate["price"], chain)
        if not contract:
            return False

        premium = contract.ask_price
        if premium <= 0:
            return False

        max_risk = self.portfolio.total_portfolio_value * self.MAX_POSITION_RISK_PCT
        qty = min(max(int(max_risk / (premium * 100)), 0), 5)
        if qty < 1:
            return False

        self.market_order(contract.symbol, qty)

        self.positions[contract.symbol] = {
            "underlying": ticker,
            "entry_price": premium,
            "high_water": premium,
            "qty": qty,
            "entry_time": self.time,
            "expiry": contract.expiry,
            "stop_loss_strikes": 0,
        }
        self.debug(
            f"{ticker}: opened CALL score={candidate['rule_score']:.1f} conf={candidate['confidence']:.2f} "
            f"strike={contract.strike} exp={contract.expiry.date()} premium={premium:.2f} qty={qty}"
        )
        return True

    def _check_exits(self) -> None:
        if self.is_warming_up:
            return
        for symbol, pos in list(self.positions.items()):
            security = self.securities.get(symbol)
            if security is None:
                continue
            current = security.bid_price if security.bid_price > 0 else security.price
            if current <= 0:
                continue

            entry = pos["entry_price"]
            if current > pos["high_water"]:
                pos["high_water"] = current
            pnl_pct = (current - entry) / entry

            reason = None
            if pnl_pct >= self.TAKE_PROFIT_PCT:
                reason = f"TAKE_PROFIT ({pnl_pct:+.1%})"
            elif pnl_pct <= self.STOP_LOSS_PCT:
                pos["stop_loss_strikes"] = pos.get("stop_loss_strikes", 0) + 1
                if pos["stop_loss_strikes"] >= 2:
                    reason = f"STOP_LOSS ({pnl_pct:+.1%})"
            else:
                pos["stop_loss_strikes"] = 0

            if not reason:
                dte = (pos["expiry"].date() - self.time.date()).days
                if dte <= self.MIN_DTE_EXIT:
                    reason = f"DTE_EXIT ({dte}d)"

            if not reason and pos["high_water"] > entry * (1 + self.TRAILING_ARM_GAIN):
                drawdown = (current - pos["high_water"]) / pos["high_water"]
                if drawdown <= -self.TRAILING_STOP_PCT:
                    reason = f"TRAILING_STOP ({drawdown:+.1%})"

            if reason:
                self.liquidate(symbol)
                self.debug(f"{pos['underlying']}: closed {symbol} — {reason}")
                del self.positions[symbol]

    def on_order_event(self, order_event) -> None:
        if order_event.status == OrderStatus.FILLED:
            self.debug(str(order_event))
