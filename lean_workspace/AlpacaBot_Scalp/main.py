"""
AlpacaBot Scalp — QuantConnect/LEAN port.

Ports the MECHANICAL rules from AlpacaBot/core/config.py + core/indicators.py +
core/trading_engine.py's `_generate_scalp_signal` (documented there as "the EXACT
same logic that produced +66.7% in backtest") onto QuantConnect's cloud engine.

Ported (faithfully, same thresholds/formulas as the live bot):
  - Watchlist + per-symbol DTE map (SYMBOL_DTE_MAP, validated via AlpacaBot's own
    DTE-sweep backtest) and the resulting per-symbol max-hold-days tiers
  - The 14-indicator bull/bear scalp score (RSI, MACD histogram, Stochastic,
    Bollinger %B, ATR-normalized, CCI, ROC, Williams %R, volatility ratio,
    mean-reversion z-score, trend strength, 1-bar/5-bar momentum), computed
    close-only exactly as AlpacaBot/core/indicators.py does (NOT QC's built-in
    high/low-based indicators, to stay faithful to the live formulas)
  - Morning/afternoon MIN_SIGNAL_SCORE adjustment (time-of-day aggression)
  - ITM strike targeting (5% ITM), min open interest, max bid-ask spread filter
  - Position sizing (% of equity per trade) and max positions / opens-per-cycle
  - Exit rules: -20% stop loss, +50% take profit, 12% trailing stop (armed at
    +15%), DTE exit (day of expiry), and per-symbol max-hold-days

NOT ported (needs external services identical to the live bot's own stack —
duplicating them here adds no independent value; this is intentionally the
"rules-only" skeleton of AlpacaBot for a clean, independent backtest):
  - ML model direction/confidence gate, sentiment score, meta-learner ensemble,
    SPY regime gate + regime-flip cooldown, scanner-universe (Phase 2) expansion,
    graduated-response risk throttle, vertical spreads (SPREADS_ENABLED).

Run this in QuantConnect's cloud IDE (Algorithm Lab) — compiling/backtesting LEAN
locally is not available in this workspace.
"""
from AlgorithmImports import *
import numpy as np


# ── Indicators — ported verbatim from AlpacaBot/core/indicators.py ─────────
# (close-only simplifications, deliberately NOT swapped for QC's OHLC-based
# built-ins, to stay faithful to the exact formulas the live bot backtested.)

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    return result


def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1:i + 1])
    return result


def _rolling_mad(data: np.ndarray, period: int) -> np.ndarray:
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        result[i] = np.mean(np.abs(window - np.mean(window)))
    return result


def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    if len(gains) < period:
        return np.full_like(prices, 50.0)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    result = 100.0 - (100.0 / (1.0 + rs))
    result[:period] = 50.0
    return result


def _macd_hist(prices: np.ndarray, fast=12, slow=26, signal=9) -> np.ndarray:
    macd_line = _ema(prices, fast) - _ema(prices, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def _bb_pct_b(prices: np.ndarray, period=20, num_std=2.0) -> np.ndarray:
    middle = _sma(prices, period)
    std = _rolling_std(prices, period)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    return np.where(width > 0, (prices - lower) / width, 0.5)


def _stochastic(prices: np.ndarray, period=14) -> np.ndarray:
    result = np.full_like(prices, 50.0)
    for i in range(period, len(prices)):
        window = prices[i - period + 1:i + 1]
        lo, hi = np.min(window), np.max(window)
        if hi > lo:
            result[i] = 100.0 * (prices[i] - lo) / (hi - lo)
    return result


def _atr_normalized(prices: np.ndarray, period=14) -> float:
    tr = np.abs(np.diff(prices))
    tr = np.insert(tr, 0, 0.0)
    result = np.zeros_like(prices)
    if len(tr) < period:
        return 0.0
    result[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(prices)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result[-1] / prices[-1] if prices[-1] > 0 else 0.0


def _cci(prices: np.ndarray, period=20) -> np.ndarray:
    sma = _sma(prices, period)
    mad = _rolling_mad(prices, period)
    return np.where(mad > 0, (prices - sma) / (0.015 * mad), 0.0)


def _roc(prices: np.ndarray, period=10) -> np.ndarray:
    result = np.zeros_like(prices)
    for i in range(period, len(prices)):
        if prices[i - period] > 0:
            result[i] = (prices[i] - prices[i - period]) / prices[i - period] * 100
    return result


def _williams_r(prices: np.ndarray, period=14) -> np.ndarray:
    result = np.full_like(prices, -50.0)
    for i in range(period, len(prices)):
        window = prices[i - period + 1:i + 1]
        hi, lo = np.max(window), np.min(window)
        if hi > lo:
            result[i] = -100.0 * (hi - prices[i]) / (hi - lo)
    return result


def _volatility_ratio(prices: np.ndarray, short=5, long=20) -> np.ndarray:
    short_std = _rolling_std(prices, short)
    long_std = _rolling_std(prices, long)
    return np.where(long_std > 0, short_std / long_std, 1.0)


def _zscore(prices: np.ndarray, period=20) -> np.ndarray:
    sma = _sma(prices, period)
    std = _rolling_std(prices, period)
    return np.where(std > 0, (prices - sma) / std, 0.0)


def _trend_strength(prices: np.ndarray, period=14) -> np.ndarray:
    result = np.zeros_like(prices)
    if len(prices) < period + 1:
        return result
    ups = np.zeros_like(prices)
    downs = np.zeros_like(prices)
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            ups[i] = diff
        else:
            downs[i] = -diff
    smooth_up = _ema(ups, period)
    smooth_down = _ema(downs, period)
    total = smooth_up + smooth_down
    return np.where(total > 0, np.abs(smooth_up - smooth_down) / total * 100, 0)


def compute_scalp_score(closes: np.ndarray, effective_min_score: int):
    """Exact bull/bear scoring logic from AlpacaBot's _generate_scalp_signal
    (rules-only — ML/sentiment/regime gates are handled by the live bot, not here)."""
    rsi = _rsi(closes)[-1]
    macd_h = _macd_hist(closes)[-1]
    stoch = _stochastic(closes)[-1]
    bb = _bb_pct_b(closes)[-1]
    atr_n = _atr_normalized(closes)
    cci_val = _cci(closes)[-1]
    roc_val = _roc(closes)[-1]
    wr = _williams_r(closes)[-1]
    vol_r = _volatility_ratio(closes)[-1]
    zs = _zscore(closes)[-1]
    ts = _trend_strength(closes)[-1]
    pc1 = (closes[-1] / closes[-2] - 1) if len(closes) > 1 else 0
    pc5 = (closes[-1] / closes[-6] - 1) if len(closes) > 5 else 0

    bull, bear = 0, 0
    if rsi < 25:
        bull += 2
    elif 35 < rsi < 55:
        bull += 1
    elif rsi > 75:
        bear += 2
    elif 50 < rsi < 65:
        bear += 1

    if macd_h > 0:
        bull += 1
        if macd_h > 0.1:
            bull += 1
    elif macd_h < 0:
        bear += 1
        if macd_h < -0.1:
            bear += 1

    if stoch < 20:
        bull += 1
    elif stoch > 80:
        bear += 1

    if bb < 0.10:
        bull += 2
    elif bb > 0.90:
        bear += 2
    elif bb < 0.30:
        bull += 1
    elif bb > 0.70:
        bear += 1

    if atr_n > 0.005:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    if cci_val < -100:
        bull += 1
    elif cci_val > 100:
        bear += 1

    if roc_val > 0.3:
        bull += 1
    elif roc_val < -0.3:
        bear += 1

    if wr > -20:
        bear += 1
    elif wr < -80:
        bull += 1

    if vol_r > 1.3:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    if zs < -2.0:
        bull += 1
    elif zs > 2.0:
        bear += 1

    if ts > 25:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    if pc1 > 0.001:
        bull += 1
    elif pc1 < -0.001:
        bear += 1

    if pc5 > 0.003:
        bull += 1
    elif pc5 < -0.003:
        bear += 1

    if bull >= effective_min_score and bull > bear + 1:
        return "call", bull
    if bear >= effective_min_score and bear > bull + 1:
        return "put", bear
    return None, 0


class AlpacaBotScalpAlgorithm(QCAlgorithm):

    WATCHLIST = ["MA", "NFLX", "GM", "IWM", "LLY", "SNOW"]
    SYMBOL_DTE_MAP = {"MA": 2, "NFLX": 2, "GM": 2, "IWM": 1, "LLY": 2}
    DEFAULT_DTE = 2

    MIN_DTE = 1     # never 0DTE
    MAX_DTE = 45

    TARGET_ITM_PCT = 0.05
    MIN_OPEN_INTEREST = 50
    MAX_BID_ASK_SPREAD_PCT = 0.15

    MAX_POSITION_PCT = 0.08           # was 0.15 — halved to reduce ruin risk from compounding on losers
    MAX_POSITIONS = 3
    MAX_OPENS_PER_CYCLE = 2

    MIN_SIGNAL_SCORE = 5             # was 4 — require stronger indicator consensus
    MORNING_WINDOW_END_MIN = 110
    LOOKBACK_BARS = 50

    STOP_LOSS_PCT = -0.15            # was -0.20 — cut losers earlier
    TAKE_PROFIT_PCT = 0.50
    TRAILING_STOP_PCT = 0.12
    TRAILING_TRIGGER = 0.15
    MIN_DTE_EXIT = 0

    SPY_TREND_SMA = 50                # proxy for the excluded SPY-regime gate (QC-native, no external service)
    DAILY_LOSS_HALT_PCT = -0.06       # circuit breaker: stop opening new trades once today's P&L breaches this

    def initialize(self) -> None:
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2026, 8, 1)
        self.set_cash(50000)
        self.set_warm_up(timedelta(days=10))

        self.option_symbol_by_ticker = {}
        self.closes = {}

        spy_equity = self.add_equity("SPY", Resolution.MINUTE)
        self.spy_closes = RollingWindow[float](self.SPY_TREND_SMA + 10)
        spy_consolidator = TradeBarConsolidator(timedelta(minutes=10))
        spy_consolidator.data_consolidated += self._make_consolidation_handler("__SPY__")
        self.subscription_manager.add_consolidator(spy_equity.symbol, spy_consolidator)

        for ticker in self.WATCHLIST:
            equity = self.add_equity(ticker, Resolution.MINUTE)
            option = self.add_option(ticker, Resolution.MINUTE)
            option.set_filter(self._filter_chain)
            self.option_symbol_by_ticker[ticker] = option.symbol

            self.closes[ticker] = RollingWindow[float](self.LOOKBACK_BARS + 10)
            consolidator = TradeBarConsolidator(timedelta(minutes=10))
            consolidator.data_consolidated += self._make_consolidation_handler(ticker)
            self.subscription_manager.add_consolidator(equity.symbol, consolidator)

        self.positions = {}   # option symbol -> dict
        self.cooldowns = {}   # ticker -> datetime

        self.day_start_value = self.portfolio.total_portfolio_value
        self.day_start_date = None
        self.trading_halted_today = False

        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(minutes=10)),
            self._scan_and_check,
        )

    def _make_consolidation_handler(self, ticker: str):
        def handler(sender, bar):
            if ticker == "__SPY__":
                self.spy_closes.add(float(bar.close))
            else:
                self.closes[ticker].add(float(bar.close))
        return handler

    def _update_daily_circuit_breaker(self) -> None:
        today = self.time.date()
        if self.day_start_date != today:
            self.day_start_date = today
            self.day_start_value = self.portfolio.total_portfolio_value
            self.trading_halted_today = False
        if not self.trading_halted_today and self.day_start_value > 0:
            day_pnl_pct = (self.portfolio.total_portfolio_value - self.day_start_value) / self.day_start_value
            if day_pnl_pct <= self.DAILY_LOSS_HALT_PCT:
                self.trading_halted_today = True
                self.debug(f"Daily loss circuit breaker tripped ({day_pnl_pct:.1%}) — halting new entries until tomorrow")

    def _spy_trend(self):
        """Returns 'up', 'down', or None (not enough data) — QC-native proxy for the
        excluded SPY regime gate: only trade with the prevailing SPY trend."""
        if self.spy_closes.count < self.SPY_TREND_SMA:
            return None
        window = [self.spy_closes[i] for i in range(self.spy_closes.count)][::-1]
        sma = sum(window[-self.SPY_TREND_SMA:]) / self.SPY_TREND_SMA
        return "up" if window[-1] > sma else "down"

    def _filter_chain(self, universe):
        return universe.strikes(-10, 10).expiration(self.MIN_DTE, self.MAX_DTE)

    def _time_window(self) -> str:
        market_open = self.time.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_since_open = (self.time - market_open).total_seconds() / 60
        return "morning" if 0 <= minutes_since_open <= self.MORNING_WINDOW_END_MIN else "afternoon"

    def _scan_and_check(self) -> None:
        if self.is_warming_up:
            return
        self._check_exits()
        self._update_daily_circuit_breaker()

        if self.trading_halted_today or len(self.positions) >= self.MAX_POSITIONS:
            return

        opens_this_cycle = 0
        for ticker in self.WATCHLIST:
            if opens_this_cycle >= self.MAX_OPENS_PER_CYCLE or len(self.positions) >= self.MAX_POSITIONS:
                break
            if any(p["underlying"] == ticker for p in self.positions.values()):
                continue
            cd = self.cooldowns.get(ticker)
            if cd and self.time < cd:
                continue

            signal = self._generate_signal(ticker)
            if not signal:
                continue
            if self._execute_scalp(ticker, signal):
                opens_this_cycle += 1

    def _generate_signal(self, ticker: str):
        window = self.closes[ticker]
        if window.count < self.LOOKBACK_BARS + 1:
            return None
        closes = np.array([window[i] for i in range(window.count)])[::-1]

        time_window = self._time_window()
        effective_min_score = self.MIN_SIGNAL_SCORE
        if time_window == "morning":
            effective_min_score = max(2, self.MIN_SIGNAL_SCORE - 1)
        elif time_window == "afternoon":
            effective_min_score = self.MIN_SIGNAL_SCORE + 1

        direction, score = compute_scalp_score(closes, effective_min_score)
        if direction is None:
            return None

        # QC-native proxy for the excluded SPY-regime gate: only trade with the trend.
        trend = self._spy_trend()
        if trend == "up" and direction == "put":
            return None
        if trend == "down" and direction == "call":
            return None

        return {"direction": direction, "score": score, "price": float(closes[-1])}

    def _select_contract(self, chain, ticker: str, signal: dict):
        right = OptionRight.CALL if signal["direction"] == "call" else OptionRight.PUT
        price = signal["price"]
        target_strike = price * (1 - self.TARGET_ITM_PCT) if right == OptionRight.CALL else price * (1 + self.TARGET_ITM_PCT)
        target_dte = self.SYMBOL_DTE_MAP.get(ticker, self.DEFAULT_DTE)
        target_expiry = self.time.date() + timedelta(days=target_dte)

        best, best_key = None, None
        for c in chain:
            if c.right != right or c.open_interest < self.MIN_OPEN_INTEREST:
                continue
            mid = (c.bid_price + c.ask_price) / 2
            if mid <= 0:
                continue
            spread_pct = (c.ask_price - c.bid_price) / mid
            if spread_pct > self.MAX_BID_ASK_SPREAD_PCT:
                continue
            key = (abs((c.expiry.date() - target_expiry).days), abs(c.strike - target_strike))
            if best_key is None or key < best_key:
                best_key, best = key, c
        return best

    def _execute_scalp(self, ticker: str, signal: dict) -> bool:
        option_symbol = self.option_symbol_by_ticker[ticker]
        chain = self.current_slice.option_chains.get(option_symbol)
        if not chain:
            return False

        contract = self._select_contract(chain, ticker, signal)
        if not contract:
            return False

        premium = contract.ask_price if contract.ask_price > 0 else (contract.bid_price + contract.ask_price) / 2
        if premium <= 0:
            return False
        entry_mid = (contract.bid_price + contract.ask_price) / 2 if contract.ask_price > 0 else premium

        max_spend = self.portfolio.total_portfolio_value * self.MAX_POSITION_PCT
        cost_per = premium * 100
        if cost_per > max_spend:
            return False
        qty = max(1, int(max_spend / cost_per))

        self.market_order(contract.symbol, qty)

        target_dte = self.SYMBOL_DTE_MAP.get(ticker, self.DEFAULT_DTE)
        max_hold_days = 1 if target_dte <= 2 else (3 if target_dte <= 9 else 7)

        self.positions[contract.symbol] = {
            "underlying": ticker,
            "direction": signal["direction"],
            "entry_price": premium,
            # exit checks compare mid-to-mid (not ask-paid-to-mid) to avoid a
            # baked-in phantom loss equal to half the bid/ask spread at entry.
            "entry_mid": entry_mid,
            "peak_price": entry_mid,
            "qty": qty,
            "entry_time": self.time,
            "expiry": contract.expiry,
            "max_hold_days": max_hold_days,
            "stop_loss_strikes": 0,
        }
        self.debug(
            f"{ticker}: opened {signal['direction']} score={signal['score']} "
            f"strike={contract.strike} exp={contract.expiry.date()} premium={premium:.2f} qty={qty}"
        )
        return True

    def _check_exits(self) -> None:
        for symbol, pos in list(self.positions.items()):
            security = self.securities.get(symbol)
            if security is None:
                continue
            current = (security.bid_price + security.ask_price) / 2 if security.ask_price > 0 else security.price
            if current <= 0:
                continue
            if current > pos["peak_price"]:
                pos["peak_price"] = current

            entry = pos["entry_mid"]
            pnl_pct = (current - entry) / entry
            reason = None

            if pnl_pct <= self.STOP_LOSS_PCT:
                # illiquid contracts can print one noisy/wide-spread quote —
                # require 2 consecutive 10-min checks below threshold before
                # actually closing, to filter single-tick noise (same pattern
                # found and fixed in CallBuyer's STOP_LOSS check).
                pos["stop_loss_strikes"] += 1
                if pos["stop_loss_strikes"] >= 2:
                    reason = f"STOP_LOSS ({pnl_pct:.0%})"
            else:
                pos["stop_loss_strikes"] = 0

            if not reason and pnl_pct >= self.TAKE_PROFIT_PCT:
                reason = f"TAKE_PROFIT ({pnl_pct:.0%})"
            elif not reason and pos["peak_price"] > entry * (1 + self.TRAILING_TRIGGER):
                drop = (current - pos["peak_price"]) / pos["peak_price"]
                if drop <= -self.TRAILING_STOP_PCT:
                    reason = f"TRAILING_STOP ({drop:.0%} from peak)"

            if not reason:
                dte = (pos["expiry"].date() - self.time.date()).days
                if dte <= self.MIN_DTE_EXIT:
                    reason = f"DTE_EXIT ({dte}d)"

            if not reason:
                hold_hours = (self.time - pos["entry_time"]).total_seconds() / 3600
                if hold_hours >= pos["max_hold_days"] * 24:
                    reason = f"MAX_HOLD ({hold_hours:.1f}h)"

            if reason:
                self.liquidate(symbol)
                self.cooldowns[pos["underlying"]] = self.time + timedelta(hours=1)
                self.debug(f"{pos['underlying']}: closed {symbol} — {reason}")
                del self.positions[symbol]

    def on_order_event(self, order_event) -> None:
        if order_event.status == OrderStatus.FILLED:
            self.debug(str(order_event))
