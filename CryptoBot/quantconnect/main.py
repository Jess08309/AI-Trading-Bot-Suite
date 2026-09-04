"""
CryptoBot Momentum — QuantConnect/LEAN port.

Ports the MECHANICAL rules from CryptoBot/cryptotrades/utils/config.py +
core/trading_engine.py (calculate_trend/calculate_rsi/execute_signals/exit
logic) onto QuantConnect's cloud engine, following the same "rules-only"
pattern already used for AlpacaBot/CallBuyer/PutSeller's QC ports.

Ported (faithfully, same thresholds/formulas as the live bot):
  - calculate_trend(): 20-bar linear-regression slope normalized by mean
    price, UP/DOWN/SIDE classification via MIN_TREND_SLOPE=0.0005
  - RSI(14), MAX_RSI_LONG=68 / MIN_RSI_SHORT=32 quality gates
  - SIDE_MARKET_FILTER: skip entries when trend == SIDE (no ML override
    available in a rules-only port, so this is a hard skip here)
  - A 0-10 mechanical rule score (trend strength, RSI zone, MACD histogram
    sign, 20-bar SMA breakout) gated by MIN_RULE_SCORE, standing in for the
    live bot's excluded ML ensemble/confidence gate (see below)
  - Exit rules: STOP_LOSS_PCT=-1.5%, TAKE_PROFIT_PCT=+1.5%,
    TRAILING_STOP_PCT=0.8% arming after TRAILING_ACTIVATE_PCT=+0.6% gain,
    MAX_HOLD_HOURS_SPOT=8h flat-close (|pnl| < MAX_HOLD_FLAT_BAND_PCT=0.5%),
    MAX_HOLD_FORCED_HOURS_SPOT=12h hard close regardless of P&L
  - Circuit breaker: CB_MAX_CONSECUTIVE_LOSSES=5, CB_DAILY_LOSS_LIMIT_PCT=-4%,
    CB_MAX_DRAWDOWN_PCT=-8%, with the live bot's CB_COOLDOWN_MINUTES=60
    pause-then-retry semantics (utils/circuit_breaker.py) -- NOT a permanent
    halt. Consecutive-losses/drawdown trips pause new entries for 60 minutes
    then allow retrying (can re-trip if losses continue); only the daily-loss
    trip is day-scoped (clears at the next UTC day boundary).
  - Position sizing: MAX_POSITION_PCT=0.12 of equity, MAX_POSITIONS_SPOT=4,
    MAX_POSITIONS_PER_SYMBOL_SPOT=1

NOT ported (needs external services identical to the live bot's own stack, or
data QuantConnect's free tier can't provide — duplicating them here adds no
independent research value; this is intentionally the "rules-only" skeleton
of CryptoBot for a clean, independent backtest):
  - ML meta-model win-probability prediction (MIN_ML_CONFIDENCE=0.62 gate),
    news/social sentiment (USE_SENTIMENT), RL agent shadow-trading, Kraken
    perpetual FUTURES leg entirely (QC's free-tier data library does not
    carry Kraken perpetuals) — this port is SPOT-LONG-ONLY on QC's Coinbase
    (GDAX) crypto market, since real Coinbase spot cannot short-sell.
    DIRECTION_MODE="both" in the live bot relies on the futures leg for
    SHORT exposure; the excluded SHORT side is therefore this port's
    biggest structural difference from the live bot, not an oversight.
  - Symbol Performance Gate / Direction Performance Tracker (auto-pause
    logic) — these are adaptive safety nets tuned against live paper-account
    behavior, not testable in a single historical backtest run the same way.

Run this in QuantConnect's cloud IDE (Algorithm Lab) — compiling/backtesting
LEAN locally is not available in this workspace.
"""
from AlgorithmImports import *
import numpy as np


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Matches CryptoBot's calculate_rsi() — simple mean-based RSI."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-12
    rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
    return float(100.0 - 100.0 / (1.0 + rs))


def _trend(closes: np.ndarray, lookback: int, min_slope: float) -> tuple:
    """Ported verbatim from TradingEngine.calculate_trend()."""
    if len(closes) < lookback:
        return "SIDE", 0.0
    recent = closes[-lookback:]
    x = np.arange(len(recent))
    slope, _ = np.polyfit(x, recent, 1)
    normalized_slope = slope / np.mean(recent)
    if normalized_slope > min_slope:
        return "UP", normalized_slope
    elif normalized_slope < -min_slope:
        return "DOWN", normalized_slope
    else:
        return "SIDE", normalized_slope


def _macd_histogram(closes: np.ndarray) -> float:
    def _ema(data, period):
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    if len(closes) < 35:
        return 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line[-9:], 9)[-1]
    return float(macd_line[-1] - signal_line)


def compute_rule_score(rsi: float, trend: str, slope: float, macd_hist: float,
                        close: float, sma20: float) -> float:
    """0-10 mechanical proxy for the live bot's excluded ML/ensemble gate."""
    score = 0.0
    # Trend strength (steeper slope = stronger conviction)
    if trend == "UP":
        score += 3.0 if slope > 0.0015 else 2.0
    # RSI zone (favor pullback-to-momentum, not overbought)
    if 45 <= rsi <= 60:
        score += 3.0
    elif 60 < rsi <= 68:
        score += 2.0
    elif 38 <= rsi < 45:
        score += 1.0
    # MACD histogram agreeing with the LONG direction
    if macd_hist > 0:
        score += 2.0
    # Breakout above the 20-bar SMA
    if close > sma20:
        score += 2.0
    return min(score, 10.0)


class CryptoBotMomentumAlgorithm(QCAlgorithm):

    SYMBOLS = ["BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD", "ADAUSD"]

    TREND_LOOKBACK = 20
    MIN_TREND_SLOPE = 0.0005
    MAX_RSI_LONG = 68.0

    MIN_RULE_SCORE = 5.0

    MAX_POSITION_PCT = 0.12
    MAX_POSITIONS = 4

    STOP_LOSS_PCT = -1.5
    TAKE_PROFIT_PCT = 1.5
    TRAILING_STOP_PCT = 0.8
    TRAILING_ACTIVATE_PCT = 0.6

    MAX_HOLD_FLAT_BAND_PCT = 0.5
    MAX_HOLD_HOURS = 8.0
    MAX_HOLD_FORCED_HOURS = 12.0

    CB_MAX_CONSECUTIVE_LOSSES = 5
    CB_DAILY_LOSS_LIMIT_PCT = -4.0
    CB_MAX_DRAWDOWN_PCT = -8.0
    CB_COOLDOWN_MINUTES = 60

    def initialize(self) -> None:
        self.set_start_date(2021, 6, 1)
        self.set_end_date(2026, 8, 1)
        self.set_cash(10000)
        self.set_warm_up(timedelta(days=10))

        self.crypto_symbols = {}
        for ticker in self.SYMBOLS:
            crypto = self.add_crypto(ticker, Resolution.HOUR, Market.GDAX)
            self.crypto_symbols[ticker] = crypto.symbol

        self.positions = {}       # symbol -> dict
        self.consecutive_losses = 0
        self.paused_until = None  # datetime, cleared once cooldown elapses
        self.day_start_equity = self.portfolio.total_portfolio_value
        self.peak_equity = self.portfolio.total_portfolio_value
        self.current_day = self.time.date()

        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(hours=1)),
            self._trade_cycle,
        )

    def _reset_daily_if_needed(self) -> None:
        if self.time.date() != self.current_day:
            self.current_day = self.time.date()
            self.day_start_equity = self.portfolio.total_portfolio_value

    def _is_paused(self) -> bool:
        """Entry-gate check: only the pause timer + day-scoped daily limit."""
        equity = self.portfolio.total_portfolio_value
        if equity > self.peak_equity:
            self.peak_equity = equity
        daily_pnl_pct = (equity / self.day_start_equity - 1.0) * 100.0 if self.day_start_equity else 0.0
        if daily_pnl_pct <= self.CB_DAILY_LOSS_LIMIT_PCT:
            return True
        return self.paused_until is not None and self.time < self.paused_until

    def _check_trip(self) -> None:
        """Evaluate trip conditions once at trade-close time, matching the live
        bot's CircuitBreaker.record_trade() -- arms a cooldown pause, does NOT
        permanently block."""
        equity = self.portfolio.total_portfolio_value
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown_pct = (equity / self.peak_equity - 1.0) * 100.0 if self.peak_equity else 0.0
        if self.consecutive_losses >= self.CB_MAX_CONSECUTIVE_LOSSES or \
                drawdown_pct <= self.CB_MAX_DRAWDOWN_PCT:
            self.paused_until = self.time + timedelta(minutes=self.CB_COOLDOWN_MINUTES)

    def _trade_cycle(self) -> None:
        if self.is_warming_up:
            return
        self._reset_daily_if_needed()
        self._check_exits()
        if not self._is_paused():
            self._scan_entries()

    def _check_exits(self) -> None:
        for symbol, pos in list(self.positions.items()):
            security = self.securities[symbol]
            price = security.price
            if price <= 0:
                continue
            entry = pos["entry_price"]
            pnl_pct = (price / entry - 1.0) * 100.0

            if pnl_pct > pos["max_pnl_pct"]:
                pos["max_pnl_pct"] = pnl_pct

            hours_held = (self.time - pos["entry_time"]).total_seconds() / 3600.0

            exit_reason = None
            if pnl_pct <= self.STOP_LOSS_PCT:
                exit_reason = "STOP_LOSS"
            elif pnl_pct >= self.TAKE_PROFIT_PCT:
                exit_reason = "TAKE_PROFIT"
            elif pos["max_pnl_pct"] >= self.TRAILING_ACTIVATE_PCT and \
                    (pos["max_pnl_pct"] - pnl_pct) >= self.TRAILING_STOP_PCT:
                exit_reason = "TRAILING_STOP"
            elif hours_held >= self.MAX_HOLD_FORCED_HOURS:
                exit_reason = "MAX_HOLD_FORCED"
            elif hours_held >= self.MAX_HOLD_HOURS and abs(pnl_pct) < self.MAX_HOLD_FLAT_BAND_PCT:
                exit_reason = "MAX_HOLD_FLAT"

            if exit_reason:
                self.liquidate(symbol)
                if pnl_pct < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
                del self.positions[symbol]
                self._check_trip()

    def _scan_entries(self) -> None:
        if len(self.positions) >= self.MAX_POSITIONS:
            return
        for ticker, symbol in self.crypto_symbols.items():
            if symbol in self.positions:
                continue
            if len(self.positions) >= self.MAX_POSITIONS:
                return

            history = self.history(symbol, self.TREND_LOOKBACK + 40, Resolution.HOUR)
            if history.empty or "close" not in history.columns:
                continue
            closes = history["close"].values
            if len(closes) < self.TREND_LOOKBACK + 20:
                continue

            trend, slope = _trend(closes, self.TREND_LOOKBACK, self.MIN_TREND_SLOPE)
            if trend != "UP":
                continue

            rsi = _rsi(closes, 14)
            if rsi > self.MAX_RSI_LONG:
                continue

            macd_hist = _macd_histogram(closes)
            sma20 = float(np.mean(closes[-20:]))
            score = compute_rule_score(rsi, trend, slope, macd_hist, closes[-1], sma20)
            if score < self.MIN_RULE_SCORE:
                continue

            price = self.securities[symbol].price
            if price <= 0:
                continue

            target_value = self.portfolio.total_portfolio_value * self.MAX_POSITION_PCT
            quantity = target_value / price
            self.market_order(symbol, quantity)
            self.positions[symbol] = {
                "entry_price": price,
                "entry_time": self.time,
                "max_pnl_pct": 0.0,
            }
