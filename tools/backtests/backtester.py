"""
Backtester Module - Replay trading strategy on historical price data.

Tests the dual-timeframe strategy:
- 1-minute risk monitoring (stop-loss, trailing-stop, take-profit)
- 10-minute candle trade decisions (ML scoring, signal exits)

Reports: total return, win rate, max drawdown, Sharpe ratio, trade count.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random

try:
    from .config import config as live_config
    from .execution_model import execution_price, sample_fill_ratio, estimate_funding_cost
except Exception:  # pragma: no cover
    try:
        from utils.config import config as live_config
        from utils.execution_model import execution_price, sample_fill_ratio, estimate_funding_cost
    except Exception:  # pragma: no cover
        live_config = None

logger = logging.getLogger("backtester")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class BacktestTrade:
    """Record of a single completed trade."""
    symbol: str
    side: str           # 'spot' or 'futures'
    direction: str      # 'long' or 'short'
    entry_price: float
    exit_price: float
    entry_idx: int      # index into price series
    exit_idx: int
    amount: float       # units of asset
    cost: float         # USD cost at entry
    pnl_pct: float      # percent gain/loss
    pnl_usd: float      # USD gain/loss
    exit_reason: str     # TAKE_PROFIT, STOP_LOSS, TRAILING_STOP, RSI_OVERBOUGHT, ENSEMBLE_BEARISH, END_OF_DATA


@dataclass
class BacktestResult:
    """Summary of a backtest run."""
    symbol: str
    side: str
    total_return_pct: float
    total_return_usd: float
    starting_balance: float
    ending_balance: float
    num_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float   # in candle periods
    longest_win_streak: int
    longest_lose_streak: int
    trades: List[BacktestTrade] = field(default_factory=list)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"BACKTEST RESULT: {self.symbol} ({self.side})",
            f"{'='*60}",
            f"  Starting Balance:  ${self.starting_balance:,.2f}",
            f"  Ending Balance:    ${self.ending_balance:,.2f}",
            f"  Total Return:      {self.total_return_pct:+.2f}% (${self.total_return_usd:+,.2f})",
            f"  Trades:            {self.num_trades}",
            f"  Win Rate:          {self.win_rate:.1%}",
            f"  Avg Win:           {self.avg_win_pct:+.2f}%",
            f"  Avg Loss:          {self.avg_loss_pct:+.2f}%",
            f"  Max Drawdown:      {self.max_drawdown_pct:.2f}%",
            f"  Sharpe Ratio:      {self.sharpe_ratio:.3f}",
            f"  Profit Factor:     {self.profit_factor:.2f}",
            f"  Avg Hold Duration: {self.avg_trade_duration:.1f} candles",
            f"  Best Streak:       {self.longest_win_streak} wins",
            f"  Worst Streak:      {self.longest_lose_streak} losses",
        ]

        # Trade breakdown by exit reason
        if self.trades:
            reasons = {}
            for t in self.trades:
                r = t.exit_reason
                if r not in reasons:
                    reasons[r] = {"count": 0, "total_pnl": 0.0}
                reasons[r]["count"] += 1
                reasons[r]["total_pnl"] += t.pnl_pct

            lines.append(f"\n  Exit Breakdown:")
            for reason, data in sorted(reasons.items()):
                avg_pnl = data["total_pnl"] / data["count"]
                lines.append(f"    {reason:20s}: {data['count']:3d} trades, "
                             f"avg {avg_pnl:+.2f}%")

        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ============================================================
# CANDLE AGGREGATION (mirrors trading_engine.py)
# ============================================================

def aggregate_to_candles(one_min_prices: List[float], candle_size: int = 10) -> List[float]:
    """Aggregate 1-minute close prices into N-minute candle closes."""
    if len(one_min_prices) < candle_size:
        return list(one_min_prices)

    candles = []
    for i in range(candle_size - 1, len(one_min_prices), candle_size):
        candles.append(one_min_prices[i])
    return candles


# ============================================================
# SPOT BACKTESTER
# ============================================================

class SpotBacktester:
    """Backtest the spot trading strategy on historical price data.

    Uses the same logic as trading_engine.py v3.0:
    - ML prediction on 10-min candle aggregated prices
    - Buy scoring system (max 15 points, threshold >= 4)
    - Risk monitoring: stop-loss, trailing-stop, take-profit
    - Signal exits: RSI overbought, ensemble bearish
    """

    def __init__(
        self,
        market_predictor,
        candle_size: int = 10,
        starting_balance: float = 2500.0,
        # Risk parameters (checked every 1-min bar) - v3.1 tuned
        take_profit_high: float = 2.0,    # TP when high confidence (was 3.0)
        take_profit_low: float = 1.5,     # TP when normal confidence (was 2.0)
        stop_loss_high: float = -3.0,     # SL when high confidence (was -2.0)
        stop_loss_low: float = -2.5,      # SL when normal confidence (was -1.5)
        trailing_stop: float = 0.5,       # trailing drawdown from max (was 0.8)
        trailing_activate: float = 0.3,   # profit needed to activate trailing (was 0.5)
        # Trade decision parameters - v3.1 tuned
        buy_score_threshold: int = 6,     # was 4
        ensemble_buy_threshold: float = 0.55,  # was 0.52
        ensemble_sell_threshold: float = 0.42,  # was 0.40
        rsi_overbought: float = 80.0,
        # Position sizing
        position_pct: float = 0.15,       # 15% of balance per trade
        max_positions: int = 5,           # max simultaneous positions
        # Sentiment (constant for backtest, no live sentiment)
        default_sentiment: float = 0.0,

        # Execution realism (defaults pulled from live config when available)
        enable_execution_costs: Optional[bool] = None,
        spot_slippage_bps: Optional[float] = None,
        futures_slippage_bps: Optional[float] = None,
        spot_fee_rate: Optional[float] = None,
        futures_fee_rate: Optional[float] = None,
        enable_partial_fills: Optional[bool] = None,
        partial_fill_prob: Optional[float] = None,
        partial_fill_min: Optional[float] = None,
        partial_fill_max: Optional[float] = None,
        enable_funding_costs: Optional[bool] = None,
        futures_funding_rate_per_8h: Optional[float] = None,
        execution_seed: int = 1337,
    ):
        self.predictor = market_predictor
        self.candle_size = candle_size
        self.starting_balance = starting_balance
        self.take_profit_high = take_profit_high
        self.take_profit_low = take_profit_low
        self.stop_loss_high = stop_loss_high
        self.stop_loss_low = stop_loss_low
        self.trailing_stop = trailing_stop
        self.trailing_activate = trailing_activate
        self.buy_score_threshold = buy_score_threshold
        self.ensemble_buy_threshold = ensemble_buy_threshold
        self.ensemble_sell_threshold = ensemble_sell_threshold
        self.rsi_overbought = rsi_overbought
        self.position_pct = position_pct
        self.max_positions = max_positions
        self.default_sentiment = default_sentiment

        self.enable_execution_costs = (
            bool(live_config.ENABLE_EXECUTION_COSTS)
            if enable_execution_costs is None and live_config is not None
            else bool(enable_execution_costs) if enable_execution_costs is not None else False
        )
        self.spot_slippage_bps = (
            float(live_config.SPOT_SLIPPAGE_BPS)
            if spot_slippage_bps is None and live_config is not None
            else float(spot_slippage_bps) if spot_slippage_bps is not None else 0.0
        )
        self.futures_slippage_bps = (
            float(live_config.FUTURES_SLIPPAGE_BPS)
            if futures_slippage_bps is None and live_config is not None
            else float(futures_slippage_bps) if futures_slippage_bps is not None else 0.0
        )
        self.spot_fee_rate = (
            float(live_config.SPOT_FEE_RATE)
            if spot_fee_rate is None and live_config is not None
            else float(spot_fee_rate) if spot_fee_rate is not None else 0.0
        )
        self.futures_fee_rate = (
            float(live_config.FUTURES_FEE_RATE)
            if futures_fee_rate is None and live_config is not None
            else float(futures_fee_rate) if futures_fee_rate is not None else 0.0
        )
        self.enable_partial_fills = (
            bool(live_config.ENABLE_PARTIAL_FILLS)
            if enable_partial_fills is None and live_config is not None
            else bool(enable_partial_fills) if enable_partial_fills is not None else False
        )
        self.partial_fill_prob = (
            float(live_config.PARTIAL_FILL_PROB)
            if partial_fill_prob is None and live_config is not None
            else float(partial_fill_prob) if partial_fill_prob is not None else 0.0
        )
        self.partial_fill_min = (
            float(live_config.PARTIAL_FILL_MIN)
            if partial_fill_min is None and live_config is not None
            else float(partial_fill_min) if partial_fill_min is not None else 1.0
        )
        self.partial_fill_max = (
            float(live_config.PARTIAL_FILL_MAX)
            if partial_fill_max is None and live_config is not None
            else float(partial_fill_max) if partial_fill_max is not None else 1.0
        )
        self.enable_funding_costs = (
            bool(live_config.ENABLE_FUNDING_COSTS)
            if enable_funding_costs is None and live_config is not None
            else bool(enable_funding_costs) if enable_funding_costs is not None else False
        )
        self.futures_funding_rate_per_8h = (
            float(live_config.FUTURES_FUNDING_RATE_PER_8H)
            if futures_funding_rate_per_8h is None and live_config is not None
            else float(futures_funding_rate_per_8h) if futures_funding_rate_per_8h is not None else 0.0
        )
        self.exec_rng = random.Random(int(execution_seed))

    def _compute_ensemble(self, ml_confidence: float, sentiment: float = 0.0) -> float:
        """Simple ensemble: ML 55% + Sentiment 45% (mirrors meta_learner)."""
        sentiment_score = (sentiment + 1.0) / 2.0  # normalize -1..1 to 0..1
        return 0.55 * ml_confidence + 0.45 * sentiment_score

    def _compute_buy_score(self, prediction: dict, ensemble_pred: float,
                           sentiment: float) -> int:
        """Score a buy opportunity, matching engine logic. Max 15 points."""
        score = 0

        # 1. Ensemble direction (0-4)
        if ensemble_pred > 0.65:
            score += 4
        elif ensemble_pred > 0.55:
            score += 2
        elif ensemble_pred > 0.50:
            score += 1

        # 2. RSI zone (0-3)
        rsi = prediction.get("rsi", 50.0)
        if rsi < 30:
            score += 3
        elif rsi < 40:
            score += 2
        elif rsi < 50:
            score += 1

        # 3. MACD crossover (0-2)
        macd = prediction.get("macd_crossover", 0.0)
        if macd > 0:
            score += 2

        # 4. Bollinger Band position (0-2)
        bb = prediction.get("bb_position", 0.5)
        if bb < 0.2:
            score += 2
        elif bb < 0.4:
            score += 1

        # 5. Sentiment (0-2)
        if sentiment > 0.2:
            score += 2
        elif sentiment > 0.0:
            score += 1

        # 6. Mean reversion (0-2)
        mr = prediction.get("mean_reversion", 0.0)
        if mr < -1.5:
            score += 2
        elif mr < -0.5:
            score += 1

        return score

    def run(self, symbol: str, one_min_prices: List[float],
            verbose: bool = False) -> BacktestResult:
        """Run backtest on a single symbol's 1-minute price series.

        Args:
            symbol: Symbol name (for reporting)
            one_min_prices: List of 1-minute close prices (oldest first)
            verbose: Print trade-by-trade detail

        Returns:
            BacktestResult with full performance metrics
        """
        balance = self.starting_balance
        peak_balance = balance
        max_drawdown = 0.0
        trades: List[BacktestTrade] = []
        returns_per_trade: List[float] = []

        # Active position (single position per symbol in backtest)
        position = None  # {entry_price, amount, cost, max_price, entry_idx, confidence}

        # We need enough history for ML (min 15 candles * candle_size)
        min_history = max(15 * self.candle_size, 150)

        if len(one_min_prices) < min_history + 50:
            logger.warning(f"{symbol}: Not enough data ({len(one_min_prices)} bars, "
                           f"need {min_history + 50})")
            return self._empty_result(symbol, "spot")

        if verbose:
            print(f"\nBacktesting {symbol} on {len(one_min_prices)} 1-min bars "
                  f"({len(one_min_prices) / 60:.1f} hours)")

        # Walk through each 1-minute bar
        for i in range(min_history, len(one_min_prices)):
            price = one_min_prices[i]
            is_trade_cycle = (i % self.candle_size == 0)

            # ----------------------------------------
            # RISK MONITORING (every 1-min bar)
            # ----------------------------------------
            if position is not None:
                entry_price = position["entry_price"]
                max_price = position["max_price"]

                impact_bps = position.get(
                    "impact_bps",
                    self.spot_slippage_bps if self.enable_execution_costs else 0.0,
                )
                exit_exec_price = (
                    execution_price(price, "SELL", impact_bps)
                    if (self.enable_execution_costs and impact_bps)
                    else price
                )
                proceeds = position["amount"] * exit_exec_price
                exit_fee = proceeds * self.spot_fee_rate
                pnl_usd = proceeds - exit_fee - position["cost"]
                profit_pct = (pnl_usd / max(position["cost"], 1e-12)) * 100

                # Update trailing max
                if price > max_price:
                    position["max_price"] = price
                    max_price = price

                drawdown_from_max = ((max_price - price) / max_price) * 100
                confidence = position.get("confidence", 0.5)

                # Dynamic thresholds
                tp = self.take_profit_high if confidence > 0.6 else self.take_profit_low
                sl = self.stop_loss_high if confidence > 0.6 else self.stop_loss_low

                exit_reason = None
                if profit_pct > tp:
                    exit_reason = "TAKE_PROFIT"
                elif profit_pct < sl:
                    exit_reason = "STOP_LOSS"
                elif (drawdown_from_max > self.trailing_stop and
                      profit_pct > self.trailing_activate):
                    exit_reason = "TRAILING_STOP"

                if exit_reason:
                    trade = self._close_position(
                        position, price, i, exit_reason, symbol, "spot", "long")
                    trades.append(trade)
                    returns_per_trade.append(trade.pnl_pct)
                    balance += trade.pnl_usd + trade.cost
                    if verbose:
                        print(f"  [{i}] SELL {exit_reason}: "
                              f"${entry_price:.2f} -> ${price:.2f} "
                              f"({trade.pnl_pct:+.2f}%)")
                    position = None

            # ----------------------------------------
            # TRADE DECISIONS (every N-min candle)
            # ----------------------------------------
            if is_trade_cycle:
                # Build candle series from history up to this point
                history = one_min_prices[:i + 1]
                candle_prices = aggregate_to_candles(history, self.candle_size)

                if len(candle_prices) < 15:
                    continue

                # ML prediction on candle data
                prediction = self.predictor.predict(candle_prices)
                ml_confidence = prediction["confidence"]
                ml_rsi = prediction.get("rsi", 50.0)

                # Ensemble
                ensemble_pred = self._compute_ensemble(
                    ml_confidence, self.default_sentiment)

                if position is None:
                    # --- BUY DECISION ---
                    buy_score = self._compute_buy_score(
                        prediction, ensemble_pred, self.default_sentiment)

                    if (buy_score >= self.buy_score_threshold and
                            ensemble_pred >= self.ensemble_buy_threshold):
                        # Open position
                        cost = balance * self.position_pct
                        if cost > 10.0 and balance > cost:
                            impact_bps = self.spot_slippage_bps if self.enable_execution_costs else 0.0
                            entry_exec_price = (
                                execution_price(price, "BUY", impact_bps)
                                if (self.enable_execution_costs and impact_bps)
                                else price
                            )
                            fill_ratio = sample_fill_ratio(
                                self.exec_rng,
                                enabled=self.enable_partial_fills,
                                partial_fill_prob=self.partial_fill_prob,
                                min_ratio=self.partial_fill_min,
                                max_ratio=self.partial_fill_max,
                            )
                            entry_notional = cost * fill_ratio
                            entry_fee = entry_notional * self.spot_fee_rate
                            total_cost = entry_notional + entry_fee
                            if total_cost <= 0 or balance <= total_cost:
                                continue
                            amount = entry_notional / max(entry_exec_price, 1e-12)
                            balance -= total_cost
                            position = {
                                "entry_price": entry_exec_price,
                                "entry_mid_price": price,
                                "amount": amount,
                                "cost": total_cost,
                                "entry_notional": entry_notional,
                                "entry_fee": entry_fee,
                                "impact_bps": impact_bps,
                                "fill_ratio": fill_ratio,
                                "max_price": price,
                                "entry_idx": i,
                                "confidence": ensemble_pred,
                            }
                            if verbose:
                                print(f"  [{i}] BUY score={buy_score}/15 "
                                      f"ens={ensemble_pred:.3f} "
                                      f"@ ${price:.2f} (${cost:.2f})")

                else:
                    # --- SIGNAL-BASED SELL ---
                    entry_price = position["entry_price"]

                    impact_bps = position.get(
                        "impact_bps",
                        self.spot_slippage_bps if self.enable_execution_costs else 0.0,
                    )
                    exit_exec_price = (
                        execution_price(price, "SELL", impact_bps)
                        if (self.enable_execution_costs and impact_bps)
                        else price
                    )
                    proceeds = position["amount"] * exit_exec_price
                    exit_fee = proceeds * self.spot_fee_rate
                    pnl_usd = proceeds - exit_fee - position["cost"]
                    profit_pct = (pnl_usd / max(position["cost"], 1e-12)) * 100

                    exit_reason = None
                    if ml_rsi > self.rsi_overbought and profit_pct > 0:
                        exit_reason = "RSI_OVERBOUGHT"
                    elif (ensemble_pred < self.ensemble_sell_threshold and
                          profit_pct > -0.5):
                        exit_reason = "ENSEMBLE_BEARISH"

                    if exit_reason:
                        trade = self._close_position(
                            position, price, i, exit_reason, symbol, "spot", "long")
                        trades.append(trade)
                        returns_per_trade.append(trade.pnl_pct)
                        balance += trade.pnl_usd + trade.cost
                        if verbose:
                            print(f"  [{i}] SELL {exit_reason}: "
                                  f"${entry_price:.2f} -> ${price:.2f} "
                                  f"({trade.pnl_pct:+.2f}%)")
                        position = None

            # Track drawdown
            current_equity = balance
            if position:
                impact_bps = position.get(
                    "impact_bps",
                    self.spot_slippage_bps if self.enable_execution_costs else 0.0,
                )
                exit_exec_price = (
                    execution_price(price, "SELL", impact_bps)
                    if (self.enable_execution_costs and impact_bps)
                    else price
                )
                proceeds = position["amount"] * exit_exec_price
                exit_fee = proceeds * self.spot_fee_rate
                current_equity += (proceeds - exit_fee)
            peak_balance = max(peak_balance, current_equity)
            dd = ((peak_balance - current_equity) / peak_balance) * 100
            max_drawdown = max(max_drawdown, dd)

        # Close any remaining position at end
        if position:
            final_price = one_min_prices[-1]
            trade = self._close_position(
                position, final_price, len(one_min_prices) - 1,
                "END_OF_DATA", symbol, "spot", "long")
            trades.append(trade)
            returns_per_trade.append(trade.pnl_pct)
            balance += trade.pnl_usd + trade.cost

        return self._build_result(
            symbol, "spot", balance, trades, returns_per_trade, max_drawdown)

    def _close_position(self, position: dict, exit_price: float,
                        exit_idx: int, reason: str, symbol: str,
                        side: str, direction: str) -> BacktestTrade:
        """Create a trade record for closing a position."""
        entry_price = position["entry_price"]
        amount = position["amount"]
        cost = position["cost"]

        impact_bps = position.get(
            "impact_bps",
            self.spot_slippage_bps if self.enable_execution_costs else 0.0,
        )
        exit_exec_price = (
            execution_price(exit_price, "SELL", impact_bps)
            if (self.enable_execution_costs and impact_bps)
            else exit_price
        )
        proceeds = amount * exit_exec_price
        exit_fee = proceeds * self.spot_fee_rate
        pnl_usd = proceeds - exit_fee - cost
        pnl_pct = (pnl_usd / max(cost, 1e-12)) * 100

        return BacktestTrade(
            symbol=symbol,
            side=side,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_exec_price,
            entry_idx=position["entry_idx"],
            exit_idx=exit_idx,
            amount=amount,
            cost=cost,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            exit_reason=reason,
        )

    def _empty_result(self, symbol: str, side: str) -> BacktestResult:
        """Return an empty result for insufficient data."""
        return BacktestResult(
            symbol=symbol, side=side,
            total_return_pct=0.0, total_return_usd=0.0,
            starting_balance=self.starting_balance,
            ending_balance=self.starting_balance,
            num_trades=0, win_rate=0.0,
            avg_win_pct=0.0, avg_loss_pct=0.0,
            max_drawdown_pct=0.0, sharpe_ratio=0.0,
            profit_factor=0.0, avg_trade_duration=0.0,
            longest_win_streak=0, longest_lose_streak=0,
            trades=[],
        )

    def _build_result(self, symbol: str, side: str, ending_balance: float,
                      trades: List[BacktestTrade],
                      returns: List[float],
                      max_drawdown: float) -> BacktestResult:
        """Compute final metrics from trade list."""
        total_return_usd = ending_balance - self.starting_balance
        total_return_pct = (total_return_usd / self.starting_balance) * 100

        if not trades:
            return self._empty_result(symbol, side)

        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        win_rate = len(wins) / len(trades)

        avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0

        # Sharpe ratio (annualized, assuming returns are per-trade)
        if len(returns) >= 2:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std_ret = math.sqrt(variance) if variance > 0 else 1e-6
            # Annualize: assume ~365 * 24 * 6 = 52,560 ten-minute candles per year
            # with avg ~N trades, trades_per_year ≈ (52560 / data_length) * num_trades
            sharpe = (mean_ret / std_ret) * math.sqrt(len(returns))
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 1e-6
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Average trade duration (in bars between entry and exit)
        durations = [t.exit_idx - t.entry_idx for t in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # Streaks
        longest_win = 0
        longest_lose = 0
        current_win = 0
        current_lose = 0
        for t in trades:
            if t.pnl_pct > 0:
                current_win += 1
                current_lose = 0
                longest_win = max(longest_win, current_win)
            else:
                current_lose += 1
                current_win = 0
                longest_lose = max(longest_lose, current_lose)

        return BacktestResult(
            symbol=symbol,
            side=side,
            total_return_pct=total_return_pct,
            total_return_usd=total_return_usd,
            starting_balance=self.starting_balance,
            ending_balance=ending_balance,
            num_trades=len(trades),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_trade_duration=avg_duration,
            longest_win_streak=longest_win,
            longest_lose_streak=longest_lose,
            trades=trades,
        )


# ============================================================
# FUTURES BACKTESTER
# ============================================================

class FuturesBacktester(SpotBacktester):
    """Backtest futures strategy (long AND short).

    Extends SpotBacktester with:
    - Short selling capability
    - Leverage support
    - Separate long/short thresholds matching engine logic
    """

    def __init__(
        self,
        market_predictor,
        leverage: float = 2.0,
        # Futures-specific thresholds (matching trading_engine.py v3.1)
        long_threshold: float = 0.60,    # was 0.55
        short_threshold: float = 0.40,   # was 0.45
        take_profit: float = 4.0,        # was 5.0
        stop_loss: float = -2.5,         # was -3.0
        trailing_stop: float = 0.5,      # was 0.6
        trailing_activate: float = 0.2,  # was 0.3
        **kwargs,
    ):
        super().__init__(market_predictor, **kwargs)
        self.leverage = leverage
        # Futures thresholds (now match live engine defaults)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.futures_take_profit = take_profit
        self.futures_stop_loss = stop_loss
        self.futures_trailing_stop = trailing_stop
        self.futures_trailing_activate = trailing_activate

    def run(self, symbol: str, one_min_prices: List[float],
            verbose: bool = False) -> BacktestResult:
        """Run futures backtest (long and short trades)."""
        balance = self.starting_balance
        peak_balance = balance
        max_drawdown = 0.0
        trades: List[BacktestTrade] = []
        returns_per_trade: List[float] = []

        position = None  # {entry_price, amount, cost, entry_idx, direction, confidence, notional, margin_required, open_fee, impact_bps, fill_ratio}

        min_history = max(15 * self.candle_size, 150)

        if len(one_min_prices) < min_history + 50:
            return self._empty_result(symbol, "futures")

        if verbose:
            print(f"\nBacktesting {symbol} FUTURES (leverage={self.leverage}x) "
                  f"on {len(one_min_prices)} bars")

        for i in range(min_history, len(one_min_prices)):
            price = one_min_prices[i]
            is_trade_cycle = (i % self.candle_size == 0)

            # ----------------------------------------
            # RISK MONITORING (every bar)
            # ----------------------------------------
            if position is not None:
                entry_price = position["entry_price"]
                direction = position["direction"]

                impact_bps = position.get(
                    "impact_bps",
                    self.futures_slippage_bps if self.enable_execution_costs else 0.0,
                )
                close_side = "SELL" if direction == "long" else "BUY"
                exit_exec_price = (
                    execution_price(price, close_side, impact_bps)
                    if (self.enable_execution_costs and impact_bps)
                    else price
                )
                notional = position["notional"]
                margin_required = position["margin_required"]
                open_fee = position["open_fee"]
                close_fee = notional * self.futures_fee_rate
                hold_hours = (i - position["entry_idx"]) / 60.0
                funding_cost = estimate_funding_cost(
                    notional,
                    hold_hours,
                    enabled=self.enable_funding_costs,
                    rate_per_8h=self.futures_funding_rate_per_8h,
                )

                denom = max(entry_price, 1e-12)
                if direction == "long":
                    gross_pnl = (exit_exec_price - entry_price) / denom * notional
                else:
                    gross_pnl = (entry_price - exit_exec_price) / denom * notional

                pnl_usd = gross_pnl - open_fee - close_fee - funding_cost
                profit_pct = (pnl_usd / max(margin_required, 1e-12)) * 100

                max_profit = position.get("max_profit", 0.0)
                if profit_pct > max_profit:
                    position["max_profit"] = profit_pct
                    max_profit = profit_pct

                drawdown_from_peak = max_profit - profit_pct

                exit_reason = None
                if profit_pct > self.futures_take_profit:
                    exit_reason = "TAKE_PROFIT"
                elif profit_pct < self.futures_stop_loss:
                    exit_reason = "STOP_LOSS"
                elif (drawdown_from_peak > self.futures_trailing_stop and
                      profit_pct > self.futures_trailing_activate):
                    exit_reason = "TRAILING_STOP"

                if exit_reason:
                    trade = BacktestTrade(
                        symbol=symbol, side="futures", direction=direction,
                        entry_price=entry_price, exit_price=exit_exec_price,
                        entry_idx=position["entry_idx"], exit_idx=i,
                        amount=position["amount"], cost=position["cost"],
                        pnl_pct=profit_pct, pnl_usd=pnl_usd,
                        exit_reason=exit_reason,
                    )
                    trades.append(trade)
                    returns_per_trade.append(profit_pct)
                    balance += trade.cost + trade.pnl_usd
                    if verbose:
                        dir_str = "LONG" if direction == "long" else "SHORT"
                        print(f"  [{i}] CLOSE {dir_str} {exit_reason}: "
                              f"${entry_price:.2f} -> ${price:.2f} "
                              f"({profit_pct:+.2f}% w/ {self.leverage}x)")
                    position = None

            # ----------------------------------------
            # TRADE DECISIONS (every N candles)
            # ----------------------------------------
            if is_trade_cycle:
                history = one_min_prices[:i + 1]
                candle_prices = aggregate_to_candles(history, self.candle_size)

                if len(candle_prices) < 15:
                    continue

                prediction = self.predictor.predict(candle_prices)
                ml_confidence = prediction["confidence"]
                ml_rsi = prediction.get("rsi", 50.0)

                ensemble_pred = self._compute_ensemble(
                    ml_confidence, self.default_sentiment)

                if position is None:
                    # --- ENTRY DECISION ---
                    margin_target = balance * self.position_pct
                    if margin_target < 10.0:
                        continue

                    impact_bps = self.futures_slippage_bps if self.enable_execution_costs else 0.0
                    fill_ratio = sample_fill_ratio(
                        self.exec_rng,
                        enabled=self.enable_partial_fills,
                        partial_fill_prob=self.partial_fill_prob,
                        min_ratio=self.partial_fill_min,
                        max_ratio=self.partial_fill_max,
                    )
                    margin_required = margin_target * fill_ratio
                    if margin_required <= 0:
                        continue
                    notional = margin_required * self.leverage
                    open_fee = notional * self.futures_fee_rate
                    total_cost = margin_required + open_fee
                    if balance < total_cost:
                        continue

                    if ensemble_pred >= self.long_threshold:
                        # Open LONG
                        entry_exec_price = (
                            execution_price(price, "BUY", impact_bps)
                            if (self.enable_execution_costs and impact_bps)
                            else price
                        )
                        amount = notional / max(entry_exec_price, 1e-12)
                        balance -= total_cost
                        position = {
                            "entry_price": entry_exec_price,
                            "entry_mid_price": price,
                            "amount": amount,
                            "cost": total_cost,
                            "max_profit": 0.0,
                            "entry_idx": i,
                            "direction": "long",
                            "confidence": ensemble_pred,
                            "notional": notional,
                            "margin_required": margin_required,
                            "open_fee": open_fee,
                            "impact_bps": impact_bps,
                            "fill_ratio": fill_ratio,
                        }
                        if verbose:
                            print(f"  [{i}] LONG @ ${price:.2f} "
                                  f"ens={ensemble_pred:.3f} "
                                  f"(${margin_required:.2f} x{self.leverage})")

                    elif ensemble_pred <= self.short_threshold:
                        # Open SHORT
                        entry_exec_price = (
                            execution_price(price, "SELL", impact_bps)
                            if (self.enable_execution_costs and impact_bps)
                            else price
                        )
                        amount = notional / max(entry_exec_price, 1e-12)
                        balance -= total_cost
                        position = {
                            "entry_price": entry_exec_price,
                            "entry_mid_price": price,
                            "amount": amount,
                            "cost": total_cost,
                            "max_profit": 0.0,
                            "entry_idx": i,
                            "direction": "short",
                            "confidence": 1.0 - ensemble_pred,
                            "notional": notional,
                            "margin_required": margin_required,
                            "open_fee": open_fee,
                            "impact_bps": impact_bps,
                            "fill_ratio": fill_ratio,
                        }
                        if verbose:
                            print(f"  [{i}] SHORT @ ${price:.2f} "
                                  f"ens={ensemble_pred:.3f} "
                                  f"(${margin_required:.2f} x{self.leverage})")

                else:
                    # --- SIGNAL EXIT ---
                    direction = position["direction"]
                    entry_price = position["entry_price"]

                    impact_bps = position.get(
                        "impact_bps",
                        self.futures_slippage_bps if self.enable_execution_costs else 0.0,
                    )
                    close_side = "SELL" if direction == "long" else "BUY"
                    exit_exec_price = (
                        execution_price(price, close_side, impact_bps)
                        if (self.enable_execution_costs and impact_bps)
                        else price
                    )
                    notional = position["notional"]
                    margin_required = position["margin_required"]
                    open_fee = position["open_fee"]
                    close_fee = notional * self.futures_fee_rate
                    hold_hours = (i - position["entry_idx"]) / 60.0
                    funding_cost = estimate_funding_cost(
                        notional,
                        hold_hours,
                        enabled=self.enable_funding_costs,
                        rate_per_8h=self.futures_funding_rate_per_8h,
                    )
                    denom = max(entry_price, 1e-12)
                    if direction == "long":
                        gross_pnl = (exit_exec_price - entry_price) / denom * notional
                    else:
                        gross_pnl = (entry_price - exit_exec_price) / denom * notional
                    pnl_usd = gross_pnl - open_fee - close_fee - funding_cost
                    profit_pct = (pnl_usd / max(margin_required, 1e-12)) * 100

                    exit_reason = None
                    if direction == "long":
                        if ml_rsi > self.rsi_overbought and profit_pct > 0:
                            exit_reason = "RSI_OVERBOUGHT"
                        elif ensemble_pred < self.ensemble_sell_threshold:
                            exit_reason = "SIGNAL_REVERSAL"
                    else:
                        if ml_rsi < 20 and profit_pct > 0:
                            exit_reason = "RSI_OVERSOLD"
                        elif ensemble_pred > 0.60:
                            exit_reason = "SIGNAL_REVERSAL"

                    if exit_reason:
                        trade = BacktestTrade(
                            symbol=symbol, side="futures", direction=direction,
                            entry_price=entry_price, exit_price=exit_exec_price,
                            entry_idx=position["entry_idx"], exit_idx=i,
                            amount=position["amount"], cost=position["cost"],
                            pnl_pct=profit_pct, pnl_usd=pnl_usd,
                            exit_reason=exit_reason,
                        )
                        trades.append(trade)
                        returns_per_trade.append(profit_pct)
                        balance += trade.cost + trade.pnl_usd
                        if verbose:
                            dir_str = "LONG" if direction == "long" else "SHORT"
                            print(f"  [{i}] CLOSE {dir_str} {exit_reason}: "
                                  f"${entry_price:.2f} -> ${price:.2f} "
                                  f"({profit_pct:+.2f}%)")
                        position = None

            # Track drawdown
            current_equity = balance
            if position:
                ep = position["entry_price"]
                d = position["direction"]
                impact_bps = position.get(
                    "impact_bps",
                    self.futures_slippage_bps if self.enable_execution_costs else 0.0,
                )
                close_side = "SELL" if d == "long" else "BUY"
                exit_exec_price = (
                    execution_price(price, close_side, impact_bps)
                    if (self.enable_execution_costs and impact_bps)
                    else price
                )
                notional = position["notional"]
                margin_required = position["margin_required"]
                open_fee = position["open_fee"]
                close_fee = notional * self.futures_fee_rate
                hold_hours = (i - position["entry_idx"]) / 60.0
                funding_cost = estimate_funding_cost(
                    notional,
                    hold_hours,
                    enabled=self.enable_funding_costs,
                    rate_per_8h=self.futures_funding_rate_per_8h,
                )
                denom = max(ep, 1e-12)
                if d == "long":
                    gross_pnl = (exit_exec_price - ep) / denom * notional
                else:
                    gross_pnl = (ep - exit_exec_price) / denom * notional
                pnl_usd = gross_pnl - open_fee - close_fee - funding_cost
                current_equity += position["cost"] + pnl_usd
            peak_balance = max(peak_balance, current_equity)
            dd = ((peak_balance - current_equity) / peak_balance) * 100
            max_drawdown = max(max_drawdown, dd)

        # Close remaining position
        if position:
            final_price = one_min_prices[-1]
            direction = position["direction"]
            entry_price = position["entry_price"]

            impact_bps = position.get(
                "impact_bps",
                self.futures_slippage_bps if self.enable_execution_costs else 0.0,
            )
            close_side = "SELL" if direction == "long" else "BUY"
            exit_exec_price = (
                execution_price(final_price, close_side, impact_bps)
                if (self.enable_execution_costs and impact_bps)
                else final_price
            )
            notional = position["notional"]
            margin_required = position["margin_required"]
            open_fee = position["open_fee"]
            close_fee = notional * self.futures_fee_rate
            hold_hours = (len(one_min_prices) - 1 - position["entry_idx"]) / 60.0
            funding_cost = estimate_funding_cost(
                notional,
                hold_hours,
                enabled=self.enable_funding_costs,
                rate_per_8h=self.futures_funding_rate_per_8h,
            )
            denom = max(entry_price, 1e-12)
            if direction == "long":
                gross_pnl = (exit_exec_price - entry_price) / denom * notional
            else:
                gross_pnl = (entry_price - exit_exec_price) / denom * notional
            pnl_usd = gross_pnl - open_fee - close_fee - funding_cost
            profit_pct = (pnl_usd / max(margin_required, 1e-12)) * 100

            trade = BacktestTrade(
                symbol=symbol, side="futures", direction=direction,
                entry_price=entry_price, exit_price=exit_exec_price,
                entry_idx=position["entry_idx"],
                exit_idx=len(one_min_prices) - 1,
                amount=position["amount"], cost=position["cost"],
                pnl_pct=profit_pct, pnl_usd=pnl_usd,
                exit_reason="END_OF_DATA",
            )
            trades.append(trade)
            returns_per_trade.append(profit_pct)
            balance += trade.cost + trade.pnl_usd

        return self._build_result(
            symbol, "futures", balance, trades, returns_per_trade, max_drawdown)


# ============================================================
# MULTI-SYMBOL BACKTEST RUNNER
# ============================================================

def run_full_backtest(
    market_predictor,
    price_data: Dict[str, List[float]],
    futures_data: Dict[str, List[float]] = None,
    candle_size: int = 10,
    starting_balance_spot: float = 2500.0,
    starting_balance_futures: float = 2500.0,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, BacktestResult]:
    """Run backtest across multiple symbols for both spot and futures.

    Args:
        market_predictor: MarketPredictor instance with trained model
        price_data: {symbol: [1-min prices]} for spot
        futures_data: {symbol: [1-min prices]} for futures (optional)
        candle_size: Minutes per candle (default 10)
        starting_balance_spot: Starting USD for spot per-symbol
        starting_balance_futures: Starting USD for futures per-symbol
        verbose: Print trade details
        **kwargs: Additional params passed to backtester constructors

    Returns:
        Dict of {symbol: BacktestResult}
    """
    results = {}

    # Spot backtest
    spot_bt = SpotBacktester(
        market_predictor,
        candle_size=candle_size,
        starting_balance=starting_balance_spot,
        **kwargs,
    )

    for symbol, prices in price_data.items():
        if len(prices) < 200:
            continue
        result = spot_bt.run(symbol, prices, verbose=verbose)
        results[f"SPOT_{symbol}"] = result
        if result.num_trades > 0:
            logger.info(result.summary())

    # Futures backtest
    if futures_data:
        futures_bt = FuturesBacktester(
            market_predictor,
            candle_size=candle_size,
            starting_balance=starting_balance_futures,
            **kwargs,
        )
        for symbol, prices in futures_data.items():
            if len(prices) < 200:
                continue
            result = futures_bt.run(symbol, prices, verbose=verbose)
            results[f"FUTURES_{symbol}"] = result
            if result.num_trades > 0:
                logger.info(result.summary())

    return results


def print_aggregate_summary(results: Dict[str, BacktestResult]):
    """Print a combined summary across all symbols."""
    if not results:
        print("\nNo backtest results to display.")
        return

    total_trades = 0
    total_pnl = 0.0
    total_starting = 0.0
    all_wins = 0
    all_trades = 0
    spot_pnl = 0.0
    futures_pnl = 0.0
    max_dd = 0.0

    print(f"\n{'='*70}")
    print(f"{'AGGREGATE BACKTEST SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Symbol':<25} {'Trades':>7} {'Win%':>7} {'Return%':>9} "
          f"{'Return$':>10} {'MaxDD%':>8} {'Sharpe':>8}")
    print(f"{'-'*70}")

    for key in sorted(results.keys()):
        r = results[key]
        if r.num_trades == 0:
            continue
        print(f"{key:<25} {r.num_trades:>7} {r.win_rate:>6.1%} "
              f"{r.total_return_pct:>+8.2f}% ${r.total_return_usd:>+9.2f} "
              f"{r.max_drawdown_pct:>7.2f}% {r.sharpe_ratio:>+7.3f}")

        total_trades += r.num_trades
        total_pnl += r.total_return_usd
        total_starting += r.starting_balance
        all_wins += int(r.win_rate * r.num_trades)
        all_trades += r.num_trades
        max_dd = max(max_dd, r.max_drawdown_pct)

        if key.startswith("SPOT"):
            spot_pnl += r.total_return_usd
        else:
            futures_pnl += r.total_return_usd

    print(f"{'-'*70}")
    overall_wr = all_wins / all_trades if all_trades > 0 else 0.0
    overall_ret = (total_pnl / total_starting * 100) if total_starting > 0 else 0.0
    print(f"{'TOTAL':<25} {total_trades:>7} {overall_wr:>6.1%} "
          f"{overall_ret:>+8.2f}% ${total_pnl:>+9.2f} "
          f"{max_dd:>7.2f}%")
    print(f"\n  Spot P/L:    ${spot_pnl:+,.2f}")
    print(f"  Futures P/L: ${futures_pnl:+,.2f}")
    print(f"  Combined:    ${total_pnl:+,.2f}")
    print(f"{'='*70}")
