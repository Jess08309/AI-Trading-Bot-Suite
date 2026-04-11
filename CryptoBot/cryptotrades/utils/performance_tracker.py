"""
Track trading performance metrics and calculate risk-adjusted returns.

Metrics:
- Sharpe Ratio (risk-adjusted returns vs risk-free rate)
- Sortino Ratio (penalizes only downside volatility)
- Max Drawdown (peak-to-trough watermark)
- Profit Factor (gross profit / gross loss)
- Win Rate, Total Return
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import os
import math


class PerformanceTracker:
    """Track trades and calculate performance metrics."""

    def __init__(self, initial_balance: float = 5000.0):
        self.initial_balance = initial_balance
        self.trades: List[Dict] = []
        self.daily_returns: Dict[str, float] = {}
        self.current_date = None
        self.daily_balance = initial_balance
        # Watermark tracking for proper drawdown
        self.balance_history: List[float] = [initial_balance]
        self.peak_balance: float = initial_balance

    def log_trade(self, symbol: str, side: str, price: float, amount: float,
                  usd_balance: float, asset_balance: float, pnl: Optional[float] = None):
        """Log a trade execution."""
        now = datetime.now(timezone.utc)
        trade = {
            "timestamp": now.isoformat(),
            "symbol": symbol,
            "side": side,
            "price": price,
            "amount": amount,
            "usd_balance": usd_balance,
            "asset_balance": asset_balance,
            "pnl": pnl,
        }
        self.trades.append(trade)

        today = now.strftime("%Y-%m-%d")
        if today != self.current_date:
            self.current_date = today
            self.daily_returns[today] = 0.0

        # Track balance for drawdown
        if usd_balance > 0:
            self.balance_history.append(usd_balance)
            if usd_balance > self.peak_balance:
                self.peak_balance = usd_balance

    def update_pnl(self, pnl: float):
        """Update daily P&L."""
        if self.current_date:
            self.daily_returns[self.current_date] = (
                self.daily_returns.get(self.current_date, 0.0) + pnl
            )

    def update_balance(self, balance: float):
        """Update balance for drawdown tracking."""
        self.balance_history.append(balance)
        if balance > self.peak_balance:
            self.peak_balance = balance

    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe Ratio.
        Formula: (mean_daily_return - daily_rf) / std_dev * sqrt(252)
        """
        if not self.daily_returns or len(self.daily_returns) < 2:
            return 0.0

        returns = list(self.daily_returns.values())
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return 0.0

        annual_return = mean_return * 252
        annual_std = std_dev * (252 ** 0.5)
        sharpe = (annual_return - risk_free_rate) / annual_std
        return float(sharpe)

    def get_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino Ratio -- like Sharpe but only penalizes downside.
        Better for strategies that have asymmetric return profiles.
        Formula: (mean_return - rf) / downside_deviation
        """
        if not self.daily_returns or len(self.daily_returns) < 2:
            return 0.0

        returns = list(self.daily_returns.values())
        mean_return = sum(returns) / len(returns)

        # Downside deviation: std of only negative returns
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf') if mean_return > 0 else 0.0

        downside_variance = sum(r ** 2 for r in negative_returns) / len(returns)
        downside_dev = math.sqrt(downside_variance)

        if downside_dev == 0:
            return 0.0

        annual_return = mean_return * 252
        annual_downside = downside_dev * (252 ** 0.5)
        sortino = (annual_return - risk_free_rate) / annual_downside
        return float(sortino)

    def get_profit_factor(self) -> float:
        """
        Gross profit / Gross loss.
        > 1.0 = profitable, > 1.5 = good, > 2.0 = excellent
        """
        gross_profit = 0.0
        gross_loss = 0.0

        for t in self.trades:
            pnl = t.get("pnl")
            if pnl is None:
                continue
            if pnl > 0:
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += abs(pnl)

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def get_win_rate(self) -> float:
        """Calculate % of profitable trades (only counts trades with P&L)."""
        trades_with_pnl = [t for t in self.trades if t.get("pnl") is not None
                           and t.get("pnl") != 0.0]
        if not trades_with_pnl:
            return 0.0
        profitable = sum(1 for t in trades_with_pnl if t["pnl"] > 0)
        return (profitable / len(trades_with_pnl) * 100)

    def get_total_return(self) -> float:
        """Calculate total return %."""
        total_pnl = sum(
            t.get("pnl", 0.0) or 0.0
            for t in self.trades
        )
        if self.initial_balance > 0:
            return (total_pnl / self.initial_balance * 100)
        return 0.0

    def get_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown % using proper watermark method.
        Tracks peak-to-trough, not just global max/min.
        """
        if len(self.balance_history) < 2:
            return 0.0

        peak = self.balance_history[0]
        max_drawdown = 0.0

        for balance in self.balance_history:
            if balance > peak:
                peak = balance
            drawdown = ((balance - peak) / peak * 100) if peak > 0 else 0.0
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        return float(max_drawdown)

    def get_avg_trade_return(self) -> float:
        """Average return per trade (%)."""
        trades_with_pnl = [t for t in self.trades if t.get("pnl") is not None]
        if not trades_with_pnl:
            return 0.0
        return sum(t["pnl"] for t in trades_with_pnl) / len(trades_with_pnl)

    def get_expectancy(self) -> float:
        """
        Expected value per trade = (win_rate * avg_win) - (loss_rate * avg_loss).
        Positive = profitable system.
        """
        wins = [t["pnl"] for t in self.trades if t.get("pnl") and t["pnl"] > 0]
        losses = [t["pnl"] for t in self.trades if t.get("pnl") and t["pnl"] < 0]

        if not wins and not losses:
            return 0.0

        total = len(wins) + len(losses)
        win_rate = len(wins) / total
        loss_rate = len(losses) / total

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

        return (win_rate * avg_win) - (loss_rate * avg_loss)

    def get_full_report(self) -> Dict:
        """Get comprehensive performance report."""
        return {
            "total_trades": len(self.trades),
            "win_rate": round(self.get_win_rate(), 1),
            "total_return_pct": round(self.get_total_return(), 2),
            "max_drawdown_pct": round(self.get_max_drawdown(), 2),
            "sharpe_ratio": round(self.get_sharpe_ratio(), 2),
            "sortino_ratio": round(self.get_sortino_ratio(), 2),
            "profit_factor": round(self.get_profit_factor(), 2),
            "avg_trade_return": round(self.get_avg_trade_return(), 3),
            "expectancy": round(self.get_expectancy(), 3),
            "peak_balance": round(self.peak_balance, 2),
            "daily_returns_count": len(self.daily_returns),
        }

    def export_trades(self, filename: str = "detailed_trades.json"):
        """Export trades to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.trades, f, indent=2)
