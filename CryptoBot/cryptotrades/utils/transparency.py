"""
Transparency Module
Tracks performance, exit reasons, and signal strength for complete visibility
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


class TransparencyTracker:
    """Tracks all trade details for full transparency"""
    
    def __init__(self, state_dir="data/state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.trades_file = self.state_dir / "trade_transparency.json"
        self.symbol_stats_file = self.state_dir / "symbol_performance.json"
        self.daily_summary_file = self.state_dir / "daily_summary.json"
        
        self.trades = self._load_trades()
        self.symbol_stats = self._load_symbol_stats()
    
    def _load_trades(self):
        """Load trade history"""
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _load_symbol_stats(self):
        """Load per-symbol statistics"""
        if self.symbol_stats_file.exists():
            try:
                with open(self.symbol_stats_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_trades(self):
        """Save trade history"""
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def _save_symbol_stats(self):
        """Save symbol statistics"""
        with open(self.symbol_stats_file, 'w') as f:
            json.dump(self.symbol_stats, f, indent=2)
    
    def log_entry(self, symbol, direction, entry_price, size, signal_data, costs=None):
        """
        Log a trade entry with signal details
        
        Args:
            symbol: Trading pair (BTC/USD, ETH/USD, etc.)
            direction: LONG or SHORT
            entry_price: Entry price
            size: Position size in USD
            signal_data: Dict with {
                'confidence': float,
                'volatility': float,
                'trend': str,
                'correlation': float,
                'regime': str,
                'multiplier': int (1, 2, or 3),
                'score': int (0-6)
            }
        """
        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_time': datetime.now().isoformat(),
            'entry_price': entry_price,
            'size': size,
            'signal': signal_data,
            'costs': costs or {},
            'status': 'OPEN'
        }
        
        self.trades.append(trade)
        self._save_trades()
        
        return len(self.trades) - 1  # Return trade ID
    
    def log_exit(self, symbol, exit_price, exit_reason, pnl_usd, pnl_pct, costs=None):
        """
        Log a trade exit
        
        Args:
            symbol: Trading pair
            exit_price: Exit price
            exit_reason: TAKE_PROFIT, STOP_LOSS, TRAILING_STOP, ML_SIGNAL, MAX_HOLD
            pnl_usd: Profit/loss in USD
            pnl_pct: Profit/loss percentage
        """
        # Find the most recent open trade for this symbol
        for trade in reversed(self.trades):
            if trade['symbol'] == symbol and trade.get('status') == 'OPEN':
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = exit_price
                trade['exit_reason'] = exit_reason
                trade['pnl_usd'] = pnl_usd
                trade['pnl_pct'] = pnl_pct
                if costs:
                    # Merge so earlier entry costs aren't lost
                    merged = dict(trade.get('costs') or {})
                    merged.update(costs)
                    trade['costs'] = merged
                trade['status'] = 'CLOSED'
                
                # Calculate hold time
                entry_time = datetime.fromisoformat(trade['entry_time'])
                exit_time = datetime.fromisoformat(trade['exit_time'])
                hold_minutes = (exit_time - entry_time).total_seconds() / 60
                trade['hold_minutes'] = round(hold_minutes, 1)
                
                self._save_trades()
                self._update_symbol_stats(symbol, trade)
                break
    
    def _update_symbol_stats(self, symbol, trade):
        """Update per-symbol performance statistics"""
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl_usd': 0,
                'total_pnl_pct': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'win_rate': 0,
                'best_trade_pct': 0,
                'worst_trade_pct': 0,
                'avg_hold_minutes': 0,
                'exit_reasons': {}
            }
        
        stats = self.symbol_stats[symbol]
        stats['total_trades'] += 1
        stats['total_pnl_usd'] += trade['pnl_usd']
        stats['total_pnl_pct'] += trade['pnl_pct']
        
        # Track win/loss (zero is neutral)
        if trade['pnl_usd'] > 0:
            stats['wins'] += 1
        elif trade['pnl_usd'] < 0:
            stats['losses'] += 1
        
        # Update win rate
        stats['win_rate'] = (stats['wins'] / stats['total_trades']) * 100

        # Track entry sources (for ML/RL credibility audits)
        entry_source = (trade.get('signal') or {}).get('decision_source')
        if entry_source:
            if 'entry_sources' not in stats:
                stats['entry_sources'] = {}
            stats['entry_sources'][entry_source] = stats['entry_sources'].get(entry_source, 0) + 1
        
        # Track exit reasons
        reason = trade['exit_reason']
        stats['exit_reasons'][reason] = stats['exit_reasons'].get(reason, 0) + 1
        
        # Update best/worst
        if trade['pnl_pct'] > stats['best_trade_pct']:
            stats['best_trade_pct'] = trade['pnl_pct']
        if trade['pnl_pct'] < stats['worst_trade_pct']:
            stats['worst_trade_pct'] = trade['pnl_pct']
        
        # Update averages
        win_trades = [t for t in self.trades if t['symbol'] == symbol and t.get('pnl_usd', 0) > 0 and t.get('status') == 'CLOSED']
        loss_trades = [t for t in self.trades if t['symbol'] == symbol and t.get('pnl_usd', 0) <= 0 and t.get('status') == 'CLOSED']
        
        if win_trades:
            stats['avg_win_pct'] = sum(t['pnl_pct'] for t in win_trades) / len(win_trades)
        if loss_trades:
            stats['avg_loss_pct'] = sum(t['pnl_pct'] for t in loss_trades) / len(loss_trades)
        
        # Average hold time
        closed_trades = [t for t in self.trades if t['symbol'] == symbol and t.get('status') == 'CLOSED']
        if closed_trades:
            stats['avg_hold_minutes'] = sum(t.get('hold_minutes', 0) for t in closed_trades) / len(closed_trades)
        
        self._save_symbol_stats()
    
    def get_daily_summary(self):
        """Generate daily performance summary"""
        today = datetime.now().date()
        today_trades = [
            t for t in self.trades 
            if t.get('status') == 'CLOSED' and 
            datetime.fromisoformat(t['exit_time']).date() == today
        ]
        
        if not today_trades:
            return {
                'date': today.isoformat(),
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl_usd': 0,
                'win_rate': 0,
                'best_trade': None,
                'worst_trade': None
            }
        
        wins = [t for t in today_trades if t['pnl_usd'] > 0]
        losses = [t for t in today_trades if t['pnl_usd'] <= 0]
        
        best_trade = max(today_trades, key=lambda t: t['pnl_pct'])
        worst_trade = min(today_trades, key=lambda t: t['pnl_pct'])
        
        summary = {
            'date': today.isoformat(),
            'total_trades': len(today_trades),
            'wins': len(wins),
            'losses': len(losses),
            'total_pnl_usd': sum(t['pnl_usd'] for t in today_trades),
            'total_pnl_pct': sum(t['pnl_pct'] for t in today_trades),
            'win_rate': (len(wins) / len(today_trades)) * 100 if today_trades else 0,
            'avg_win': sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0,
            'best_trade': {
                'symbol': best_trade['symbol'],
                'pnl_pct': best_trade['pnl_pct'],
                'pnl_usd': best_trade['pnl_usd'],
                'exit_reason': best_trade['exit_reason']
            },
            'worst_trade': {
                'symbol': worst_trade['symbol'],
                'pnl_pct': worst_trade['pnl_pct'],
                'pnl_usd': worst_trade['pnl_usd'],
                'exit_reason': worst_trade['exit_reason']
            },
            'exit_reasons': {}
        }
        
        # Count exit reasons
        for trade in today_trades:
            reason = trade['exit_reason']
            summary['exit_reasons'][reason] = summary['exit_reasons'].get(reason, 0) + 1
        
        # Save daily summary
        with open(self.daily_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def get_symbol_performance(self, symbol=None):
        """Get performance stats for a symbol or all symbols"""
        if symbol:
            return self.symbol_stats.get(symbol, {})
        return self.symbol_stats
    
    def get_recent_trades(self, count=10):
        """Get most recent closed trades"""
        closed = [t for t in self.trades if t.get('status') == 'CLOSED']
        return sorted(closed, key=lambda t: t['exit_time'], reverse=True)[:count]
    
    def get_open_trades(self):
        """Get all open trades"""
        return [t for t in self.trades if t.get('status') == 'OPEN']
