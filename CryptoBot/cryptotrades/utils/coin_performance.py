"""
Coin Performance Tracker
Tracks metrics per coin to enable dynamic portfolio rotation
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class CoinPerformanceTracker:
    def __init__(self, data_file="data/state/coin_performance.json"):
        self.data_file = data_file
        self.performance = self._load_performance()
        
    def _load_performance(self) -> Dict:
        """Load performance data from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_performance(self):
        """Save performance data to JSON file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.performance, f, indent=2)
    
    def record_trade(self, pair: str, action: str, profit_pct: float, timestamp: Optional[str] = None):
        """Record trade outcome for a coin"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
            
        if pair not in self.performance:
            self.performance[pair] = {
                "trades": [],
                "total_profit": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "last_updated": timestamp
            }
        
        # Record trade
        self.performance[pair]["trades"].append({
            "action": action,
            "profit_pct": profit_pct,
            "timestamp": timestamp
        })
        
        # Update stats
        self.performance[pair]["total_profit"] += profit_pct
        # Only count non-zero outcomes as win/loss.
        # (Entry-side bookkeeping like profit_pct=0.0 should not be treated as a loss.)
        if profit_pct > 0:
            self.performance[pair]["win_count"] += 1
        elif profit_pct < 0:
            self.performance[pair]["loss_count"] += 1
        self.performance[pair]["last_updated"] = timestamp
        
        # Keep only last 100 trades per coin
        if len(self.performance[pair]["trades"]) > 100:
            self.performance[pair]["trades"] = self.performance[pair]["trades"][-100:]
        
        self._save_performance()
    
    def record_volatility(self, pair: str, volatility: float, timestamp: Optional[str] = None):
        """Record current volatility measurement for a coin"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
            
        if pair not in self.performance:
            self.performance[pair] = {
                "trades": [],
                "total_profit": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "last_updated": timestamp
            }
        
        if "volatility_samples" not in self.performance[pair]:
            self.performance[pair]["volatility_samples"] = []
        
        self.performance[pair]["volatility_samples"].append({
            "volatility": volatility,
            "timestamp": timestamp
        })
        
        # Keep only last 24 hours of samples
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        self.performance[pair]["volatility_samples"] = [
            s for s in self.performance[pair]["volatility_samples"]
            if s["timestamp"] > cutoff
        ]
        
        self._save_performance()
    
    def get_win_rate(self, pair: str, days: int = 7) -> float:
        """Calculate win rate for a coin over last N days"""
        if pair not in self.performance:
            return 0.5  # Default neutral
        
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        recent_trades = [
            t for t in self.performance[pair]["trades"]
            if t["timestamp"] > cutoff
        ]
        
        if not recent_trades:
            return 0.5
        
        wins = sum(1 for t in recent_trades if t["profit_pct"] > 0)
        return wins / len(recent_trades)
    
    def get_avg_profit(self, pair: str, days: int = 7) -> float:
        """Calculate average profit % per trade over last N days"""
        if pair not in self.performance:
            return 0.0
        
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        recent_trades = [
            t for t in self.performance[pair]["trades"]
            if t["timestamp"] > cutoff
        ]
        
        if not recent_trades:
            return 0.0
        
        return sum(t["profit_pct"] for t in recent_trades) / len(recent_trades)
    
    def get_avg_volatility(self, pair: str, hours: int = 24) -> float:
        """Calculate average volatility over last N hours"""
        if pair not in self.performance or "volatility_samples" not in self.performance[pair]:
            return 0.02  # Default 2%
        
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        recent_samples = [
            s for s in self.performance[pair]["volatility_samples"]
            if s["timestamp"] > cutoff
        ]
        
        if not recent_samples:
            return 0.02
        
        return sum(s["volatility"] for s in recent_samples) / len(recent_samples)
    
    def get_trade_count(self, pair: str, days: int = 7) -> int:
        """Get number of trades for coin in last N days"""
        if pair not in self.performance:
            return 0
        
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        recent_trades = [
            t for t in self.performance[pair]["trades"]
            if t["timestamp"] > cutoff
        ]
        
        return len(recent_trades)
    
    def calculate_coin_score(self, pair: str, sentiment: float = 0.0, ml_confidence: float = 0.5) -> float:
        """
        Calculate composite score for a coin (0-100)
        Higher score = better performance/opportunity
        """
        # Win rate component (0-30 points)
        win_rate = self.get_win_rate(pair, days=7)
        win_score = win_rate * 30
        
        # Average profit component (0-25 points)
        avg_profit = self.get_avg_profit(pair, days=7)
        profit_score = min(max(avg_profit * 500, 0), 25)  # 5% avg = full points
        
        # Volatility component (0-20 points) - higher volatility = more opportunity
        volatility = self.get_avg_volatility(pair, hours=24)
        vol_score = min(volatility * 400, 20)  # 5% vol = full points
        
        # Trade frequency component (0-15 points) - active coins preferred
        trade_count = self.get_trade_count(pair, days=7)
        freq_score = min(trade_count / 2, 15)  # 30 trades/week = full points
        
        # Sentiment component (-10 to +10 points)
        sent_score = sentiment * 10
        
        # ML confidence component (0-10 points)
        conf_score = ml_confidence * 10
        
        total_score = win_score + profit_score + vol_score + freq_score + sent_score + conf_score
        return max(0, min(100, total_score))  # Clamp to 0-100
    
    def rank_coins(self, coin_list: List[str], sentiment_map: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        Rank coins by performance score
        Returns list of (pair, score) tuples sorted by score descending
        """
        if sentiment_map is None:
            sentiment_map = {}
        
        scores = []
        for pair in coin_list:
            sentiment = sentiment_map.get(pair, 0.0)
            score = self.calculate_coin_score(pair, sentiment=sentiment)
            scores.append((pair, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def get_rotation_candidates(self, active_coins: List[str], watchlist: List[str], 
                                 bottom_n: int = 3) -> Dict[str, List[str]]:
        """
        Identify coins to rotate out and potential replacements
        Returns dict with 'drop' and 'add' lists
        """
        # Rank active coins
        active_ranked = self.rank_coins(active_coins)
        
        # Find underperformers (bottom N)
        underperformers = [coin for coin, score in active_ranked[-bottom_n:]]
        
        # Get non-active coins from watchlist
        candidates = [c for c in watchlist if c not in active_coins]
        
        # Rank candidates
        candidate_ranked = self.rank_coins(candidates)
        
        # Find top candidates
        top_candidates = [coin for coin, score in candidate_ranked[:bottom_n]]
        
        # Only rotate if candidates are significantly better
        rotation = {"drop": [], "add": []}
        for i in range(min(len(underperformers), len(top_candidates))):
            underperformer_score = next(s for c, s in active_ranked if c == underperformers[i])
            candidate_score = next(s for c, s in candidate_ranked if c == top_candidates[i])
            
            # Require 20% better score to rotate
            if candidate_score > underperformer_score * 1.2:
                rotation["drop"].append(underperformers[i])
                rotation["add"].append(top_candidates[i])
        
        return rotation
    
    def get_stats_summary(self, pair: str) -> str:
        """Get formatted stats summary for a coin"""
        win_rate = self.get_win_rate(pair, days=7)
        avg_profit = self.get_avg_profit(pair, days=7)
        volatility = self.get_avg_volatility(pair, hours=24)
        trade_count = self.get_trade_count(pair, days=7)
        score = self.calculate_coin_score(pair)
        
        return (f"{pair}: Score={score:.1f} | WR={win_rate*100:.1f}% | "
                f"AvgProfit={avg_profit:.2f}% | Vol={volatility*100:.2f}% | Trades={trade_count}")
