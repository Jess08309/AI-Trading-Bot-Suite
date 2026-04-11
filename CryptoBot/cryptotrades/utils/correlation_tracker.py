"""
Asset Correlation Tracker.
Monitors rolling correlations between assets to prevent over-concentration
in highly correlated positions.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


class CorrelationTracker:
    """Track rolling price correlations between trading pairs."""

    def __init__(self, window: int = 30, max_correlation: float = 0.85,
                 save_path: str = "data/state/correlations.json"):
        """
        Args:
            window: Number of periods for rolling correlation
            max_correlation: Max allowed correlation with existing positions
            save_path: Path to persist correlation data
        """
        self.window = window
        self.max_correlation = max_correlation
        self.save_path = save_path
        self.price_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}

    def update_price(self, pair: str, price: float):
        """Add a new price observation for a pair."""
        if pair not in self.price_history:
            self.price_history[pair] = []
        self.price_history[pair].append(price)
        # Keep only what we need
        if len(self.price_history[pair]) > self.window * 2:
            self.price_history[pair] = self.price_history[pair][-self.window * 2:]

    def calculate_returns(self, pair: str) -> Optional[np.ndarray]:
        """Calculate returns for a pair."""
        if pair not in self.price_history or len(self.price_history[pair]) < 3:
            return None
        prices = np.array(self.price_history[pair][-self.window:])
        # Guard against zero prices (API fluke) causing division by zero
        denom = prices[:-1]
        safe = denom != 0
        returns = np.zeros(len(denom))
        returns[safe] = np.diff(prices)[safe] / denom[safe]
        return returns

    def get_correlation(self, pair1: str, pair2: str) -> float:
        """Calculate rolling correlation between two pairs."""
        r1 = self.calculate_returns(pair1)
        r2 = self.calculate_returns(pair2)

        if r1 is None or r2 is None:
            return 0.0

        # Align lengths
        min_len = min(len(r1), len(r2))
        if min_len < 5:
            return 0.0

        r1 = r1[-min_len:]
        r2 = r2[-min_len:]

        # Guard against zero-variance arrays (flat prices → division by zero in corrcoef)
        if np.std(r1) < 1e-12 or np.std(r2) < 1e-12:
            return 0.0

        # Pearson correlation
        corr = np.corrcoef(r1, r2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def update_correlation_matrix(self, pairs: List[str]):
        """Recalculate correlation matrix for all pairs."""
        self.correlation_matrix = {}
        for i, p1 in enumerate(pairs):
            self.correlation_matrix[p1] = {}
            for j, p2 in enumerate(pairs):
                if i == j:
                    self.correlation_matrix[p1][p2] = 1.0
                elif j < i and p2 in self.correlation_matrix:
                    self.correlation_matrix[p1][p2] = self.correlation_matrix[p2].get(p1, 0.0)
                else:
                    self.correlation_matrix[p1][p2] = self.get_correlation(p1, p2)

    def get_portfolio_correlation(self, candidate: str,
                                  open_positions: List[str]) -> float:
        """Get average correlation between candidate and all open positions.
        Returns value between -1 and 1.
        """
        if not open_positions:
            return 0.0

        correlations = []
        for pos in open_positions:
            corr = self.get_correlation(candidate, pos)
            correlations.append(abs(corr))  # Use absolute correlation

        return float(np.mean(correlations)) if correlations else 0.0

    def should_allow_trade(self, candidate: str,
                           open_positions: List[str]) -> Tuple[bool, float, str]:
        """Check if adding this position would create too much correlation risk.

        Returns:
            (allowed, avg_correlation, reason)
        """
        if not open_positions:
            return True, 0.0, "no_existing_positions"

        avg_corr = self.get_portfolio_correlation(candidate, open_positions)

        if avg_corr > self.max_correlation:
            return False, avg_corr, f"too_correlated ({avg_corr:.2f} > {self.max_correlation})"

        return True, avg_corr, "ok"

    def get_diversification_score(self, open_positions: List[str]) -> float:
        """Portfolio diversification score (0=perfectly correlated, 1=uncorrelated).
        Higher is better.
        """
        if len(open_positions) < 2:
            return 1.0

        correlations = []
        for i, p1 in enumerate(open_positions):
            for j, p2 in enumerate(open_positions):
                if j > i:
                    correlations.append(abs(self.get_correlation(p1, p2)))

        if not correlations:
            return 1.0

        avg_corr = np.mean(correlations)
        return float(max(0, 1.0 - avg_corr))

    def save_state(self):
        """Persist correlation data."""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump({
                    "correlation_matrix": self.correlation_matrix,
                    "price_counts": {k: len(v) for k, v in self.price_history.items()}
                }, f, indent=2)
        except Exception:
            pass

    def load_state(self):
        """Load persisted correlation data."""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.correlation_matrix = data.get("correlation_matrix", {})
        except Exception:
            pass
