"""
Position Sizing Engine.
Kelly-criterion inspired sizing that scales with ML confidence and volatility.
"""
from __future__ import annotations
from typing import Dict, Optional


class PositionSizer:
    """Calculate position sizes based on confidence, volatility, and portfolio state."""

    def __init__(self, max_position_pct: float = 0.15,
                 min_position_pct: float = 0.02,
                 max_portfolio_risk: float = 0.50,
                 kelly_fraction: float = 0.25):
        """
        Args:
            max_position_pct: Max % of balance for single position (15%)
            min_position_pct: Min % of balance for single position (2%)
            max_portfolio_risk: Max % of balance in all positions (50%)
            kelly_fraction: Fraction of Kelly criterion to use (quarter-Kelly for safety)
        """
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.kelly_fraction = kelly_fraction

    def calculate_kelly(self, win_rate: float, avg_win: float,
                        avg_loss: float) -> float:
        """Calculate Kelly Criterion fraction.
        f* = (bp - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        if avg_loss == 0 or win_rate <= 0:
            return 0.0
        b = abs(avg_win / avg_loss)
        p = min(1.0, max(0.0, win_rate))
        q = 1.0 - p
        kelly = (b * p - q) / b
        # Apply fraction for safety (quarter-Kelly)
        return max(0.0, kelly * self.kelly_fraction)

    def calculate_size(self, balance: float, confidence: float,
                       volatility: float, existing_exposure: float,
                       win_rate: float = 0.5,
                       avg_win: float = 0.02,
                       avg_loss: float = 0.02,
                       num_positions: int = 0) -> Dict[str, float]:
        """Calculate position size based on all factors.

        Args:
            balance: Available trading balance
            confidence: ML model confidence (0.0 to 1.0)
            volatility: Current asset volatility
            existing_exposure: Total $ already in positions
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average win percentage
            avg_loss: Average loss percentage
            num_positions: Current number of open positions

        Returns:
            Dict with position_size ($), position_pct, confidence_factor, vol_factor
        """
        if balance <= 0:
            return {"position_size": 0, "position_pct": 0,
                    "confidence_factor": 0, "vol_factor": 0, "reason": "no_balance"}

        # 1. Kelly-based starting point
        kelly_pct = self.calculate_kelly(win_rate, avg_win, avg_loss)
        if kelly_pct <= 0:
            kelly_pct = self.min_position_pct  # Minimum bet

        # 2. Confidence scaling: higher confidence = larger position
        # Scale from 0.5x at confidence=0.5 to 1.5x at confidence=0.9
        confidence_factor = 0.5 + (confidence - 0.5) * 2.5
        confidence_factor = max(0.3, min(1.5, confidence_factor))

        # 3. Volatility scaling: higher volatility = smaller position
        # Scale from 1.0x at vol=0 to 0.3x at vol=0.05+
        vol_factor = max(0.3, 1.0 - volatility * 15)

        # 4. Position count scaling: more positions = smaller each
        count_factor = max(0.5, 1.0 - num_positions * 0.05)

        # Combine factors
        raw_pct = kelly_pct * confidence_factor * vol_factor * count_factor

        # Clamp to min/max
        position_pct = max(self.min_position_pct, min(self.max_position_pct, raw_pct))

        # Check portfolio risk limit
        remaining_capacity = max(0, self.max_portfolio_risk * balance - existing_exposure)
        position_size = min(balance * position_pct, remaining_capacity)
        position_size = max(0, position_size)

        return {
            "position_size": round(position_size, 2),
            "position_pct": round(position_pct, 4),
            "confidence_factor": round(confidence_factor, 3),
            "vol_factor": round(vol_factor, 3),
            "kelly_pct": round(kelly_pct, 4),
            "reason": "ok" if position_size > 0 else "capacity_limit"
        }

    def calculate_futures_size(self, balance: float, confidence: float,
                               volatility: float, leverage: int = 2,
                               num_positions: int = 0) -> Dict[str, float]:
        """Calculate futures position size with leverage consideration.
        More conservative than spot due to leverage risk.
        """
        # Futures use tighter limits
        max_pct = self.max_position_pct / leverage  # Divide by leverage
        min_pct = self.min_position_pct / leverage

        confidence_factor = 0.5 + (confidence - 0.5) * 2.0
        confidence_factor = max(0.3, min(1.3, confidence_factor))

        vol_factor = max(0.2, 1.0 - volatility * 20)  # More sensitive to vol

        count_factor = max(0.4, 1.0 - num_positions * 0.08)

        raw_pct = max_pct * 0.5 * confidence_factor * vol_factor * count_factor
        position_pct = max(min_pct, min(max_pct, raw_pct))

        contract_value = balance * position_pct * leverage
        margin_required = contract_value / leverage

        return {
            "contract_value": round(contract_value, 2),
            "margin_required": round(margin_required, 2),
            "position_pct": round(position_pct, 4),
            "confidence_factor": round(confidence_factor, 3),
            "vol_factor": round(vol_factor, 3),
            "reason": "ok"
        }

    def calculate_contract_multiplier(self, confidence: float, volatility: float,
                                       regime: str = "neutral", 
                                       correlation: float = 0.5,
                                       rsi: float = 50.0) -> Dict[str, any]:
        """Determine position size multiplier (1-3x) based on setup quality.
        
        Use this for futures positions when multiple signals align strongly.
        Returns 1x for standard setups, 2x for strong, 3x for perfect.
        
        Args:
            confidence: ML model confidence (0.0 to 1.0)
            volatility: Current volatility (lower is better for sizing up)
            regime: Market regime ("trending", "ranging", "neutral")
            correlation: Average correlation to open positions (0.0 to 1.0)
            rsi: RSI indicator (extreme values get bonus for mean reversion)
        
        Returns:
            Dict with multiplier (1-3), score (0-6), and breakdown
        """
        score = 0
        breakdown = []
        
        # 1. High confidence: +2 points
        if confidence > 0.75:
            score += 2
            breakdown.append("confidence_high")
        elif confidence > 0.65:
            score += 1
            breakdown.append("confidence_med")
        
        # 2. Low volatility: +2 points (safer to size up)
        if volatility < 0.015:
            score += 2
            breakdown.append("volatility_low")
        elif volatility < 0.025:
            score += 1
            breakdown.append("volatility_med")
        
        # 3. Trending regime: +1 point
        if regime == "trending":
            score += 1
            breakdown.append("regime_trend")
        
        # 4. Low correlation: +1 point (good diversification)
        if correlation < 0.3:
            score += 1
            breakdown.append("correlation_low")
        elif correlation < 0.5:
            breakdown.append("correlation_med")
        
        # Score to multiplier mapping:
        # 0-2: 1x (standard - most trades)
        # 3-4: 2x (strong setup - ~20% of trades)
        # 5-6: 3x (perfect setup - rare, <5% of trades)
        
        if score >= 5:
            multiplier = 3
            quality = "perfect"
        elif score >= 3:
            multiplier = 2
            quality = "strong"
        else:
            multiplier = 1
            quality = "standard"
        
        return {
            "multiplier": multiplier,
            "score": score,
            "quality": quality,
            "breakdown": breakdown,
            "reason": f"{quality}_setup_{score}pts"
        }

    def calculate_futures_size_scaled(self, balance: float, confidence: float,
                                       volatility: float, leverage: int = 2,
                                       num_positions: int = 0,
                                       regime: str = "neutral",
                                       correlation: float = 0.5,
                                       rsi: float = 50.0) -> Dict[str, float]:
        """Calculate futures position with intelligent scaling (1-3x multiplier).
        
        This combines the base position sizing with quality-based multipliers.
        Perfect setups get 3x, strong setups get 2x, standard get 1x.
        """
        # Get base position size
        base_sizing = self.calculate_futures_size(
            balance=balance,
            confidence=confidence,
            volatility=volatility,
            leverage=leverage,
            num_positions=num_positions
        )
        
        # Calculate multiplier based on setup quality
        multiplier_data = self.calculate_contract_multiplier(
            confidence=confidence,
            volatility=volatility,
            regime=regime,
            correlation=correlation,
            rsi=rsi
        )
        
        multiplier = multiplier_data["multiplier"]
        
        # Apply multiplier to position size
        scaled_contract_value = base_sizing["contract_value"] * multiplier
        scaled_margin = base_sizing["margin_required"] * multiplier
        
        # Safety check: don't exceed max position limit even with multiplier.
        # `max_position_pct` is treated as max contract notional (not margin).
        max_contract_value = balance * self.max_position_pct
        if scaled_contract_value > max_contract_value:
            scaled_contract_value = max_contract_value
            scaled_margin = max_contract_value / leverage
            actual_multiplier = scaled_contract_value / base_sizing["contract_value"]
        else:
            actual_multiplier = multiplier
        
        return {
            "contract_value": round(scaled_contract_value, 2),
            "margin_required": round(scaled_margin, 2),
            "position_pct": round(base_sizing["position_pct"] * actual_multiplier, 4),
            "confidence_factor": base_sizing["confidence_factor"],
            "vol_factor": base_sizing["vol_factor"],
            "multiplier": multiplier,
            "actual_multiplier": round(actual_multiplier, 2),
            "quality": multiplier_data["quality"],
            "score": multiplier_data["score"],
            "breakdown": multiplier_data["breakdown"],
            "reason": f"{multiplier_data['quality']}_{multiplier}x"
        }
