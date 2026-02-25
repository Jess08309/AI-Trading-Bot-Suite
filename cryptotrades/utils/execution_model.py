"""Execution realism helpers.

This module provides a lightweight execution-cost model for paper trading and
backtests, covering:
- Slippage / spread impact (in bps) via worse-than-mid execution prices
- Optional partial fills (stochastic fill ratio)
- Optional futures funding cost (simple notional-based approximation)

The goal is not to perfectly simulate an exchange, but to avoid systematically
overstating paper performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

OrderSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class ExecutionResult:
    exec_price: float
    impact_bps: float
    fill_ratio: float


def bps_to_rate(bps: float) -> float:
    return float(bps) / 10_000.0


def execution_price(mid_price: float, side: OrderSide, impact_bps: float) -> float:
    """Return an execution price worse than the mid, based on impact in bps."""
    mid = float(mid_price)
    impact = bps_to_rate(impact_bps)
    if side == "BUY":
        return mid * (1.0 + impact)
    if side == "SELL":
        return mid * (1.0 - impact)
    raise ValueError(f"Unknown side: {side}")


def sample_fill_ratio(
    rng: random.Random,
    *,
    enabled: bool,
    partial_fill_prob: float,
    min_ratio: float,
    max_ratio: float,
) -> float:
    if not enabled:
        return 1.0

    p = max(0.0, min(1.0, float(partial_fill_prob)))
    lo = max(0.0, min(1.0, float(min_ratio)))
    hi = max(0.0, min(1.0, float(max_ratio)))
    if hi < lo:
        lo, hi = hi, lo

    if rng.random() < p:
        return float(rng.uniform(lo, hi))
    return 1.0


def estimate_funding_cost(
    notional_usd: float,
    hold_hours: float,
    *,
    enabled: bool,
    rate_per_8h: float,
) -> float:
    """Approximate futures funding cost as a notional charge.

    This is intentionally conservative/simplified: it always charges cost
    (doesn't attempt to predict pay/receive).
    """
    if not enabled:
        return 0.0
    notional = max(0.0, float(notional_usd))
    hours = max(0.0, float(hold_hours))
    rate = max(0.0, float(rate_per_8h))
    return notional * rate * (hours / 8.0)
