"""
Risk-Based Position Sizing.

Sizes a trade from the distance to its stop-loss so that a full stop-out
loses at most `risk_pct` of equity, independent of whatever confidence/
Kelly-based sizing an upstream model suggests. Also enforces a hard
notional cap (% of equity) per asset so a single symbol can never account
for an outsized share of the book regardless of stop placement.

This is intended to be applied as a final safety cap on top of (not a
replacement for) existing confidence/volatility-based sizing.
"""
from __future__ import annotations
from typing import Dict


def calculate_risk_capped_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.01,
    notional_cap_pct: float = 0.10,
) -> Dict[str, float]:
    """Calculate the max position notional ($) allowed by risk and notional caps.

    Args:
        equity: Total account equity ($).
        entry_price: Planned entry price.
        stop_price: Planned stop-loss price (same side as entry; must differ
            from entry_price to derive a risk-based cap).
        risk_pct: Max fraction of equity to risk on a full stop-out (default 1%).
        notional_cap_pct: Max fraction of equity to allocate to a single
            asset's notional, regardless of stop distance (default 10%).

    Returns:
        Dict with:
            notional: final capped position size in $
            risk_based_notional: size implied by risk budget / stop distance
            notional_cap: the hard notional ceiling ($)
            limiting_factor: "risk", "notional_cap", or "none" (no equity)
    """
    if equity <= 0 or entry_price <= 0:
        return {
            "notional": 0.0,
            "risk_based_notional": 0.0,
            "notional_cap": 0.0,
            "limiting_factor": "none",
        }

    notional_cap = equity * max(0.0, notional_cap_pct)

    stop_distance_pct = abs(entry_price - stop_price) / entry_price
    risk_budget = equity * max(0.0, risk_pct)

    if stop_distance_pct <= 0:
        # No usable stop distance — fall back to the notional cap only.
        risk_based_notional = notional_cap
    else:
        risk_based_notional = risk_budget / stop_distance_pct

    notional = min(risk_based_notional, notional_cap)
    limiting_factor = "risk" if risk_based_notional <= notional_cap else "notional_cap"

    return {
        "notional": round(max(0.0, notional), 2),
        "risk_based_notional": round(max(0.0, risk_based_notional), 2),
        "notional_cap": round(notional_cap, 2),
        "limiting_factor": limiting_factor,
    }
