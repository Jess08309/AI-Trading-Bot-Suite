"""
Execution Guard - Pre-trade spread/price gating and ATR-based exit levels.

Replaces blind market-order execution with limit-order pricing at the
passive side of the book (bid for buys, ask for sells), and enforces a
maximum-spread gate so wide-spread meme coins (TRUMP, PEPE, SHIB, BONK...)
can't eat the whole edge of a scalp before it even starts.

Also enforces a fee/spread-adjusted minimum take-profit floor so a trade
with gross profit below round-trip cost is never taken, and derives
ATR-scaled stop-loss/take-profit levels per position.
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict


def compute_spread_bps(bid: float, ask: float) -> float:
    """Return the bid/ask spread in basis points of the mid price.

    Returns a large sentinel value (float('inf')) for degenerate quotes
    (non-positive bid/ask, or crossed/locked books) so callers always
    treat them as failing any spread gate.
    """
    if bid is None or ask is None:
        return float("inf")
    if bid <= 0 or ask <= 0 or ask < bid:
        return float("inf")
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return float("inf")
    return (ask - bid) / mid * 10000.0


def passes_spread_gate(bid: float, ask: float, max_spread_bps: float = 15.0) -> Tuple[bool, float]:
    """Check whether the current spread is tight enough to trade.

    Returns (passes, spread_bps).
    """
    spread_bps = compute_spread_bps(bid, ask)
    return spread_bps <= max_spread_bps, spread_bps


def limit_price_for_side(bid: float, ask: float, side: str) -> float:
    """Return the passive limit price for a side: bid for buys, ask for sells.

    `side` is case-insensitive and accepts "buy"/"sell" or "long"/"short".
    """
    normalized = side.strip().lower()
    if normalized in ("buy", "long"):
        return bid
    if normalized in ("sell", "short"):
        return ask
    raise ValueError(f"Unknown order side: {side!r}")


def min_required_take_profit_pct(
    spread_bps: float,
    fee_rate_pct: float = 0.1,
    floor_pct: float = 1.0,
) -> float:
    """Minimum gross take-profit percentage required to clear round-trip costs.

    Round-trip cost = 2x the taker/maker fee rate (one fee per leg) plus the
    spread paid crossing in and out of the position. The absolute floor
    (default 1%) always applies even when fees/spread are negligible, per
    the "no scalps below round-trip cost" requirement.
    """
    if spread_bps is None or spread_bps == float("inf"):
        spread_bps = 0.0
    round_trip_cost_pct = (2 * max(0.0, fee_rate_pct)) + (max(0.0, spread_bps) / 100.0)
    return max(floor_pct, round_trip_cost_pct)


def meets_min_take_profit(
    expected_gain_pct: float,
    spread_bps: float,
    fee_rate_pct: float = 0.1,
    floor_pct: float = 1.0,
) -> bool:
    """Whether a planned trade's expected gross gain clears the min-profit floor."""
    required = min_required_take_profit_pct(spread_bps, fee_rate_pct, floor_pct)
    return expected_gain_pct >= required


def atr_stop_take_profit(
    entry_price: float,
    atr: float,
    side: str,
    stop_mult: float = 1.5,
    tp_mult: float = 2.0,
    min_tp_pct: float = 1.0,
) -> Dict[str, float]:
    """Derive ATR-scaled stop-loss/take-profit prices for a position.

    The take-profit distance is widened (never narrowed) so it never falls
    below `min_tp_pct` of the entry price, enforcing the min-profit floor
    even when ATR is very small (e.g. in a quiet market).

    Returns a dict with stop_price, take_profit_price, stop_pct, tp_pct.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    atr = max(0.0, atr)
    normalized = side.strip().lower()
    is_long = normalized in ("buy", "long")
    is_short = normalized in ("sell", "short")
    if not (is_long or is_short):
        raise ValueError(f"Unknown position side: {side!r}")

    stop_distance = atr * stop_mult
    tp_distance = atr * tp_mult

    min_tp_distance = entry_price * (min_tp_pct / 100.0)
    if tp_distance < min_tp_distance:
        tp_distance = min_tp_distance

    if is_long:
        stop_price = entry_price - stop_distance
        take_profit_price = entry_price + tp_distance
    else:
        stop_price = entry_price + stop_distance
        take_profit_price = entry_price - tp_distance

    stop_pct = abs(stop_distance) / entry_price * 100.0
    tp_pct = abs(tp_distance) / entry_price * 100.0

    return {
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "stop_pct": stop_pct,
        "tp_pct": tp_pct,
    }
