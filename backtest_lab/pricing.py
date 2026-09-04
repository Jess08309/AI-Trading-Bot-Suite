"""Black-Scholes option pricing + delta, used to synthesize option prices from real
underlying data instead of paying for historical options tick data.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    """European option price. T in years, sigma annualized. right is 'call' or 'put'."""
    if T <= 0:
        return max(0.0, (S - K) if right == "call" else (K - S))
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if right == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    """Option delta. Short-put/short-call strategies select strikes by |delta|."""
    if T <= 0:
        return (1.0 if S > K else 0.0) if right == "call" else (-1.0 if S < K else 0.0)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if right == "call" else norm.cdf(d1) - 1.0


def strike_for_delta(S: float, T: float, r: float, sigma: float, right: str, target_delta: float) -> float:
    """Solve for the strike whose delta matches target_delta (e.g. 0.15 for a 0.15-delta short put)."""
    target = target_delta if right == "call" else -target_delta

    def f(K):
        return bs_delta(S, K, T, r, sigma, right) - target

    lo, hi = S * 0.3, S * 3.0
    try:
        return brentq(f, lo, hi)
    except ValueError:
        # target delta unreachable at extreme T/sigma -- fall back to nearest boundary
        return lo if f(lo) > 0 else hi


def synthetic_bid_ask(mid: float, bid_ask_pct: float) -> tuple:
    """Simulate a bid/ask spread as +/- half of bid_ask_pct around the BS mid price."""
    half = mid * bid_ask_pct / 2.0
    return max(0.01, mid - half), mid + half


def implied_vol(price: float, S: float, K: float, T: float, r: float, right: str) -> float | None:
    """Invert Black-Scholes to get the IV implied by a REAL observed option price.
    Returns None if no solution exists in a sane vol range (e.g. price outside
    no-arbitrage bounds, common for stale/illiquid daily prints)."""
    if T <= 0 or price <= 0:
        return None
    intrinsic = max(0.0, (S - K) if right == "call" else (K - S))
    if price < intrinsic:
        return None

    def f(sigma):
        return bs_price(S, K, T, r, sigma, right) - price

    try:
        return brentq(f, 1e-4, 5.0)
    except ValueError:
        return None
