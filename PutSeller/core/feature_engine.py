"""
PutSeller Feature Engine — Credit Spread Quality Features

12 features optimized for evaluating PUT SPREAD QUALITY.
NO overlap with CallBuyer features — these target PREMIUM HARVESTING,
not directional momentum.

Features:
 1. iv_hv_ratio        — IV/HV20 ratio (want HIGH = fat premiums)
 2. support_distance   — price above 20-day low as % (cushion)
 3. rsi_level          — raw RSI normalized (avoid oversold = assignment risk)
 4. credit_quality     — credit / spread_width ratio
 5. dte_score          — proximity to target DTE (sweet spot for theta)
 6. oi_liquidity       — open interest of short leg (fill quality)
 7. sector_strength    — stock 20-day return vs SPY (relative strength)
 8. mean_reversion     — z-score of price vs 50-day SMA (overextended?)
 9. atr_ratio          — ATR(14) / ATR(50) (vol expansion = caution)
10. trend_stability    — % of last 20 days that closed above SMA20
11. otm_buffer         — % OTM of short strike (safety margin)
12. put_volume_ratio   — put volume / total volume (sentiment proxy)
"""
import logging
import json
import os
from typing import Dict, Optional, Any

import numpy as np

log = logging.getLogger("putseller.features")

FEATURE_NAMES = [
    "iv_hv_ratio", "support_distance", "rsi_level", "credit_quality",
    "dte_score", "oi_liquidity", "sector_strength", "mean_reversion",
    "atr_ratio", "trend_stability", "otm_buffer", "put_volume_ratio",
]


class PutSellerFeatureEngine:
    """Extracts 12 credit-spread-quality features from market data."""

    def build_features(self, daily_bars: list, spread_info: dict,
                       spy_bars: Optional[list] = None) -> Optional[np.ndarray]:
        """Build a 12-element feature vector for a put spread candidate.

        Args:
            daily_bars: List of bar objects with .open, .close, .high, .low, .volume
            spread_info: Dict with keys: iv_premium, credit_pct, dte, target_dte,
                         max_dte, short_oi, otm_pct, short_strike, price
            spy_bars: SPY daily bars for relative strength (optional)

        Returns:
            np.ndarray of shape (12,) or None if insufficient data
        """
        if len(daily_bars) < 50:
            return None

        closes = np.array([float(b.close) for b in daily_bars])
        highs = np.array([float(b.high) for b in daily_bars])
        lows = np.array([float(b.low) for b in daily_bars])
        opens = np.array([float(b.open) for b in daily_bars])
        volumes = np.array([float(b.volume) for b in daily_bars])

        f = np.zeros(12)

        # 1. IV/HV ratio — want elevated IV for premium selling
        iv_premium = spread_info.get("iv_premium")
        if iv_premium and iv_premium > 0:
            # Normalize: 1.0 = fair, >1.5 = rich, <0.8 = cheap
            f[0] = np.clip((iv_premium - 0.8) / 1.2, 0, 1)
        else:
            # Proxy: use recent vol spike as proxy
            rv5 = np.std(np.diff(np.log(closes[-6:]))) * np.sqrt(252) if len(closes) > 6 else 0
            rv20 = np.std(np.diff(np.log(closes[-21:]))) * np.sqrt(252) if len(closes) > 21 else rv5
            proxy = rv5 / rv20 if rv20 > 0 else 1.0
            f[0] = np.clip((proxy - 0.8) / 1.2, 0, 1)

        # 2. Support distance — how far price is above 20-day low
        low_20 = np.min(lows[-20:])
        f[1] = np.clip((closes[-1] - low_20) / closes[-1] if closes[-1] > 0 else 0, 0, 0.15) / 0.15

        # 3. RSI level — avoid oversold (< 30) = assignment risk
        rsi = self._rsi(closes, 14)
        # Score: 0 = RSI 20 (dangerous), 1 = RSI 60 (comfortable)
        f[2] = np.clip((rsi - 20) / 50, 0, 1)

        # 4. Credit quality — credit as % of spread width
        credit_pct = spread_info.get("credit_pct", 0)
        f[3] = np.clip(credit_pct / 0.40, 0, 1)  # 40% credit = perfect score

        # 5. DTE score — proximity to target DTE
        dte = spread_info.get("dte", 35)
        target_dte = spread_info.get("target_dte", 35)
        max_dte = spread_info.get("max_dte", 50)
        f[4] = 1.0 - abs(dte - target_dte) / max_dte

        # 6. OI liquidity — open interest indicates fill quality
        oi = spread_info.get("short_oi", 0)
        f[5] = np.clip(oi / 1000.0, 0, 1)

        # 7. Sector relative strength — stock Return(20) vs SPY Return(20)
        stock_ret20 = (closes[-1] / closes[-21] - 1) if len(closes) > 21 and closes[-21] > 0 else 0
        spy_ret20 = 0
        if spy_bars and len(spy_bars) > 21:
            spy_c = np.array([float(b.close) for b in spy_bars])
            spy_ret20 = (spy_c[-1] / spy_c[-21] - 1) if spy_c[-21] > 0 else 0
        rel_strength = stock_ret20 - spy_ret20
        # Positive = outperforming (good for put selling)
        f[6] = np.clip(rel_strength / 0.10 * 0.5 + 0.5, 0, 1)

        # 8. Mean reversion — z-score of price vs 50-day SMA
        sma50 = np.mean(closes[-50:])
        std50 = np.std(closes[-50:])
        z_score = (closes[-1] - sma50) / std50 if std50 > 0 else 0
        # Z > 0 means above mean (good for puts), Z < -2 = dangerous
        f[7] = np.clip((z_score + 2) / 4, 0, 1)  # center -2 to +2

        # 9. ATR ratio — ATR(14) / ATR(50) — vol expansion warning
        atr14 = self._atr(highs, lows, closes, 14)
        atr50 = self._atr(highs, lows, closes, min(50, len(closes) - 1))
        ratio = atr14 / atr50 if atr50 > 0 else 1.0
        # Ratio > 1.5 = expanding vol = caution; < 1 = contracting = good
        f[8] = np.clip(1.0 - (ratio - 0.5) / 1.5, 0, 1)

        # 10. Trend stability — % of last 20 days above SMA20
        sma20 = np.mean(closes[-20:])
        above_sma = np.sum(closes[-20:] > sma20) / 20
        f[9] = above_sma

        # 11. OTM buffer — % OTM of short strike
        otm_pct = spread_info.get("otm_pct", 0)
        # More OTM = safer = higher score
        f[10] = np.clip(otm_pct / 0.15, 0, 1)

        # 12. Put volume ratio — put volume / total volume (sentiment)
        # Proxy: use recent volume trend (declining vol = less panic = good for puts)
        vol_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
        vol_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else vol_5
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0
        # Low ratio = quiet = good for selling premium
        f[11] = np.clip(1.0 - (vol_ratio - 0.5) / 1.5, 0, 1)

        return np.nan_to_num(f, nan=0.5)

    def compute_rule_score(self, features: np.ndarray) -> float:
        """Compute a rules-based spread quality score (0-10 scale)."""
        score = 0.0

        # Elevated IV (fat premiums)
        if features[0] > 0.5:
            score += 1.5
        elif features[0] > 0.3:
            score += 0.5

        # Good support cushion
        if features[1] > 0.5:
            score += 1.0
        elif features[1] > 0.3:
            score += 0.5

        # RSI not oversold (safe for short puts)
        if features[2] > 0.5:
            score += 1.5
        elif features[2] > 0.3:
            score += 0.5

        # Good credit quality
        if features[3] > 0.5:
            score += 2.0
        elif features[3] > 0.3:
            score += 1.0

        # DTE near sweet spot
        if features[4] > 0.7:
            score += 1.0

        # Good liquidity
        if features[5] > 0.3:
            score += 0.5

        # Stock outperforming market
        if features[6] > 0.6:
            score += 0.5

        # Low volatility expansion (stable market)
        if features[8] > 0.5:
            score += 1.0

        # Strong trend stability
        if features[9] > 0.6:
            score += 0.5

        # Good OTM buffer
        if features[10] > 0.3:
            score += 0.5

        return min(score, 10.0)

    def log_features(self, symbol: str, features: np.ndarray,
                     outcome: Optional[float] = None,
                     features_file: str = "data/state/features_log.json"):
        """Append features + outcome to log for future model training."""
        entry = {
            "symbol": symbol,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "features": features.tolist(),
            "feature_names": FEATURE_NAMES,
            "outcome": outcome,
        }
        try:
            os.makedirs(os.path.dirname(features_file), exist_ok=True)
            existing = []
            if os.path.exists(features_file):
                with open(features_file) as f:
                    existing = json.load(f)
            existing.append(entry)
            if len(existing) > 1000:
                existing = existing[-1000:]
            with open(features_file, "w") as f:
                json.dump(existing, f, indent=1)
        except Exception as e:
            log.warning(f"Could not log features: {e}")

    # ── Indicator Helpers ────────────────────────────────
    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        return float(100 - 100 / (1 + rs))

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int) -> float:
        if len(closes) < period + 1:
            return 0.0
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-(period + 1):-1]),
                np.abs(lows[-period:] - closes[-(period + 1):-1])
            )
        )
        return float(np.mean(tr))
