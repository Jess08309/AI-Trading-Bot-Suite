"""
CallBuyer Feature Engine — Momentum Breakout Features

12 features optimized for call buying on momentum breakouts.
NO overlap with PutSeller features — these target ENTRY TIMING for directional
long calls, not spread quality.

Features:
 1. rsi_14             — momentum oscillator (want 50-75 = bullish not exhausted)
 2. momentum_composite — weighted multi-timeframe ROC
 3. volume_surge       — volume vs 20-day avg (breakout confirmation)
 4. breakout_score     — % above 20-day high (0 if below)
 5. sma_alignment      — 20 > 50 > 200 alignment score (trend quality)
 6. atr_pct            — ATR(14) as % of price (expected move magnitude)
 7. iv_rank            — IV percentile (want LOW = cheap calls)
 8. sector_momentum    — SPY 5-day ROC (broad market tailwind)
 9. range_position     — price position in 20-day range (0=bottom, 1=top)
10. gap_magnitude      — |open - prev_close| / prev_close
11. consecutive_green  — count of consecutive green candles (momentum persistence)
12. macd_histogram     — MACD histogram value (trend acceleration)
"""
import logging
import json
import os
from typing import Dict, List, Optional, Any

import numpy as np

log = logging.getLogger("callbuyer.features")

FEATURE_NAMES = [
    "rsi_14", "momentum_composite", "volume_surge", "breakout_score",
    "sma_alignment", "atr_pct", "iv_rank", "sector_momentum",
    "range_position", "gap_magnitude", "consecutive_green", "macd_histogram",
]


class CallBuyerFeatureEngine:
    """Extracts 12 momentum-focused features from price data."""

    def __init__(self):
        self._spy_cache: Optional[Dict] = None
        self._spy_cache_ts: float = 0

    def build_features(self, daily_bars: list, iv_current: Optional[float] = None,
                       hv20: Optional[float] = None,
                       spy_bars: Optional[list] = None) -> Optional[np.ndarray]:
        """Build a 12-element feature vector from daily bars.

        Args:
            daily_bars: List of bar objects with .open, .close, .high, .low, .volume
            iv_current: Current implied volatility of ATM call (optional)
            hv20: 20-day historical volatility (optional)
            spy_bars: SPY daily bars for sector momentum (optional)

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

        # 1. RSI(14) — normalized to 0-1
        f[0] = self._rsi(closes, 14) / 100.0

        # 2. Momentum composite — weighted multi-timeframe ROC
        roc5 = (closes[-1] / closes[-6] - 1) if closes[-6] != 0 else 0
        roc10 = (closes[-1] / closes[-11] - 1) if closes[-11] != 0 else 0
        roc20 = (closes[-1] / closes[-21] - 1) if closes[-21] != 0 else 0
        f[1] = np.clip(0.5 * roc5 + 0.3 * roc10 + 0.2 * roc20, -0.15, 0.15) / 0.15

        # 3. Volume surge — current vol vs 20-day avg
        avg_vol = np.mean(volumes[-21:-1]) if len(volumes) > 21 else np.mean(volumes[:-1])
        f[2] = np.clip(volumes[-1] / avg_vol if avg_vol > 0 else 1.0, 0.3, 4.0) / 4.0

        # 4. Breakout score — % above 20-day high
        high_20 = np.max(highs[-21:-1])
        f[3] = np.clip((closes[-1] - high_20) / high_20 if high_20 > 0 else 0, 0, 0.10) / 0.10

        # 5. SMA alignment — 20 > 50 > 200 trend quality
        sma20 = np.mean(closes[-20:])
        sma50 = np.mean(closes[-50:])
        sma200 = np.mean(closes[-min(200, len(closes)):]) if len(closes) >= 200 else np.mean(closes)
        alignment = 0.0
        if closes[-1] > sma20:
            alignment += 0.25
        if sma20 > sma50:
            alignment += 0.25
        if sma50 > sma200:
            alignment += 0.25
        if closes[-1] > sma200:
            alignment += 0.25
        f[4] = alignment

        # 6. ATR% — ATR(14) as % of price (expected move magnitude)
        tr_arr = np.maximum(
            highs[-14:] - lows[-14:],
            np.maximum(
                np.abs(highs[-14:] - closes[-15:-1]),
                np.abs(lows[-14:] - closes[-15:-1])
            )
        )
        atr = np.mean(tr_arr)
        f[5] = np.clip(atr / closes[-1] if closes[-1] > 0 else 0, 0, 0.08) / 0.08

        # 7. IV rank — want LOW for cheap calls
        # If IV available, compare to HV; otherwise estimate from price action
        if iv_current and hv20 and hv20 > 0:
            iv_ratio = iv_current / hv20
            f[6] = 1.0 - np.clip(iv_ratio, 0.5, 2.0) / 2.0  # invert: low IV = high score
        else:
            # Proxy: use recent realized vol vs longer-term
            rv5 = np.std(np.diff(np.log(closes[-6:]))) * np.sqrt(252) if len(closes) > 6 else 0
            rv20 = np.std(np.diff(np.log(closes[-21:]))) * np.sqrt(252) if len(closes) > 21 else rv5
            ratio = rv5 / rv20 if rv20 > 0 else 1.0
            f[6] = 1.0 - np.clip(ratio, 0.5, 2.0) / 2.0

        # 8. Sector momentum — SPY 5-day ROC
        if spy_bars and len(spy_bars) >= 6:
            spy_closes = np.array([float(b.close) for b in spy_bars])
            spy_roc = (spy_closes[-1] / spy_closes[-6] - 1) if spy_closes[-6] != 0 else 0
            f[7] = np.clip(spy_roc, -0.05, 0.05) / 0.05 * 0.5 + 0.5  # center at 0.5
        else:
            f[7] = 0.5  # neutral if no SPY data

        # 9. Range position — price in 20-day range
        low_20 = np.min(lows[-20:])
        high_20_all = np.max(highs[-20:])
        range_width = high_20_all - low_20
        f[8] = (closes[-1] - low_20) / range_width if range_width > 0 else 0.5

        # 10. Gap magnitude — |open - prev_close| / prev_close
        if closes[-2] > 0:
            gap = abs(opens[-1] - closes[-2]) / closes[-2]
        else:
            gap = 0
        f[9] = np.clip(gap, 0, 0.05) / 0.05

        # 11. Consecutive green candles — momentum persistence
        greens = 0
        for i in range(1, min(11, len(closes))):
            if closes[-i] > opens[-i]:
                greens += 1
            else:
                break
        f[10] = np.clip(greens / 5.0, 0, 1.0)

        # 12. MACD histogram — trend acceleration
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema_from_array(macd_line[-9:], 9) if len(macd_line) >= 9 else macd_line[-1]
        histogram = macd_line[-1] - signal_line
        # Normalize: histogram as % of price
        f[11] = np.clip(histogram / closes[-1] * 100 if closes[-1] > 0 else 0, -1, 1) * 0.5 + 0.5

        return np.nan_to_num(f, nan=0.5)

    def compute_rule_score(self, features: np.ndarray) -> float:
        """Compute a rules-based momentum score from features (0-10 scale).

        This is the 'rules' component of the ensemble, analogous to AlpacaBot's
        indicator-based bull/bear scoring.
        """
        score = 0.0

        # RSI in bullish zone (50-75) or momentum breakout (75-85)
        rsi = features[0] * 100
        if 50 <= rsi <= 75:
            score += 2.0
        elif 75 < rsi <= 85:
            score += 2.0  # Momentum breakout — reward, don't penalize (was 1.0)
        elif 45 <= rsi < 50:
            score += 1.0

        # Positive momentum composite
        if features[1] > 0.3:
            score += 2.0
        elif features[1] > 0.1:
            score += 1.0

        # Volume surge above average
        if features[2] > 0.4:  # > 1.6x avg
            score += 1.5
        elif features[2] > 0.3:  # > 1.2x avg
            score += 0.5

        # Breakout (above 20-day high)
        if features[3] > 0.01:
            score += 1.5

        # SMA alignment (strong trend)
        if features[4] >= 0.75:
            score += 1.5
        elif features[4] >= 0.50:
            score += 0.5

        # Cheap IV (below historical)
        if features[6] > 0.6:
            score += 0.5

        # Sector tailwind (SPY positive)
        if features[7] > 0.6:
            score += 0.5

        # MACD accelerating
        if features[11] > 0.6:
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
            "outcome": outcome,  # filled in later when trade closes
        }
        try:
            os.makedirs(os.path.dirname(features_file), exist_ok=True)
            existing = []
            if os.path.exists(features_file):
                with open(features_file) as f:
                    existing = json.load(f)
            existing.append(entry)
            # Keep last 1000 entries
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
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average returning full array."""
        alpha = 2 / (period + 1)
        result = np.zeros(len(data))
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _ema_from_array(data: np.ndarray, period: int) -> float:
        """EMA of a small array, returns final value."""
        if len(data) == 0:
            return 0.0
        alpha = 2 / (period + 1)
        val = data[0]
        for i in range(1, len(data)):
            val = alpha * data[i] + (1 - alpha) * val
        return float(val)
