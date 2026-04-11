"""
IronCondor Sentiment Analyzer — Multi-Source Market & Per-Symbol Sentiment
============================================================================
Purpose: Adds sentiment intelligence to PutSeller/IronCondor, which previously
         traded purely on delta/IV/regime filters with ZERO sentiment awareness.

Sources (4 categories):

  Market-Wide:
    1. VIX via VIXY ETF   (0.35) — implied volatility gauge (Alpaca bars)
    2. SPY trend           (0.30) — 5-day + 1-day return vs 20d MA (Alpaca bars)
    3. Fear & Greed Index  (0.20) — CNN market sentiment gauge
    4. Market breadth      (0.15) — gainers vs losers (Alpaca screener)

  Per-Symbol:
    1. Alpaca News          (0.40) — Benzinga headlines, keyword-scored
    2. Options Flow         (0.35) — unusual volume/OI ratio from Alpaca
    3. Finnhub Sentiment    (0.25) — analyst recs + news sentiment score

Sentinel score: -1.0 (extreme bearish) to +1.0 (extreme bullish)
Market-wide cached 5 min; per-symbol cached 15-20 min.
"""
from __future__ import annotations
import logging
import time
import re
import json
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("ironcondor.sentiment")

# ── Keyword scoring lists ─────────────────────────────────────────────────────
BULLISH_WORDS = [
    "upgrade", "upgrades", "upgraded",
    "outperform", "overweight", "buy",
    "beat", "beats", "topped", "exceeded", "surpassed",
    "raises guidance", "raised guidance", "raised forecast",
    "strong", "strength", "record", "all-time high", "breakout",
    "bullish", "positive", "upside", "momentum",
    "deal", "acquisition", "buyback", "dividend", "special dividend",
    "fda approval", "approved", "approval",
    "partnership", "contract", "wins contract",
]

BEARISH_WORDS = [
    "downgrade", "downgrades", "downgraded",
    "underperform", "underweight", "sell",
    "miss", "misses", "missed", "below estimates", "fell short",
    "cuts guidance", "cut guidance", "lowers guidance", "reduced forecast",
    "weak", "weakness", "concern", "warning", "caution",
    "bearish", "negative", "downside",
    "recall", "investigation", "lawsuit", "subpoena", "sec probe",
    "layoffs", "restructuring", "charges", "write-down", "impairment",
    "loss", "losses", "shortfall", "decline", "slump",
]

_BULL_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BULLISH_WORDS) + r")\b", re.IGNORECASE
)
_BEAR_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BEARISH_WORDS) + r")\b", re.IGNORECASE
)


def _score_text(text: str, weight: float = 1.0) -> float:
    """Score a piece of text. Returns ±weight per matched word, clipped to ±1."""
    bull = len(_BULL_RE.findall(text))
    bear = len(_BEAR_RE.findall(text))
    if bull == 0 and bear == 0:
        return 0.0
    return float(np.clip((bull - bear) / max(bull + bear, 1), -1.0, 1.0)) * weight


class IronCondorSentiment:
    """Multi-source sentiment analyzer for the IronCondor/PutSeller engine."""

    MARKET_CACHE_TTL = 300     # 5 min for VIX/SPY/breadth/fear-greed
    NEWS_CACHE_TTL   = 1200    # 20 min per-symbol news
    FLOW_CACHE_TTL   = 600     # 10 min per-symbol options flow
    FINNHUB_CACHE_TTL = 900    # 15 min per-symbol finnhub data

    def __init__(self, api_client=None, finnhub_key: str = ""):
        self.api = api_client
        self._finnhub_key = finnhub_key

        # Market-wide cache
        self._composite: float = 0.0
        self._signals: Dict[str, float] = {}
        self._last_market_fetch: float = 0.0

        # Per-symbol caches
        self._news_cache: Dict[str, Tuple[float, float]] = {}
        self._flow_cache: Dict[str, Tuple[float, float]] = {}
        self._finnhub_cache: Dict[str, Tuple[float, float]] = {}

        # Lazy-init
        self._news_client = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_sentiment(self) -> float:
        """Get composite market-wide sentiment score. Cached 5 minutes.
        Returns: -1.0 (extreme bearish) to +1.0 (extreme bullish)
        """
        if time.time() - self._last_market_fetch < self.MARKET_CACHE_TTL:
            return self._composite

        signals: Dict[str, float] = {}

        try:
            signals["vix"] = self._fetch_vix_signal()
        except Exception as e:
            log.debug(f"Sentiment: VIX fetch failed: {e}")

        try:
            signals["spy_trend"] = self._fetch_spy_trend()
        except Exception as e:
            log.debug(f"Sentiment: SPY trend failed: {e}")

        try:
            signals["fear_greed"] = self._fetch_fear_greed()
        except Exception as e:
            log.debug(f"Sentiment: Fear & Greed failed: {e}")

        try:
            signals["breadth"] = self._fetch_market_breadth()
        except Exception as e:
            log.debug(f"Sentiment: Breadth fetch failed: {e}")

        if not signals:
            return self._composite

        weights = {"vix": 0.35, "spy_trend": 0.30, "fear_greed": 0.20, "breadth": 0.15}
        ws, wt = 0.0, 0.0
        for k, w in weights.items():
            if k in signals:
                ws += signals[k] * w
                wt += w

        if wt > 0:
            self._composite = float(np.clip(ws / wt, -1.0, 1.0))

        self._signals = signals
        self._last_market_fetch = time.time()

        log.info(
            f"[Sentiment] composite={self._composite:+.3f} | "
            + " | ".join(f"{k}={v:+.3f}" for k, v in signals.items())
        )
        return self._composite

    def get_per_symbol_sentiment(self, symbol: str) -> float:
        """Get per-symbol sentiment combining news + options flow + finnhub.
        Returns: -0.40 to +0.40 adjustment to add to global sentiment.
        """
        news_score = self._get_symbol_news_score(symbol)
        flow_score = self._get_options_flow_signal(symbol)
        finnhub_score = self._get_finnhub_score(symbol)

        # Weight: 40% news, 35% options flow, 25% finnhub
        combined = news_score * 0.40 + flow_score * 0.35 + finnhub_score * 0.25
        return float(np.clip(combined, -0.40, 0.40))

    def get_combined_score(self, symbol: str) -> float:
        """Get the full combined sentiment: market-wide + per-symbol.
        Returns: -1.0 to +1.0
        """
        market = self.get_sentiment()
        per_sym = self.get_per_symbol_sentiment(symbol)
        return float(np.clip(market + per_sym, -1.0, 1.0))

    def status(self) -> Dict:
        """Return sentiment status for dashboard/logging."""
        return {
            "composite": round(self._composite, 3),
            "sources": {k: round(v, 3) for k, v in self._signals.items()},
            "source_note": "VIX + SPY + Fear&Greed + breadth + news + options flow + finnhub",
            "last_update": (
                datetime.fromtimestamp(self._last_market_fetch).isoformat()
                if self._last_market_fetch > 0 else None
            ),
            "finnhub_enabled": bool(self._finnhub_key),
            "caches": {
                "news": len(self._news_cache),
                "flow": len(self._flow_cache),
                "finnhub": len(self._finnhub_cache),
            },
        }

    # ── Market-wide signals ───────────────────────────────────────────────────

    def _fetch_vix_signal(self) -> float:
        """VIXY ETF bars as VIX proxy. Higher VIXY = more fear = bearish."""
        if self.api is None:
            return 0.0
        from alpaca.data.timeframe import TimeFrame
        bars = self.api.get_bars("VIXY", TimeFrame.Day, days=10)
        if not bars or len(bars) < 3:
            return 0.0
        closes = [float(b.close) for b in bars]
        current = closes[-1]
        avg = float(np.mean(closes[-5:]))
        if avg <= 0:
            return 0.0
        ratio = current / avg
        return float(np.clip(-(ratio - 1.0) * 8.0, -1.0, 1.0))

    def _fetch_spy_trend(self) -> float:
        """SPY trend: 5-day return + 1-day return + distance from 20d MA."""
        if self.api is None:
            return 0.0
        from alpaca.data.timeframe import TimeFrame
        bars = self.api.get_bars("SPY", TimeFrame.Day, days=30)
        if not bars or len(bars) < 6:
            return 0.0
        closes = [float(b.close) for b in bars]
        ret_5d = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0.0
        ret_1d = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0.0
        ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else closes[-1]
        vs_ma = (closes[-1] - ma20) / ma20 if ma20 > 0 else 0.0
        signal = ret_5d * 8.0 + ret_1d * 15.0 + vs_ma * 10.0
        return float(np.clip(signal, -1.0, 1.0))

    def _fetch_fear_greed(self) -> float:
        """CNN Fear & Greed Index — market-wide sentiment gauge.
        0 = extreme fear, 100 = extreme greed.
        Maps to -1.0 (fear) to +1.0 (greed).
        """
        try:
            import requests as req
            resp = req.get(
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                },
                timeout=8,
            )
            if resp.status_code != 200:
                return 0.0
            data = resp.json()
            score = data.get("fear_and_greed", {}).get("score", 50)
            # Map 0-100 to -1.0 to +1.0
            normalized = (float(score) - 50) / 50
            return float(np.clip(normalized, -1.0, 1.0))
        except Exception as e:
            log.debug(f"Fear & Greed fetch failed: {e}")
            return 0.0

    def _fetch_market_breadth(self) -> float:
        """Gainers vs losers ratio from Alpaca screener."""
        if self.api is None:
            return 0.0
        try:
            from alpaca.data.historical.screener import ScreenerClient
            from alpaca.data.requests import MarketMoversRequest
            sc = ScreenerClient(
                api_key=self.api.config.API_KEY,
                secret_key=self.api.config.API_SECRET,
            )
            movers = sc.get_market_movers(MarketMoversRequest(top=20))
            gainers = len(getattr(movers, "gainers", []) or [])
            losers = len(getattr(movers, "losers", []) or [])
            total = gainers + losers
            if total == 0:
                return 0.0
            return float(np.clip((gainers - losers) / total, -1.0, 1.0))
        except Exception as e:
            log.debug(f"Breadth fetch failed: {e}")
            return 0.0

    # ── Per-symbol: Alpaca News ───────────────────────────────────────────────

    def _get_news_client(self):
        """Lazy-init Alpaca NewsClient."""
        if self._news_client is None and self.api is not None:
            try:
                from alpaca.data.historical.news import NewsClient
                self._news_client = NewsClient(
                    api_key=self.api.config.API_KEY,
                    secret_key=self.api.config.API_SECRET,
                )
            except Exception as e:
                log.debug(f"NewsClient init failed: {e}")
        return self._news_client

    def _get_symbol_news_score(self, symbol: str) -> float:
        """Fetch and score last 4h of headlines for a symbol. Cached 20 min."""
        now = time.time()
        cached = self._news_cache.get(symbol)
        if cached and (now - cached[1]) < self.NEWS_CACHE_TTL:
            return cached[0]
        score = self._fetch_news_score(symbol)
        self._news_cache[symbol] = (score, now)
        return score

    def _fetch_news_score(self, symbol: str) -> float:
        """Fetch Alpaca news for symbol and keyword-score headlines."""
        client = self._get_news_client()
        if client is None:
            return 0.0
        try:
            from alpaca.data.requests import NewsRequest
            start = datetime.now(timezone.utc) - timedelta(hours=4)
            req = NewsRequest(
                symbols=symbol, start=start, limit=10,
                sort="desc", include_content=False,
            )
            news_set = client.get_news(req)
            articles: List = list(news_set) if news_set else []
            if not articles:
                return 0.0

            raw_scores = []
            for article in articles[:8]:
                headline = getattr(article, "headline", "") or ""
                summary = getattr(article, "summary", "") or ""
                s = _score_text(headline, 1.0) + _score_text(summary, 0.5)
                raw_scores.append(s)

            if not raw_scores:
                return 0.0
            final = float(np.clip(np.mean(raw_scores), -1.0, 1.0))
            log.info(f"[News] {symbol}: {len(articles)} articles -> score {final:+.2f}")
            return final
        except Exception as e:
            log.debug(f"[News] {symbol}: fetch failed ({e})")
            return 0.0

    # ── Per-symbol: Options Flow ──────────────────────────────────────────────

    def _get_options_flow_signal(self, symbol: str) -> float:
        """Detect unusual options activity. Cached 10 min."""
        now = time.time()
        cache_key = f"_flow_{symbol}"
        cached = self._flow_cache.get(cache_key)
        if cached and (now - cached[1]) < self.FLOW_CACHE_TTL:
            return cached[0]
        score = self._fetch_options_flow(symbol)
        self._flow_cache[cache_key] = (score, now)
        return score

    def _fetch_options_flow(self, symbol: str) -> float:
        """Fetch options chain snapshot and analyze volume vs OI."""
        if self.api is None:
            return 0.0
        try:
            import requests as req
            headers = {
                "APCA-API-KEY-ID": self.api.config.API_KEY,
                "APCA-API-SECRET-KEY": self.api.config.API_SECRET,
                "accept": "application/json",
            }
            # Snapshots for volume data
            snap_url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}"
            snap_resp = req.get(
                snap_url, headers=headers,
                params={"limit": 40, "feed": "indicative"},
                timeout=10,
            )
            snap_resp.raise_for_status()
            snapshots = snap_resp.json().get("snapshots", {})
            if not snapshots:
                return 0.0

            # Contracts for OI
            base_url = getattr(self.api.config, "BASE_URL", "https://paper-api.alpaca.markets")
            contract_url = f"{base_url}/v2/options/contracts"
            contract_resp = req.get(
                contract_url, headers=headers,
                params={"underlying_symbols": symbol, "status": "active", "limit": 100},
                timeout=10,
            )
            oi_lookup: dict = {}
            if contract_resp.status_code == 200:
                for c in contract_resp.json().get("option_contracts", []):
                    oi_lookup[c.get("symbol", "")] = int(c.get("open_interest", 0) or 0)

            call_volume, put_volume = 0, 0
            call_oi, put_oi = 0, 0
            unusual_contracts = 0

            for sym, snap in snapshots.items():
                daily_bar = snap.get("dailyBar", {}) or {}
                vol = int(daily_bar.get("v", 0) or 0)
                oi = oi_lookup.get(sym, 0)

                suffix = sym[len(symbol):] if sym.startswith(symbol) else sym
                is_call = "C" in suffix[:8]

                if is_call:
                    call_volume += vol
                    call_oi += oi
                else:
                    put_volume += vol
                    put_oi += oi

                if oi > 0 and vol > oi * 2:
                    unusual_contracts += 1

            total_vol = call_volume + put_volume
            total_oi = call_oi + put_oi
            if total_vol == 0 and total_oi == 0:
                return 0.0

            vol_oi_ratio = total_vol / max(total_oi, 1)

            if total_vol > 0:
                call_pct = call_volume / total_vol
                direction_signal = (call_pct - 0.5) * 4.0
            else:
                direction_signal = 0.0

            intensity = min(2.0, vol_oi_ratio) / 2.0
            flow_score = float(np.clip(direction_signal * max(0.3, intensity), -1.0, 1.0))

            if unusual_contracts > 0 or vol_oi_ratio > 0.5:
                log.info(
                    f"[Flow] {symbol}: C_vol={call_volume} P_vol={put_volume} "
                    f"C_OI={call_oi} P_OI={put_oi} | V/OI={vol_oi_ratio:.2f} "
                    f"| unusual={unusual_contracts} | signal={flow_score:+.2f}"
                )
            return flow_score
        except Exception as e:
            log.debug(f"[Flow] {symbol}: fetch failed ({e})")
            return 0.0

    # ── Per-symbol: Finnhub ───────────────────────────────────────────────────

    def _get_finnhub_score(self, symbol: str) -> float:
        """Finnhub analyst recommendations + news sentiment. Cached 15 min."""
        if not self._finnhub_key:
            return 0.0
        now = time.time()
        cached = self._finnhub_cache.get(symbol)
        if cached and (now - cached[1]) < self.FINNHUB_CACHE_TTL:
            return cached[0]
        score = self._fetch_finnhub(symbol)
        self._finnhub_cache[symbol] = (score, now)
        return score

    def _fetch_finnhub(self, symbol: str) -> float:
        """Fetch Finnhub analyst recommendation trends + news sentiment.

        Analyst recs: strongBuy/buy/hold/sell/strongSell counts -> bullish/bearish ratio
        News sentiment: average sentiment score from recent articles
        """
        try:
            import requests as req
            base = "https://finnhub.io/api/v1"
            params = {"symbol": symbol, "token": self._finnhub_key}

            # 1. Analyst recommendation trends
            rec_score = 0.0
            try:
                resp = req.get(f"{base}/stock/recommendation", params=params, timeout=8)
                if resp.status_code == 200:
                    recs = resp.json()
                    if recs and len(recs) > 0:
                        latest = recs[0]  # most recent period
                        sb = latest.get("strongBuy", 0)
                        b = latest.get("buy", 0)
                        h = latest.get("hold", 0)
                        s = latest.get("sell", 0)
                        ss = latest.get("strongSell", 0)
                        total = sb + b + h + s + ss
                        if total > 0:
                            bullish = sb * 2 + b
                            bearish = ss * 2 + s
                            rec_score = float(np.clip((bullish - bearish) / total, -1.0, 1.0))
            except Exception as e:
                log.debug(f"[Finnhub] {symbol} recs failed: {e}")

            # 2. News sentiment
            news_score = 0.0
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                resp = req.get(
                    f"{base}/company-news",
                    params={"symbol": symbol, "from": week_ago, "to": today, "token": self._finnhub_key},
                    timeout=8,
                )
                if resp.status_code == 200:
                    articles = resp.json()
                    if articles:
                        scores = []
                        for a in articles[:10]:
                            headline = a.get("headline", "")
                            summary = a.get("summary", "")
                            s = _score_text(headline, 1.0) + _score_text(summary, 0.5)
                            scores.append(s)
                        if scores:
                            news_score = float(np.clip(np.mean(scores), -1.0, 1.0))
            except Exception as e:
                log.debug(f"[Finnhub] {symbol} news failed: {e}")

            # Combine: 60% analyst recs (more reliable), 40% news sentiment
            combined = rec_score * 0.60 + news_score * 0.40
            final = float(np.clip(combined, -1.0, 1.0))

            if abs(final) > 0.1:
                log.info(f"[Finnhub] {symbol}: recs={rec_score:+.2f} news={news_score:+.2f} -> {final:+.2f}")
            return final
        except Exception as e:
            log.debug(f"[Finnhub] {symbol}: fetch failed ({e})")
            return 0.0
