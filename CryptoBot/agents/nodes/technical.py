"""Technical Analysis Agent — evaluates price action, indicators, and chart patterns."""
from __future__ import annotations
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents import config as acfg
from agents.state import TradingState, AgentAnalysis
import json


SYSTEM_PROMPT = """You are an expert crypto technical analyst with deep knowledge of market 
microstructure. You receive real-time market data for a cryptocurrency AND broader market 
context from live data feeds.

Analyze:
1. RSI context (overbought >70, oversold <30, momentum zone 40-60)
2. Trend alignment (is the proposed direction WITH or AGAINST the trend?)
3. Volatility regime (high vol = wider stops needed, low vol = breakout potential)
4. Price momentum (1h and 4h change — accelerating or decelerating?)
5. Market regime context (trending vs ranging — different strategies apply)
6. Funding rates — positive = crowded longs (contrarian short edge), negative = crowded shorts
7. Open interest changes — rising OI + price rise = strong trend, rising OI + price fall = forced selling
8. BTC dominance trend — rising BTC dom often means altcoin weakness
9. Market breadth — what % of top coins are green? Broad rally vs isolated move?
10. Where does this coin sit relative to the top gainers/losers today?
11. BTC detailed metrics — ATH distance, volume/mcap ratio, 24h range (volatility proxy)
12. Reddit headlines — what is the crypto community talking about right now? Any catalysts?

IMPORTANT — OVER-RELIANCE SAFEGUARD:
External market data is CONTEXT, not a veto. A strong technical setup (good RSI, clear trend,
high ML confidence) should NOT be killed just because one external indicator looks uncertain.
Weight your analysis: 70% technical indicators + 30% external context. If data quality is
DEGRADED or POOR, reduce external data weight to 10-15% and rely more on chart data.

Respond with EXACTLY this JSON (no markdown, no extra text):
{
  "verdict": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence technical analysis incorporating market data",
  "flags": ["list", "of", "warning", "flags"]
}

Verdict meanings:
- STRONG_BUY/SELL: Technical setup is excellent, high conviction
- BUY/SELL: Decent setup with some caveats
- HOLD: Mixed signals, not enough edge to justify the trade"""


def _build_prompt(signal: Dict, market: Dict, memory: Dict = None) -> str:
    mkt = market.get("summary", {})
    funding = market.get("funding_rates", {})
    mem = memory or {}
    
    # Find funding rate for this specific coin
    coin_base = signal['symbol'].replace("/USD", "").replace("USD", "").replace("PI_", "").replace("XBTUSD", "BTC")
    coin_funding_key = f"{coin_base}USDT"
    coin_funding = funding.get(coin_funding_key, None)
    funding_str = f"{coin_funding:+.4f}%" if coin_funding is not None else "unavailable"
    
    return f"""Evaluate this proposed {signal['direction']} trade on {signal['symbol']}:

PROPOSED TRADE:
- Direction: {signal['direction']}
- ML Confidence: {signal['confidence']:.2%}
- ML Directional Score: {signal['ml_score']:.3f}
- Entry Reason: {signal['reason']}

TECHNICAL DATA:
- RSI(14): {signal['rsi']:.1f}
- Trend: {signal['trend']} (slope: {signal['trend_slope']:.4f})
- Volatility: {signal['volatility']:.4f}
- Current Price: ${signal['current_price']:,.2f}
- 1h Change: {signal['price_change_1h']:+.2f}%
- 4h Change: {signal['price_change_4h']:+.2f}%

MARKET REGIME: {signal['regime']}

LIVE MARKET DATA (from external feeds):
- Fear & Greed Index: {mkt.get('fear_greed_value', '?')} ({mkt.get('fear_greed_label', '?')})
- BTC Dominance: {mkt.get('btc_dominance', '?')}%
- Total Crypto Market Cap: ${mkt.get('total_market_cap_billion', '?')}B ({mkt.get('market_cap_change_24h_pct', '?'):+}% 24h)
- Market Breadth: {mkt.get('breadth_pct_positive', '?')}% of top 20 coins are green
- Top Gainers: {mkt.get('top_gainers', 'none')}
- Top Losers: {mkt.get('top_losers', 'none')}
- This Coin's Funding Rate: {funding_str}
- All Funding Rates: {mkt.get('funding_rates', 'unavailable')}
- Open Interest: {mkt.get('open_interest', 'unavailable')}
- Trending on CoinGecko: {mkt.get('trending_coins', 'none')}

BTC DETAILED METRICS:
{mkt.get('btc_metrics', 'unavailable')}

REDDIT CRYPTO SENTIMENT (r/CryptoCurrency + r/Bitcoin):
{mkt.get('reddit_sentiment', 'unavailable')}
Top Headlines:
{mkt.get('reddit_headlines', 'unavailable')}

DATA QUALITY: {mkt.get('data_quality', 'UNKNOWN')}
{mkt.get('data_quality_warning', '')}

{mem.get('technical', 'AGENT MEMORY: No history yet.')}
Symbol history: {mem.get('symbols', dict()).get(signal['symbol'], 'No history for this symbol.')}

Is this trade technically justified? Consider both the coin's indicators AND the broader market picture."""


def technical_analyst(state: TradingState) -> Dict:
    """LangGraph node: analyze each signal from a technical perspective."""
    import time as _t; _start = _t.time()
    import logging; logging.getLogger("agent_advisor").info(f"[PARALLEL] technical_analyst START (thread={__import__('threading').current_thread().name})")
    if not acfg.OPENAI_API_KEY:
        return {"technical_analyses": [
            AgentAnalysis(
                agent="technical",
                verdict="HOLD",
                confidence=0.5,
                reasoning="No API key configured — skipping technical analysis.",
                flags=["no_api_key"],
            )
            for _ in state["signals"]
        ]}

    llm = ChatOpenAI(
        model=acfg.TECHNICAL_MODEL,
        api_key=acfg.OPENAI_API_KEY,
        temperature=acfg.AGENT_TEMPERATURE,
        max_tokens=acfg.AGENT_MAX_TOKENS,
        timeout=acfg.AGENT_TIMEOUT,
    )

    analyses = []
    for signal in state["signals"]:
        try:
            resp = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=_build_prompt(signal, state.get("market_context", {}), state.get("agent_memory", {}))),
            ])
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            analyses.append(AgentAnalysis(
                agent="technical",
                verdict=data.get("verdict", "HOLD"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                flags=data.get("flags", []),
            ))
        except Exception as e:
            analyses.append(AgentAnalysis(
                agent="technical",
                verdict="HOLD",
                confidence=0.5,
                reasoning=f"Analysis failed: {e}",
                flags=["error"],
            ))

    import logging as _log; _log.getLogger("agent_advisor").info(f"[PARALLEL] technical_analyst END ({_t.time()-_start:.1f}s)")
    return {"technical_analyses": analyses}
