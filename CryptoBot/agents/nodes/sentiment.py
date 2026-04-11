"""Sentiment Analysis Agent — evaluates news, social signals, and market mood."""
from __future__ import annotations
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents import config as acfg
from agents.state import TradingState, AgentAnalysis
import json


SYSTEM_PROMPT = """You are an expert crypto sentiment and macro analyst with access to LIVE 
market data feeds. You receive real-time Fear & Greed data, funding rates, trending coins, 
market breadth, and a proposed trade. Your job is to evaluate whether market CONDITIONS 
actually support this trade.

You know how crypto markets work:
- Extreme fear (F&G < 25) historically = buying opportunities (contrarian)
- Extreme greed (F&G > 75) historically = distribution zones (contrarian)
- Positive funding = longs are overleveraged (crowded trade risk)
- Negative funding = shorts are overleveraged (short squeeze risk)
- Rising BTC dominance = "risk off" (altcoin weakness, capital flows to BTC)
- Falling BTC dominance = "risk on" (altcoin season, speculative capital rotating)
- Trending coins on CoinGecko = retail FOMO attention (often lagging indicator)
- High breadth (>70% green) = broad rally = healthier trend
- Low breadth (<30% green) = narrow or bearish = defensive posture

Analyze:
1. Fear & Greed alignment — does the index support the trade direction?
2. Funding rate signal — is the crowd positioned WITH or AGAINST this trade?
3. Market breadth — is the broader market confirming or diverging?
4. BTC dominance trend — risk-on or risk-off environment?
5. Trending/retail attention — FOMO top risk or genuine momentum?
6. Portfolio stress (consecutive losses, daily P&L) — should we be aggressive or defensive?
7. Macro regime — put it all together: what is the MACRO THESIS right now?

Respond with EXACTLY this JSON (no markdown, no extra text):
{
  "verdict": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence sentiment analysis using the live data",
  "flags": ["list", "of", "warning", "flags"]
}

Be contrarian when sentiment is extreme. Be cautious during high consecutive losses.
Use the ACTUAL data to support your reasoning — don't just echo the bot's internal score.

IMPORTANT — OVER-RELIANCE SAFEGUARD:
External data is CONTEXT, not a veto. A trade with strong internal signals (high ML confidence,
good regime, solid RSI) should NOT be killed just because one sentiment indicator looks bad.
Weight your analysis: 60% live sentiment data + 40% internal signals. Never override a clear
setup based on a single external metric. If data quality is DEGRADED or POOR, reduce external
data weight to 25-30% and note which feeds you trust less."""


def _build_prompt(signal: Dict, market: Dict, memory: Dict = None) -> str:
    mkt = market.get("summary", {})
    funding = market.get("funding_rates", {})
    mem = memory or {}
    
    # Find funding rate for this specific coin
    coin_base = signal['symbol'].replace("/USD", "").replace("USD", "").replace("PI_", "").replace("XBTUSD", "BTC")
    coin_funding_key = f"{coin_base}USDT"
    coin_funding = funding.get(coin_funding_key, None)
    funding_str = f"{coin_funding:+.4f}%" if coin_funding is not None else "unavailable"
    
    trending = market.get("trending", [])
    coin_is_trending = coin_base in [t.upper() for t in trending]
    
    return f"""Evaluate sentiment for this proposed {signal['direction']} trade on {signal['symbol']}:

PROPOSED TRADE:
- Direction: {signal['direction']}
- ML Confidence: {signal['confidence']:.2%}

BOT'S INTERNAL SENTIMENT SCORE: {signal['sentiment']:.2f} (range: -1.0 bearish to +1.0 bullish)
MARKET REGIME: {signal['regime']}

LIVE FEAR & GREED INDEX: {mkt.get('fear_greed_value', '?')} — {mkt.get('fear_greed_label', '?')}
  (0=Extreme Fear, 25=Fear, 50=Neutral, 75=Greed, 100=Extreme Greed)

LIVE MARKET BREADTH:
- {mkt.get('breadth_pct_positive', '?')}% of top 20 coins are green (24h)
- Total crypto market cap: ${mkt.get('total_market_cap_billion', '?')}B ({mkt.get('market_cap_change_24h_pct', '?'):+}% 24h)
- 24h volume: ${mkt.get('total_volume_24h_billion', '?')}B

BTC DOMINANCE: {mkt.get('btc_dominance', '?')}%  (rising = risk-off, falling = altcoin season)

FUNDING RATES (positive = longs pay shorts = crowded long):
- This coin ({coin_base}): {funding_str}
- All rates: {mkt.get('funding_rates', 'unavailable')}

TRENDING ON COINGECKO: {mkt.get('trending_coins', 'none')}
  This coin is {'TRENDING (retail FOMO attention)' if coin_is_trending else 'NOT trending'}

TOP MOVERS:
- Gainers: {mkt.get('top_gainers', 'none')}
- Losers: {mkt.get('top_losers', 'none')}

REDDIT CRYPTO SENTIMENT (r/CryptoCurrency + r/Bitcoin):
{mkt.get('reddit_sentiment', 'unavailable')}
Top Headlines:
{mkt.get('reddit_headlines', 'unavailable')}

DeFi ECOSYSTEM:
- {mkt.get('defi_tvl', 'unavailable')}
- Top Chains: {mkt.get('defi_top_chains', 'unavailable')}
- {mkt.get('stablecoin_yields', 'unavailable')} (high yields = risk appetite, low = risk aversion)

BTC DETAILED METRICS:
{mkt.get('btc_metrics', 'unavailable')}

DATA QUALITY: {mkt.get('data_quality', 'UNKNOWN')}
{mkt.get('data_quality_warning', '')}

PORTFOLIO STRESS:
- Open Positions: {signal['open_position_count']}
- Current Exposure: {signal['existing_exposure']}
- Daily P&L: {signal['daily_pnl_pct']:+.2f}%
- Consecutive Losses: {signal['consecutive_losses']}

Given ALL of this real-time data, does market sentiment support this {signal['direction']} on {signal['symbol']}?
What is the macro thesis right now, and does this trade align with it?

{mem.get('sentiment', 'AGENT MEMORY: No history yet.')}
Symbol history: {mem.get('symbols', dict()).get(signal['symbol'], 'No history for this symbol.')}"""


def sentiment_analyst(state: TradingState) -> Dict:
    """LangGraph node: analyze each signal from a sentiment/macro perspective."""
    import time as _t; _start = _t.time()
    import logging; logging.getLogger("agent_advisor").info(f"[PARALLEL] sentiment_analyst START (thread={__import__('threading').current_thread().name})")
    if not acfg.OPENAI_API_KEY:
        return {"sentiment_analyses": [
            AgentAnalysis(
                agent="sentiment",
                verdict="HOLD",
                confidence=0.5,
                reasoning="No API key configured — skipping sentiment analysis.",
                flags=["no_api_key"],
            )
            for _ in state["signals"]
        ]}

    llm = ChatOpenAI(
        model=acfg.SENTIMENT_MODEL,
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
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            analyses.append(AgentAnalysis(
                agent="sentiment",
                verdict=data.get("verdict", "HOLD"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                flags=data.get("flags", []),
            ))
        except Exception as e:
            analyses.append(AgentAnalysis(
                agent="sentiment",
                verdict="HOLD",
                confidence=0.5,
                reasoning=f"Analysis failed: {e}",
                flags=["error"],
            ))

    import logging as _log; _log.getLogger("agent_advisor").info(f"[PARALLEL] sentiment_analyst END ({_t.time()-_start:.1f}s)")
    return {"sentiment_analyses": analyses}
