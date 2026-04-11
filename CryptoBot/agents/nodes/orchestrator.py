"""Orchestrator Agent — aggregates all analyst opinions into a final recommendation."""
from __future__ import annotations
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents import config as acfg
from agents.state import TradingState
import json


SYSTEM_PROMPT = """You are the chief trading strategist and final decision maker. 
You receive analyses from three specialist agents who have each reviewed LIVE market data:
1. Technical Analyst — chart patterns, indicators, price action, funding rates, market breadth
2. Sentiment Analyst — Fear & Greed Index, funding rates, BTC dominance, trending coins, macro thesis
3. Risk Manager — portfolio risk, leverage risk, liquidation risk, systemic market risk

You also have direct access to the live market summary. Use it to catch anything the specialists missed.

Rules:
- If Risk Manager says SELL or STRONG_SELL, you should heavily weight that (capital preservation first)
- If Technical and Sentiment disagree, lean toward the more confident one
- If all three agree, boost confidence
- If on a losing streak (>3 consecutive losses), bias toward SKIP unless conviction is very high
- If Fear & Greed is extreme (>75 or <25), factor that into sizing
- If funding rates are extreme for this coin, that's a contrarian signal worth noting
- Always explain WHY you agree or disagree with each analyst
- Reference the ACTUAL market data in your reasoning (e.g., "F&G at 82 signals greed...")
- Check the DATA QUALITY indicator — if DEGRADED or POOR, discount external-data-based arguments

IMPORTANT — OVER-RELIANCE SAFEGUARD:
You are the final decision maker. External market data is SUPPLEMENTARY intelligence.
A strong consensus from your analysts based on solid technical/risk analysis should NOT
be overridden by a single external data point (e.g., one bad Reddit headline or one
stale DeFi metric). If data quality is POOR, note it and weight internal analysis higher.

Respond with EXACTLY this JSON array (no markdown, no extra text):
[
  {
    "symbol": "BTC/USD",
    "action": "TAKE|SKIP|REDUCE_SIZE",
    "confidence": 0.0-1.0,
    "reasoning": "3-4 sentence synthesis explaining the decision with specific data points",
    "size_modifier": 0.5-1.5,
    "tuning_suggestions": [{"param": "PARAM_NAME", "direction": "increase|decrease", "reasoning": "why"}]
  }
]

Actions:
- TAKE: Execute the trade as proposed
- SKIP: Do not take this trade
- REDUCE_SIZE: Take the trade but with reduced position size (use size_modifier < 1.0)

size_modifier: 1.0 = normal size, 0.5 = half size, 1.5 = 150% size (only if all agents agree strongly)

tuning_suggestions: OPTIONAL. Only include if you see a clear pattern that a parameter should change.
Use the ADAPTIVE PARAMETER TUNING section below for available parameters and bounds.
Only suggest changes after observing consistent patterns across multiple cycles."""


def _build_prompt(state: TradingState) -> str:
    lines = []
    
    # Lead with the live market snapshot
    mkt = state.get("market_context", {}).get("summary", {})
    if mkt:
        lines.append("═══ LIVE MARKET SNAPSHOT ═══")
        lines.append(f"Fear & Greed: {mkt.get('fear_greed_value', '?')} ({mkt.get('fear_greed_label', '?')})")
        lines.append(f"BTC Dominance: {mkt.get('btc_dominance', '?')}%")
        lines.append(f"Total Market Cap: ${mkt.get('total_market_cap_billion', '?')}B ({mkt.get('market_cap_change_24h_pct', '?'):+}% 24h)")
        lines.append(f"Market Breadth: {mkt.get('breadth_pct_positive', '?')}% of top 20 coins green")
        lines.append(f"Funding Rates: {mkt.get('funding_rates', 'unavailable')}")
        lines.append(f"Open Interest: {mkt.get('open_interest', 'unavailable')}")
        lines.append(f"Trending: {mkt.get('trending_coins', 'none')}")
        lines.append(f"Top Gainers: {mkt.get('top_gainers', 'none')}")
        lines.append(f"Top Losers: {mkt.get('top_losers', 'none')}")
        lines.append("")
        lines.append("═══ EXTENDED DATA ═══")
        lines.append(f"Reddit Sentiment: {mkt.get('reddit_sentiment', 'unavailable')}")
        if mkt.get('reddit_headlines'):
            lines.append(f"Reddit Headlines:\n{mkt.get('reddit_headlines')}")
        lines.append(f"DeFi TVL: {mkt.get('defi_tvl', 'unavailable')}")
        lines.append(f"DeFi Top Chains: {mkt.get('defi_top_chains', 'unavailable')}")
        lines.append(f"Stablecoin Yields: {mkt.get('stablecoin_yields', 'unavailable')}")
        lines.append(f"BTC Metrics: {mkt.get('btc_metrics', 'unavailable')}")
        lines.append("")
        quality = mkt.get('data_quality', 'UNKNOWN')
        warning = mkt.get('data_quality_warning', '')
        lines.append(f"DATA QUALITY: {quality}")
        if warning:
            lines.append(warning)
        lines.append("")
    
    # Agent memory context
    mem = state.get("agent_memory", {})
    if mem:
        lines.append("═══ AGENT MEMORY ═══")
        lines.append(mem.get("orchestrator", "No history yet."))
        # Per-symbol memory for signals being evaluated
        for sig in state["signals"]:
            sym = sig["symbol"]
            sym_mem = mem.get("symbols", {}).get(sym)
            if sym_mem:
                lines.append(f"  {sym_mem}")
        lines.append("")

    # Adaptive tuning context
    tuning = state.get("tuning_context", "")
    if tuning:
        lines.append(tuning)
        lines.append("")

    lines.append("Here are the specialist analyses for each proposed trade:\n")

    for i, signal in enumerate(state["signals"]):
        sym = signal["symbol"]
        direction = signal["direction"]
        lines.append(f"═══ TRADE {i+1}: {direction} {sym} (ML conf: {signal['confidence']:.2%}) ═══")

        # Technical
        if i < len(state.get("technical_analyses", [])):
            ta = state["technical_analyses"][i]
            lines.append(f"  TECHNICAL: {ta['verdict']} ({ta['confidence']:.0%})")
            lines.append(f"    {ta['reasoning']}")
            if ta["flags"]:
                lines.append(f"    Flags: {', '.join(ta['flags'])}")

        # Sentiment
        if i < len(state.get("sentiment_analyses", [])):
            sa = state["sentiment_analyses"][i]
            lines.append(f"  SENTIMENT: {sa['verdict']} ({sa['confidence']:.0%})")
            lines.append(f"    {sa['reasoning']}")
            if sa["flags"]:
                lines.append(f"    Flags: {', '.join(sa['flags'])}")

        # Risk
        if i < len(state.get("risk_analyses", [])):
            ra = state["risk_analyses"][i]
            lines.append(f"  RISK: {ra['verdict']} ({ra['confidence']:.0%})")
            lines.append(f"    {ra['reasoning']}")
            if ra["flags"]:
                lines.append(f"    Flags: {', '.join(ra['flags'])}")

        lines.append("")

    lines.append("Produce your final recommendation for each trade.")
    return "\n".join(lines)


def orchestrator(state: TradingState) -> Dict:
    """LangGraph node: synthesize all agent analyses into final recommendations."""
    import time as _t; _start = _t.time()
    import logging; logging.getLogger("agent_advisor").info(f"[PARALLEL] orchestrator START")
    if not acfg.OPENAI_API_KEY:
        return {"recommendations": [
            {
                "symbol": s["symbol"],
                "action": "SKIP",
                "confidence": 0.0,
                "reasoning": "No API key configured.",
                "size_modifier": 1.0,
            }
            for s in state["signals"]
        ]}

    llm = ChatOpenAI(
        model=acfg.ORCHESTRATOR_MODEL,
        api_key=acfg.OPENAI_API_KEY,
        temperature=acfg.AGENT_TEMPERATURE,
        max_tokens=2048,  # orchestrator needs space for 13+ symbols + tuning
        timeout=acfg.AGENT_TIMEOUT,
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_prompt(state)),
        ])
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        recs = json.loads(text)
        if not isinstance(recs, list):
            recs = [recs]
        import logging as _log; _log.getLogger("agent_advisor").info(f"[PARALLEL] orchestrator END ({_t.time()-_start:.1f}s)")
        return {"recommendations": recs}
    except Exception as e:
        import logging as _log; _log.getLogger("agent_advisor").info(f"[PARALLEL] orchestrator END-ERROR ({_t.time()-_start:.1f}s)")
        return {"recommendations": [
            {
                "symbol": s["symbol"],
                "action": "SKIP",
                "confidence": 0.0,
                "reasoning": f"Orchestrator failed: {e}",
                "size_modifier": 1.0,
            }
            for s in state["signals"]
        ]}
