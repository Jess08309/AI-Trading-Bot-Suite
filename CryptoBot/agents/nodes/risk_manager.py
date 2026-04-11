"""Risk Manager Agent — evaluates position sizing, portfolio risk, and trade safety."""
from __future__ import annotations
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents import config as acfg
from agents.state import TradingState, AgentAnalysis
import json


SYSTEM_PROMPT = """You are a professional trading risk manager with access to live market data.
You review proposed trades strictly from a RISK perspective. Your job is to PROTECT capital.

You have real-time data on funding rates, open interest, market breadth, and fear/greed.
Use this to assess SYSTEMIC risk in addition to portfolio-specific risk.

Evaluate:
1. Position concentration — how many open positions? Adding correlated risk?
2. Drawdown context — if on a losing streak, should we be adding more risk?
3. Direction balance — is the portfolio tilted too far in one direction?
4. Volatility-adjusted sizing — high volatility requires smaller positions
5. Regime awareness — is the market environment suitable for this trade type?
6. Correlation risk — is this trade adding to existing exposure in the same asset/direction?
7. Funding rate risk — extreme funding rates (|rate| > 0.03%) signal leveraged positioning 
   that often unwinds violently. Going WITH an extreme funding rate = high risk.
8. Open interest surges — rapidly rising OI + price move = potential liquidation cascade risk
9. Market breadth risk — low breadth (<30% green) means selling pressure is broad
10. Fear & Greed extreme — extreme greed (>75) = elevated crash risk, extreme fear (<25) = 
    potential capitulation but also knife-catching risk

Respond with EXACTLY this JSON (no markdown, no extra text):
{
  "verdict": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence risk assessment using live market data",
  "flags": ["list", "of", "risk", "flags"]
}

As risk manager:
- STRONG_BUY = risk is minimal, portfolio can absorb this easily
- BUY = acceptable risk
- HOLD = risk is borderline, proceed with reduced size
- SELL = risk is too high, recommend skipping
- STRONG_SELL = this trade violates risk limits, VETO it

Flag any of: "high_correlation", "overconcentrated", "drawdown_protection", 
"max_positions", "volatile_regime", "losing_streak", "extreme_funding",
"liquidation_risk", "low_breadth", "extreme_greed", "extreme_fear",
"defi_tvl_drop", "stale_data".

IMPORTANT — OVER-RELIANCE SAFEGUARD:
External data is CONTEXT for risk sizing, not an absolute veto. A good risk/reward trade
should not be blocked just because one external metric is uncertain. Weight your analysis:
60% portfolio risk + 40% systemic/external risk. If data quality is DEGRADED or POOR,
reduce your reliance on external risk signals and focus on portfolio-level risk metrics."""


def _build_prompt(signal: Dict, market: Dict, memory: Dict = None) -> str:
    mkt = market.get("summary", {})
    funding = market.get("funding_rates", {})
    mem = memory or {}
    
    # Find funding rate for this specific coin
    coin_base = signal['symbol'].replace("/USD", "").replace("USD", "").replace("PI_", "").replace("XBTUSD", "BTC")
    coin_funding_key = f"{coin_base}USDT"
    coin_funding = funding.get(coin_funding_key, None)
    funding_str = f"{coin_funding:+.4f}%" if coin_funding is not None else "unavailable"
    
    return f"""RISK REVIEW for proposed {signal['direction']} on {signal['symbol']}:

TRADE:
- Direction: {signal['direction']}
- ML Confidence: {signal['confidence']:.2%}
- Volatility: {signal['volatility']:.4f}

PORTFOLIO STATE:
- Open Positions: {signal['open_position_count']}
- Current Exposure: {signal['existing_exposure']}
- Daily P&L: {signal['daily_pnl_pct']:+.2f}%
- Consecutive Losses: {signal['consecutive_losses']}

MARKET CONTEXT:
- Regime: {signal['regime']}
- RSI: {signal['rsi']:.1f}
- 1h Price Change: {signal['price_change_1h']:+.2f}%
- 4h Price Change: {signal['price_change_4h']:+.2f}%

LIVE RISK DATA:
- Fear & Greed: {mkt.get('fear_greed_value', '?')} ({mkt.get('fear_greed_label', '?')})
- Market Breadth: {mkt.get('breadth_pct_positive', '?')}% of top coins green
- Market Cap 24h Change: {mkt.get('market_cap_change_24h_pct', '?'):+}%
- This Coin's Funding Rate: {funding_str}
- All Funding Rates: {mkt.get('funding_rates', 'unavailable')}
- Open Interest: {mkt.get('open_interest', 'unavailable')}

DeFi / SYSTEMIC RISK:
- {mkt.get('defi_tvl', 'unavailable')}
- Top Chains: {mkt.get('defi_top_chains', 'unavailable')}
- {mkt.get('stablecoin_yields', 'unavailable')}

BTC METRICS:
{mkt.get('btc_metrics', 'unavailable')}

DATA QUALITY: {mkt.get('data_quality', 'UNKNOWN')}
{mkt.get('data_quality_warning', '')}

RISK QUESTIONS:
1. Does the portfolio have room for another position?
2. Given {signal['consecutive_losses']} consecutive losses, should we add risk?
3. Is the volatility regime appropriate for this trade?
4. Does existing exposure ({signal['existing_exposure']}) create concentration risk?
5. Are funding rates signaling dangerous leverage in this direction?
6. Is the broader market showing stress (low breadth, extreme fear/greed)?

{mem.get('risk', 'AGENT MEMORY: No history yet.')}
Symbol history: {mem.get('symbols', dict()).get(signal['symbol'], 'No history for this symbol.')}"""


def risk_manager(state: TradingState) -> Dict:
    """LangGraph node: risk-review each signal considering portfolio state."""
    import time as _t; _start = _t.time()
    import logging; logging.getLogger("agent_advisor").info(f"[PARALLEL] risk_manager START")
    if not acfg.OPENAI_API_KEY:
        return {"risk_analyses": [
            AgentAnalysis(
                agent="risk",
                verdict="HOLD",
                confidence=0.5,
                reasoning="No API key configured — skipping risk analysis.",
                flags=["no_api_key"],
            )
            for _ in state["signals"]
        ]}

    llm = ChatOpenAI(
        model=acfg.RISK_MODEL,
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
                agent="risk",
                verdict=data.get("verdict", "HOLD"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                flags=data.get("flags", []),
            ))
        except Exception as e:
            analyses.append(AgentAnalysis(
                agent="risk",
                verdict="HOLD",
                confidence=0.5,
                reasoning=f"Analysis failed: {e}",
                flags=["error"],
            ))

    import logging as _log; _log.getLogger("agent_advisor").info(f"[PARALLEL] risk_manager END ({_t.time()-_start:.1f}s)")
    return {"risk_analyses": analyses}
