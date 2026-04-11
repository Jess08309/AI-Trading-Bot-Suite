"""LangGraph state schema for the multi-agent trading advisor."""
from __future__ import annotations
from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add


class SignalContext(TypedDict):
    """Snapshot of one trading signal + surrounding market context."""
    symbol: str
    direction: str           # "LONG" or "SHORT"
    confidence: float        # 0-1 from base ML model
    ml_score: float          # raw ML directional probability
    rsi: float
    trend: str               # "UP", "DOWN", "SIDE"
    sentiment: float         # -1 to +1
    volatility: float
    trend_slope: float
    reason: str              # base model's reasoning string
    # Market context
    current_price: float
    price_change_1h: float   # % change over last 60 candles
    price_change_4h: float   # % change over last 240 candles
    regime: str              # "TRENDING_UP", "RANGING", etc.
    # Portfolio context
    open_position_count: int
    existing_exposure: str   # e.g. "LONG BTC, SHORT ETH"
    daily_pnl_pct: float
    consecutive_losses: int


class AgentAnalysis(TypedDict):
    """One agent's analysis of a signal."""
    agent: str
    verdict: str             # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence: float        # 0-1 how confident the agent is
    reasoning: str           # short explanation
    flags: List[str]         # risk flags or warnings


class TradingState(TypedDict):
    """Full state that flows through the LangGraph."""
    # Input
    signals: List[SignalContext]

    # Agent outputs (appended by each node)
    technical_analyses: Annotated[List[AgentAnalysis], add]
    sentiment_analyses: Annotated[List[AgentAnalysis], add]
    risk_analyses: Annotated[List[AgentAnalysis], add]

    # Final orchestrator output
    recommendations: List[Dict]   # [{symbol, action, confidence, reasoning}]
    
    # External market context (fetched from free APIs)
    market_context: Dict          # from data_feeds.fetch_all_market_context()
    
    # Agent memory context (rolling history + accuracy stats)
    agent_memory: Dict            # from memory.AgentMemory.get_agent_context()
    
    # Adaptive tuning context (tunable params + current values)
    tuning_context: str           # from tuning.AdaptiveTuner.get_tuning_prompt()

    # Metadata
    run_id: str
    timestamp: str
