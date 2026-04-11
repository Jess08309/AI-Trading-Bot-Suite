"""LangGraph orchestration — wires the 4 agents into a directed graph.

Flow:
  START → [technical_analyst, sentiment_analyst] (parallel) → risk_manager → orchestrator → END
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from agents.state import TradingState
from agents.nodes.technical import technical_analyst
from agents.nodes.sentiment import sentiment_analyst
from agents.nodes.risk_manager import risk_manager
from agents.nodes.orchestrator import orchestrator


def build_graph() -> StateGraph:
    """Construct and compile the multi-agent trading graph."""
    builder = StateGraph(TradingState)

    # Add nodes
    builder.add_node("technical_analyst", technical_analyst)
    builder.add_node("sentiment_analyst", sentiment_analyst)
    builder.add_node("risk_manager", risk_manager)
    builder.add_node("orchestrator", orchestrator)

    # START → technical + sentiment in parallel
    builder.add_edge(START, "technical_analyst")
    builder.add_edge(START, "sentiment_analyst")

    # Both analysts → risk manager
    builder.add_edge("technical_analyst", "risk_manager")
    builder.add_edge("sentiment_analyst", "risk_manager")

    # Risk manager → orchestrator → END
    builder.add_edge("risk_manager", "orchestrator")
    builder.add_edge("orchestrator", END)

    return builder.compile()


# Singleton compiled graph (reused across cycles)
trading_graph = build_graph()
