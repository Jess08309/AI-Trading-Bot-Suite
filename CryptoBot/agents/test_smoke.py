"""Smoke test for the multi-agent advisor framework."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("MULTI-AGENT ADVISOR SMOKE TEST")
print("=" * 60)

# 1. Config
print("\n[1/5] Loading config...")
from agents.config import OPENAI_API_KEY, AGENT_ENABLED, AGENT_MODE
print(f"  API Key: {'set (' + OPENAI_API_KEY[:15] + '...)' if OPENAI_API_KEY else 'NOT SET'}")
print(f"  Enabled: {AGENT_ENABLED}")
print(f"  Mode: {AGENT_MODE}")

# 2. State schema
print("\n[2/5] Loading state schema...")
from agents.state import TradingState, SignalContext, AgentAnalysis
print("  TradingState, SignalContext, AgentAnalysis — OK")

# 3. Graph
print("\n[3/5] Building LangGraph...")
from agents.graph import trading_graph
print(f"  Graph compiled — nodes: {list(trading_graph.get_graph().nodes.keys())}")

# 4. Advisor
print("\n[4/5] Creating advisor...")
from agents.advisor import MultiAgentAdvisor
advisor = MultiAgentAdvisor()
print(f"  Advisor ready — mode={advisor.mode}, enabled={advisor.enabled}")

# 5. Run a test signal through the pipeline
print("\n[5/5] Running test signal through full pipeline...")
test_signals = [
    SignalContext(
        symbol="BTC/USD",
        direction="LONG",
        confidence=0.65,
        ml_score=0.62,
        rsi=45.3,
        trend="UP",
        sentiment=0.3,
        volatility=0.0025,
        trend_slope=0.0012,
        reason="ML:0.62|Trend:UP|RSI:45",
        current_price=84500.0,
        price_change_1h=0.45,
        price_change_4h=1.2,
        regime="TRENDING_UP",
        open_position_count=3,
        existing_exposure="LONG ETH/USD, SHORT PI_XRPUSD",
        daily_pnl_pct=-0.5,
        consecutive_losses=2,
    ),
    SignalContext(
        symbol="ETH/USD",
        direction="SHORT",
        confidence=0.58,
        ml_score=0.42,
        rsi=62.1,
        trend="SIDE",
        sentiment=-0.1,
        volatility=0.0035,
        trend_slope=-0.0005,
        reason="ML:0.42|Trend:SIDE|RSI:62",
        current_price=1820.0,
        price_change_1h=-0.3,
        price_change_4h=-0.8,
        regime="TRENDING_UP",
        open_position_count=3,
        existing_exposure="LONG ETH/USD, SHORT PI_XRPUSD",
        daily_pnl_pct=-0.5,
        consecutive_losses=2,
    ),
]

initial_state: TradingState = {
    "signals": test_signals,
    "technical_analyses": [],
    "sentiment_analyses": [],
    "risk_analyses": [],
    "recommendations": [],
    "run_id": "test-001",
    "timestamp": "2026-04-07T22:00:00Z",
}

if not OPENAI_API_KEY:
    print("  SKIPPING live test — no API key")
    print("\n  To test live, add OPENAI_API_KEY to C:\\Bot\\cryptotrades\\.env")
else:
    import time
    start = time.time()
    result = trading_graph.invoke(initial_state)
    elapsed = time.time() - start
    
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Technical analyses: {len(result.get('technical_analyses', []))}")
    print(f"  Sentiment analyses: {len(result.get('sentiment_analyses', []))}")
    print(f"  Risk analyses: {len(result.get('risk_analyses', []))}")
    print(f"  Recommendations: {len(result.get('recommendations', []))}")
    
    print("\n  === RESULTS ===")
    for i, sig in enumerate(test_signals):
        sym = sig["symbol"]
        print(f"\n  --- {sig['direction']} {sym} ---")
        
        if i < len(result.get("technical_analyses", [])):
            ta = result["technical_analyses"][i]
            print(f"  TECHNICAL: {ta['verdict']} ({ta['confidence']:.0%}) — {ta['reasoning'][:100]}")
        
        if i < len(result.get("sentiment_analyses", [])):
            sa = result["sentiment_analyses"][i]
            print(f"  SENTIMENT: {sa['verdict']} ({sa['confidence']:.0%}) — {sa['reasoning'][:100]}")
        
        if i < len(result.get("risk_analyses", [])):
            ra = result["risk_analyses"][i]
            print(f"  RISK:      {ra['verdict']} ({ra['confidence']:.0%}) — {ra['reasoning'][:100]}")
    
    print("\n  === FINAL RECOMMENDATIONS ===")
    for rec in result.get("recommendations", []):
        print(f"  {rec.get('symbol','?')}: {rec.get('action','?')} "
              f"(conf={rec.get('confidence',0):.0%}, size={rec.get('size_modifier',1.0)}) "
              f"— {rec.get('reasoning','')[:120]}")

print("\n" + "=" * 60)
print("SMOKE TEST COMPLETE")
print("=" * 60)
