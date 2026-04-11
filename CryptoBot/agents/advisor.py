"""MultiAgentAdvisor — bridge between CryptoBot's trading engine and the LangGraph agent system.

Usage in trading_engine.py:
    from agents.advisor import MultiAgentAdvisor
    self.advisor = MultiAgentAdvisor()
    
    # In the trade cycle:
    signals = self.generate_signals()
    if signals:
        signals = self.advisor.evaluate(signals, self.positions, self)
        if signals:
            self.execute_signals(signals)
"""
from __future__ import annotations
import os, json, time, logging, uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional, TYPE_CHECKING

from agents import config as acfg
from agents.state import TradingState, SignalContext
from agents.graph import trading_graph
from agents.data_feeds import fetch_all_market_context
from agents.memory import AgentMemory
from agents.tuning import AdaptiveTuner

if TYPE_CHECKING:
    pass  # avoid circular imports with trading_engine types

logger = logging.getLogger("agent_advisor")
_handler = logging.FileHandler(acfg.AGENT_LOG, encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


class MultiAgentAdvisor:
    """Wraps the LangGraph multi-agent system for use inside CryptoBot."""

    def __init__(self):
        self.enabled = acfg.AGENT_ENABLED
        self.mode = acfg.AGENT_MODE  # "advisor" (log only) or "active" (modify signals)
        self._last_run = 0.0
        self._min_interval = 60  # don't call LLMs more than once per minute
        self.memory = AgentMemory()
        self.tuner = AdaptiveTuner(mode="suggest")  # log-only until trusted
        logger.info(f"MultiAgentAdvisor initialized — mode={self.mode}, enabled={self.enabled}")

    def evaluate(
        self,
        signals: list,
        positions: dict,
        engine,
    ) -> list:
        """Evaluate signals through the multi-agent system.
        
        In 'advisor' mode: logs recommendations but returns signals unchanged.
        In 'active' mode: filters/modifies signals based on agent recommendations.
        
        Args:
            signals: List of Signal dataclass instances from generate_signals()
            positions: Dict of current open Position objects
            engine: The TradingBot instance (for accessing market_data, balances, etc.)
        
        Returns:
            List of Signal objects (potentially filtered/modified)
        """
        if not self.enabled or not acfg.OPENAI_API_KEY:
            return signals

        if not signals:
            return signals

        # Rate limit — don't hammer the API
        now = time.time()
        if now - self._last_run < self._min_interval:
            logger.debug("Skipping agent eval — too soon since last run")
            return signals
        self._last_run = now

        try:
            # Convert signals to agent-friendly format
            signal_contexts = self._build_contexts(signals, positions, engine)
            
            run_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Fetch external market data (Fear & Greed, funding rates, etc.)
            logger.info(f"[{run_id}] Fetching external market data...")
            market_context = fetch_all_market_context()
            mkt = market_context.get("summary", {})
            logger.info(
                f"[{run_id}] Market: Fear/Greed={mkt.get('fear_greed_value','?')} ({mkt.get('fear_greed_label','?')}), "
                f"BTC dom={mkt.get('btc_dominance','?')}%, "
                f"MCap 24h={mkt.get('market_cap_change_24h_pct','?'):+}%, "
                f"Breadth={mkt.get('breadth_pct_positive','?')}% green, "
                f"Trending={mkt.get('trending_coins','none')}"
            )

            # Build initial state
            initial_state: TradingState = {
                "signals": signal_contexts,
                "technical_analyses": [],
                "sentiment_analyses": [],
                "risk_analyses": [],
                "recommendations": [],
                "market_context": market_context,
                "agent_memory": {
                    "technical":   self.memory.get_agent_context("technical"),
                    "sentiment":   self.memory.get_agent_context("sentiment"),
                    "risk":        self.memory.get_agent_context("risk"),
                    "orchestrator": self.memory.get_agent_context("orchestrator"),
                    "symbols": {
                        sig["symbol"]: self.memory.get_symbol_context(sig["symbol"])
                        for sig in signal_contexts
                    },
                },
                "tuning_context": self._get_tuning_context(engine),
                "run_id": run_id,
                "timestamp": timestamp,
            }

            logger.info(f"[{run_id}] Running agent evaluation for {len(signals)} signals...")
            start = time.time()
            
            # Execute the LangGraph
            result = trading_graph.invoke(initial_state)
            
            elapsed = time.time() - start
            logger.info(f"[{run_id}] Agent evaluation completed in {elapsed:.1f}s")

            # Log all analyses
            self._log_analyses(run_id, result)

            # Apply recommendations
            if self.mode == "active":
                signals = self._apply_recommendations(signals, result.get("recommendations", []))
                logger.info(f"[{run_id}] Active mode: {len(signals)} signals after agent filtering")
            else:
                logger.info(f"[{run_id}] Advisor mode: returning all {len(signals)} signals unchanged")

            # Persist the full run to disk
            self._save_run(run_id, result, elapsed)

            # Record this cycle in agent memory (outcome filled in later)
            self.memory.record_cycle(
                run_id=run_id,
                signals=result.get("signals", []),
                technical_analyses=result.get("technical_analyses", []),
                sentiment_analyses=result.get("sentiment_analyses", []),
                risk_analyses=result.get("risk_analyses", []),
                recommendations=result.get("recommendations", []),
            )

            # Process any tuning suggestions from the orchestrator
            self._process_tuning_suggestions(result.get("recommendations", []), engine)

        except Exception as e:
            logger.error(f"Agent evaluation failed: {e}", exc_info=True)
            # On failure, return signals unchanged (fail-open for advisor mode)

        return signals

    def record_outcome(self, symbol: str, pnl_pct: float, hold_minutes: float = 0):
        """Record a trade outcome so agents can learn from results.
        Called by the trading engine when a position is closed."""
        try:
            self.memory.record_outcome(symbol, pnl_pct, hold_minutes)
        except Exception as e:
            logger.warning(f"Failed to record outcome for {symbol}: {e}")

    def _get_tuning_context(self, engine) -> str:
        """Build tuning context string for the orchestrator prompt."""
        try:
            # Get the config object from the engine
            config_obj = None
            if hasattr(engine, '_config'):
                config_obj = engine._config
            elif hasattr(engine, 'config'):
                config_obj = engine.config
            else:
                # Try the module-level cfg
                from cryptotrades.core.trading_engine import cfg
                config_obj = cfg

            if config_obj is None:
                return ""

            header = self.tuner.get_tuning_prompt()
            values = self.tuner.get_current_values_str(config_obj)
            return f"{header}\n{values}"
        except Exception as e:
            logger.debug(f"Tuning context generation failed: {e}")
            return ""

    def _process_tuning_suggestions(self, recommendations: List[Dict], engine):
        """Extract and record tuning suggestions from orchestrator recommendations."""
        try:
            for rec in recommendations:
                tuning = rec.get("tuning_suggestions", [])
                if not tuning:
                    continue
                for suggestion in tuning:
                    param = suggestion.get("param", "")
                    direction = suggestion.get("direction", "")
                    reasoning = suggestion.get("reasoning", "")
                    if param and direction:
                        self.tuner.record_suggestion(param, direction, reasoning)

            # Process accumulated suggestions
            config_obj = None
            if hasattr(engine, '_config'):
                config_obj = engine._config
            elif hasattr(engine, 'config'):
                config_obj = engine.config
            else:
                from cryptotrades.core.trading_engine import cfg
                config_obj = cfg

            if config_obj:
                resolved = sum(
                    1 for c in self.memory.cycles
                    for s in c.get("signals", [])
                    if s.get("outcome") is not None
                )
                self.tuner.process_suggestions(config_obj, resolved)
        except Exception as e:
            logger.debug(f"Tuning suggestion processing failed: {e}")

    def _build_contexts(self, signals, positions, engine) -> List[SignalContext]:
        """Convert Signal dataclass instances into SignalContext dicts for the agents."""
        contexts = []
        
        # Build exposure string
        exposure_parts = []
        for key, pos in positions.items():
            exposure_parts.append(f"{pos.direction} {pos.symbol}")
        exposure_str = ", ".join(exposure_parts[:10]) or "none"
        
        # Get portfolio state
        daily_pnl_pct = 0.0
        consecutive_losses = 0
        try:
            if hasattr(engine, 'risk_manager'):
                rm = engine.risk_manager
                if hasattr(rm, 'daily_pnl_pct'):
                    daily_pnl_pct = rm.daily_pnl_pct
                if hasattr(rm, 'consecutive_losses'):
                    consecutive_losses = rm.consecutive_losses
            elif hasattr(engine, '_daily_pnl_pct'):
                daily_pnl_pct = engine._daily_pnl_pct
            if hasattr(engine, '_consecutive_losses'):
                consecutive_losses = engine._consecutive_losses
        except Exception:
            pass

        # Get regime
        regime = "UNKNOWN"
        try:
            if hasattr(engine, '_current_regime') and engine._current_regime:
                regime = str(engine._current_regime)
        except Exception:
            pass

        for sig in signals:
            # Calculate price changes from history
            price_change_1h = 0.0
            price_change_4h = 0.0
            current_price = 0.0
            try:
                hist = engine.market_data.price_history.get(sig.symbol, [])
                if hist:
                    current_price = hist[-1]
                    if len(hist) > 60:
                        price_change_1h = ((hist[-1] - hist[-61]) / hist[-61]) * 100
                    if len(hist) > 240:
                        price_change_4h = ((hist[-1] - hist[-241]) / hist[-241]) * 100
            except Exception:
                pass

            contexts.append(SignalContext(
                symbol=sig.symbol,
                direction=sig.direction,
                confidence=sig.confidence,
                ml_score=sig.ml_score,
                rsi=sig.rsi,
                trend=sig.trend,
                sentiment=sig.sentiment,
                volatility=sig.volatility,
                trend_slope=sig.trend_slope,
                reason=sig.reason,
                current_price=current_price,
                price_change_1h=price_change_1h,
                price_change_4h=price_change_4h,
                regime=regime,
                open_position_count=len(positions),
                existing_exposure=exposure_str,
                daily_pnl_pct=daily_pnl_pct,
                consecutive_losses=consecutive_losses,
            ))

        return contexts

    def _apply_recommendations(self, signals, recommendations: List[Dict]) -> list:
        """Filter signals based on orchestrator recommendations (active mode only)."""
        if not recommendations:
            return signals

        rec_by_symbol = {r["symbol"]: r for r in recommendations}
        filtered = []

        for sig in signals:
            rec = rec_by_symbol.get(sig.symbol)
            if not rec:
                filtered.append(sig)
                continue

            action = rec.get("action", "TAKE").upper()
            if action == "SKIP":
                logger.info(f"  SKIPPED {sig.symbol} {sig.direction}: {rec.get('reasoning', '')}")
                continue
            elif action == "REDUCE_SIZE":
                # We can't directly modify Signal (it's frozen), so we adjust confidence
                # which will reduce position size via Kelly criterion
                modifier = rec.get("size_modifier", 0.7)
                sig.confidence = sig.confidence * modifier
                logger.info(f"  REDUCED {sig.symbol} confidence to {sig.confidence:.2f} (modifier={modifier})")
            # else TAKE — pass through unchanged

            filtered.append(sig)

        return filtered

    def _log_analyses(self, run_id: str, result: TradingState):
        """Log all agent analyses for review."""
        for i, sig in enumerate(result.get("signals", [])):
            sym = sig["symbol"]
            direction = sig["direction"]
            logger.info(f"[{run_id}] ═══ {direction} {sym} (ML: {sig['confidence']:.2%}) ═══")

            # Technical
            if i < len(result.get("technical_analyses", [])):
                ta = result["technical_analyses"][i]
                logger.info(f"[{run_id}]   TECHNICAL: {ta['verdict']} ({ta['confidence']:.0%}) — {ta['reasoning']}")
                if ta["flags"]:
                    logger.info(f"[{run_id}]     Flags: {', '.join(ta['flags'])}")

            # Sentiment
            if i < len(result.get("sentiment_analyses", [])):
                sa = result["sentiment_analyses"][i]
                logger.info(f"[{run_id}]   SENTIMENT: {sa['verdict']} ({sa['confidence']:.0%}) — {sa['reasoning']}")
                if sa["flags"]:
                    logger.info(f"[{run_id}]     Flags: {', '.join(sa['flags'])}")

            # Risk
            if i < len(result.get("risk_analyses", [])):
                ra = result["risk_analyses"][i]
                logger.info(f"[{run_id}]   RISK: {ra['verdict']} ({ra['confidence']:.0%}) — {ra['reasoning']}")
                if ra["flags"]:
                    logger.info(f"[{run_id}]     Flags: {', '.join(ra['flags'])}")

        # Final recommendations
        for rec in result.get("recommendations", []):
            logger.info(
                f"[{run_id}]   >>> {rec.get('symbol','?')}: {rec.get('action','?')} "
                f"(conf={rec.get('confidence',0):.0%}, size_mod={rec.get('size_modifier',1.0)}) "
                f"— {rec.get('reasoning','')}"
            )

    def _save_run(self, run_id: str, result: TradingState, elapsed: float):
        """Persist full agent run to JSON for audit."""
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "agent_runs")
        os.makedirs(save_dir, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y%m%d")
        save_path = os.path.join(save_dir, f"run_{date_str}_{run_id}.json")
        
        try:
            save_data = {
                "run_id": run_id,
                "timestamp": result.get("timestamp", ""),
                "elapsed_seconds": elapsed,
                "mode": self.mode,
                "signal_count": len(result.get("signals", [])),
                "signals": result.get("signals", []),
                "technical_analyses": result.get("technical_analyses", []),
                "sentiment_analyses": result.get("sentiment_analyses", []),
                "risk_analyses": result.get("risk_analyses", []),
                "recommendations": result.get("recommendations", []),
                "market_context_summary": result.get("market_context", {}).get("summary", {}),
            }
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, default=str)
            logger.info(f"[{run_id}] Run saved to {save_path}")
        except Exception as e:
            logger.warning(f"[{run_id}] Failed to save run: {e}")
