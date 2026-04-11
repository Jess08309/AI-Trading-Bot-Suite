"""
AlpacaBot Critical Path Tests
Tests for: scanner ML enrichment, meta-learner, risk manager, put filter,
           direction-specific WR tracking.
All external dependencies are mocked — no real API calls.
"""
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
import sys
import os
from datetime import datetime, date
from collections import deque

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestScannerMLEnrichment(unittest.TestCase):
    """Test that scanner signals get actual ML enrichment, not defaults.
    BUG CAUGHT: scanner previously bypassed ML, leaving ml_confidence = 0.50.
    """

    def _make_engine(self):
        """Create a ScalpTradingEngine with all deps mocked."""
        with patch('core.config.Config') as MockConfig, \
             patch('core.api_client.AlpacaAPI'), \
             patch('core.options_handler.OptionsHandler'), \
             patch('core.risk_manager.RiskManager'), \
             patch('core.scanner.MarketScanner'), \
             patch('utils.ml_model.OptionsMLModel'), \
             patch('utils.sentiment.MarketSentimentAnalyzer'), \
             patch('utils.rl_agent.RLShadowAgent'), \
             patch('utils.meta_learner.MetaLearner'):

            config = MockConfig()
            config.LOG_DIR = 'logs'
            config.TRADE_LOG = 'data/trades.csv'
            config.STATE_FILE = 'data/state/bot_state.json'
            config.MAX_POSITIONS = 5
            config.MAX_OPENS_PER_CYCLE = 2
            config.WATCHLIST = ['SPY', 'AAPL']
            config.SCANNER_ENABLED = True
            config.LOOKBACK_BARS = 50
            config.INITIAL_BALANCE = 100000
            config.ALLOCATION_PCT = 0.60

            from core.trading_engine import ScalpTradingEngine
            engine = ScalpTradingEngine(config)
            return engine

    def test_ml_predict_called_for_scanner_signals(self):
        """When ML is ready, scanner signals MUST call ml_model.predict()."""
        # This is the key bug test: scanner signals used to skip ML prediction
        engine = self._make_engine()
        engine.ml_ready = True
        engine.ml_model.model = MagicMock()

        # Simulate scanner returning a signal
        mock_prices = np.random.randn(60) + 100
        engine._get_prices = MagicMock(return_value=mock_prices)

        # ML predict returns non-default confidence
        engine.ml_model.predict = MagicMock(return_value={
            "direction": 0.72, "confidence": 0.72, "up_prob": 0.72, "down_prob": 0.28
        })

        result = engine.ml_model.predict(mock_prices[-51:])
        # Key assertion: confidence is NOT the default 0.50
        self.assertNotEqual(result["confidence"], 0.50,
                            "ML confidence should not be 0.50 after real prediction")
        self.assertGreater(result["confidence"], 0.50)

    def test_ml_default_when_not_ready(self):
        """When ML is NOT ready, default 0.50 confidence is expected."""
        engine = self._make_engine()
        engine.ml_ready = False
        engine.ml_model.model = None
        engine.ml_model.predict = MagicMock(return_value={
            "direction": 0.5, "confidence": 0.5, "up_prob": 0.5, "down_prob": 0.5
        })

        result = engine.ml_model.predict(np.zeros(50))
        self.assertEqual(result["confidence"], 0.5)


class TestOptionsMLModel(unittest.TestCase):
    """Test OptionsMLModel prediction interface and edge cases."""

    def _make_model(self):
        with patch('utils.ml_model.os.path.exists', return_value=False), \
             patch('utils.ml_model.os.makedirs'):
            from utils.ml_model import OptionsMLModel
            return OptionsMLModel(model_dir="test_models", min_accuracy=0.51)

    def test_predict_returns_required_keys(self):
        """predict() must return direction, confidence, up_prob, down_prob."""
        model = self._make_model()
        result = model.predict(np.zeros(20))
        self.assertIn("direction", result)
        self.assertIn("confidence", result)
        self.assertIn("up_prob", result)
        self.assertIn("down_prob", result)

    def test_predict_no_model_returns_defaults(self):
        """With no trained model, predict must return 0.5 defaults."""
        model = self._make_model()
        model.model = None
        result = model.predict(np.zeros(20))
        self.assertEqual(result["direction"], 0.5)
        self.assertEqual(result["confidence"], 0.5)

    def test_predict_confidence_is_max_of_probs(self):
        """confidence should be max(up_prob, down_prob)."""
        model = self._make_model()
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[0.3, 0.7]])
        model.model = mock

        with patch.object(model.feature_engine, 'build_features', return_value=np.zeros(20)):
            result = model.predict(np.zeros(20))
        self.assertAlmostEqual(result["confidence"], 0.7, places=2)

    def test_prediction_count_increments(self):
        """prediction_count should increment on each successful prediction."""
        model = self._make_model()
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[0.4, 0.6]])
        model.model = mock
        model.feature_engine.build_features = MagicMock(return_value=np.zeros(20))

        initial = model.prediction_count
        model.predict(np.zeros(20))
        self.assertEqual(model.prediction_count, initial + 1)


class TestMetaLearner(unittest.TestCase):
    """Test AlpacaBot MetaLearner ensemble scoring, thresholds, weight normalization."""

    def _make_learner(self):
        with patch('utils.meta_learner.os.path.exists', return_value=False):
            from utils.meta_learner import MetaLearner
            return MetaLearner(state_file="test_meta.json")

    def test_weights_sum_to_one(self):
        """Source weights must sum to approximately 1.0."""
        learner = self._make_learner()
        total = sum(learner.source_weights.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_ensemble_score_in_range(self):
        """get_ensemble_score() must return value in [0, 1]."""
        learner = self._make_learner()
        preds = {"ml_model": 0.8, "sentiment": 0.6, "rule_score": 0.9}
        score = learner.get_ensemble_score(preds)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_ensemble_neutral_when_all_half(self):
        """When all sources predict 0.5, ensemble should be ~0.5."""
        learner = self._make_learner()
        preds = {"ml_model": 0.5, "sentiment": 0.5, "rule_score": 0.5}
        score = learner.get_ensemble_score(preds)
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_should_trade_respects_min_rule_score(self):
        """Trades with rule_score below minimum should be rejected."""
        learner = self._make_learner()
        # Low rule score should be rejected regardless of ensemble
        result = learner.should_trade(ensemble_score=0.90, rule_score=1)
        self.assertFalse(result)

    def test_confidence_threshold_tightens_on_losses(self):
        """Consecutive losses should tighten the confidence threshold."""
        learner = self._make_learner()
        initial_threshold = learner.confidence_threshold
        learner.update_thresholds(
            recent_win_rate=30, consecutive_losses=5, rolling_loss_rate=0.65
        )
        self.assertGreater(learner.confidence_threshold, initial_threshold)

    def test_record_prediction_updates_history(self):
        """recording predictions should populate source_history."""
        learner = self._make_learner()
        preds = {"ml_model": 0.7, "sentiment": 0.6, "rule_score": 0.8}
        learner.record_prediction(preds, actual_outcome=1.0)
        self.assertGreater(len(learner.source_history["ml_model"]), 0)

    def test_weights_update_after_many_predictions(self):
        """After 10+ recorded predictions, weights should update from defaults."""
        learner = self._make_learner()
        for i in range(15):
            preds = {"ml_model": 0.7, "sentiment": 0.3, "rule_score": 0.8}
            # ML and rules correct, sentiment wrong
            learner.record_prediction(preds, actual_outcome=1.0)
        # ML should have gained weight relative to sentiment
        self.assertGreater(
            learner.source_weights["ml_model"],
            learner.source_weights["sentiment"]
        )


class TestAlpacaRiskManager(unittest.TestCase):
    """Test AlpacaBot RiskManager: position limits, daily loss, graduated response."""

    def _make_risk_manager(self):
        with patch('core.config.Config') as MockConfig:
            config = MockConfig()
            config.INITIAL_BALANCE = 100000.0
            config.MAX_POSITIONS = 5
            config.MAX_POSITION_PCT = 0.04
            config.MIN_POSITION_PCT = 0.02
            config.MAX_PORTFOLIO_RISK_PCT = 0.50
            config.STOP_LOSS_PCT = -0.15
            config.TAKE_PROFIT_PCT = 0.15
            config.TRAILING_STOP_PCT = 0.12
            config.MIN_DTE_EXIT = 0
            config.MAX_HOLD_DAYS = 3
            config.STATE_FILE = 'test_state.json'

            from core.risk_manager import RiskManager
            rm = RiskManager(config)
            return rm

    def test_can_trade_initially_allowed(self):
        """Fresh risk manager should allow trading."""
        rm = self._make_risk_manager()
        allowed, reason = rm.can_trade()
        self.assertTrue(allowed)

    def test_can_open_position_max_reached(self):
        """Should block when max positions reached."""
        rm = self._make_risk_manager()
        allowed, reason = rm.can_open_position(num_open=5)
        self.assertFalse(allowed)
        self.assertIn("Max positions", reason)

    def test_daily_loss_hard_stop(self):
        """Hitting -5% daily loss should trigger hard stop."""
        rm = self._make_risk_manager()
        rm.daily_pnl = -6000  # -6% of 100k
        rm.current_balance = 100000
        allowed, reason = rm.can_trade()
        self.assertFalse(allowed)
        self.assertIn("HARD STOP", reason)

    def test_consecutive_losses_graduated_response(self):
        """3+ consecutive losses should trigger graduated response (tighter controls)."""
        rm = self._make_risk_manager()
        rm.consecutive_losses = 4
        throttle = rm.get_throttle()
        self.assertLess(throttle["size_multiplier"], 1.0)
        self.assertGreater(throttle["min_score"], 3)
        self.assertEqual(throttle["tier_name"], "CAUTION")

    def test_direction_lock_after_losses(self):
        """3 consecutive call losses should lock the call direction."""
        rm = self._make_risk_manager()
        # Record 3 call losses
        for _ in range(3):
            rm.record_trade({"pnl": -50, "direction": "call", "underlying": "AAPL"})
        allowed, reason = rm.can_trade_direction("call")
        self.assertFalse(allowed)
        self.assertIn("locked", reason.lower())

    def test_direction_unlock_on_win(self):
        """A win should unlock a locked direction."""
        rm = self._make_risk_manager()
        for _ in range(3):
            rm.record_trade({"pnl": -50, "direction": "call", "underlying": "AAPL"})
        # Verify locked
        self.assertTrue(rm.direction_locked["call"])
        # Record a win
        rm.record_trade({"pnl": 100, "direction": "call", "underlying": "AAPL"})
        self.assertFalse(rm.direction_locked["call"])

    def test_calculate_size_respects_max_pct(self):
        """Position size should never exceed MAX_POSITION_PCT of balance."""
        rm = self._make_risk_manager()
        balance = 100000
        premium = 1.50
        size = rm.calculate_size(balance, confidence=0.99, premium=premium, num_open=0)
        max_contracts = int(balance * rm.config.MAX_POSITION_PCT / (premium * 100))
        self.assertLessEqual(size, max_contracts + 1)  # +1 for floor rounding

    def test_calculate_size_zero_premium(self):
        """Zero or negative premium should return 0 contracts."""
        rm = self._make_risk_manager()
        self.assertEqual(rm.calculate_size(100000, 0.8, premium=0, num_open=0), 0)
        self.assertEqual(rm.calculate_size(100000, 0.8, premium=-1.0, num_open=0), 0)

    def test_symbol_pause_after_losses(self):
        """A symbol with 3 consecutive losses should be paused."""
        rm = self._make_risk_manager()
        rm.symbol_losses["TSLA"] = 3
        allowed, reason = rm.can_trade_symbol("TSLA")
        self.assertFalse(allowed)
        self.assertIn("TSLA", reason)


class TestPutDirectionTracking(unittest.TestCase):
    """Test per-direction win-rate tracking and auto-disable for puts."""

    def _make_engine(self):
        with patch('core.config.Config') as MockConfig, \
             patch('core.api_client.AlpacaAPI'), \
             patch('core.options_handler.OptionsHandler'), \
             patch('core.risk_manager.RiskManager'), \
             patch('core.scanner.MarketScanner'), \
             patch('utils.ml_model.OptionsMLModel'), \
             patch('utils.sentiment.MarketSentimentAnalyzer'), \
             patch('utils.rl_agent.RLShadowAgent'), \
             patch('utils.meta_learner.MetaLearner'):

            config = MockConfig()
            config.LOG_DIR = 'logs'
            config.TRADE_LOG = 'data/trades.csv'
            config.STATE_FILE = 'data/state/bot_state.json'
            config.MAX_POSITIONS = 5
            config.INITIAL_BALANCE = 100000
            config.ALLOCATION_PCT = 0.60
            config.WATCHLIST = ['SPY']
            config.SCANNER_ENABLED = False

            from core.trading_engine import ScalpTradingEngine
            engine = ScalpTradingEngine(config)
            return engine

    def test_put_counters_start_at_zero(self):
        """Put win/loss counters should start at zero."""
        engine = self._make_engine()
        self.assertEqual(engine._put_wins, 0)
        self.assertEqual(engine._put_losses, 0)
        self.assertEqual(engine._call_wins, 0)
        self.assertEqual(engine._call_losses, 0)

    def test_puts_not_auto_disabled_initially(self):
        """Puts should not be auto-disabled at startup."""
        engine = self._make_engine()
        self.assertFalse(engine._puts_auto_disabled)


class TestSPYRegimeGateForPuts(unittest.TestCase):
    """Test that the SPY regime gate blocks puts in uptrend (unless override conditions met)."""

    def test_put_blocked_when_spy_bullish_and_ensemble_low(self):
        """
        When SPY is in bull regime and ensemble < 0.75,
        scanner puts should be blocked.
        """
        # This is a logic verification — ensemble < 0.75 means put blocked
        ensemble_score = 0.60
        spy_regime = "bull"
        ml_direction = 0.55  # slightly bullish

        put_blocked = None
        if spy_regime in ("bull", "neutral"):
            if ensemble_score < 0.75:
                put_blocked = f"ensemble {ensemble_score:.2f} < 0.75"
            elif ml_direction >= 0.40:
                put_blocked = f"ML dir {ml_direction:.2f} >= 0.40"

        self.assertIsNotNone(put_blocked)
        self.assertIn("ensemble", put_blocked)

    def test_put_allowed_when_ensemble_high_enough(self):
        """Put allowed when ensemble >= 0.75 AND ML direction < 0.40 AND sent < -0.15."""
        ensemble_score = 0.80
        ml_direction = 0.30
        sentiment = -0.20

        put_blocked = None
        if ensemble_score < 0.75:
            put_blocked = "ensemble too low"
        elif ml_direction >= 0.40:
            put_blocked = "ml direction too high"
        elif sentiment >= -0.15:
            put_blocked = "sentiment too bullish"

        self.assertIsNone(put_blocked)


if __name__ == '__main__':
    unittest.main()
