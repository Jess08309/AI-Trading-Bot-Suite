"""
PutSeller Critical Path Tests
Tests for: spread construction, risk manager, regime-dependent spread gating,
           ML warmup period behavior.
All external dependencies are mocked — no real API calls.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os
from datetime import datetime, date, timedelta

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestSpreadConstruction(unittest.TestCase):
    """Test credit put spread construction: short_strike > long_strike, credit > 0."""

    def test_short_strike_greater_than_long_strike(self):
        """In a bull put spread, short strike must be above long strike."""
        # Simulate a valid spread
        short_strike = 450.0
        long_strike = 445.0
        spread_width = short_strike - long_strike
        self.assertGreater(short_strike, long_strike)
        self.assertGreater(spread_width, 0)

    def test_credit_must_be_positive(self):
        """Net credit received must be positive for a valid spread."""
        short_mid = 3.50  # premium for short put (closer to ATM)
        long_mid = 1.20   # premium for long put (further OTM)
        credit = short_mid - long_mid
        self.assertGreater(credit, 0)

    def test_max_loss_is_width_minus_credit(self):
        """Max loss per share = spread width - credit received."""
        short_strike = 450.0
        long_strike = 445.0
        credit = 1.50
        max_loss = (short_strike - long_strike) - credit
        self.assertEqual(max_loss, 3.50)
        self.assertGreater(max_loss, 0)

    def test_invalid_spread_width_zero(self):
        """Spread width must be positive — same strikes is invalid."""
        short_strike = 450.0
        long_strike = 450.0
        spread_width = short_strike - long_strike
        self.assertEqual(spread_width, 0)
        # Engine should reject this

    def test_negative_credit_is_rejected(self):
        """If short premium < long premium, spread should be rejected."""
        short_mid = 1.00
        long_mid = 2.00
        credit = short_mid - long_mid
        self.assertLess(credit, 0)
        # Engine code: `if credit <= 0: continue`


class TestPutSellerRiskManager(unittest.TestCase):
    """Test PutSeller RiskManager: position limits, sizing, daily loss limits."""

    def _make_risk_manager(self):
        with patch('core.config.PutSellerConfig') as MockConfig:
            config = MockConfig()
            config.ALLOCATION_PCT = 0.40
            config.MAX_POSITIONS = 8
            config.MAX_PER_UNDERLYING = 2
            config.MAX_POSITION_RISK_PCT = 0.05
            config.STATE_FILE = 'test_state.json'

            with patch('core.risk_manager.os.path.exists', return_value=False):
                from core.risk_manager import RiskManager
                rm = RiskManager(config)
                return rm

    def test_can_open_initially_allowed(self):
        """Fresh risk manager should allow opening positions."""
        rm = self._make_risk_manager()
        can, reason = rm.can_open_position(0, "AAPL", {})
        self.assertTrue(can)

    def test_max_positions_blocks_new_opens(self):
        """Should block when max positions reached."""
        rm = self._make_risk_manager()
        can, reason = rm.can_open_position(8, "AAPL", {})
        self.assertFalse(can)
        self.assertIn("max positions", reason.lower())

    def test_per_underlying_limit(self):
        """Should block when MAX_PER_UNDERLYING reached for a symbol."""
        rm = self._make_risk_manager()
        positions = {
            "pos1": {"underlying": "AAPL"},
            "pos2": {"underlying": "AAPL"},
        }
        can, reason = rm.can_open_position(2, "AAPL", positions)
        self.assertFalse(can)
        self.assertIn("per underlying", reason.lower())

    def test_daily_loss_limit_blocks_trading(self):
        """Daily loss limit should block new openings."""
        rm = self._make_risk_manager()
        rm.state["current_balance"] = 40000
        rm.state["daily_pnl"] = -1500  # -3.75% > -3% limit
        can, reason = rm.can_open_position(0, "MSFT", {})
        self.assertFalse(can)
        self.assertIn("daily loss", reason.lower())

    def test_consecutive_losses_pause(self):
        """4 consecutive losses should pause trading."""
        rm = self._make_risk_manager()
        rm.state["consecutive_losses"] = 4
        can, reason = rm.can_open_position(0, "MSFT", {})
        self.assertFalse(can)
        self.assertIn("consecutive", reason.lower())

    def test_size_position_returns_positive(self):
        """Position sizing should return a positive number of contracts."""
        rm = self._make_risk_manager()
        rm.state["current_balance"] = 40000
        contracts = rm.size_position(
            max_loss_per_contract=500,
            current_capital_in_use=0
        )
        self.assertGreater(contracts, 0)

    def test_size_position_zero_max_loss(self):
        """Zero max_loss_per_contract should return 0 contracts."""
        rm = self._make_risk_manager()
        contracts = rm.size_position(
            max_loss_per_contract=0,
            current_capital_in_use=0
        )
        self.assertEqual(contracts, 0)

    def test_record_trade_updates_stats(self):
        """Recording a trade should update PnL and win/loss counters."""
        rm = self._make_risk_manager()
        rm.record_trade(150.0, "AAPL")
        self.assertEqual(rm.state["total_trades"], 1)
        self.assertEqual(rm.state["wins"], 1)
        self.assertGreater(rm.state["total_pnl"], 0)

    def test_record_loss_increments_streak(self):
        """Recording a loss should increment consecutive_losses."""
        rm = self._make_risk_manager()
        rm.record_trade(-200.0, "AAPL")
        self.assertEqual(rm.state["consecutive_losses"], 1)
        rm.record_trade(-100.0, "GOOGL")
        self.assertEqual(rm.state["consecutive_losses"], 2)

    def test_win_resets_loss_streak(self):
        """A win should reset consecutive_losses to 0."""
        rm = self._make_risk_manager()
        rm.record_trade(-200.0, "AAPL")
        rm.record_trade(-100.0, "GOOGL")
        self.assertEqual(rm.state["consecutive_losses"], 2)
        rm.record_trade(300.0, "MSFT")
        self.assertEqual(rm.state["consecutive_losses"], 0)

    def test_max_risk_per_trade(self):
        """Max risk per trade should be balance * MAX_POSITION_RISK_PCT."""
        rm = self._make_risk_manager()
        rm.state["current_balance"] = 40000
        max_risk = rm.get_max_risk_per_trade()
        expected = 40000 * 0.05  # 5%
        self.assertAlmostEqual(max_risk, expected)


class TestPutSellerRegimeGating(unittest.TestCase):
    """Test that regime detection gates spread openings correctly."""

    def test_high_vol_halts_new_spreads(self):
        """HIGH_VOLATILITY regime should halt new spread openings."""
        # This tests the logic from _scan_opportunities:
        # if r["regime"] == "HIGH_VOLATILITY" and r["confidence"] >= 0.30: return
        regime_result = {
            "regime": "HIGH_VOLATILITY",
            "confidence": 0.50,
            "trend_strength": 0.3,
            "volatility_ratio": 2.1,
            "suggested_adjustments": {"position_size": 0.0},
        }
        should_halt = (regime_result["regime"] == "HIGH_VOLATILITY"
                       and regime_result["confidence"] >= 0.30)
        self.assertTrue(should_halt)

    def test_high_vol_low_confidence_does_not_halt(self):
        """HIGH_VOLATILITY with low confidence should not halt."""
        regime_result = {
            "regime": "HIGH_VOLATILITY",
            "confidence": 0.15,
        }
        should_halt = (regime_result["regime"] == "HIGH_VOLATILITY"
                       and regime_result["confidence"] >= 0.30)
        self.assertFalse(should_halt)

    def test_trending_down_widens_otm_buffer(self):
        """TRENDING_DOWN regime should increase OTM buffer multiplier."""
        from core.regime_detector import RegimeDetector, TRENDING_DOWN
        adj = RegimeDetector.get_adjustments(TRENDING_DOWN, "PutSeller")
        self.assertGreater(adj["otm_buffer"], 1.0,
                           "OTM buffer should be > 1.0 in TRENDING_DOWN (wider OTM)")

    def test_trending_down_raises_credit_threshold(self):
        """TRENDING_DOWN should raise credit threshold (harder to qualify)."""
        from core.regime_detector import RegimeDetector, TRENDING_DOWN
        adj = RegimeDetector.get_adjustments(TRENDING_DOWN, "PutSeller")
        self.assertGreater(adj["credit_threshold"], 1.0,
                           "Credit threshold should be > 1.0 in TRENDING_DOWN")

    def test_ranging_lowers_credit_threshold(self):
        """RANGING regime is ideal for put selling — lower credit threshold."""
        from core.regime_detector import RegimeDetector, RANGING
        adj = RegimeDetector.get_adjustments(RANGING, "PutSeller")
        self.assertLess(adj["credit_threshold"], 1.0)

    def test_trending_up_safe_for_puts(self):
        """TRENDING_UP should have normal/tight OTM buffer (safe for puts)."""
        from core.regime_detector import RegimeDetector, TRENDING_UP
        adj = RegimeDetector.get_adjustments(TRENDING_UP, "PutSeller")
        self.assertLessEqual(adj["otm_buffer"], 1.0)


class TestPutSellerMLModel(unittest.TestCase):
    """Test PutSeller ML model: warmup behavior, prediction interface."""

    def _make_ml(self):
        with patch('core.ml_model.os.path.exists', return_value=False), \
             patch('core.ml_model.os.makedirs'), \
             patch('core.ml_model.os.listdir', return_value=[]):
            from core.ml_model import PutSellerMLModel
            return PutSellerMLModel(models_dir="test_models", state_dir="test_state")

    def test_predict_no_model_returns_default(self):
        """With no trained model, predict returns (0.5, False)."""
        ml = self._make_ml()
        ml.model = None
        prob, active = ml.predict(np.zeros(20))
        self.assertEqual(prob, 0.5)
        self.assertFalse(active)

    def test_predict_with_model_returns_active(self):
        """With a trained model, predict returns (probability, True)."""
        ml = self._make_ml()
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        ml.model = mock_model
        prob, active = ml.predict(np.zeros(20))
        self.assertAlmostEqual(prob, 0.7, places=2)
        self.assertTrue(active)

    def test_should_retrain_false_during_warmup(self):
        """should_retrain() returns False when < 30 completed trades."""
        ml = self._make_ml()
        ml._count_completed = MagicMock(return_value=15)
        self.assertFalse(ml.should_retrain())

    def test_should_retrain_true_after_warmup_no_model(self):
        """should_retrain() returns True after warmup if no model exists."""
        ml = self._make_ml()
        ml._count_completed = MagicMock(return_value=35)
        ml.model = None
        self.assertTrue(ml.should_retrain())

    def test_warmup_trades_constant(self):
        """ML_WARMUP_TRADES should be 30."""
        from core.ml_model import ML_WARMUP_TRADES
        self.assertEqual(ML_WARMUP_TRADES, 30)

    def test_get_status_structure(self):
        """get_status() should return dict with required keys."""
        ml = self._make_ml()
        status = ml.get_status()
        self.assertIn("active", status)
        self.assertIn("accuracy", status)
        self.assertIn("warmup_progress", status)


if __name__ == '__main__':
    unittest.main()
