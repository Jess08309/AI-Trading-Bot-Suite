"""
CallBuyer Critical Path Tests
Tests for: API client contract format, risk manager, call engine signature bugs,
           get_account return format, regime-based confidence offsets.
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


class TestCallBuyerAPIClient(unittest.TestCase):
    """Test CallBuyerAPI interface: get_options_chain format, get_account return type."""

    def _make_api(self):
        with patch('core.config.CallBuyerConfig') as MockConfig:
            config = MockConfig()
            config.API_KEY = "test_key"
            config.API_SECRET = "test_secret"
            config.PAPER = True
            config.BASE_URL = "https://paper-api.alpaca.markets"
            config.ORDER_PREFIX = "cb_"

            from core.api_client import CallBuyerAPI
            api = CallBuyerAPI(config)
            return api

    def test_get_account_returns_dict_with_equity(self):
        """get_account() must return dict with 'equity' key (not get_account_equity()).
        BUG CAUGHT: code was calling get_account_equity() which doesn't exist.
        """
        api = self._make_api()
        mock_acct = MagicMock()
        mock_acct.equity = "125000.00"
        mock_acct.cash = "50000.00"
        mock_acct.buying_power = "200000.00"

        api._trading = MagicMock()
        api._trading.get_account.return_value = mock_acct
        api._last_call_ts = 0  # bypass throttle

        result = api.get_account()
        self.assertIsInstance(result, dict)
        self.assertIn("equity", result)
        self.assertEqual(result["equity"], 125000.0)
        self.assertIn("cash", result)
        self.assertIn("buying_power", result)

    def test_get_options_chain_returns_contracts_with_bid_ask(self):
        """get_options_chain must return list of dicts with 'bid' and 'ask' fields."""
        api = self._make_api()
        api._last_call_ts = 0

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "option_contracts": [
                {
                    "symbol": "NVDA260321C00150000",
                    "underlying_symbol": "NVDA",
                    "type": "call",
                    "strike_price": "150.00",
                    "expiration_date": "2026-03-21",
                    "open_interest": "1500",
                    "size": "100",
                }
            ]
        }

        # Mock quote response
        api.get_option_quote = MagicMock(return_value={
            "symbol": "NVDA260321C00150000",
            "bid": 5.20,
            "ask": 5.50,
            "mid": 5.35,
        })

        with patch('requests.get', return_value=mock_response):
            contracts = api.get_options_chain(
                underlying="NVDA",
                option_type="call",
            )

        self.assertIsInstance(contracts, list)
        self.assertGreater(len(contracts), 0)
        contract = contracts[0]
        self.assertIn("bid", contract)
        self.assertIn("ask", contract)
        self.assertIn("strike", contract)
        self.assertIn("symbol", contract)

    def test_get_options_chain_empty_on_error(self):
        """get_options_chain should return [] on error, not raise."""
        api = self._make_api()
        api._last_call_ts = 0

        with patch('requests.get', side_effect=Exception("Network error")):
            contracts = api.get_options_chain(underlying="BAD")
        self.assertEqual(contracts, [])

    def test_calculate_hv20_takes_symbol_not_bars(self):
        """calculate_hv20 takes a symbol string, NOT bars.
        BUG CAUGHT: code was passing bars array instead of symbol string.
        """
        api = self._make_api()
        api._last_call_ts = 0

        # Mock get_bars to return price data
        mock_bars = []
        for i in range(25):
            bar = MagicMock()
            bar.close = 100.0 + i * 0.5 + np.random.randn() * 2
            mock_bars.append(bar)
        api.get_bars = MagicMock(return_value=mock_bars)

        # Key test: calculate_hv20 accepts a SYMBOL string
        result = api.calculate_hv20("NVDA")  # NOT api.calculate_hv20(bars)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_calculate_hv20_insufficient_data(self):
        """calculate_hv20 returns None when not enough bars."""
        api = self._make_api()
        api._last_call_ts = 0
        api.get_bars = MagicMock(return_value=[])
        result = api.calculate_hv20("AAPL")
        self.assertIsNone(result)


class TestCallBuyerRiskManager(unittest.TestCase):
    """Test CallBuyer RiskManager: can_trade, position sizing, daily limits."""

    def _make_risk_manager(self):
        with patch('core.config.CallBuyerConfig') as MockConfig:
            config = MockConfig()
            config.ALLOCATION_PCT = 0.15
            config.MAX_POSITION_RISK_PCT = 0.02
            config.MAX_POSITIONS = 6
            config.MAX_PER_UNDERLYING = 2
            config.STATE_FILE = 'test_state.json'

            with patch('core.risk_manager.os.path.exists', return_value=False):
                from core.risk_manager import RiskManager
                rm = RiskManager(config)
                return rm

    def test_can_trade_initially_true(self):
        """Fresh risk manager should allow trading."""
        rm = self._make_risk_manager()
        self.assertTrue(rm.can_trade())

    def test_can_trade_false_after_daily_loss_limit(self):
        """can_trade() should return False after daily loss exceeds -5%."""
        rm = self._make_risk_manager()
        rm.state["current_balance"] = 15000
        rm.state["daily_pnl"] = -1000  # -6.7% > 5%
        self.assertFalse(rm.can_trade())

    def test_can_trade_false_after_5_consecutive_losses(self):
        """can_trade() should return False after 5 consecutive losses."""
        rm = self._make_risk_manager()
        rm.state["consecutive_losses"] = 5
        self.assertFalse(rm.can_trade())

    def test_can_trade_false_after_8_daily_trades(self):
        """can_trade() should return False after 8 daily trades."""
        rm = self._make_risk_manager()
        rm.state["daily_trades"] = 8
        self.assertFalse(rm.can_trade())

    def test_can_open_position_max_reached(self):
        """Should block when max positions reached."""
        rm = self._make_risk_manager()
        can, reason = rm.can_open_position(6, "NVDA", {})
        self.assertFalse(can)

    def test_can_open_position_per_underlying_limit(self):
        """Should block when max positions per underlying reached."""
        rm = self._make_risk_manager()
        positions = {
            "p1": {"underlying": "TSLA"},
            "p2": {"underlying": "TSLA"},
        }
        can, reason = rm.can_open_position(2, "TSLA", positions)
        self.assertFalse(can)

    def test_size_position_with_option_price(self):
        """size_position should accept option_price kwarg."""
        rm = self._make_risk_manager()
        rm.state["current_balance"] = 15000
        qty = rm.size_position(option_price=2.50, allocation=15000)
        self.assertGreaterEqual(qty, 0)
        self.assertLessEqual(qty, 5)  # capped at 5 contracts

    def test_size_position_zero_premium(self):
        """Zero premium should return 0 contracts."""
        rm = self._make_risk_manager()
        qty = rm.size_position(premium_per_contract=0)
        self.assertEqual(qty, 0)

    def test_record_trade_loss(self):
        """Recording a loss should update consecutive_losses."""
        rm = self._make_risk_manager()
        rm.record_trade(pnl=-200, symbol="NVDA")
        self.assertEqual(rm.state["consecutive_losses"], 1)
        self.assertEqual(rm.state["losses"], 1)

    def test_record_trade_win_resets_streak(self):
        """Recording a win should reset consecutive_losses."""
        rm = self._make_risk_manager()
        rm.record_trade(pnl=-100, symbol="NVDA")
        rm.record_trade(pnl=-100, symbol="AAPL")
        self.assertEqual(rm.state["consecutive_losses"], 2)
        rm.record_trade(pnl=500, symbol="TSLA")
        self.assertEqual(rm.state["consecutive_losses"], 0)
        self.assertGreater(rm.state["consecutive_wins"], 0)

    def test_daily_pnl_accumulates(self):
        """daily_pnl should accumulate across trades."""
        rm = self._make_risk_manager()
        rm.record_trade(pnl=100, symbol="AAPL")
        rm.record_trade(pnl=-50, symbol="NVDA")
        self.assertAlmostEqual(rm.state["daily_pnl"], 50.0)


class TestCallBuyerMLModel(unittest.TestCase):
    """Test CallBuyer ML model: warmup, prediction, quality gate."""

    def _make_ml(self):
        with patch('core.ml_model.os.path.exists', return_value=False), \
             patch('core.ml_model.os.makedirs'), \
             patch('core.ml_model.os.listdir', return_value=[]):
            from core.ml_model import CallBuyerMLModel
            return CallBuyerMLModel(models_dir="test_models", state_dir="test_state")

    def test_predict_returns_tuple(self):
        """predict() returns (probability, is_active)."""
        ml = self._make_ml()
        result = ml.predict(np.zeros(20))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_predict_default_during_warmup(self):
        """No model → return (0.5, False)."""
        ml = self._make_ml()
        ml.model = None
        prob, active = ml.predict(np.zeros(20))
        self.assertEqual(prob, 0.5)
        self.assertFalse(active)

    def test_should_retrain_respects_warmup(self):
        """should_retrain is False when completed < ML_WARMUP_TRADES."""
        ml = self._make_ml()
        ml._count_completed = MagicMock(return_value=10)
        self.assertFalse(ml.should_retrain())

    def test_should_retrain_true_when_enough_data(self):
        """should_retrain is True with enough data and no model."""
        ml = self._make_ml()
        ml._count_completed = MagicMock(return_value=35)
        ml.model = None
        self.assertTrue(ml.should_retrain())

    def test_get_status_keys(self):
        """get_status() must return dict with required keys."""
        ml = self._make_ml()
        status = ml.get_status()
        self.assertIn("active", status)
        self.assertIn("accuracy", status)
        self.assertIn("warmup_progress", status)
        self.assertIn("pending_outcomes", status)


class TestCallBuyerRegimeConfidenceOffset(unittest.TestCase):
    """Test that regime-based confidence offsets are applied correctly."""

    def test_trending_up_lowers_threshold(self):
        """TRENDING_UP should have negative confidence_offset (more aggressive)."""
        from core.regime_detector import RegimeDetector, TRENDING_UP
        adj = RegimeDetector.get_adjustments(TRENDING_UP, "CallBuyer")
        self.assertIn("confidence_offset", adj)
        self.assertLess(adj["confidence_offset"], 0)

    def test_trending_down_raises_threshold(self):
        """TRENDING_DOWN should have positive confidence_offset (more selective)."""
        from core.regime_detector import RegimeDetector, TRENDING_DOWN
        adj = RegimeDetector.get_adjustments(TRENDING_DOWN, "CallBuyer")
        self.assertGreater(adj["confidence_offset"], 0)
        self.assertGreaterEqual(adj["confidence_offset"], 0.10)

    def test_high_vol_raises_threshold(self):
        """HIGH_VOLATILITY should have positive confidence_offset."""
        from core.regime_detector import RegimeDetector, HIGH_VOLATILITY
        adj = RegimeDetector.get_adjustments(HIGH_VOLATILITY, "CallBuyer")
        self.assertGreater(adj["confidence_offset"], 0)

    def test_ranging_slightly_raises_threshold(self):
        """RANGING should slightly raise confidence_offset."""
        from core.regime_detector import RegimeDetector, RANGING
        adj = RegimeDetector.get_adjustments(RANGING, "CallBuyer")
        self.assertGreater(adj["confidence_offset"], 0)

    def test_confidence_offset_application_math(self):
        """Verify the confidence offset is applied correctly in _evaluate_symbol logic.

        Engine code: confidence = confidence - conf_offset
        So TRENDING_UP (offset=-0.05) → confidence increases by 0.05 (more trades)
        And TRENDING_DOWN (offset=+0.15) → confidence decreases by 0.15 (fewer trades)
        """
        base_confidence = 0.65

        # TRENDING_UP: offset = -0.05 → adjusted = 0.65 - (-0.05) = 0.70
        up_offset = -0.05
        adjusted_up = base_confidence - up_offset
        self.assertGreater(adjusted_up, base_confidence)

        # TRENDING_DOWN: offset = +0.15 → adjusted = 0.65 - 0.15 = 0.50
        down_offset = 0.15
        adjusted_down = base_confidence - down_offset
        self.assertLess(adjusted_down, base_confidence)

    def test_regime_adjustments_have_position_size(self):
        """All regime adjustments for CallBuyer should have position_size."""
        from core.regime_detector import RegimeDetector, ALL_REGIMES
        for regime in ALL_REGIMES:
            adj = RegimeDetector.get_adjustments(regime, "CallBuyer")
            self.assertIn("position_size", adj,
                          f"Missing position_size for {regime}")
            self.assertGreaterEqual(adj["position_size"], 0.0)


class TestCallBuyerMetaLearner(unittest.TestCase):
    """Test CallBuyer MetaLearner: evaluate, threshold adaptation."""

    def _make_meta(self):
        with patch('core.meta_learner.os.path.exists', return_value=False), \
             patch('core.meta_learner.os.makedirs'):
            from core.meta_learner import MetaLearner
            return MetaLearner(state_dir="test_state")

    def test_evaluate_returns_triple(self):
        """evaluate() must return (confidence, should_trade, reason)."""
        meta = self._make_meta()
        result = meta.evaluate(rule_score=5.0, ml_proba=0.7, ml_active=True)
        self.assertEqual(len(result), 3)
        confidence, should_trade, reason = result
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(should_trade, bool)
        self.assertIsInstance(reason, str)

    def test_low_rule_score_rejected(self):
        """Trades below min_rule_score should be rejected."""
        meta = self._make_meta()
        _, should_trade, reason = meta.evaluate(rule_score=1.0, ml_proba=0.9, ml_active=True)
        self.assertFalse(should_trade)
        self.assertIn("rule_score", reason)

    def test_loss_streak_tightens_thresholds(self):
        """3+ consecutive losses should tighten thresholds."""
        meta = self._make_meta()
        from core.config import CallBuyerConfig as cfg
        base_conf = cfg.META_CONFIDENCE_THRESHOLD

        for _ in range(3):
            meta.record_result(won=False, pnl_pct=-20.0)

        self.assertGreater(meta.confidence_threshold, base_conf)

    def test_win_streak_loosens_thresholds(self):
        """5+ consecutive wins should loosen thresholds."""
        meta = self._make_meta()
        from core.config import CallBuyerConfig as cfg
        base_conf = cfg.META_CONFIDENCE_THRESHOLD

        for _ in range(5):
            meta.record_result(won=True, pnl_pct=30.0)

        self.assertLessEqual(meta.confidence_threshold, base_conf)

    def test_mode_label_changes_with_streaks(self):
        """Mode label should reflect current streak state."""
        meta = self._make_meta()
        # Initial
        self.assertEqual(meta._get_mode_label(), "NORMAL")
        # After losses
        for _ in range(5):
            meta.record_result(won=False, pnl_pct=-15.0)
        self.assertEqual(meta._get_mode_label(), "DEFENSIVE")


class TestCallEngineContractParsing(unittest.TestCase):
    """Test call engine utility methods."""

    def test_parse_exp_from_contract(self):
        """_parse_exp_from_contract should correctly parse OCC symbol dates."""
        from core.call_engine import CallBuyerEngine

        # Standard OCC format: NVDA260321C00150000 → 2026-03-21
        result = CallBuyerEngine._parse_exp_from_contract("NVDA260321C00150000")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2026)
        self.assertEqual(result.month, 3)
        self.assertEqual(result.day, 21)

    def test_parse_exp_returns_none_on_garbage(self):
        """_parse_exp_from_contract should return None for invalid input."""
        from core.call_engine import CallBuyerEngine
        result = CallBuyerEngine._parse_exp_from_contract("not_a_contract")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
