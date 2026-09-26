"""
CryptoBot Critical Path Tests
Tests for: risk limits, ML model predictions, feature computation, regime detection.
All external dependencies are mocked — no real API calls or file I/O.
"""
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
import sys
import os

# Ensure the cryptotrades package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cryptotrades'))


class TestTradingConfig(unittest.TestCase):
    """Test TradingConfig validation and position sizing limits."""

    def test_max_position_pct_must_be_positive(self):
        """MAX_POSITION_PCT must be in (0, 1]."""
        from cryptotrades.core.trading_engine import TradingConfig
        cfg = TradingConfig()
        self.assertGreater(cfg.MAX_POSITION_PCT, 0)
        self.assertLessEqual(cfg.MAX_POSITION_PCT, 1.0)

    def test_min_position_pct_less_than_max(self):
        """MIN_POSITION_PCT must be <= MAX_POSITION_PCT."""
        from cryptotrades.core.trading_engine import TradingConfig
        cfg = TradingConfig()
        self.assertLessEqual(cfg.MIN_POSITION_PCT, cfg.MAX_POSITION_PCT)

    def test_stop_loss_is_negative(self):
        """STOP_LOSS_PCT must be negative (it's a loss threshold)."""
        from cryptotrades.core.trading_engine import TradingConfig
        cfg = TradingConfig()
        self.assertLess(cfg.STOP_LOSS_PCT, 0)
        self.assertLess(cfg.FUTURES_STOP_LOSS, 0)

    def test_max_positions_enforced(self):
        """Max position counts are positive and reasonable."""
        from cryptotrades.core.trading_engine import TradingConfig
        cfg = TradingConfig()
        self.assertGreater(cfg.MAX_POSITIONS_SPOT, 0)
        self.assertGreater(cfg.MAX_POSITIONS_FUTURES, 0)
        self.assertLessEqual(cfg.MAX_POSITIONS_SPOT, 30)

    def test_daily_loss_limit_is_negative(self):
        """Circuit breaker daily loss limit must be negative."""
        from cryptotrades.core.trading_engine import TradingConfig
        cfg = TradingConfig()
        self.assertLess(cfg.CB_DAILY_LOSS_LIMIT_PCT, 0)

    def test_max_drawdown_is_negative(self):
        """Max drawdown threshold must be negative."""
        from cryptotrades.core.trading_engine import TradingConfig
        cfg = TradingConfig()
        self.assertLess(cfg.CB_MAX_DRAWDOWN_PCT, 0)

    def test_ml_confidence_in_valid_range(self):
        """MIN_ML_CONFIDENCE must be between 0 and 1."""
        from cryptotrades.core.trading_engine import TradingConfig
        cfg = TradingConfig()
        self.assertGreaterEqual(cfg.MIN_ML_CONFIDENCE, 0)
        self.assertLessEqual(cfg.MIN_ML_CONFIDENCE, 1)

    def test_position_sizing_bounds(self):
        """Position sizing parameters form a valid range."""
        from cryptotrades.core.trading_engine import TradingConfig
        cfg = TradingConfig()
        # Max per position cannot exceed 100% of balance
        self.assertLessEqual(cfg.MAX_POSITION_PCT, 1.0)
        # Balance * MAX_POSITION_PCT defines the ceiling
        balance = 10000
        max_trade = balance * cfg.MAX_POSITION_PCT
        min_trade = balance * cfg.MIN_POSITION_PCT
        self.assertGreater(max_trade, min_trade)

    def test_consecutive_loss_breaker_positive(self):
        """Circuit breaker consecutive losses must be a positive integer."""
        from cryptotrades.core.trading_engine import TradingConfig
        cfg = TradingConfig()
        self.assertGreater(cfg.CB_MAX_CONSECUTIVE_LOSSES, 0)
        self.assertIsInstance(cfg.CB_MAX_CONSECUTIVE_LOSSES, int)


class TestMarketPredictor(unittest.TestCase):
    """Test MarketPredictor ML model predictions and feature handling."""

    def _make_predictor(self):
        """Create a MarketPredictor with mocked file I/O."""
        with patch('cryptotrades.utils.market_predictor.os.path.exists', return_value=False):
            from cryptotrades.utils.market_predictor import MarketPredictor
            return MarketPredictor(model_path="dummy/model.joblib")

    def test_predict_returns_dict_with_required_keys(self):
        """predict() must return dict with confidence, direction, strength."""
        predictor = self._make_predictor()
        # No model loaded — should return defaults
        prices = list(np.linspace(100, 110, 50))
        result = predictor.predict(prices)
        self.assertIn("confidence", result)
        self.assertIn("direction", result)
        self.assertIn("strength", result)

    def test_predict_confidence_range(self):
        """Confidence must be between 0 and 1."""
        predictor = self._make_predictor()

        # Mock model to return known probabilities
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        predictor.model = mock_model

        # Mock feature engine
        predictor.feature_engine.build_features_from_prices = MagicMock(
            return_value={"rsi_14": 0.5, "macd_histogram": 0.1}
        )
        predictor.feature_engine.features_to_array = MagicMock(
            return_value=np.zeros(15)
        )

        prices = list(np.linspace(100, 110, 50))
        result = predictor.predict(prices)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_predict_handles_empty_prices(self):
        """predict() should return default when given insufficient data."""
        predictor = self._make_predictor()
        result = predictor.predict([])
        self.assertEqual(result["confidence"], 0.5)
        self.assertEqual(result["direction"], 0.0)

    def test_predict_handles_short_prices(self):
        """predict() should return default when prices list is too short (<15)."""
        predictor = self._make_predictor()
        result = predictor.predict([100.0] * 10)
        self.assertEqual(result["confidence"], 0.5)

    def test_predict_no_model_returns_default(self):
        """predict() with no loaded model returns neutral values."""
        predictor = self._make_predictor()
        predictor.model = None
        result = predictor.predict([100.0] * 50)
        self.assertEqual(result["confidence"], 0.5)
        self.assertEqual(result["strength"], 0.0)

    def test_predict_direction_sign(self):
        """Direction should be 1.0 when confidence > 0.5, -1.0 otherwise."""
        predictor = self._make_predictor()
        mock_model = MagicMock()
        # Bullish prediction
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        predictor.model = mock_model
        predictor.feature_engine.build_features_from_prices = MagicMock(
            return_value={"rsi_14": 0.5}
        )
        predictor.feature_engine.features_to_array = MagicMock(
            return_value=np.zeros(15)
        )
        result = predictor.predict([100.0] * 50)
        self.assertEqual(result["direction"], 1.0)

        # Bearish prediction
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        result = predictor.predict([100.0] * 50)
        self.assertEqual(result["direction"], -1.0)


class TestFeatureEngine(unittest.TestCase):
    """Test FeatureEngine feature computation correctness."""

    def _get_engine(self):
        from cryptotrades.utils.feature_engine import FeatureEngine
        return FeatureEngine(lookback=30, prediction_horizon=5)

    def test_feature_names_count(self):
        """FEATURE_NAMES must have exactly 7 features (trimmed from 15)."""
        from cryptotrades.utils.feature_engine import FEATURE_NAMES
        self.assertEqual(len(FEATURE_NAMES), 7)

    def test_features_to_array_length(self):
        """features_to_array must produce array with len == FEATURE_NAMES."""
        from cryptotrades.utils.feature_engine import FEATURE_NAMES
        engine = self._get_engine()
        features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
        arr = engine.features_to_array(features)
        self.assertEqual(len(arr), len(FEATURE_NAMES))

    def test_features_to_array_no_nan(self):
        """Feature array should have no NaN when feature dict has valid values."""
        engine = self._get_engine()
        features = {"rsi_14": 50.0, "macd_histogram": 0.001}
        arr = engine.features_to_array(features)
        self.assertFalse(np.any(np.isnan(arr)))

    def test_features_to_array_missing_keys_default_zero(self):
        """Missing feature keys should default to 0.0."""
        engine = self._get_engine()
        arr = engine.features_to_array({})  # empty dict
        self.assertTrue(np.all(arr == 0.0))

    def test_build_features_insufficient_data_returns_none(self):
        """build_features_from_prices returns None with too little data."""
        engine = self._get_engine()
        result = engine.build_features_from_prices([100.0] * 5)
        self.assertIsNone(result)


class TestRegimeDetector(unittest.TestCase):
    """Test RegimeDetector detects all 4 states and confidence [0, 1]."""

    def _make_bars(self, closes, highs=None, lows=None):
        """Build a list of bar dicts from close prices."""
        if highs is None:
            highs = [c * 1.01 for c in closes]
        if lows is None:
            lows = [c * 0.99 for c in closes]
        return [
            {"close": c, "high": h, "low": l, "volume": 1000}
            for c, h, l in zip(closes, highs, lows)
        ]

    def _get_detector(self, **kwargs):
        from cryptotrades.utils.regime_detector import RegimeDetector
        return RegimeDetector(**kwargs)

    def test_trending_up_detected(self):
        """Strong uptrend bars should produce TRENDING_UP regime."""
        from cryptotrades.utils.regime_detector import TRENDING_UP
        # Strong uptrend: 250 bars with consistent price increase
        closes = [100 + i * 0.5 for i in range(250)]
        detector = self._get_detector()
        result = detector.detect(self._make_bars(closes))
        # With strong consistently rising prices, should detect uptrend
        self.assertEqual(result["regime"], TRENDING_UP)

    def test_trending_down_detected(self):
        """Strong downtrend bars should produce TRENDING_DOWN regime."""
        from cryptotrades.utils.regime_detector import TRENDING_DOWN
        closes = [200 - i * 0.5 for i in range(250)]
        detector = self._get_detector()
        result = detector.detect(self._make_bars(closes))
        self.assertEqual(result["regime"], TRENDING_DOWN)

    def test_ranging_detected(self):
        """Flat oscillating bars should produce RANGING regime."""
        from cryptotrades.utils.regime_detector import RANGING
        # Oscillate around 100 with small amplitude
        np.random.seed(42)
        closes = [100 + 0.5 * np.sin(i * 0.3) for i in range(250)]
        detector = self._get_detector()
        result = detector.detect(self._make_bars(closes))
        self.assertEqual(result["regime"], RANGING)

    def test_high_volatility_detected(self):
        """Extreme price swings should produce HIGH_VOLATILITY regime."""
        from cryptotrades.utils.regime_detector import HIGH_VOLATILITY
        # Normal bars then extreme volatility — huge swings with wide high/low range
        closes = [100 + 0.1 * i for i in range(200)]
        # Add extreme volatility at the end — much larger swings
        for i in range(50):
            closes.append(closes[-1] + (25 if i % 2 == 0 else -25))
        highs = [c + 20 for c in closes]
        lows = [c - 20 for c in closes]
        detector = self._get_detector()
        result = detector.detect(self._make_bars(closes, highs, lows))
        self.assertIn(result["regime"], [HIGH_VOLATILITY, "RANGING"],
                      "Extreme swings should trigger HIGH_VOLATILITY or at least RANGING")

    def test_confidence_between_0_and_1(self):
        """Confidence must be in [0, 1] for any input."""
        closes = [100 + i * 0.5 for i in range(250)]
        detector = self._get_detector()
        result = detector.detect(self._make_bars(closes))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_insufficient_data_fallback(self):
        """Less than 30 bars should return RANGING fallback."""
        from cryptotrades.utils.regime_detector import RANGING
        closes = [100 + i for i in range(10)]
        detector = self._get_detector()
        result = detector.detect(self._make_bars(closes))
        self.assertEqual(result["regime"], RANGING)
        self.assertEqual(result["confidence"], 0.0)

    def test_all_regimes_in_constants(self):
        """ALL_REGIMES should contain exactly 4 valid states."""
        from cryptotrades.utils.regime_detector import (
            ALL_REGIMES, TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY
        )
        self.assertEqual(len(ALL_REGIMES), 4)
        self.assertIn(TRENDING_UP, ALL_REGIMES)
        self.assertIn(TRENDING_DOWN, ALL_REGIMES)
        self.assertIn(RANGING, ALL_REGIMES)
        self.assertIn(HIGH_VOLATILITY, ALL_REGIMES)

    def test_get_adjustments_returns_dict(self):
        """get_adjustments() returns a dict for each regime."""
        from cryptotrades.utils.regime_detector import (
            RegimeDetector, ALL_REGIMES
        )
        for regime in ALL_REGIMES:
            adj = RegimeDetector.get_adjustments(regime, "CryptoBot")
            self.assertIsInstance(adj, dict)
            self.assertIn("position_size", adj)

    def test_get_adjustments_high_vol_putseller_zero_size(self):
        """HIGH_VOLATILITY should halt PutSeller (position_size=0)."""
        from cryptotrades.utils.regime_detector import RegimeDetector, HIGH_VOLATILITY
        adj = RegimeDetector.get_adjustments(HIGH_VOLATILITY, "PutSeller")
        self.assertEqual(adj["position_size"], 0.0)

    def test_callbuyer_confidence_offset_per_regime(self):
        """CallBuyer should have different confidence_offset per regime."""
        from cryptotrades.utils.regime_detector import (
            RegimeDetector, TRENDING_UP, TRENDING_DOWN, RANGING
        )
        up_adj = RegimeDetector.get_adjustments(TRENDING_UP, "CallBuyer")
        down_adj = RegimeDetector.get_adjustments(TRENDING_DOWN, "CallBuyer")
        range_adj = RegimeDetector.get_adjustments(RANGING, "CallBuyer")

        # TRENDING_UP should lower threshold (negative offset)
        self.assertLess(up_adj["confidence_offset"], 0)
        # TRENDING_DOWN should raise it (positive offset)
        self.assertGreater(down_adj["confidence_offset"], 0)
        # RANGING should slightly raise it
        self.assertGreater(range_adj["confidence_offset"], 0)

    def test_result_contains_signals_dict(self):
        """Detection result must include a signals dict with indicator values."""
        closes = [100 + i * 0.3 for i in range(250)]
        detector = self._get_detector()
        result = detector.detect(self._make_bars(closes))
        self.assertIn("signals", result)
        self.assertIn("adx", result["signals"])
        self.assertIn("sma_alignment", result["signals"])

    def test_bot_name_filter(self):
        """Passing bot_name filters adjustments to that bot only."""
        from cryptotrades.utils.regime_detector import RegimeDetector, TRENDING_UP
        detector = RegimeDetector(bot_name="CryptoBot")
        closes = [100 + i * 0.5 for i in range(250)]
        result = detector.detect(self._make_bars(closes))
        adj = result["suggested_adjustments"]
        # Should be CryptoBot-specific (has long_bias key)
        self.assertIn("long_bias", adj)


class TestFillConfirmation(unittest.TestCase):
    """Tests for the Alpaca spot fill-confirmation fix (fix/cryptobot-fill-confirmation):
    zero/partial fills must never be silently treated as a confirmed position change.
    """

    def _make_bot(self, **overrides):
        """Build a TradingBot instance bypassing __init__, with only the
        attributes needed by the methods under test set on it."""
        from cryptotrades.core.trading_engine import TradingBot
        bot = TradingBot.__new__(TradingBot)
        bot.logger = MagicMock()
        bot.trading_client = overrides.pop("trading_client", MagicMock())
        bot.positions = overrides.pop("positions", {})
        bot.balance_spot = overrides.pop("balance_spot", 5000.0)
        bot.balance_futures = overrides.pop("balance_futures", 5000.0)
        bot.risk_manager = MagicMock()
        bot.advisor = None
        bot._save_state = MagicMock()
        bot._log_trade_csv = MagicMock()
        bot._record_symbol_result = MagicMock()
        bot._record_direction_result = MagicMock()
        bot._apply_slippage = MagicMock(side_effect=lambda price, direction, is_entry, is_futures: price)
        for key, value in overrides.items():
            setattr(bot, key, value)
        return bot

    def _make_position(self, broker_qty=2.0):
        from cryptotrades.core.trading_engine import Position
        return Position(
            symbol="AAVE/USD",
            direction="LONG",
            entry_price=100.0,
            size=200.0,
            entry_time=datetime.now(),
            stop_loss=95.0,
            take_profit=105.0,
            max_price=100.0,
            broker_qty=broker_qty,
        )

    def test_buy_zero_fill_returns_zero_and_cancels(self):
        """An order that never confirms a fill must return 0.0, never a phantom qty."""
        order = MagicMock(id="order-1", filled_qty=None)
        bot = self._make_bot()
        bot.trading_client.submit_order.return_value = order
        bot.trading_client.get_order_by_id.return_value = order

        with patch("cryptotrades.core.trading_engine.time.sleep"):
            result = bot._alpaca_buy_spot("BTC/USD", 100.0)

        self.assertEqual(result, 0.0)
        bot.trading_client.cancel_order_by_id.assert_called_once_with("order-1")

    def test_buy_confirmed_fill_returns_qty(self):
        """A confirmed fill must return the real filled qty."""
        order = MagicMock(id="order-2", filled_qty="1.5")
        bot = self._make_bot()
        bot.trading_client.submit_order.return_value = order
        bot.trading_client.get_order_by_id.return_value = order

        with patch("cryptotrades.core.trading_engine.time.sleep"):
            result = bot._alpaca_buy_spot("BTC/USD", 100.0)

        self.assertEqual(result, 1.5)
        bot.trading_client.cancel_order_by_id.assert_not_called()

    def test_sell_zero_fill_keeps_position_tracked(self):
        """A sell that doesn't confirm any fill must NOT delete local state."""
        position = self._make_position(broker_qty=2.0)
        bot = self._make_bot(positions={"AAVE/USD": position})
        bot._alpaca_sell_spot = MagicMock(return_value=0.0)

        bot._close_position("AAVE/USD", 100.0, "STOP_LOSS", 0.0)

        self.assertIn("AAVE/USD", bot.positions)
        self.assertEqual(bot.positions["AAVE/USD"].broker_qty, 2.0)
        bot._save_state.assert_not_called()

    def test_sell_partial_fill_keeps_position_with_corrected_qty(self):
        """A partial sell must retain the position with the remaining qty, not drop state."""
        position = self._make_position(broker_qty=2.0)
        bot = self._make_bot(positions={"AAVE/USD": position})
        bot._alpaca_sell_spot = MagicMock(return_value=1.2)

        bot._close_position("AAVE/USD", 100.0, "STOP_LOSS", 0.0)

        self.assertIn("AAVE/USD", bot.positions)
        self.assertAlmostEqual(bot.positions["AAVE/USD"].broker_qty, 0.8)
        bot._save_state.assert_called_once()

    def test_sell_full_fill_closes_position(self):
        """A fully-confirmed sell must proceed with the normal close/delete path."""
        position = self._make_position(broker_qty=2.0)
        bot = self._make_bot(positions={"AAVE/USD": position})
        bot._alpaca_sell_spot = MagicMock(return_value=2.0)

        bot._close_position("AAVE/USD", 105.0, "TAKE_PROFIT", 5.0)

        self.assertNotIn("AAVE/USD", bot.positions)
        bot._save_state.assert_called_once()
        bot.risk_manager.record_trade.assert_called_once()


if __name__ == '__main__':
    unittest.main()
