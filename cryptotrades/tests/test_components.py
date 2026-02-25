"""
Unit tests for the crypto trading bot's critical components.

Run with: python -m pytest tests/ -v
"""
import sys
import os
import json
import tempfile
import shutil

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pytest


# ============================================================
# Circuit Breaker Tests
# ============================================================
class TestCircuitBreaker:
    """Test circuit breaker safety mechanisms."""

    def _make_cb(self, **kwargs):
        from utils.circuit_breaker import CircuitBreaker
        defaults = {
            "max_consecutive_losses": 3,
            "daily_loss_limit_pct": -5.0,
            "max_drawdown_pct": -10.0,
            "cooldown_minutes": 30,
            "save_path": os.path.join(tempfile.mkdtemp(), "cb_test.json"),
        }
        defaults.update(kwargs)
        return CircuitBreaker(**defaults)

    def test_can_trade_initially(self):
        cb = self._make_cb()
        ok, reason = cb.can_trade("spot")
        assert ok is True
        assert reason == "ok"

    def test_consecutive_losses_trigger(self):
        cb = self._make_cb(max_consecutive_losses=3)
        # 3 losses in a row
        cb.record_trade("spot", -1.0, 4900)
        cb.record_trade("spot", -1.5, 4800)
        triggered, reason = cb.record_trade("spot", -2.0, 4700)
        assert triggered is True
        assert "CONSECUTIVE_LOSSES" in reason

        # Should be paused now
        ok, reason = cb.can_trade("spot")
        assert ok is False
        assert "PAUSED" in reason

    def test_consecutive_losses_reset_on_win(self):
        cb = self._make_cb(max_consecutive_losses=3)
        cb.record_trade("spot", -1.0, 4900)
        cb.record_trade("spot", -1.0, 4800)
        # Win resets counter
        cb.record_trade("spot", 2.0, 4900)
        triggered, reason = cb.record_trade("spot", -1.0, 4850)
        assert triggered is False

    def test_daily_loss_limit(self):
        cb = self._make_cb(daily_loss_limit_pct=-5.0)
        cb.record_trade("spot", -3.0, 4900)
        triggered, reason = cb.record_trade("spot", -3.0, 4800)
        assert triggered is True
        assert "DAILY_LOSS_LIMIT" in reason

    def test_max_drawdown(self):
        cb = self._make_cb(max_drawdown_pct=-10.0)
        cb.peak_balance = 5000.0
        # Drop to 4400 = -12% drawdown
        triggered, reason = cb.record_trade("spot", -5.0, 4400)
        assert triggered is True
        assert "MAX_DRAWDOWN" in reason

    def test_futures_independent(self):
        cb = self._make_cb(max_consecutive_losses=2)
        cb.record_trade("spot", -1.0, 4900)
        cb.record_trade("spot", -1.0, 4800)  # triggers spot
        # Futures should still be ok
        ok, reason = cb.can_trade("futures")
        assert ok is True

    def test_save_load_state(self):
        cb = self._make_cb()
        cb.record_trade("spot", -1.0, 4900)
        cb.record_trade("futures", 2.0, 5100)
        cb.save_state()

        cb2 = self._make_cb(save_path=cb.save_path)
        cb2.load_state()
        assert cb2.consecutive_losses["spot"] == 1
        assert cb2.consecutive_losses["futures"] == 0
        assert cb2.consecutive_wins["futures"] == 1

    def test_get_status(self):
        cb = self._make_cb()
        cb.peak_balance = 5000
        cb.current_balance = 4800
        status = cb.get_status()
        assert status["drawdown_pct"] == pytest.approx(-4.0, abs=0.1)
        assert status["spot_paused"] is False


# ============================================================
# Performance Tracker Tests
# ============================================================
class TestPerformanceTracker:
    """Test performance metrics calculations."""

    def _make_tracker(self, balance=5000.0):
        from utils.performance_tracker import PerformanceTracker
        return PerformanceTracker(initial_balance=balance)

    def test_win_rate(self):
        pt = self._make_tracker()
        pt.log_trade("BTC", "sell", 100, 1, 5100, 0, pnl=2.0)
        pt.log_trade("ETH", "sell", 50, 1, 5050, 0, pnl=-1.0)
        pt.log_trade("SOL", "sell", 30, 1, 5080, 0, pnl=1.5)
        assert pt.get_win_rate() == pytest.approx(66.7, abs=0.1)

    def test_max_drawdown_watermark(self):
        """The key bug fix: watermark-based drawdown, not global min/max."""
        pt = self._make_tracker(balance=1000)
        # Balance goes: 1000 -> 1200 -> 900 -> 1100 -> 800
        pt.balance_history = [1000, 1200, 900, 1100, 800]
        dd = pt.get_max_drawdown()
        # Worst drawdown: 1200 -> 800 = -33.3% (peak was 1200, trough 800)
        # Actually: 1200->900=-25%, then 1100->800=-27.3%, but peak remains 1200
        # So 1200->800 = -33.3%
        assert dd == pytest.approx(-33.3, abs=0.1)

    def test_max_drawdown_no_loss(self):
        pt = self._make_tracker(balance=1000)
        pt.balance_history = [1000, 1100, 1200, 1300]
        dd = pt.get_max_drawdown()
        assert dd == 0.0

    def test_profit_factor(self):
        pt = self._make_tracker()
        pt.log_trade("BTC", "sell", 100, 1, 5200, 0, pnl=3.0)
        pt.log_trade("ETH", "sell", 50, 1, 5100, 0, pnl=-1.0)
        pt.log_trade("SOL", "sell", 30, 1, 5250, 0, pnl=2.0)
        # Gross profit: 5.0, gross loss: 1.0
        assert pt.get_profit_factor() == pytest.approx(5.0, abs=0.1)

    def test_sortino_ratio_no_downside(self):
        pt = self._make_tracker()
        pt.daily_returns = {"2024-01-01": 1.0, "2024-01-02": 2.0, "2024-01-03": 0.5}
        sortino = pt.get_sortino_ratio()
        # All positive returns -> infinite sortino
        assert sortino == float('inf')

    def test_expectancy(self):
        pt = self._make_tracker()
        pt.log_trade("BTC", "sell", 100, 1, 5200, 0, pnl=4.0)
        pt.log_trade("ETH", "sell", 50, 1, 5100, 0, pnl=-2.0)
        # Win rate 50%, avg_win=4, avg_loss=2
        # Expectancy = 0.5 * 4 - 0.5 * 2 = 1.0
        assert pt.get_expectancy() == pytest.approx(1.0, abs=0.1)

    def test_full_report_structure(self):
        pt = self._make_tracker()
        report = pt.get_full_report()
        expected_keys = [
            "total_trades", "win_rate", "total_return_pct",
            "max_drawdown_pct", "sharpe_ratio", "sortino_ratio",
            "profit_factor", "avg_trade_return", "expectancy",
            "peak_balance", "daily_returns_count",
        ]
        for key in expected_keys:
            assert key in report, f"Missing key: {key}"


# ============================================================
# Retry Decorator Tests
# ============================================================
class TestRetryDecorator:
    """Test retry with backoff."""

    def test_succeeds_first_try(self):
        from utils.retry import retry_with_backoff
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def always_works():
            nonlocal call_count
            call_count += 1
            return 42

        result = always_works()
        assert result == 42
        assert call_count == 1

    def test_retries_then_succeeds(self):
        from utils.retry import retry_with_backoff
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network error")
            return "ok"

        result = fails_twice()
        assert result == "ok"
        assert call_count == 3

    def test_returns_none_after_all_retries(self):
        from utils.retry import retry_with_backoff

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            raise ValueError("bad")

        result = always_fails()
        assert result is None

    def test_specific_exception_types(self):
        from utils.retry import retry_with_backoff

        @retry_with_backoff(max_retries=2, base_delay=0.01,
                            exceptions=(ConnectionError,))
        def raises_value_error():
            raise ValueError("wrong type")

        # Should NOT retry ValueError — should propagate the exception
        with pytest.raises(ValueError):
            raises_value_error()


# ============================================================
# Config Tests
# ============================================================
class TestConfig:
    """Test configuration management."""

    def test_config_defaults(self):
        from utils.config import TradingConfig
        cfg = TradingConfig()
        assert cfg.PAPER_TRADING is True
        assert cfg.TRADE_INTERVAL == 10
        assert cfg.CHECK_INTERVAL == 60
        assert cfg.CB_MAX_CONSECUTIVE_LOSSES == 5

    def test_config_summary(self):
        from utils.config import TradingConfig
        cfg = TradingConfig()
        summary = cfg.summary()
        assert "Paper Trading" in summary
        assert "Circuit Breaker" in summary

    def test_singleton_import(self):
        from utils.config import config
        assert config is not None
        assert hasattr(config, "POPULAR_PAIRS")


# ============================================================
# Position Sizer Tests
# ============================================================
class TestPositionSizer:
    """Test Kelly criterion position sizing."""

    def test_basic_sizing(self):
        from utils.position_sizer import PositionSizer
        ps = PositionSizer()
        result = ps.calculate_size(
            balance=5000,
            confidence=0.65,
            volatility=0.02,
            existing_exposure=0,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.5,
            num_positions=0,
        )
        assert result["position_size"] > 0
        assert result["position_pct"] > 0
        assert result["position_pct"] <= 1.0

    def test_zero_balance(self):
        from utils.position_sizer import PositionSizer
        ps = PositionSizer()
        result = ps.calculate_size(
            balance=0,
            confidence=0.65,
            volatility=0.02,
            existing_exposure=0,
        )
        assert result["position_size"] == 0

    def test_futures_sizing(self):
        from utils.position_sizer import PositionSizer
        ps = PositionSizer()
        result = ps.calculate_futures_size(
            balance=5000,
            confidence=0.65,
            volatility=0.02,
            leverage=2,
            num_positions=0,
        )
        assert result["contract_value"] > 0
        assert result["margin_required"] > 0


# ============================================================
# Alerting Tests
# ============================================================
class TestAlerting:
    """Test alerting module (without actual Discord calls)."""

    def test_is_configured_false_by_default(self):
        from utils import alerting
        # Unless DISCORD_WEBHOOK_URL is set in env, should be False
        old_url = alerting._webhook_url
        alerting._webhook_url = ""
        assert alerting.is_configured() is False
        alerting._webhook_url = old_url

    def test_is_configured_true_when_set(self):
        from utils import alerting
        old_url = alerting._webhook_url
        alerting.set_webhook_url("https://discord.com/api/webhooks/test")
        assert alerting.is_configured() is True
        alerting._webhook_url = old_url

    def test_send_does_nothing_without_url(self):
        """Calling alert functions without a URL should not raise."""
        from utils import alerting
        old_url = alerting._webhook_url
        alerting._webhook_url = ""
        # These should all be no-ops
        alerting.alert_circuit_breaker("spot", "test")
        alerting.alert_large_loss("BTC", -5.0, -250)
        alerting.alert_bot_startup(0, 0, 5000, 5000)
        alerting.alert_bot_shutdown(100, 0, 0)
        alerting.alert_error("test error")
        alerting._webhook_url = old_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
