"""
Unit tests for the crypto trading bot's critical components.

Run with: python -m pytest tests/ -v
"""
import sys
import os
import csv
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


# ============================================================
# Execution Guard Tests (spread gate, limit pricing, ATR exits)
# ============================================================
class TestExecutionGuard:
    def test_spread_gate_passes_tight_spread(self):
        from utils.execution_guard import passes_spread_gate
        # 10 bps spread on a $100 mid -> passes a 15 bps gate
        ok, spread_bps = passes_spread_gate(bid=99.95, ask=100.05, max_spread_bps=15.0)
        assert ok is True
        assert spread_bps == pytest.approx(10.0, abs=0.01)

    def test_spread_gate_blocks_wide_spread(self):
        from utils.execution_guard import passes_spread_gate
        # Meme-coin style wide spread: ~50 bps
        ok, spread_bps = passes_spread_gate(bid=99.75, ask=100.25, max_spread_bps=15.0)
        assert ok is False
        assert spread_bps > 15.0

    def test_spread_gate_handles_crossed_book(self):
        from utils.execution_guard import passes_spread_gate
        ok, spread_bps = passes_spread_gate(bid=101.0, ask=100.0, max_spread_bps=15.0)
        assert ok is False
        assert spread_bps == float("inf")

    def test_limit_price_buy_uses_bid(self):
        from utils.execution_guard import limit_price_for_side
        assert limit_price_for_side(bid=99.9, ask=100.1, side="buy") == 99.9
        assert limit_price_for_side(bid=99.9, ask=100.1, side="LONG") == 99.9

    def test_limit_price_sell_uses_ask(self):
        from utils.execution_guard import limit_price_for_side
        assert limit_price_for_side(bid=99.9, ask=100.1, side="sell") == 100.1
        assert limit_price_for_side(bid=99.9, ask=100.1, side="SHORT") == 100.1

    def test_limit_price_rejects_unknown_side(self):
        from utils.execution_guard import limit_price_for_side
        with pytest.raises(ValueError):
            limit_price_for_side(bid=99.9, ask=100.1, side="sideways")

    def test_min_take_profit_floor_applies_even_with_tight_spread(self):
        from utils.execution_guard import min_required_take_profit_pct
        required = min_required_take_profit_pct(spread_bps=1.0, fee_rate_pct=0.05, floor_pct=1.0)
        assert required == 1.0  # floor dominates tiny fee/spread cost

    def test_min_take_profit_scales_with_fees_and_spread(self):
        from utils.execution_guard import min_required_take_profit_pct
        required = min_required_take_profit_pct(spread_bps=80.0, fee_rate_pct=0.5, floor_pct=1.0)
        # 2*0.5 fee + 0.8 spread = 1.8% > 1.0% floor
        assert required == pytest.approx(1.8, abs=0.01)

    def test_meets_min_take_profit(self):
        from utils.execution_guard import meets_min_take_profit
        assert meets_min_take_profit(expected_gain_pct=1.2, spread_bps=10.0) is True
        assert meets_min_take_profit(expected_gain_pct=0.3, spread_bps=10.0) is False

    def test_atr_stop_take_profit_long(self):
        from utils.execution_guard import atr_stop_take_profit
        result = atr_stop_take_profit(entry_price=100.0, atr=2.0, side="long", stop_mult=1.5, tp_mult=2.0)
        assert result["stop_price"] == pytest.approx(97.0)
        assert result["take_profit_price"] == pytest.approx(104.0)
        assert result["tp_pct"] == pytest.approx(4.0)

    def test_atr_stop_take_profit_short(self):
        from utils.execution_guard import atr_stop_take_profit
        result = atr_stop_take_profit(entry_price=100.0, atr=2.0, side="short", stop_mult=1.5, tp_mult=2.0)
        assert result["stop_price"] == pytest.approx(103.0)
        assert result["take_profit_price"] == pytest.approx(96.0)

    def test_atr_stop_take_profit_enforces_min_tp_floor(self):
        from utils.execution_guard import atr_stop_take_profit
        # Tiny ATR in a quiet market: 2x ATR = 0.2% << 1% floor
        result = atr_stop_take_profit(
            entry_price=100.0, atr=0.1, side="long", stop_mult=1.5, tp_mult=2.0, min_tp_pct=1.0,
        )
        assert result["tp_pct"] == pytest.approx(1.0)
        assert result["take_profit_price"] == pytest.approx(101.0)


# ============================================================
# Risk-Based Position Sizing Tests
# ============================================================
class TestRiskSizing:
    def test_risk_based_cap_binds_on_wide_stop(self):
        from utils.risk_sizing import calculate_risk_capped_size
        # $10k equity, 1% risk = $100 budget. Stop 20% away -> $100/0.20 = $500
        # notional, which is below the $1000 (10%) notional cap -> risk binds.
        result = calculate_risk_capped_size(
            equity=10000, entry_price=100.0, stop_price=80.0,
            risk_pct=0.01, notional_cap_pct=0.10,
        )
        assert result["notional"] == pytest.approx(500.0)
        assert result["limiting_factor"] == "risk"

    def test_notional_cap_binds_on_tight_stop(self):
        from utils.risk_sizing import calculate_risk_capped_size
        # Stop only 0.1% away -> risk budget implies $100k notional, way above the 10% cap ($1000).
        result = calculate_risk_capped_size(
            equity=10000, entry_price=100.0, stop_price=99.9,
            risk_pct=0.01, notional_cap_pct=0.10,
        )
        assert result["notional"] == pytest.approx(1000.0)
        assert result["limiting_factor"] == "notional_cap"

    def test_zero_equity_returns_zero(self):
        from utils.risk_sizing import calculate_risk_capped_size
        result = calculate_risk_capped_size(equity=0, entry_price=100.0, stop_price=95.0)
        assert result["notional"] == 0.0
        assert result["limiting_factor"] == "none"

    def test_zero_stop_distance_falls_back_to_notional_cap(self):
        from utils.risk_sizing import calculate_risk_capped_size
        result = calculate_risk_capped_size(
            equity=10000, entry_price=100.0, stop_price=100.0,
            risk_pct=0.01, notional_cap_pct=0.10,
        )
        assert result["notional"] == pytest.approx(1000.0)


# ============================================================
# Order Retry / Backoff Tests
# ============================================================
class TestOrderRetryManager:
    def _make_manager(self, **kwargs):
        from utils.order_retry import OrderRetryManager
        defaults = {
            "max_retries": 3,
            "base_delay": 2.0,
            "max_delay": 60.0,
            "backoff_factor": 2.0,
            "cooldown_minutes": 30.0,
        }
        defaults.update(kwargs)
        return OrderRetryManager(**defaults)

    def test_backoff_delay_grows_exponentially(self):
        mgr = self._make_manager()
        assert mgr.backoff_delay(1) == pytest.approx(2.0)
        assert mgr.backoff_delay(2) == pytest.approx(4.0)
        assert mgr.backoff_delay(3) == pytest.approx(8.0)

    def test_backoff_delay_caps_at_max_delay(self):
        mgr = self._make_manager(max_delay=5.0)
        assert mgr.backoff_delay(5) == pytest.approx(5.0)

    def test_not_halted_initially(self):
        mgr = self._make_manager()
        assert mgr.is_halted("HYPE/USD") is False

    def test_halts_after_max_retries(self):
        mgr = self._make_manager(max_retries=3)
        r1 = mgr.record_failure("HYPE/USD")
        assert r1["halted"] is False
        assert r1["attempts"] == 1
        r2 = mgr.record_failure("HYPE/USD")
        assert r2["halted"] is False
        r3 = mgr.record_failure("HYPE/USD")
        assert r3["halted"] is True
        assert r3["attempts"] == 3
        assert mgr.is_halted("HYPE/USD") is True

    def test_halt_blocks_further_retries_until_cooldown_expires(self):
        from datetime import datetime, timedelta
        now = {"t": datetime(2025, 1, 1, 0, 0, 0)}

        def clock():
            return now["t"]

        mgr = self._make_manager(max_retries=3, cooldown_minutes=30.0, clock=clock)
        for _ in range(3):
            mgr.record_failure("HYPE/USD")
        assert mgr.is_halted("HYPE/USD") is True

        # Still within cooldown
        now["t"] += timedelta(minutes=10)
        assert mgr.is_halted("HYPE/USD") is True

        # Cooldown has expired — halt clears and streak resets
        now["t"] += timedelta(minutes=25)
        assert mgr.is_halted("HYPE/USD") is False
        assert mgr.attempts("HYPE/USD") == 0

    def test_success_resets_streak(self):
        mgr = self._make_manager(max_retries=3)
        mgr.record_failure("HYPE/USD")
        mgr.record_failure("HYPE/USD")
        mgr.record_success("HYPE/USD")
        assert mgr.attempts("HYPE/USD") == 0
        assert mgr.is_halted("HYPE/USD") is False


# ============================================================
# Symbol Expectancy Tracker Tests
# ============================================================
class TestSymbolExpectancyTracker:
    def _make_tracker(self, **kwargs):
        from utils.expectancy_tracker import SymbolExpectancyTracker
        defaults = {
            "max_consecutive_losses": 3,
            "expectancy_window": 20,
            "save_path": os.path.join(tempfile.mkdtemp(), "expectancy_test.json"),
        }
        defaults.update(kwargs)
        return SymbolExpectancyTracker(**defaults)

    def test_disables_after_consecutive_losses(self):
        tracker = self._make_tracker(max_consecutive_losses=3)
        tracker.record_round_trip("LTC/USD", -1.0)
        tracker.record_round_trip("LTC/USD", -0.8)
        just_disabled, reason = tracker.record_round_trip("LTC/USD", -1.6)
        assert just_disabled is True
        disabled, _ = tracker.is_disabled("LTC/USD")
        assert disabled is True
        assert "CONSECUTIVE_LOSSES" in reason

    def test_win_resets_consecutive_losses(self):
        tracker = self._make_tracker(max_consecutive_losses=3)
        tracker.record_round_trip("LINK/USD", -1.0)
        tracker.record_round_trip("LINK/USD", -1.0)
        tracker.record_round_trip("LINK/USD", 0.5)  # resets streak
        tracker.record_round_trip("LINK/USD", -1.0)
        disabled, _ = tracker.is_disabled("LINK/USD")
        assert disabled is False

    def test_disables_on_negative_20_trade_expectancy(self):
        tracker = self._make_tracker(max_consecutive_losses=99, expectancy_window=20)
        # Alternate win/loss so consecutive-loss trigger never fires, but
        # average P&L is still negative.
        for i in range(20):
            pnl = 0.2 if i % 2 == 0 else -0.5
            tracker.record_round_trip("SIDE/USD", pnl)
        disabled, reason = tracker.is_disabled("SIDE/USD")
        assert disabled is True
        assert "NEGATIVE_EXPECTANCY" in reason

    def test_stays_enabled_with_positive_expectancy(self):
        tracker = self._make_tracker(max_consecutive_losses=3, expectancy_window=20)
        for i in range(20):
            pnl = 1.0 if i % 2 == 0 else -0.3
            tracker.record_round_trip("BTC/USD", pnl)
        disabled, _ = tracker.is_disabled("BTC/USD")
        assert disabled is False

    def test_save_and_load_state(self):
        save_path = os.path.join(tempfile.mkdtemp(), "expectancy_persist.json")
        tracker = self._make_tracker(max_consecutive_losses=3, save_path=save_path)
        tracker.record_round_trip("ETH/USD", -1.0)
        tracker.record_round_trip("ETH/USD", -1.0)
        tracker.record_round_trip("ETH/USD", -1.0)
        tracker.save_state()

        from utils.expectancy_tracker import SymbolExpectancyTracker
        reloaded = SymbolExpectancyTracker(max_consecutive_losses=3, save_path=save_path)
        reloaded.load_state()
        disabled, _ = reloaded.is_disabled("ETH/USD")
        assert disabled is True


# ============================================================
# Trade Journal Tests
# ============================================================
class TestTradeJournal:
    def test_slippage_bps_buy_adverse_when_paying_above_mid(self):
        from utils.trade_journal import compute_slippage_bps
        slip = compute_slippage_bps(mid_price_at_fill=100.0, fill_price=100.1, side="buy")
        assert slip == pytest.approx(10.0, abs=0.01)

    def test_slippage_bps_sell_adverse_when_receiving_below_mid(self):
        from utils.trade_journal import compute_slippage_bps
        slip = compute_slippage_bps(mid_price_at_fill=100.0, fill_price=99.9, side="sell")
        assert slip == pytest.approx(10.0, abs=0.01)

    def test_record_writes_csv_and_sqlite(self):
        from utils.trade_journal import TradeJournal, TradeJournalEntry
        tmpdir = tempfile.mkdtemp()
        journal = TradeJournal(
            csv_path=os.path.join(tmpdir, "journal.csv"),
            sqlite_path=os.path.join(tmpdir, "journal.db"),
        )
        entry = TradeJournalEntry(
            symbol="BTC/USD", signal="ml_long", side="buy",
            entry_price=50000.0, exit_price=50600.0, mid_price_at_fill=50010.0,
            size_usd=500.0, pnl_usd=6.0, pnl_pct=1.2, exit_reason="take_profit",
        )
        journal.record(entry)

        with open(os.path.join(tmpdir, "journal.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["symbol"] == "BTC/USD"
        assert rows[0]["exit_reason"] == "take_profit"

        import sqlite3
        conn = sqlite3.connect(os.path.join(tmpdir, "journal.db"))
        count = conn.execute("SELECT COUNT(*) FROM trade_journal").fetchone()[0]
        conn.close()
        assert count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
