"""
Tests for cancelling opposite-side (BUY) orders before an Alpaca spot SELL,
which avoids wash-trade rejection 40310000 ("opposite side market/stop order exists").

Run with: python -m pytest tests/ -v
"""
import sys
import os
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pytest

pytest.importorskip("alpaca")

from core import trading_engine  # noqa: E402
from core.trading_engine import TradingBot  # noqa: E402

SYMBOL = "SHIB/USD"


def _order(order_id, side="buy", symbol=SYMBOL, filled_qty=None):
    return SimpleNamespace(id=order_id, side=side, symbol=symbol, filled_qty=filled_qty)


def _make_bot(open_orders_seq):
    """Bare TradingBot with a mocked Alpaca client.

    `open_orders_seq` is the list of results returned by successive
    trading_client.get_orders() calls.
    """
    bot = TradingBot.__new__(TradingBot)
    bot.logger = logging.getLogger("test_wash_trade_cancel")
    client = MagicMock()
    client.get_orders.side_effect = list(open_orders_seq)
    client.submit_order.return_value = _order("sell-1", side="sell", filled_qty="10")
    bot.trading_client = client
    bot._alpaca_spot_available_qty = MagicMock(return_value=(10.0, True))
    bot._fetch_spot_quote = MagicMock(return_value=(1.0, 1.0001))
    return bot, client


class TestCancelOpposingBuysBeforeSell:
    def test_sell_proceeds_when_no_open_orders(self):
        bot, client = _make_bot([[]])
        with patch.object(trading_engine.time, "sleep") as sleep:
            filled = bot._alpaca_sell_spot(SYMBOL, 10.0)
        assert filled == 10.0
        assert client.get_orders.call_count == 1  # one cheap filtered query
        client.cancel_order_by_id.assert_not_called()
        sleep.assert_not_called()
        client.submit_order.assert_called_once()

    def test_open_buy_cancelled_before_sell_submitted(self):
        bot, client = _make_bot([[_order("buy-1")], []])
        calls = []
        client.cancel_order_by_id.side_effect = lambda oid: calls.append(("cancel", oid))
        submit_rv = client.submit_order.return_value
        client.submit_order.side_effect = lambda req: calls.append(("submit", req.side.value)) or submit_rv
        with patch.object(trading_engine.time, "sleep") as sleep:
            filled = bot._alpaca_sell_spot(SYMBOL, 10.0)
        assert filled == 10.0
        assert calls == [("cancel", "buy-1"), ("submit", "sell")]
        sleep.assert_called_once_with(trading_engine._WASH_TRADE_CANCEL_WAIT_SEC)
        assert client.get_orders.call_count == 2  # initial query + single re-check

    def test_cancel_failure_skips_sell_for_retry(self):
        # Cancel raises and the BUY is still open on re-check: submitting the
        # SELL would just be rejected as a wash trade, so it is skipped this
        # cycle (caller keeps the position tracked and retries next loop).
        bot, client = _make_bot([[_order("buy-1")], [_order("buy-1")]])
        client.cancel_order_by_id.side_effect = Exception("cancel rejected")
        with patch.object(trading_engine.time, "sleep"):
            filled = bot._alpaca_sell_spot(SYMBOL, 10.0)
        assert filled == 0.0
        client.submit_order.assert_not_called()

    def test_cancel_failure_but_order_gone_on_recheck_proceeds(self):
        # e.g. the BUY was already filled/cancelled — re-check is authoritative.
        bot, client = _make_bot([[_order("buy-1")], []])
        client.cancel_order_by_id.side_effect = Exception("order not cancelable")
        with patch.object(trading_engine.time, "sleep"):
            filled = bot._alpaca_sell_spot(SYMBOL, 10.0)
        assert filled == 10.0
        client.submit_order.assert_called_once()

    def test_query_failure_falls_back_to_previous_behavior(self):
        bot, client = _make_bot([Exception("api down")])
        with patch.object(trading_engine.time, "sleep"):
            filled = bot._alpaca_sell_spot(SYMBOL, 10.0)
        assert filled == 10.0
        client.cancel_order_by_id.assert_not_called()
        client.submit_order.assert_called_once()

    def test_open_sell_orders_on_symbol_are_not_cancelled(self):
        bot, client = _make_bot([[_order("sell-old", side="sell"), _order("buy-x", symbol="BTC/USD")]])
        with patch.object(trading_engine.time, "sleep"):
            bot._alpaca_sell_spot(SYMBOL, 10.0)
        client.cancel_order_by_id.assert_not_called()
        client.submit_order.assert_called_once()
