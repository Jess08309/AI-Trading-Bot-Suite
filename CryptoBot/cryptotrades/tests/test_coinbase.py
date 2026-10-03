"""Offline Coinbase client and production spot broker integration tests."""
import base64
import json
import logging
import os
import sys
from types import SimpleNamespace
from datetime import datetime
from unittest.mock import MagicMock, patch

import jwt
import pytest
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from core.coinbase_client import (
    CoinbaseClient, CoinbaseOrderRejected, select_spot_broker,
    to_alpaca_symbol, to_coinbase_symbol,
)
from core import trading_engine
from utils.config import TradingConfig

KEY_NAME = "organizations/test-org/apiKeys/test-key"


@pytest.fixture
def client():
    private_key = ed25519.Ed25519PrivateKey.generate()
    secret = base64.b64encode(private_key.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )).decode()
    client = CoinbaseClient(KEY_NAME, secret, paper_trading=False, min_request_interval=0)
    client._session = MagicMock()
    return client


def response(payload, status=200):
    result = MagicMock()
    result.status_code = status
    result.json.return_value = payload
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(response=result)
    return result


@pytest.mark.parametrize("key_type,escaped", [("ed25519", False), ("p256", True)])
def test_cdp_jwt_signature_and_claims(key_type, escaped):
    key = ed25519.Ed25519PrivateKey.generate() if key_type == "ed25519" else ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    if escaped:
        pem = pem.replace("\n", "\\n")
    client = CoinbaseClient(KEY_NAME, pem)
    token = client._jwt("GET", "/api/v3/brokerage/accounts")
    algorithm = "EdDSA" if key_type == "ed25519" else "ES256"
    claims = jwt.decode(token, key.public_key(), algorithms=[algorithm])
    assert claims["sub"] == KEY_NAME
    assert claims["iss"] == "cdp"
    assert claims["exp"] - claims["nbf"] == 120
    assert claims["uri"] == "GET api.coinbase.com/api/v3/brokerage/accounts"
    assert jwt.get_unverified_header(token)["kid"] == KEY_NAME
    assert jwt.get_unverified_header(token)["nonce"]


def test_64_byte_ed25519_secret(client):
    key = client._private_key
    raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                            serialization.NoEncryption())
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    other = CoinbaseClient(KEY_NAME, base64.b64encode(raw + public).decode())
    jwt.decode(other._jwt("GET", "/test"), key.public_key(), algorithms=["EdDSA"])


@pytest.mark.parametrize("secret", ["invalid key", "YQ==", ""])
def test_invalid_key_is_rejected_without_echoing_secret(secret):
    with pytest.raises(ValueError) as error:
        CoinbaseClient(KEY_NAME, secret)
    if secret:
        assert secret not in str(error.value)


@pytest.mark.parametrize("method,args,verb,path", [
    ("get_accounts", (), "GET", "/accounts"),
    ("get_product", ("BTC/USD",), "GET", "/products/BTC-USD"),
    ("list_products", (), "GET", "/products"),
    ("get_order", ("order-1",), "GET", "/orders/historical/order-1"),
    ("cancel_orders", (["order-1"],), "POST", "/orders/batch_cancel"),
    ("get_best_bid_ask", (["BTC/USD", "ETH/USD"],), "GET", "/best_bid_ask"),
    ("get_product_book", ("BTC/USD",), "GET", "/product_book"),
])
def test_http_methods(client, method, args, verb, path):
    client._session.request.return_value = response({"accounts": [], "products": []})
    getattr(client, method)(*args)
    call = client._session.request.call_args
    assert call.args == (verb, "https://api.coinbase.com/api/v3/brokerage" + path)
    assert call.kwargs["timeout"] == 10
    token = call.kwargs["headers"]["Authorization"].split()[1]
    claims = jwt.decode(token, client._private_key.public_key(), algorithms=["EdDSA"])
    assert claims["uri"] == verb + " api.coinbase.com/api/v3/brokerage" + path
    if method == "get_best_bid_ask":
        assert call.kwargs["params"] == {"product_ids": ["BTC-USD", "ETH-USD"]}


@pytest.mark.parametrize("side,size_key", [("buy", "quote_size"), ("SELL", "base_size")])
def test_market_order_body(client, side, size_key):
    client._session.request.return_value = response({"success": True, "success_response": {"order_id": "1"}})
    client.place_market_order("BTC/USD", side, **{size_key: "12.500"})
    body = client._session.request.call_args.kwargs["json"]
    assert body["product_id"] == "BTC-USD"
    assert body["side"] == side.upper()
    assert body["client_order_id"]
    assert body["order_configuration"] == {"market_market_ioc": {size_key: "12.500"}}


@pytest.mark.parametrize("kwargs", [
    {}, {"quote_size": 1, "base_size": 1}, {"quote_size": 0},
    {"base_size": -1}, {"quote_size": float("nan")}, {"quote_size": "bad"},
    {"base_size": float("inf")},
])
def test_invalid_sizes_do_not_send_orders(client, kwargs):
    with pytest.raises(ValueError):
        client.place_market_order("BTC/USD", "BUY", **kwargs)
    client._session.request.assert_not_called()


def test_invalid_side(client):
    with pytest.raises(ValueError):
        client.place_market_order("BTC/USD", "HOLD", quote_size=1)
    client._session.request.assert_not_called()


def test_order_rejection(client):
    client._session.request.return_value = response({"success": False})
    with pytest.raises(CoinbaseOrderRejected):
        client.place_market_order("BTC/USD", "BUY", quote_size=10)


def test_malformed_order_response_is_ambiguous_not_rejected(client):
    client._session.request.return_value = response({})
    with pytest.raises(RuntimeError, match="ambiguous"):
        client.place_market_order("BTC/USD", "BUY", quote_size=10)


def test_paper_reads_live_but_never_mutates(client, monkeypatch, caplog):
    monkeypatch.setenv("PAPER_TRADING", "true")
    secret = base64.b64encode(client._private_key.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )).decode()
    paper = CoinbaseClient(KEY_NAME, secret, min_request_interval=0)
    paper._session = client._session
    paper._session.request.return_value = response({"accounts": []})
    paper.get_accounts()
    paper._session.request.reset_mock()
    with caplog.at_level(logging.INFO):
        assert paper.place_market_order("BTC/USD", "BUY", quote_size=20)["paper"]
        assert paper.cancel_orders(["1"])["paper"]
    assert "Would BUY BTC-USD" in caplog.text
    with pytest.raises(RuntimeError):
        paper._request("POST", "/orders")
    paper._session.request.assert_not_called()


@pytest.mark.parametrize("failure", [response({}, 429), response({}, 503), requests.Timeout()])
def test_retry_backoff_and_idempotency(client, failure):
    client._session.request.side_effect = [failure, failure, response({"success": True})]
    with patch("core.coinbase_client.time.sleep") as sleep:
        client.place_market_order("BTC/USD", "BUY", quote_size=100)
    assert [c.args[0] for c in sleep.call_args_list] == [0.5, 1.0]
    calls = client._session.request.call_args_list
    assert len({c.kwargs["json"]["client_order_id"] for c in calls}) == 1
    assert len({c.kwargs["headers"]["Authorization"] for c in calls}) == 3


@pytest.mark.parametrize("status", [400, 401, 403])
def test_permanent_errors_not_retried(client, status):
    client._session.request.return_value = response({}, status)
    with pytest.raises(requests.HTTPError):
        client.get_accounts()
    assert client._session.request.call_count == 1


def test_retry_exhaustion(client):
    client._session.request.side_effect = requests.ConnectionError()
    with patch("core.coinbase_client.time.sleep"), pytest.raises(requests.ConnectionError):
        client.get_accounts()
    assert client._session.request.call_count == 3


def test_rate_limit(client):
    client.min_request_interval = 0.1
    client._last_request = 10
    with patch("core.coinbase_client.time.monotonic", return_value=10.04), patch("core.coinbase_client.time.sleep") as sleep:
        client._throttle()
    assert sleep.call_args.args[0] == pytest.approx(0.06)


def test_account_pagination(client):
    client._session.request.side_effect = [
        response({"accounts": [{"currency": "USD"}], "has_next": True, "cursor": "next"}),
        response({"accounts": [{"currency": "BTC"}], "has_next": False}),
    ]
    assert len(client.get_accounts()["accounts"]) == 2
    assert client._session.request.call_args.kwargs["params"]["cursor"] == "next"


def test_product_pagination(client):
    client._session.request.side_effect = [
        response({"products": [{}] * 250, "num_products": 251}),
        response({"products": [{}], "num_products": 251}),
    ]
    assert len(client.list_products()["products"]) == 251
    assert client._session.request.call_args.kwargs["params"]["offset"] == 250


def test_entire_symbol_universe():
    universe = set(TradingConfig().POPULAR_PAIRS + TradingConfig().SPOT_WATCHLIST +
                   list(trading_engine.cfg.SPOT_SYMBOLS))
    for symbol in universe:
        assert to_coinbase_symbol(symbol) == symbol.replace("/", "-")
        assert to_alpaca_symbol(to_coinbase_symbol(symbol)) == symbol
    assert to_coinbase_symbol(" btc-usd ") == "BTC-USD"
    with pytest.raises(ValueError):
        to_coinbase_symbol("PI_XBTUSD")


@pytest.mark.parametrize("enabled,key,secret,broker", [
    (False, "key", "secret", "alpaca"), (True, "", "secret", "alpaca"),
    (True, "key", " ", "alpaca"), (True, "key", "secret", "coinbase"),
])
def test_broker_selection(enabled, key, secret, broker):
    assert select_spot_broker(enabled, key, secret) == broker


def test_config_environment(monkeypatch):
    monkeypatch.setenv("ENABLE_COINBASE", "true")
    monkeypatch.setenv("COINBASE_API_KEY", KEY_NAME)
    monkeypatch.setenv("COINBASE_API_SECRET", "test-placeholder")
    monkeypatch.setenv("PAPER_TRADING", "false")
    for config_type in [TradingConfig, trading_engine.TradingConfig]:
        config = config_type()
        assert config.ENABLE_COINBASE and not config.PAPER_TRADING
        assert config.COINBASE_API_KEY == KEY_NAME
        assert config.COINBASE_API_SECRET == "test-placeholder"
        assert "test-placeholder" not in repr(config)
    monkeypatch.delenv("ENABLE_COINBASE")
    assert not TradingConfig().ENABLE_COINBASE
    assert not trading_engine.TradingConfig().ENABLE_COINBASE


@pytest.fixture
def bot(tmp_path):
    bot = trading_engine.TradingBot.__new__(trading_engine.TradingBot)
    bot.spot_broker = "coinbase"
    bot.coinbase_client = MagicMock()
    bot.trading_client = None
    bot.logger = logging.getLogger("test_coinbase")
    bot.positions = {}
    bot._coinbase_pending_path = str(tmp_path / "pending.json")
    bot.coinbase_client.get_product.return_value = {
        "price": "50000", "quote_increment": "0.01", "quote_min_size": "1",
        "base_increment": "0.00001", "base_min_size": "0.00001",
    }
    bot.coinbase_client.get_best_bid_ask.return_value = {
        "pricebooks": [{"product_id": "BTC-USD", "bids": [{"price": "50000"}],
                        "asks": [{"price": "50001"}]}],
    }
    return bot


def test_coinbase_startup_verifies_auth_without_alpaca(bot, monkeypatch, caplog):
    monkeypatch.setenv("ENABLE_COINBASE", "true")
    monkeypatch.setenv("COINBASE_API_KEY", KEY_NAME)
    monkeypatch.setenv("COINBASE_API_SECRET", "test-placeholder")
    with patch.object(trading_engine, "CoinbaseClient", return_value=bot.coinbase_client) as constructor, \
            patch.object(bot, "_init_kraken_futures"), patch.object(trading_engine, "TradingClient") as alpaca, \
            caplog.at_level(logging.INFO):
        bot._init_api()
    assert bot.spot_broker == "coinbase"
    constructor.return_value.get_accounts.assert_called_once()
    alpaca.assert_not_called()
    assert "Active spot broker: COINBASE" in caplog.text


def test_alpaca_routing_unchanged(bot):
    bot.spot_broker = "alpaca"
    bot._alpaca_buy_spot = MagicMock(return_value=1)
    bot._alpaca_sell_spot = MagicMock(return_value=2)
    bot._alpaca_spot_available_qty = MagicMock(return_value=(3, True))
    assert bot._buy_spot("BTC/USD", 100) == 1
    assert bot._sell_spot("BTC/USD", 2) == 2
    assert bot._spot_available_qty("BTC/USD") == (3, True)
    bot.coinbase_client.place_market_order.assert_not_called()


def test_paper_engine_never_places_orders(bot, monkeypatch, caplog):
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", True)
    with caplog.at_level(logging.INFO):
        assert bot._buy_spot("BTC/USD", 100) == 0
        assert bot._sell_spot("BTC/USD", 1) == 0
    assert "Would BUY" in caplog.text and "Would SELL" in caplog.text
    bot.coinbase_client.place_market_order.assert_not_called()
    assert not os.path.exists(bot._coinbase_pending_path)


def test_coinbase_market_data(bot):
    assert bot._fetch_public_spot_price("BTC/USD") == 50000
    assert bot._fetch_spot_quote("BTC/USD") == (50000, 50001)
    bot.coinbase_client.get_product_book.return_value = {
        "pricebook": {"bids": [{"price": "50000", "size": "2"}],
                      "asks": [{"price": "50001", "size": "1"}]},
    }
    assert bot._fetch_spot_depth_usd("BTC/USD") == 50001


def test_coinbase_balance_errors_never_confirm_zero(bot):
    bot.coinbase_client.get_accounts.return_value = {
        "accounts": [{"currency": "BTC", "available_balance": {"value": "0"},
                      "hold": {"value": "0"}}],
    }
    assert bot._spot_available_qty("BTC/USD") == (0, True)
    assert bot._confirm_zero_broker_qty("BTC/USD")
    bot.coinbase_client.get_accounts.side_effect = requests.Timeout()
    assert not bot._confirm_zero_broker_qty("BTC/USD")


@pytest.mark.parametrize("side,expected", [("BUY", "100.12"), ("SELL", "0.5")])
def test_live_orders_confirm_actual_partial_fills(bot, monkeypatch, side, expected):
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", False)
    bot.coinbase_client.get_accounts.return_value = {
        "accounts": [{"currency": "BTC", "available_balance": {"value": "0.5"}}],
    }
    bot.coinbase_client.place_market_order.return_value = {"success_response": {"order_id": "1"}}
    bot.coinbase_client.get_order.return_value = {"order": {"status": "FILLED", "filled_size": "0.001"}}
    size = 100.129 if side == "BUY" else 0.6
    assert bot._coinbase_order_spot("BTC/USD", side, size) == 0.001
    kwargs = bot.coinbase_client.place_market_order.call_args.kwargs
    assert kwargs["quote_size" if side == "BUY" else "base_size"] == expected
    assert os.path.exists(bot._coinbase_pending_path)
    assert bot._coinbase_order_resolved


def test_unresolved_order_blocks_restart(bot, monkeypatch):
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", False)
    bot.coinbase_client.place_market_order.side_effect = requests.Timeout()
    with pytest.raises(SystemExit, match="unresolved"):
        bot._buy_spot("BTC/USD", 100)
    assert os.path.exists(bot._coinbase_pending_path)
    with patch.object(trading_engine.os.path, "exists", return_value=True), \
            pytest.raises(RuntimeError, match="Reconcile"):
        bot._init_api()


def test_reject_clears_pending_marker(bot, monkeypatch):
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", False)
    bot.coinbase_client.place_market_order.side_effect = CoinbaseOrderRejected()
    assert bot._buy_spot("BTC/USD", 100) == 0
    assert not os.path.exists(bot._coinbase_pending_path)


def test_broker_switch_protects_existing_positions(bot):
    bot.positions = {"BTC/USD": SimpleNamespace(broker_qty=1, spot_broker="alpaca")}
    with pytest.raises(RuntimeError, match="switching"):
        bot._validate_spot_broker_positions()


@pytest.mark.parametrize("enabled,key,secret", [
    ("false", KEY_NAME, "test-placeholder"),
    ("true", "", "test-placeholder"),
    ("true", KEY_NAME, ""),
])
def test_startup_falls_back_to_alpaca(bot, monkeypatch, enabled, key, secret):
    monkeypatch.setenv("ENABLE_COINBASE", enabled)
    monkeypatch.setenv("COINBASE_API_KEY", key)
    monkeypatch.setenv("COINBASE_API_SECRET", secret)
    monkeypatch.setenv("ALPACA_API_KEY", "test-placeholder")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-placeholder")
    with patch.object(trading_engine, "CoinbaseClient") as coinbase, \
            patch.object(trading_engine, "TradingClient") as alpaca, \
            patch.object(trading_engine.requests, "Session"), patch.object(bot, "_init_kraken_futures"):
        bot._init_api()
    assert bot.spot_broker == "alpaca"
    alpaca.assert_called_once()
    coinbase.assert_not_called()


def test_startup_coinbase_live_requires_confirmation(bot, monkeypatch):
    monkeypatch.setenv("ENABLE_COINBASE", "true")
    monkeypatch.setenv("COINBASE_API_KEY", KEY_NAME)
    monkeypatch.setenv("COINBASE_API_SECRET", "test-placeholder")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.delenv("LIVE_TRADING_CONFIRM", raising=False)
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", True)
    with patch.object(trading_engine, "CoinbaseClient") as constructor, \
            pytest.raises(RuntimeError, match="Live trading blocked"):
        bot._init_api()
    constructor.assert_not_called()


def test_paper_env_overrides_live_locked_profile(bot, monkeypatch):
    monkeypatch.setenv("ENABLE_COINBASE", "true")
    monkeypatch.setenv("COINBASE_API_KEY", KEY_NAME)
    monkeypatch.setenv("COINBASE_API_SECRET", "test-placeholder")
    monkeypatch.setenv("PAPER_TRADING", "true")
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", False)
    with patch.object(trading_engine, "CoinbaseClient", return_value=bot.coinbase_client) as constructor, \
            patch.object(bot, "_init_kraken_futures"):
        bot._init_api()
    assert trading_engine.cfg.PAPER_TRADING
    assert constructor.call_args.kwargs["paper_trading"]


def test_filled_order_marker_cleared_only_after_positions_saved(bot, monkeypatch, tmp_path):
    monkeypatch.setattr(trading_engine, "__file__", str(tmp_path / "core" / "trading_engine.py"))
    bot._coinbase_order_resolved = True
    with open(bot._coinbase_pending_path, "w") as pending:
        json.dump({"client_order_id": "test-order"}, pending)
    # Other state is deliberately absent: the position ledger must be persisted
    # before clearing the latch, even if a later balance/history write fails.
    bot._save_state()
    assert json.loads((tmp_path / "data" / "state" / "positions.json").read_text()) == {}
    assert not os.path.exists(bot._coinbase_pending_path)
    assert not bot._coinbase_order_resolved


def test_failed_position_save_keeps_marker(bot, monkeypatch):
    bot._coinbase_order_resolved = True
    with open(bot._coinbase_pending_path, "w") as pending:
        pending.write("{}")
    with patch.object(trading_engine.os, "makedirs", side_effect=OSError()):
        bot._save_state()
    assert os.path.exists(bot._coinbase_pending_path)


def test_runtime_fingerprint_never_serializes_credentials(bot, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trading_engine.cfg, "COINBASE_API_KEY", KEY_NAME)
    monkeypatch.setattr(trading_engine.cfg, "COINBASE_API_SECRET", "test-placeholder")
    bot.ml_model = SimpleNamespace(model_path="missing-model")
    for name in ["direction_bias", "direction_bias_strength", "enable_futures_data",
                 "enable_kraken_futures_fallback", "rl_shadow_mode", "rl_shadow_min_multiplier",
                 "rl_live_size_control", "rl_live_size_min_mult", "rl_live_size_max_mult",
                 "use_locked_profile", "locked_profile_path"]:
        setattr(bot, name, False)
    bot._save_runtime_fingerprint()
    for path in (tmp_path / "data" / "state").glob("runtime_fingerprint*"):
        text = path.read_text()
        assert KEY_NAME not in text
        assert "test-placeholder" not in text
        assert "COINBASE_API_SECRET" not in text
    assert (tmp_path / "data" / "state" / "runtime_fingerprint_latest.json").exists()


def test_production_paper_signal_tracks_simulated_position(bot, monkeypatch, caplog):
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", True)
    bot.risk_manager = MagicMock(consecutive_wins=0, consecutive_losses=0)
    bot.risk_manager.can_trade.return_value = (True, "")
    bot.risk_manager.calculate_position_size.return_value = 100
    bot.market_data = MagicMock(price_history={"BTC/USD": [50000]})
    bot.market_data.calculate_volatility.return_value = 0.01
    bot.balance_spot = bot.balance_futures = 1000
    bot._current_regime = None
    bot._regime_flip_state = {}
    bot.rl_live_size_control = False
    bot.ml_challenger_shadow_mode = False
    bot._is_symbol_paused = MagicMock(return_value=False)
    bot._rl_shadow_decision = MagicMock(return_value={})
    bot._passes_correlation_gate = MagicMock(return_value=(True, 0))
    bot._passes_liquidity_gate = MagicMock(return_value=(True, 10000))
    bot._shadow_open_position = MagicMock()
    bot._shadow_record_event = MagicMock()
    bot._save_state = MagicMock()
    signal = SimpleNamespace(symbol="BTC/USD", direction="LONG", confidence=0.9,
                             reason="test", timestamp=datetime.now())
    with caplog.at_level(logging.INFO):
        bot.execute_signals([signal])
    assert bot.positions["BTC/USD"].broker_qty == 0
    assert bot.positions["BTC/USD"].spot_broker == "coinbase"
    assert "COINBASE PAPER: Would BUY" in caplog.text
    bot.coinbase_client.place_market_order.assert_not_called()


def test_held_coinbase_funds_keep_position_tracked(bot, monkeypatch):
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", False)
    bot.coinbase_client.get_accounts.return_value = {
        "accounts": [{"currency": "BTC", "available_balance": {"value": "0"},
                      "hold": {"value": "1"}}],
    }
    assert bot._spot_available_qty("BTC/USD") == (0, True)
    assert not bot._confirm_zero_broker_qty("BTC/USD")
    bot.positions = {"BTC/USD": trading_engine.Position(
        symbol="BTC/USD", direction="LONG", entry_price=50000, size=50000,
        entry_time=datetime.now(), stop_loss=49000, take_profit=52000,
        max_price=50000, broker_qty=1, spot_broker="coinbase",
    )}
    bot._sell_spot = MagicMock(return_value=0)
    bot._save_state = MagicMock()
    bot._should_reconcile_dust = MagicMock(return_value=False)
    for _ in range(3):
        bot._close_position("BTC/USD", 50000, "STOP_LOSS", 0)
    assert bot.positions["BTC/USD"].broker_qty == 1
    assert bot.positions["BTC/USD"].broker_zero_qty_confirmations == 0


def test_alpaca_fallback_live_requires_early_confirmation(bot, monkeypatch):
    monkeypatch.setenv("ENABLE_COINBASE", "false")
    monkeypatch.delenv("LIVE_TRADING_CONFIRM", raising=False)
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", False)
    with patch.object(trading_engine, "TradingClient") as alpaca, \
            pytest.raises(RuntimeError, match="Live trading blocked"):
        bot._init_api()
    alpaca.assert_not_called()


def test_coinbase_short_does_not_charge_paper_fee(bot, monkeypatch):
    monkeypatch.setattr(trading_engine.cfg, "PAPER_TRADING", True)
    bot.balance_spot = bot.balance_futures = 1000
    bot.risk_manager = MagicMock()
    bot.risk_manager.can_trade.return_value = (True, "")
    bot._rl_shadow_decision = MagicMock()
    bot.execute_signals([SimpleNamespace(symbol="BTC/USD", direction="SHORT", confidence=0.9)])
    assert bot.balance_spot == 1000
    assert not bot.positions
    bot._rl_shadow_decision.assert_not_called()
    bot.coinbase_client.place_market_order.assert_not_called()
