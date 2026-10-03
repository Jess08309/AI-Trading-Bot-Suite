"""Coinbase Advanced Trade spot REST client with CDP JWT authentication."""
from __future__ import annotations

import base64
import logging
import os
import re
import secrets
import threading
import time
import uuid
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)
API_HOST = "api.coinbase.com"
API_PATH = "/api/v3/brokerage"


class CoinbaseOrderRejected(ValueError):
    """An explicit rejection, unlike an ambiguous transport failure."""


def to_coinbase_symbol(symbol: str) -> str:
    """Map the bot's BASE/USD universe to Coinbase BASE-USD product IDs."""
    product = symbol.strip().upper().replace("/", "-")
    if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", product):
        raise ValueError("Expected a spot pair such as BTC/USD or BTC-USD")
    return product


def to_alpaca_symbol(product_id: str) -> str:
    return to_coinbase_symbol(product_id).replace("-", "/")


def select_spot_broker(enable_coinbase: bool, api_key: str, api_secret: str) -> str:
    return "coinbase" if enable_coinbase and api_key.strip() and api_secret.strip() else "alpaca"


class CoinbaseClient:
    """Signed live reads; paper mode never sends a mutating API request."""

    def __init__(self, api_key=None, api_secret=None, paper_trading=None,
                 timeout=10.0, retry_attempts=3, retry_base_delay=0.5,
                 min_request_interval=0.1):
        self.api_key = (api_key if api_key is not None else os.getenv("COINBASE_API_KEY", "")).strip()
        secret = (api_secret if api_secret is not None else os.getenv("COINBASE_API_SECRET", "")).strip()
        if not self.api_key or not secret:
            raise ValueError("Coinbase CDP credentials are required")
        self.paper_trading = (
            os.getenv("PAPER_TRADING", "true").strip().lower() in {"true", "1", "yes", "on"}
            if paper_trading is None else paper_trading
        )
        self.timeout = timeout
        self.retry_attempts = max(1, retry_attempts)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self.min_request_interval = max(0.0, min_request_interval)
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._private_key, self._algorithm = self._load_key(secret)

    @staticmethod
    def _load_key(secret):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec, ed25519

        secret = secret.replace("\\n", "\n")
        try:
            if "-----BEGIN" in secret:
                key = serialization.load_pem_private_key(secret.encode(), None)
            else:
                raw = base64.b64decode("".join(secret.split()), validate=True)
                # CDP Ed25519 secrets contain a 32-byte seed, optionally followed
                # by the 32-byte public key.
                if len(raw) not in (32, 64):
                    raise ValueError("Invalid Ed25519 key length")
                key = ed25519.Ed25519PrivateKey.from_private_bytes(raw[:32])
        except (ValueError, TypeError):
            raise ValueError("Invalid Coinbase CDP private key") from None
        if isinstance(key, ed25519.Ed25519PrivateKey):
            return key, "EdDSA"
        if isinstance(key, ec.EllipticCurvePrivateKey) and isinstance(key.curve, ec.SECP256R1):
            return key, "ES256"
        raise ValueError("Coinbase requires an Ed25519 or P-256 private key")

    def _jwt(self, method, path):
        import jwt

        now = int(time.time())
        return jwt.encode(
            {"sub": self.api_key, "iss": "cdp", "nbf": now, "exp": now + 120,
             "uri": f"{method} {API_HOST}{path}"},
            self._private_key, algorithm=self._algorithm,
            headers={"kid": self.api_key, "nonce": secrets.token_hex()},
        )

    def _throttle(self):
        with self._lock:
            delay = self.min_request_interval - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()

    def _request(self, method, endpoint, params=None, body=None):
        if self.paper_trading and method != "GET":
            raise RuntimeError("Coinbase mutations are blocked in paper mode")
        path = API_PATH + endpoint
        for attempt in range(self.retry_attempts):
            self._throttle()
            try:
                token = self._jwt(method, path)
                response = self._session.request(
                    method, f"https://{API_HOST}{path}", params=params, json=body,
                    headers={"Authorization": "Bearer " + token},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                status = exc.response.status_code
                if status != 429 and status < 500:
                    raise
                if attempt + 1 == self.retry_attempts:
                    raise
            except (requests.ConnectionError, requests.Timeout):
                if attempt + 1 == self.retry_attempts:
                    raise
            time.sleep(self.retry_base_delay * 2 ** attempt)

    def get_accounts(self):
        accounts = []
        cursor = None
        while True:
            params = {"limit": 250}
            if cursor:
                params["cursor"] = cursor
            result = self._request("GET", "/accounts", params=params)
            accounts.extend(result.get("accounts", []))
            if not result.get("has_next"):
                return {**result, "accounts": accounts}
            next_cursor = result.get("cursor")
            if not next_cursor or next_cursor == cursor:
                raise RuntimeError("Coinbase returned an invalid account cursor")
            cursor = next_cursor

    def get_product(self, product_id):
        return self._request("GET", "/products/" + to_coinbase_symbol(product_id))

    def list_products(self):
        products = []
        offset = 0
        while True:
            result = self._request("GET", "/products",
                                   params={"product_type": "SPOT", "limit": 250, "offset": offset})
            page = result.get("products", [])
            products.extend(page)
            offset += len(page)
            if len(page) < 250 or offset >= result.get("num_products", float("inf")):
                return {**result, "products": products}

    @staticmethod
    def _size(value):
        try:
            size = Decimal(str(value))
        except InvalidOperation:
            raise ValueError("Order size must be positive and finite") from None
        if not size.is_finite() or size <= 0:
            raise ValueError("Order size must be positive and finite")
        return format(size, "f")

    def place_market_order(self, product_id, side, quote_size=None, base_size=None,
                           client_order_id=None):
        product_id = to_coinbase_symbol(product_id)
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Order side must be BUY or SELL")
        if (quote_size is None) == (base_size is None):
            raise ValueError("Provide exactly one of quote_size or base_size")
        size_key = "quote_size" if quote_size is not None else "base_size"
        size = self._size(quote_size if quote_size is not None else base_size)
        # Reuse this ID across transport retries to prevent duplicate purchases.
        body = {"client_order_id": client_order_id or str(uuid.uuid4()), "product_id": product_id, "side": side,
                "order_configuration": {"market_market_ioc": {size_key: size}}}
        if self.paper_trading:
            logger.info("COINBASE PAPER: Would %s %s %s=%s", side, product_id, size_key, size)
            return {"paper": True, "intended_order": body}
        result = self._request("POST", "/orders", body=body)
        if not result.get("success"):
            raise CoinbaseOrderRejected("Coinbase rejected the market order")
        return result

    def get_order(self, order_id):
        return self._request("GET", "/orders/historical/" + quote(str(order_id), safe=""))

    def cancel_orders(self, order_ids):
        order_ids = [str(order_id) for order_id in order_ids]
        if self.paper_trading:
            logger.info("COINBASE PAPER: Would cancel %d orders", len(order_ids))
            return {"paper": True, "order_ids": order_ids}
        return self._request("POST", "/orders/batch_cancel", body={"order_ids": order_ids})

    def get_best_bid_ask(self, product_ids):
        return self._request("GET", "/best_bid_ask",
                             params={"product_ids": [to_coinbase_symbol(p) for p in product_ids]})

    def get_product_book(self, product_id, limit=10):
        return self._request("GET", "/product_book",
                             params={"product_id": to_coinbase_symbol(product_id), "limit": limit})
