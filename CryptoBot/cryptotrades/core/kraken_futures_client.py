"""
Kraken Futures REST client — authenticated order execution.

Implements Kraken Futures' HMAC signing scheme:
    message   = postData + nonce + endpointPath (endpoint without the
                leading "/derivatives" segment)
    sha256    = SHA256(message)
    signature = base64(HMAC-SHA512(base64_decode(api_secret), sha256))
Headers: APIKey, Nonce, Authent.

Reference: https://docs.kraken.com/api/docs/futures-api/trading
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
import urllib.parse
from typing import Optional

import requests

DEMO_BASE_URL = "https://demo-futures.kraken.com/derivatives"
LIVE_BASE_URL = "https://futures.kraken.com/derivatives"


class KrakenFuturesClient:
    """Minimal authenticated client for Kraken Futures order management."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = DEMO_BASE_URL, timeout: float = 10.0):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _sign(self, path: str, post_data: str, nonce: str) -> str:
        # path must be the endpoint path without the "/derivatives" prefix, e.g. "/api/v3/sendorder".
        message = (post_data + nonce + path).encode("utf-8")
        sha256_digest = hashlib.sha256(message).digest()
        secret_decoded = base64.b64decode(self.api_secret)
        mac = hmac.new(secret_decoded, sha256_digest, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _request(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        """path is relative to base_url, e.g. '/api/v3/sendorder'."""
        params = params or {}
        post_data = urllib.parse.urlencode(params)
        nonce = str(int(time.time() * 1000))
        signature = self._sign(path, post_data, nonce)
        headers = {
            "APIKey": self.api_key,
            "Nonce": nonce,
            "Authent": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        url = self.base_url + path
        if method == "GET":
            resp = self._session.get(url, headers=headers, params=params, timeout=self.timeout)
        else:
            resp = self._session.post(url, headers=headers, data=post_data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_accounts(self) -> dict:
        """Read-only account/balance check — used to verify credentials without placing orders."""
        return self._request("GET", "/api/v3/accounts")

    def get_open_positions(self) -> dict:
        return self._request("GET", "/api/v3/openpositions")

    def send_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "mkt",
        limit_price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> dict:
        """Submit a Kraken Futures order. side: 'buy' or 'sell'. size is in contracts."""
        params = {
            "orderType": order_type,
            "symbol": symbol,
            "side": side,
            "size": size,
        }
        if limit_price is not None:
            params["limitPrice"] = limit_price
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._request("POST", "/api/v3/sendorder", params)

    def cancel_order(self, order_id: str) -> dict:
        return self._request("POST", "/api/v3/cancelorder", {"order_id": order_id})
