from __future__ import annotations

import json
import threading
import time
from typing import Dict, Iterable, Optional, Tuple

import websocket


class KrakenFuturesWS:
    def __init__(self, symbols: Iterable[str], url: str = "wss://futures.kraken.com/ws/v1"):
        self._symbols = list(symbols)
        self._url = url
        self._prices: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ws = websocket.WebSocketApp(
            self._url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    def get_price(self, symbol: str, max_age_seconds: float = 5.0) -> Optional[float]:
        now = time.time()
        with self._lock:
            value = self._prices.get(symbol)
        if not value:
            return None
        price, ts = value
        if (now - ts) > max_age_seconds:
            return None
        return price

    def _run(self) -> None:
        if not self._ws:
            return
        while not self._stop_event.is_set():
            try:
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                time.sleep(1.0)
            if self._stop_event.is_set():
                break
            time.sleep(1.0)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        subscribe_msg = {
            "event": "subscribe",
            "feed": "ticker",
            "product_ids": self._symbols,
        }
        ws.send(json.dumps(subscribe_msg))

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            data = json.loads(message)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        if data.get("feed") != "ticker":
            return
        symbol = data.get("product_id") or data.get("symbol")
        if not symbol:
            return
        price = data.get("last") or data.get("markPrice") or data.get("indexPrice")
        if price is None:
            return
        try:
            price_f = float(price)
        except Exception:
            return
        with self._lock:
            self._prices[symbol] = (price_f, time.time())

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        return

    def _on_close(self, ws: websocket.WebSocketApp, close_status_code, close_msg) -> None:
        return
