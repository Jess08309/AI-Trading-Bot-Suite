"""
Alpaca API Client - wraps alpaca-py for trading, account, and market data.
"""
import logging
import time as _time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    GetOrdersRequest, ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

from core.config import Config

log = logging.getLogger("alpacabot.api")


class AlpacaAPI:
    """Thin wrapper around Alpaca's SDK for stocks + options."""

    # Alpaca paper: 200 req/min.  Keep comfortably under that.
    _MIN_CALL_INTERVAL = 0.50  # seconds between API calls (~120/min max)

    def __init__(self, config: Config):
        self.config = config
        self._trading: Optional[TradingClient] = None
        self._data: Optional[StockHistoricalDataClient] = None
        self._last_call_ts: float = 0.0
        self._rate_lock = threading.Lock()

    # ── Connection ───────────────────────────────────────
    def connect(self) -> bool:
        """Initialize API clients. Returns True if account is accessible."""
        try:
            self._trading = TradingClient(
                api_key=self.config.API_KEY,
                secret_key=self.config.API_SECRET,
                paper=self.config.PAPER,
            )
            self._data = StockHistoricalDataClient(
                api_key=self.config.API_KEY,
                secret_key=self.config.API_SECRET,
            )
            acct = self._trading.get_account()
            log.info(f"Connected to Alpaca ({'PAPER' if self.config.PAPER else 'LIVE'})")
            log.info(f"Account equity: ${float(acct.equity):,.2f}, "
                     f"cash: ${float(acct.cash):,.2f}, "
                     f"buying power: ${float(acct.buying_power):,.2f}")
            return True
        except Exception as e:
            log.error(f"Alpaca connection failed: {e}")
            return False

    def _throttle(self):
        """Rate-limit: sleep if we're calling too fast."""
        with self._rate_lock:
            now = _time.monotonic()
            elapsed = now - self._last_call_ts
            if elapsed < self._MIN_CALL_INTERVAL:
                _time.sleep(self._MIN_CALL_INTERVAL - elapsed)
            self._last_call_ts = _time.monotonic()

    def _call_with_retry(self, fn, *args, retries: int = 3, **kwargs):
        """Call fn(*args, **kwargs) with exponential backoff on 429 rate-limit errors."""
        wait = 5.0
        for attempt in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "too many requests" in msg.lower() or "rate limit" in msg.lower():
                    log.warning(f"Rate limit hit (attempt {attempt+1}/{retries}) — backing off {wait:.0f}s")
                    _time.sleep(wait)
                    wait *= 2  # exponential: 5s → 10s → 20s
                else:
                    raise
        raise RuntimeError(f"Rate limit retry exhausted after {retries} attempts")

    @property
    def trading(self) -> TradingClient:
        if not self._trading:
            raise RuntimeError("Call connect() first")
        return self._trading

    @property
    def data(self) -> StockHistoricalDataClient:
        if not self._data:
            raise RuntimeError("Call connect() first")
        return self._data

    # ── Account ──────────────────────────────────────────
    def get_account(self) -> Dict[str, Any]:
        """Get account info as dict."""
        self._throttle()
        acct = self.trading.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "day_trade_count": int(acct.daytrade_count),
            "pattern_day_trader": acct.pattern_day_trader,
            "trading_blocked": acct.trading_blocked,
            "account_blocked": acct.account_blocked,
        }

    def get_balance(self) -> float:
        """Current portfolio value."""
        self._throttle()
        return float(self.trading.get_account().portfolio_value)

    # ── Positions ────────────────────────────────────────
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        self._throttle()
        positions = self.trading.get_all_positions()
        result = []
        for p in positions:
            result.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "asset_class": getattr(p, 'asset_class', 'us_equity'),
            })
        return result

    def close_position(self, symbol: str) -> bool:
        """Close entire position for a symbol via Alpaca's close endpoint."""
        try:
            self._throttle()
            self._call_with_retry(self.trading.close_position, symbol)
            log.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            msg = str(e)
            # Position already gone from Alpaca — treat as successfully closed
            if "position not found" in msg.lower() or "40410000" in msg:
                log.warning(f"Position {symbol} not found on Alpaca — treating as already closed")
                return True
            log.error(f"Failed to close {symbol}: {e}")
            return False

    def position_exists(self, symbol: str) -> bool:
        """Check if a position actually exists on Alpaca."""
        try:
            self._throttle()
            pos = self.trading.get_open_position(symbol)
            return pos is not None
        except Exception:
            return False

    # ── Orders ───────────────────────────────────────────
    def buy_option(self, symbol: str, qty: int = 1,
                   limit_price: Optional[float] = None) -> Optional[str]:
        """
        Buy an options contract.
        symbol: OCC symbol like 'SPY260320C00580000'
        Returns order ID or None on failure.
        """
        try:
            self._throttle()
            if limit_price:
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            order = self.trading.submit_order(req)
            log.info(f"BUY {qty}x {symbol} @ {'MKT' if not limit_price else f'${limit_price}'} "
                     f"→ order {order.id}")
            return str(order.id)
        except Exception as e:
            log.error(f"Order failed for {symbol}: {e}")
            return None

    def cancel_conflicting_orders(self, symbol: str) -> int:
        """Cancel any open BUY orders for a symbol before selling.
        Returns the number of orders cancelled."""
        cancelled = 0
        try:
            open_orders = self.get_open_orders()
            for order in open_orders:
                if order and order.get("symbol") == symbol and "BUY" in str(order.get("side", "")):
                    if self.cancel_order(order["id"]):
                        log.info(f"Cancelled conflicting BUY order {order['id']} for {symbol}")
                        cancelled += 1
            if cancelled:
                import time
                time.sleep(0.5)  # brief pause for cancellation to settle
        except Exception as e:
            log.warning(f"Error checking conflicting orders for {symbol}: {e}")
        return cancelled

    def sell_option(self, symbol: str, qty: int = 1,
                    limit_price: Optional[float] = None) -> Optional[str]:
        """Sell (close) an options position."""
        try:
            # Cancel any conflicting BUY orders first
            self.cancel_conflicting_orders(symbol)

            self._throttle()
            if limit_price:
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    type=OrderType.LIMIT,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            order = self._call_with_retry(self.trading.submit_order, req)
            log.info(f"SELL {qty}x {symbol} @ {'MKT' if not limit_price else f'${limit_price}'} "
                     f"→ order {order.id}")
            return str(order.id)
        except Exception as e:
            msg = str(e)
            # Position already gone — not really an error
            if "position not found" in msg.lower() or "40410000" in msg:
                log.warning(f"Sell skipped for {symbol} — position not found on Alpaca")
                return "ALREADY_CLOSED"
            log.error(f"Sell order failed for {symbol}: {e}")
            return None

    def submit_mleg_order(self, legs: List[Dict], qty: int = 1,
                          limit_price: Optional[float] = None,
                          time_in_force: str = "day") -> Optional[str]:
        """Submit a multi-leg options order (Level 3).

        Args:
            legs: list of dicts with keys: symbol, side, position_intent, ratio_qty
            qty: number of spread sets
            limit_price: net debit per share (positive), None for market
            time_in_force: 'day' or 'gtc'

        Returns:
            order_id or None on failure
        """
        try:
            self._throttle()
            import requests as req
            url = f"{self.config.BASE_URL}/v2/orders"
            headers = {
                "APCA-API-KEY-ID": self.config.API_KEY,
                "APCA-API-SECRET-KEY": self.config.API_SECRET,
                "accept": "application/json",
                "content-type": "application/json",
            }
            order_data = {
                "order_class": "mleg",
                "qty": str(qty),
                "time_in_force": time_in_force,
                "legs": [{
                    "symbol": leg["symbol"],
                    "ratio_qty": str(leg.get("ratio_qty", 1)),
                    "side": leg["side"],
                    "position_intent": leg["position_intent"],
                } for leg in legs],
            }
            if limit_price is not None:
                order_data["type"] = "limit"
                order_data["limit_price"] = str(round(limit_price, 2))
            else:
                order_data["type"] = "market"

            resp = req.post(url, headers=headers, json=order_data, timeout=15)
            if resp.status_code in (200, 201):
                data = resp.json()
                order_id = data.get("id", "")
                log.info(f"MLEG order: {qty}x {len(legs)}-leg "
                         f"@ {'MKT' if limit_price is None else f'${limit_price}'} "
                         f"-> order {order_id}")
                return order_id
            else:
                log.error(f"MLEG order failed: {resp.status_code} {resp.text[:200]}")
                return None
        except Exception as e:
            log.error(f"MLEG order error: {e}")
            return None

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Check order status."""
        try:
            self._throttle()
            order = self.trading.get_order_by_id(order_id)
            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "side": str(order.side),
                "qty": float(order.qty) if order.qty else 0,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                "status": str(order.status),
                "submitted_at": str(order.submitted_at),
            }
        except Exception as e:
            log.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open/pending orders."""
        self._throttle()
        req = GetOrdersRequest(status="open")
        orders = self.trading.get_orders(req)
        return [self.get_order(str(o.id)) for o in orders]

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._throttle()
            self.trading.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            log.error(f"Cancel failed for {order_id}: {e}")
            return False

    # ── Market Data (Underlying) ─────────────────────────
    def get_bars(self, symbol: str, timeframe: TimeFrame = TimeFrame.Minute,
                 days: int = 5) -> list:
        """Get historical bars for a symbol."""
        try:
            self._throttle()
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=datetime.now() - timedelta(days=days),
            )
            bars = self.data.get_stock_bars(req)
            try:
                result = bars[symbol]
                if result:
                    return result
            except (KeyError, IndexError):
                pass
            return []
        except Exception as e:
            log.error(f"Failed to get bars for {symbol}: {e}")
            return []

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest bar close price."""
        try:
            req = StockLatestBarRequest(symbol_or_symbols=symbol)
            bars = self.data.get_stock_latest_bar(req)
            if symbol in bars:
                return float(bars[symbol].close)
            return None
        except Exception as e:
            log.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_options_chain(self, underlying: str,
                          expiration_after: Optional[str] = None,
                          expiration_before: Optional[str] = None,
                          option_type: Optional[str] = None,
                          strike_price_gte: Optional[float] = None,
                          strike_price_lte: Optional[float] = None,
                          ) -> List[Dict[str, Any]]:
        """
        Fetch options contracts for an underlying.
        Uses Alpaca's options API endpoints.
        Returns list of contract dicts.
        """
        try:
            self._throttle()
            import requests as req
            url = f"{self.config.BASE_URL}/v2/options/contracts"
            params = {
                "underlying_symbols": underlying,
                "status": "active",
                "limit": 100,
            }
            if expiration_after:
                params["expiration_date_gte"] = expiration_after
            if expiration_before:
                params["expiration_date_lte"] = expiration_before
            if option_type:
                params["type"] = option_type  # "call" or "put"
            if strike_price_gte is not None:
                params["strike_price_gte"] = str(strike_price_gte)
            if strike_price_lte is not None:
                params["strike_price_lte"] = str(strike_price_lte)

            headers = {
                "APCA-API-KEY-ID": self.config.API_KEY,
                "APCA-API-SECRET-KEY": self.config.API_SECRET,
                "accept": "application/json",
            }
            resp = req.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            contracts = []
            for c in data.get("option_contracts", []):
                contracts.append({
                    "symbol": c.get("symbol", ""),
                    "name": c.get("name", ""),
                    "underlying": c.get("underlying_symbol", ""),
                    "type": c.get("type", ""),      # call / put
                    "strike": float(c.get("strike_price", 0)),
                    "expiration": c.get("expiration_date", ""),
                    "style": c.get("style", "american"),
                    "status": c.get("status", ""),
                    "open_interest": int(c.get("open_interest", 0)),
                    "size": int(c.get("size", 100)),  # multiplier
                })
            return contracts
        except Exception as e:
            log.error(f"Options chain fetch failed for {underlying}: {e}")
            return []

    def get_option_quote(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote (bid/ask/last) for an option contract."""
        try:
            self._throttle()
            import requests as req
            url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest"
            params = {"symbols": option_symbol, "feed": "indicative"}
            headers = {
                "APCA-API-KEY-ID": self.config.API_KEY,
                "APCA-API-SECRET-KEY": self.config.API_SECRET,
                "accept": "application/json",
            }
            resp = req.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            quotes = data.get("quotes", {})
            if option_symbol in quotes:
                q = quotes[option_symbol]
                bid = float(q.get("bp", 0))
                ask = float(q.get("ap", 0))
                return {
                    "symbol": option_symbol,
                    "bid": bid,
                    "ask": ask,
                    "mid": round((bid + ask) / 2, 2) if (bid and ask) else 0,
                    "bid_size": int(q.get("bs", 0)),
                    "ask_size": int(q.get("as", 0)),
                }
            return None
        except Exception as e:
            log.error(f"Option quote failed for {option_symbol}: {e}")
            return None

    # ── Market Status ────────────────────────────────────
    def is_market_open(self) -> bool:
        """Check if stock market is currently open."""
        try:
            clock = self.trading.get_clock()
            return clock.is_open
        except Exception:
            return False

    def next_open(self) -> Optional[datetime]:
        """When does market next open?"""
        try:
            clock = self.trading.get_clock()
            return clock.next_open
        except Exception:
            return None
