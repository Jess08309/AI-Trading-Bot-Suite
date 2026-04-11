"""
PutSeller API Client — Alpaca wrapper for credit put spreads.
Completely isolated from AlpacaBot's API client.
Same account, different allocation, different order tags.
"""
import logging
import time as _time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from core.config import PutSellerConfig

log = logging.getLogger("putseller.api")


class PutSellerAPI:
    """Alpaca API wrapper focused on credit put spreads."""

    _MIN_CALL_INTERVAL = 0.60  # slightly slower than AlpacaBot to be conservative

    def __init__(self, config: PutSellerConfig):
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
            equity = float(acct.equity)
            allocation = equity * self.config.ALLOCATION_PCT
            log.info(f"Connected to Alpaca ({'PAPER' if self.config.PAPER else 'LIVE'})")
            log.info(f"Account equity: ${equity:,.2f} | "
                     f"PutSeller allocation ({self.config.ALLOCATION_PCT:.0%}): ${allocation:,.2f}")
            return True
        except Exception as e:
            log.error(f"Alpaca connection failed: {e}")
            return False

    def _throttle(self):
        """Rate-limit: sleep if calling too fast."""
        with self._rate_lock:
            now = _time.monotonic()
            elapsed = now - self._last_call_ts
            if elapsed < self._MIN_CALL_INTERVAL:
                _time.sleep(self._MIN_CALL_INTERVAL - elapsed)
            self._last_call_ts = _time.monotonic()

    def _call_with_retry(self, fn, *args, retries: int = 3, **kwargs):
        """Call with exponential backoff on rate limit."""
        wait = 5.0
        for attempt in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate limit" in msg.lower():
                    log.warning(f"Rate limit (attempt {attempt+1}/{retries}) — waiting {wait:.0f}s")
                    _time.sleep(wait)
                    wait *= 2
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
        """Get account info."""
        self._throttle()
        acct = self.trading.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
        }

    def get_allocation(self) -> float:
        """Get PutSeller's capital allocation (equity * ALLOCATION_PCT)."""
        acct = self.get_account()
        return acct["equity"] * self.config.ALLOCATION_PCT

    # ── Market Data ──────────────────────────────────────
    def get_bars(self, symbol: str, timeframe: TimeFrame = TimeFrame.Day,
                 days: int = 40) -> list:
        """Get historical daily bars for HV calculation."""
        try:
            self._throttle()
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=datetime.now() - timedelta(days=days),
            )
            bars = self.data.get_stock_bars(req)
            try:
                return bars[symbol]
            except (KeyError, IndexError):
                return []
        except Exception as e:
            log.error(f"Bars fetch failed for {symbol}: {e}")
            return []

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest close price."""
        try:
            from alpaca.data.requests import StockLatestBarRequest
            req = StockLatestBarRequest(symbol_or_symbols=symbol)
            bars = self.data.get_stock_latest_bar(req)
            if symbol in bars:
                return float(bars[symbol].close)
            return None
        except Exception as e:
            log.error(f"Price fetch failed for {symbol}: {e}")
            return None

    def calculate_hv20(self, symbol: str) -> Optional[float]:
        """Calculate 20-day annualized historical volatility."""
        try:
            bars = self.get_bars(symbol, TimeFrame.Day, days=self.config.HV_LOOKBACK_DAYS + 5)
            if len(bars) < 21:
                return None
            closes = [float(b.close) for b in bars[-21:]]
            returns = np.diff(np.log(closes))
            hv = float(np.std(returns) * np.sqrt(252))
            return round(hv, 4)
        except Exception as e:
            log.error(f"HV calculation failed for {symbol}: {e}")
            return None

    # ── Options Chain ────────────────────────────────────
    def get_options_chain(self, underlying: str,
                          expiration_after: Optional[str] = None,
                          expiration_before: Optional[str] = None,
                          option_type: str = "put",
                          strike_price_gte: Optional[float] = None,
                          strike_price_lte: Optional[float] = None,
                          ) -> List[Dict[str, Any]]:
        """Fetch put options contracts for an underlying."""
        try:
            self._throttle()
            import requests as req
            url = f"{self.config.BASE_URL}/v2/options/contracts"
            params = {
                "underlying_symbols": underlying,
                "status": "active",
                "type": option_type,
                "limit": 100,
            }
            if expiration_after:
                params["expiration_date_gte"] = expiration_after
            if expiration_before:
                params["expiration_date_lte"] = expiration_before
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
                    "underlying": c.get("underlying_symbol", ""),
                    "type": c.get("type", ""),
                    "strike": float(c.get("strike_price", 0) or 0),
                    "expiration": c.get("expiration_date", ""),
                    "open_interest": int(c.get("open_interest", 0) or 0),
                    "size": int(c.get("size", 100) or 100),
                })
            return contracts
        except Exception as e:
            log.error(f"Options chain failed for {underlying}: {e}")
            return []

    def get_option_quote(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote for an option contract."""
        try:
            self._throttle()
            import requests as req
            url = "https://data.alpaca.markets/v1beta1/options/quotes/latest"
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
                    "mid": round((bid + ask) / 2, 2) if (bid > 0 or ask > 0) else 0,
                    "bid_size": int(q.get("bs", 0)),
                    "ask_size": int(q.get("as", 0)),
                }
            return None
        except Exception as e:
            log.error(f"Option quote failed for {option_symbol}: {e}")
            return None

    def get_option_snapshot(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """Get snapshot with greeks (delta, IV) for an option.
        Uses the batch snapshots endpoint with symbols param (single-symbol path returns 400).
        """
        try:
            self._throttle()
            import requests as req
            url = "https://data.alpaca.markets/v1beta1/options/snapshots"
            params = {"symbols": option_symbol, "feed": "indicative"}
            headers = {
                "APCA-API-KEY-ID": self.config.API_KEY,
                "APCA-API-SECRET-KEY": self.config.API_SECRET,
                "accept": "application/json",
            }
            resp = req.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Batch endpoint returns {"snapshots": {"SYMBOL": {...}}}
            snapshots = data.get("snapshots", data)
            snap = snapshots.get(option_symbol)
            if not snap:
                return None

            greeks = snap.get("greeks", {})
            quote = snap.get("latestQuote", {})
            bid = float(quote.get("bp", 0))
            ask = float(quote.get("ap", 0))

            return {
                "symbol": option_symbol,
                "delta": float(greeks.get("delta", 0)),
                "gamma": float(greeks.get("gamma", 0)),
                "theta": float(greeks.get("theta", 0)),
                "vega": float(greeks.get("vega", 0)),
                "iv": float(snap.get("impliedVolatility", 0)),
                "bid": bid,
                "ask": ask,
                "mid": round((bid + ask) / 2, 2) if (bid and ask) else 0,
            }
        except Exception as e:
            log.debug(f"Snapshot failed for {option_symbol}: {e}")
            return None

    # ── Multi-Leg Orders (Credit Spreads) ────────────────
    def submit_credit_spread(self, short_symbol: str, long_symbol: str,
                              qty: int, credit_limit: float) -> Optional[str]:
        """
        Submit a bull put spread (credit spread) as a multi-leg order.

        short_symbol: the OTM put we SELL (higher strike, more premium)
        long_symbol:  the further OTM put we BUY (lower strike, protection)
        qty: number of spreads
        credit_limit: net credit per share we want (negative limit_price for credit)

        Returns order_id or None.
        """
        try:
            self._throttle()
            import requests as req
            import uuid

            url = f"{self.config.BASE_URL}/v2/orders"
            headers = {
                "APCA-API-KEY-ID": self.config.API_KEY,
                "APCA-API-SECRET-KEY": self.config.API_SECRET,
                "accept": "application/json",
                "content-type": "application/json",
            }

            # For credit spreads, limit_price is NEGATIVE (we receive credit)
            order_data = {
                "order_class": "mleg",
                "type": "limit",
                "limit_price": str(round(-abs(credit_limit), 2)),  # negative = credit
                "qty": str(qty),
                "time_in_force": "day",
                "client_order_id": f"{self.config.ORDER_PREFIX}{uuid.uuid4().hex[:12]}",
                "legs": [
                    {
                        "symbol": short_symbol,
                        "ratio_qty": "1",
                        "side": "sell",
                        "position_intent": "sell_to_open",
                    },
                    {
                        "symbol": long_symbol,
                        "ratio_qty": "1",
                        "side": "buy",
                        "position_intent": "buy_to_open",
                    },
                ],
            }

            resp = req.post(url, headers=headers, json=order_data, timeout=15)
            if resp.status_code in (200, 201):
                data = resp.json()
                order_id = data.get("id", "")
                log.info(f"CREDIT SPREAD order: {qty}x sell {short_symbol} / buy {long_symbol} "
                         f"@ ${credit_limit:.2f} credit -> order {order_id}")
                return order_id
            else:
                log.error(f"Credit spread order failed: {resp.status_code} {resp.text[:300]}")
                if "insufficient" in resp.text.lower():
                    return "NO_BUYING_POWER"
                return None
        except Exception as e:
            log.error(f"Credit spread order error: {e}")
            return None

    def close_credit_spread(self, short_symbol: str, long_symbol: str,
                             qty: int, debit_limit: float) -> Optional[str]:
        """
        Close a bull put spread by buying back the short and selling the long.

        debit_limit: max debit per share we're willing to pay to close.
        """
        try:
            self._throttle()
            import requests as req
            import uuid

            url = f"{self.config.BASE_URL}/v2/orders"
            headers = {
                "APCA-API-KEY-ID": self.config.API_KEY,
                "APCA-API-SECRET-KEY": self.config.API_SECRET,
                "accept": "application/json",
                "content-type": "application/json",
            }

            order_data = {
                "order_class": "mleg",
                "type": "limit",
                "limit_price": str(round(abs(debit_limit), 2)),  # positive = debit
                "qty": str(qty),
                "time_in_force": "day",
                "client_order_id": f"{self.config.ORDER_PREFIX}cl_{uuid.uuid4().hex[:10]}",
                "legs": [
                    {
                        "symbol": short_symbol,
                        "ratio_qty": "1",
                        "side": "buy",
                        "position_intent": "buy_to_close",
                    },
                    {
                        "symbol": long_symbol,
                        "ratio_qty": "1",
                        "side": "sell",
                        "position_intent": "sell_to_close",
                    },
                ],
            }

            resp = req.post(url, headers=headers, json=order_data, timeout=15)
            if resp.status_code in (200, 201):
                data = resp.json()
                order_id = data.get("id", "")
                log.info(f"CLOSE SPREAD: {qty}x buy {short_symbol} / sell {long_symbol} "
                         f"@ ${debit_limit:.2f} debit -> order {order_id}")
                return order_id
            else:
                log.error(f"Close spread failed: {resp.status_code} {resp.text[:300]}")
                return None
        except Exception as e:
            log.error(f"Close spread error: {e}")
            return None

    def close_individual_legs(self, short_symbol: str, long_symbol: str,
                               qty: int) -> bool:
        """Fallback: close each leg individually if mleg close fails.
        Uses raw HTTP to include position_intent (SDK MarketOrderRequest
        omits it, causing Alpaca to default to 'open' → reversed positions).
        """
        import requests as req
        import uuid

        url = f"{self.config.BASE_URL}/v2/orders"
        headers = {
            "APCA-API-KEY-ID": self.config.API_KEY,
            "APCA-API-SECRET-KEY": self.config.API_SECRET,
            "accept": "application/json",
            "content-type": "application/json",
        }
        success = True

        try:
            # Buy back the short leg
            self._throttle()
            order_data = {
                "symbol": short_symbol,
                "qty": str(qty),
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "position_intent": "buy_to_close",
                "client_order_id": f"{self.config.ORDER_PREFIX}btc_{uuid.uuid4().hex[:10]}",
            }
            resp = req.post(url, headers=headers, json=order_data, timeout=15)
            if resp.status_code in (200, 201):
                log.info(f"Bought back short: {qty}x {short_symbol}")
            else:
                log.error(f"Failed to buy back short {short_symbol}: {resp.status_code} {resp.text[:300]}")
                success = False
        except Exception as e:
            log.error(f"Failed to buy back short {short_symbol}: {e}")
            success = False

        try:
            # Sell the long leg
            self._throttle()
            order_data = {
                "symbol": long_symbol,
                "qty": str(qty),
                "side": "sell",
                "type": "market",
                "time_in_force": "day",
                "position_intent": "sell_to_close",
                "client_order_id": f"{self.config.ORDER_PREFIX}stc_{uuid.uuid4().hex[:10]}",
            }
            resp = req.post(url, headers=headers, json=order_data, timeout=15)
            if resp.status_code in (200, 201):
                log.info(f"Sold long: {qty}x {long_symbol}")
            else:
                log.error(f"Failed to sell long {long_symbol}: {resp.status_code} {resp.text[:300]}")
                success = False
        except Exception as e:
            log.error(f"Failed to sell long {long_symbol}: {e}")
            success = False

        return success

    # ── Market Status ────────────────────────────────────
    def is_market_open(self) -> bool:
        """Check if stock market is open."""
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

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancelled successfully."""
        try:
            self._throttle()
            self.trading.cancel_order_by_id(order_id)
            log.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            log.warning(f"Cancel order {order_id} failed: {e}")
            return False

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
            }
        except Exception as e:
            log.error(f"Order check failed for {order_id}: {e}")
            return None
