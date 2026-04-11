"""
CallBuyer API Client — Alpaca wrapper for buying calls.
Isolated from AlpacaBot + PutSeller API clients.
Same account, different allocation, different order tags.
"""
import logging
import time as _time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from core.config import CallBuyerConfig

log = logging.getLogger("callbuyer.api")


class CallBuyerAPI:
    """Alpaca API wrapper for momentum call buying."""

    _MIN_CALL_INTERVAL = 0.65

    def __init__(self, config: CallBuyerConfig):
        self.config = config
        self._trading: Optional[TradingClient] = None
        self._data: Optional[StockHistoricalDataClient] = None
        self._last_call_ts: float = 0.0
        self._rate_lock = threading.Lock()

    def connect(self) -> bool:
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
                     f"CallBuyer allocation ({self.config.ALLOCATION_PCT:.0%}): ${allocation:,.2f}")
            return True
        except Exception as e:
            log.error(f"Alpaca connection failed: {e}")
            return False

    def _throttle(self):
        with self._rate_lock:
            now = _time.monotonic()
            elapsed = now - self._last_call_ts
            if elapsed < self._MIN_CALL_INTERVAL:
                _time.sleep(self._MIN_CALL_INTERVAL - elapsed)
            self._last_call_ts = _time.monotonic()

    def _call_with_retry(self, fn, *args, retries: int = 3, **kwargs):
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
        self._throttle()
        acct = self.trading.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
        }

    # ── Market Data ──────────────────────────────────────
    def get_bars(self, symbol: str, timeframe: TimeFrame = TimeFrame.Day,
                 days: int = 90) -> list:
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
                log.warning(f"get_bars: '{symbol}' not in response keys: "
                            f"{list(bars.keys()) if hasattr(bars, 'keys') else type(bars).__name__}")
                return []
        except Exception as e:
            log.error(f"Bars fetch failed for {symbol}: {e}")
            return []

    def get_intraday_bars(self, symbol: str, minutes: int = 5,
                          days: int = 5) -> list:
        """Get intraday bars for momentum analysis."""
        try:
            self._throttle()
            tf = TimeFrame.Minute if minutes == 1 else TimeFrame(minutes, "Min")
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=datetime.now() - timedelta(days=days),
            )
            bars = self.data.get_stock_bars(req)
            try:
                return bars[symbol]
            except (KeyError, IndexError):
                return []
        except Exception as e:
            # Fall back to daily bars
            log.debug(f"Intraday bars failed for {symbol}, using daily: {e}")
            return self.get_bars(symbol, TimeFrame.Day, days)

    def get_latest_price(self, symbol: str) -> Optional[float]:
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

    def get_spy_price(self) -> Optional[float]:
        """Get SPY price for sector momentum."""
        return self.get_latest_price("SPY")

    def calculate_hv20(self, symbol: str) -> Optional[float]:
        """20-day annualized historical volatility."""
        try:
            bars = self.get_bars(symbol, TimeFrame.Day, days=35)
            if len(bars) < 21:
                return None
            closes = [float(b.close) for b in bars[-21:]]
            returns = np.diff(np.log(closes))
            return float(round(np.std(returns) * np.sqrt(252), 4))
        except Exception as e:
            log.error(f"HV calc failed for {symbol}: {e}")
            return None

    # ── Options Chain ────────────────────────────────────
    def get_options_chain(self, underlying: str,
                          expiration_after: Optional[str] = None,
                          expiration_before: Optional[str] = None,
                          option_type: str = "call",
                          strike_price_gte: Optional[float] = None,
                          strike_price_lte: Optional[float] = None,
                          ) -> List[Dict[str, Any]]:
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
                contract_sym = c.get("symbol", "")
                oi = int(c.get("open_interest", 0) or 0)
                entry = {
                    "symbol": contract_sym,
                    "underlying": c.get("underlying_symbol", ""),
                    "type": c.get("type", ""),
                    "strike": float(c.get("strike_price", 0) or 0),
                    "expiration": c.get("expiration_date", ""),
                    "open_interest": oi,
                    "size": int(c.get("size", 100) or 100),
                    "bid": 0.0,
                    "ask": 0.0,
                }
                contracts.append(entry)

            # Enrich top contracts with live bid/ask quotes
            for entry in contracts[:30]:  # limit API calls
                try:
                    quote = self.get_option_quote(entry["symbol"])
                    if quote:
                        entry["bid"] = quote.get("bid", 0)
                        entry["ask"] = quote.get("ask", 0)
                except Exception:
                    pass

            return contracts
        except Exception as e:
            log.error(f"Options chain failed for {underlying}: {e}")
            return []

    def get_option_quote(self, option_symbol: str) -> Optional[Dict[str, Any]]:
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
                    "mid": round((bid + ask) / 2, 2) if (bid and ask) else 0,
                }
            return None
        except Exception as e:
            log.error(f"Option quote failed for {option_symbol}: {e}")
            return None

    def get_option_snapshot(self, option_symbol: str) -> Optional[Dict[str, Any]]:
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

    # ── Call Order Submission ────────────────────────────
    def buy_call(self, option_symbol: str, qty: int,
                 limit_price: float) -> Optional[str]:
        """Buy to open a call option (debit order)."""
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
                "symbol": option_symbol,
                "qty": str(qty),
                "side": "buy",
                "type": "limit",
                "limit_price": str(round(limit_price, 2)),
                "time_in_force": "day",
                "position_intent": "buy_to_open",
                "client_order_id": f"{self.config.ORDER_PREFIX}{uuid.uuid4().hex[:12]}",
            }
            resp = req.post(url, headers=headers, json=order_data, timeout=15)
            if resp.status_code in (200, 201):
                data = resp.json()
                order_id = data.get("id", "")
                log.info(f"BUY CALL order: {qty}x {option_symbol} @ ${limit_price:.2f} -> {order_id}")
                return order_id
            else:
                log.error(f"Buy call failed: {resp.status_code} {resp.text[:300]}")
                return None
        except Exception as e:
            log.error(f"Buy call error: {e}")
            return None

    def sell_call(self, option_symbol: str, qty: int,
                  limit_price: Optional[float] = None) -> Optional[str]:
        """Sell to close a call option."""
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
                "symbol": option_symbol,
                "qty": str(qty),
                "side": "sell",
                "time_in_force": "day",
                "position_intent": "sell_to_close",
                "client_order_id": f"{self.config.ORDER_PREFIX}cl_{uuid.uuid4().hex[:10]}",
            }
            if limit_price:
                order_data["type"] = "limit"
                order_data["limit_price"] = str(round(limit_price, 2))
            else:
                order_data["type"] = "market"

            resp = req.post(url, headers=headers, json=order_data, timeout=15)
            if resp.status_code in (200, 201):
                data = resp.json()
                order_id = data.get("id", "")
                log.info(f"SELL CALL order: {qty}x {option_symbol} -> {order_id}")
                return order_id
            else:
                log.error(f"Sell call failed: {resp.status_code} {resp.text[:300]}")
                return None
        except Exception as e:
            log.error(f"Sell call error: {e}")
            return None

    def is_market_open(self) -> bool:
        try:
            clock = self.trading.get_clock()
            return clock.is_open
        except Exception:
            return False

    def next_open(self) -> Optional[datetime]:
        try:
            clock = self.trading.get_clock()
            return clock.next_open
        except Exception:
            return None
