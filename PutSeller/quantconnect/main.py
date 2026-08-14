"""
PutSeller Iron Condor — QuantConnect/LEAN port.

Ports the MECHANICAL rules from PutSeller/core/config.py + core/put_engine.py onto
QuantConnect's cloud backtesting/live engine, so they can be validated independently
of our own Alpaca-side backtests and (eventually) cross-checked live via QC's Live
Management API as a confirming filter before PutSeller opens a spread.

Ported (faithfully, same thresholds as the live bot):
  - Watchlist, DTE window (30-60d, target 45), short-leg delta band (0.10-0.25)
  - Spread width tiers by underlying price ($5 / $10 / $25)
  - Min credit = 15% of spread width
  - Max positions (12 puts / 8 calls) and max 2 spreads per underlying
  - Exit rules: 50% take-profit, 2x credit stop-loss, 21 DTE exit, 5% emergency buffer

NOT ported (needs external data/services identical to the live bot's own stack —
porting these here would just duplicate them, not add independent value):
  - SPY/VIXY crash filter, regime detection, sentiment score, ML meta-learner,
    universe scanner. This algorithm is intentionally the "rules-only" skeleton
    of PutSeller for a clean, independent backtest comparison.

Run this in QuantConnect's cloud IDE (Algorithm Lab) — compiling/backtesting LEAN
locally is not available in this workspace.
"""
from AlgorithmImports import *


class PutSellerIronCondorAlgorithm(QCAlgorithm):

    WATCHLIST = [
        "SPY", "QQQ", "IWM",
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        "JPM", "V", "HD", "UNH",
    ]

    MIN_DTE = 30
    MAX_DTE = 60
    MIN_DTE_EXIT = 21

    SHORT_DELTA_MIN = 0.10
    SHORT_DELTA_MAX = 0.25

    MIN_CREDIT_PCT = 0.15
    TAKE_PROFIT_PCT = 0.50
    STOP_LOSS_MULT = 2.0
    EMERGENCY_BUFFER_PCT = 0.05

    MAX_POSITIONS = 12
    MAX_CALL_POSITIONS = 8
    MAX_PER_UNDERLYING = 2

    def initialize(self) -> None:
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2026, 8, 1)
        self.set_cash(100000)
        self.set_warm_up(timedelta(days=35))

        self.option_symbol_by_ticker = {}
        for ticker in self.WATCHLIST:
            self.add_equity(ticker, Resolution.MINUTE)
            option = self.add_option(ticker, Resolution.MINUTE)
            option.set_filter(self._filter_chain)
            self.option_symbol_by_ticker[ticker] = option.symbol

        # underlying -> list of open spread dicts (strategy, credit, short/long strike, expiry, right)
        self.spreads = {ticker: [] for ticker in self.WATCHLIST}

        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(minutes=15)),
            self._scan,
        )

    def _filter_chain(self, universe):
        return universe.strikes(-15, 15).expiration(self.MIN_DTE, self.MAX_DTE)

    def _spread_width(self, price: float) -> float:
        if price < 200:
            return 5.0
        elif price < 500:
            return 10.0
        return 25.0

    def _scan(self) -> None:
        if self.is_warming_up:
            return
        for ticker, option_symbol in self.option_symbol_by_ticker.items():
            chain = self.current_slice.option_chains.get(option_symbol)
            if not chain:
                continue
            self._check_exits(ticker, chain)
            self._open_new_spreads(ticker, chain)

    # ── Exit management ──────────────────────────────────────────
    def _check_exits(self, ticker: str, chain) -> None:
        underlying_price = float(chain.underlying.price)
        still_open = []
        for pos in self.spreads[ticker]:
            reason = self._check_exit(pos, chain, underlying_price)
            if reason:
                self.sell(pos["strategy"], 1)
                self.debug(f"{ticker}: closing spread ({pos['right']}) — {reason}")
            else:
                still_open.append(pos)
        self.spreads[ticker] = still_open

    def _check_exit(self, pos: dict, chain, underlying_price: float):
        short_c = self._find_contract(chain, pos["short_symbol"])
        long_c = self._find_contract(chain, pos["long_symbol"])
        if not short_c or not long_c:
            return None

        short_mid = (short_c.bid_price + short_c.ask_price) / 2
        long_mid = (long_c.bid_price + long_c.ask_price) / 2
        current_debit = short_mid - long_mid
        credit = pos["credit"]

        if credit <= 0:
            return None

        profit_pct = (credit - current_debit) / credit
        if profit_pct >= self.TAKE_PROFIT_PCT:
            return f"TAKE_PROFIT ({profit_pct:.0%})"

        if current_debit >= credit * self.STOP_LOSS_MULT:
            return f"STOP_LOSS (debit {current_debit:.2f} >= {self.STOP_LOSS_MULT}x credit {credit:.2f})"

        days_to_exp = (pos["expiry"].date() - self.time.date()).days
        if days_to_exp <= self.MIN_DTE_EXIT:
            return f"DTE_EXIT ({days_to_exp}d remaining)"

        buffer = pos["short_strike"] * self.EMERGENCY_BUFFER_PCT
        if pos["right"] == "call":
            if underlying_price >= pos["short_strike"] - buffer:
                return f"EMERGENCY_CALL (price {underlying_price:.2f} near short strike {pos['short_strike']:.2f})"
        else:
            if underlying_price <= pos["short_strike"] + buffer:
                return f"EMERGENCY_PUT (price {underlying_price:.2f} near short strike {pos['short_strike']:.2f})"

        return None

    def _find_contract(self, chain, symbol):
        for c in chain:
            if c.symbol == symbol:
                return c
        return None

    # ── Entry scanning ───────────────────────────────────────────
    def _open_new_spreads(self, ticker: str, chain) -> None:
        if len(self.spreads[ticker]) >= self.MAX_PER_UNDERLYING:
            return

        underlying_price = float(chain.underlying.price)
        width = self._spread_width(underlying_price)

        put_spread = self._find_credit_spread(chain, underlying_price, width, OptionRight.PUT)
        if put_spread and self._total_open("put") < self.MAX_POSITIONS:
            self._execute_spread(ticker, put_spread, "put")
            return  # one new spread per symbol per cycle, matches live bot's per-cycle pacing

        call_spread = self._find_credit_spread(chain, underlying_price, width, OptionRight.CALL)
        if call_spread and self._total_open("call") < self.MAX_CALL_POSITIONS:
            self._execute_spread(ticker, call_spread, "call")

    def _total_open(self, right: str) -> int:
        return sum(1 for spreads in self.spreads.values() for p in spreads if p["right"] == right)

    def _find_credit_spread(self, chain, price: float, width: float, right):
        contracts = [c for c in chain if c.right == right]
        if not contracts:
            return None

        by_expiry = {}
        for c in contracts:
            by_expiry.setdefault(c.expiry, []).append(c)

        best = None
        best_credit_pct = -1.0

        for expiry, group in by_expiry.items():
            for short_c in group:
                delta = abs(short_c.greeks.delta) if short_c.greeks else 0.0
                if delta < self.SHORT_DELTA_MIN or delta > self.SHORT_DELTA_MAX:
                    continue

                target_long_strike = (
                    short_c.strike - width if right == OptionRight.PUT else short_c.strike + width
                )
                long_c = min(group, key=lambda c: abs(c.strike - target_long_strike))
                if abs(long_c.strike - target_long_strike) > 1.0:
                    continue

                short_mid = (short_c.bid_price + short_c.ask_price) / 2
                long_mid = (long_c.bid_price + long_c.ask_price) / 2
                credit = short_mid - long_mid
                if credit <= 0:
                    continue

                actual_width = abs(short_c.strike - long_c.strike)
                if actual_width <= 0:
                    continue
                credit_pct = credit / actual_width
                if credit_pct < self.MIN_CREDIT_PCT:
                    continue

                if credit_pct > best_credit_pct:
                    best_credit_pct = credit_pct
                    best = {
                        "short": short_c, "long": long_c, "expiry": expiry,
                        "credit": credit, "width": actual_width,
                    }

        return best

    def _execute_spread(self, ticker: str, spread: dict, right: str) -> None:
        canonical = self.option_symbol_by_ticker[ticker]
        short_c, long_c = spread["short"], spread["long"]

        if right == "put":
            strategy = OptionStrategies.bull_put_spread(
                canonical, short_c.strike, long_c.strike, spread["expiry"]
            )
        else:
            strategy = OptionStrategies.bear_call_spread(
                canonical, short_c.strike, long_c.strike, spread["expiry"]
            )

        self.buy(strategy, 1)
        self.spreads[ticker].append({
            "strategy": strategy,
            "short_symbol": short_c.symbol,
            "long_symbol": long_c.symbol,
            "short_strike": short_c.strike,
            "expiry": spread["expiry"],
            "credit": spread["credit"],
            "right": right,
        })
        self.debug(
            f"{ticker}: opened {right} spread short={short_c.strike} long={long_c.strike} "
            f"exp={spread['expiry'].date()} credit={spread['credit']:.2f}"
        )

    def on_order_event(self, order_event) -> None:
        if order_event.status == OrderStatus.FILLED:
            self.debug(str(order_event))
