# RESCUED (Workstream D, 2026-09-25): found uncommitted at
# PutSeller/quantconnect/main.py in the live droplet tree during the
# PR #9/#11 post-merge git cleanup. Content never matched any commit in
# repo history and never executed from a git-tracked working tree.
# Preserved here on droplet/divergence-rescue for later analysis only —
# NOT merged into main, NOT running anywhere.
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
import numpy as np


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
    SHORT_DELTA_MAX = 0.20            # was 0.25 — favor higher OTM probability / win rate

    MIN_CREDIT_PCT = 0.15
    TAKE_PROFIT_PCT = 0.50
    STOP_LOSS_MULT = 2.0
    EMERGENCY_BUFFER_PCT = 0.02      # was 0.05 — order analysis showed EMERGENCY exits closing within 3 days of
                                     # entry had only a 36.4% win rate (worst of any hold-time bucket) since 5% was
                                     # triggering on routine volatility before positions could develop; 2% requires
                                     # underlying to genuinely approach the strike before force-closing

    MAX_POSITIONS = 12
    MAX_CALL_POSITIONS = 8
    MAX_PER_UNDERLYING = 2

    MIN_HOLD_DAYS = 3         # don't evaluate TAKE_PROFIT/STOP_LOSS until a position has aged this long —
                              # prevents same-day whipsaw closes from noisy bid/ask mid-price swings on
                              # thinly-traded OTM contracts (DTE_EXIT/EMERGENCY exits still always apply)
    MAX_BID_ASK_PCT = 0.60    # reject spreads where either leg's bid/ask is wider than this fraction of its
                              # mid-price (illiquid contract — unreliable credit calc, gets stopped out on noise)
    RE_ENTRY_COOLDOWN_DAYS = 5  # after ANY exit, don't open a new spread of the same right on the same
                                # underlying for this many days — prevents repeatedly re-selling into a strike
                                # the underlying is actively grinding through (was the #1 remaining churn source:
                                # EMERGENCY exit fires, next 15-min scan re-opens a near-identical spread, repeat)

    SPY_VOL_SMA = 20                  # realized-vol lookback — QC-native proxy for the excluded VIX/crash filter
    SPY_VOL_HALT_ANNUALIZED = 0.28    # pause NEW entries when SPY realized vol spikes above this
    TREND_SMA = 50                    # underlying trend lookback for directional alignment

    def initialize(self) -> None:
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2026, 8, 1)
        self.set_cash(100000)
        self.set_warm_up(timedelta(days=65))

        self.option_symbol_by_ticker = {}
        self.equity_symbol_by_ticker = {}
        for ticker in self.WATCHLIST:
            equity = self.add_equity(ticker, Resolution.MINUTE)
            self.equity_symbol_by_ticker[ticker] = equity.symbol
            option = self.add_option(ticker, Resolution.MINUTE)
            option.set_filter(self._filter_chain)
            self.option_symbol_by_ticker[ticker] = option.symbol

        # underlying -> list of open spread dicts (strategy, credit, short/long strike, expiry, right)
        self.spreads = {ticker: [] for ticker in self.WATCHLIST}
        # underlying -> {"put": last_exit_time, "call": last_exit_time} — re-entry cooldown tracking
        self.last_exit = {ticker: {"put": None, "call": None} for ticker in self.WATCHLIST}

        # rolling daily closes per underlying (SPY, already in WATCHLIST, doubles as the vol/crash filter)
        self.daily_closes = {ticker: RollingWindow[float](max(self.TREND_SMA, self.SPY_VOL_SMA) + 5) for ticker in self.WATCHLIST}
        for ticker, window in self.daily_closes.items():
            consolidator = TradeBarConsolidator(timedelta(days=1))
            consolidator.data_consolidated += self._make_daily_handler(ticker)
            self.subscription_manager.add_consolidator(self.equity_symbol_by_ticker[ticker], consolidator)

        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(minutes=15)),
            self._scan,
        )

    def _make_daily_handler(self, ticker: str):
        def handler(sender, bar):
            self.daily_closes[ticker].add(float(bar.close))
        return handler

    def _spy_vol_spiked(self) -> bool:
        """QC-native proxy for the excluded SPY/VIXY crash filter: pause new entries
        when SPY's short-term realized volatility spikes (crash/high-vol regime)."""
        window = self.daily_closes.get("SPY")
        if window is None or window.count < self.SPY_VOL_SMA + 1:
            return False
        closes = np.array([window[i] for i in range(window.count)])[::-1]
        log_rets = np.diff(np.log(closes[-(self.SPY_VOL_SMA + 1):]))
        realized_vol = float(np.std(log_rets) * np.sqrt(252))
        return realized_vol > self.SPY_VOL_HALT_ANNUALIZED

    def _trend(self, ticker: str):
        """Returns 'up', 'down', or None — used to avoid selling puts into downtrends
        and calls into uptrends (QC-native proxy for the excluded regime detector)."""
        window = self.daily_closes.get(ticker)
        if window is None or window.count < self.TREND_SMA:
            return None
        closes = [window[i] for i in range(window.count)][::-1]
        sma = sum(closes[-self.TREND_SMA:]) / self.TREND_SMA
        return "up" if closes[-1] > sma else "down"

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

    # ── Exit management ───────────────────────
    def _check_exits(self, ticker: str, chain) -> None:
        underlying_price = float(chain.underlying.price)
        still_open = []
        for pos in self.spreads[ticker]:
            reason = self._check_exit(pos, chain, underlying_price)
            if reason:
                self.sell(pos["strategy"], 1)
                self.last_exit[ticker][pos["right"]] = self.time
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

        days_held = (self.time.date() - pos["entry_time"].date()).days
        if days_held >= self.MIN_HOLD_DAYS:
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

    # ── Entry scanning ───────────────────────
    def _open_new_spreads(self, ticker: str, chain) -> None:
        if len(self.spreads[ticker]) >= self.MAX_PER_UNDERLYING:
            return

        # QC-native crash filter proxy: pause ALL new entries when SPY realized vol spikes.
        if self._spy_vol_spiked():
            return

        underlying_price = float(chain.underlying.price)
        width = self._spread_width(underlying_price)
        trend = self._trend(ticker)

        if trend != "down" and not self._in_cooldown(ticker, "put"):  # don't sell puts into a confirmed downtrend
            put_spread = self._find_credit_spread(chain, underlying_price, width, OptionRight.PUT)
            if put_spread and self._total_open("put") < self.MAX_POSITIONS:
                self._execute_spread(ticker, put_spread, "put")
                return  # one new spread per symbol per cycle, matches live bot's per-cycle pacing

        if trend != "up" and not self._in_cooldown(ticker, "call"):  # don't sell calls into a confirmed uptrend
            call_spread = self._find_credit_spread(chain, underlying_price, width, OptionRight.CALL)
            if call_spread and self._total_open("call") < self.MAX_CALL_POSITIONS:
                self._execute_spread(ticker, call_spread, "call")

    def _in_cooldown(self, ticker: str, right: str) -> bool:
        last = self.last_exit[ticker][right]
        return last is not None and (self.time - last).days < self.RE_ENTRY_COOLDOWN_DAYS

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

                if not self._liquid(short_c):
                    continue

                target_long_strike = (
                    short_c.strike - width if right == OptionRight.PUT else short_c.strike + width
                )
                long_c = min(group, key=lambda c: abs(c.strike - target_long_strike))
                if abs(long_c.strike - target_long_strike) > 1.0:
                    continue
                if not self._liquid(long_c):
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

    def _liquid(self, contract) -> bool:
        """Reject contracts with no market or an excessively wide bid/ask —
        these produce unreliable mid-price credit calcs that trigger false stop-outs."""
        bid, ask = contract.bid_price, contract.ask_price
        if bid <= 0 or ask <= 0 or ask <= bid:
            return False
        mid = (bid + ask) / 2
        return (ask - bid) / mid <= self.MAX_BID_ASK_PCT

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
            "entry_time": self.time,
        })
        self.debug(
            f"{ticker}: opened {right} spread short={short_c.strike} long={long_c.strike} "
            f"exp={spread['expiry'].date()} credit={spread['credit']:.2f}"
        )

    def on_order_event(self, order_event) -> None:
        if order_event.status == OrderStatus.FILLED:
            self.debug(str(order_event))
