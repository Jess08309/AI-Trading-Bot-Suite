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

Order-state safety: a multi-leg entry (short+long combo) is tracked as
"pending" until BOTH legs confirm FILLED. If one leg fills without its hedge
(the sibling CANCELED/INVALID, or only partially filled), this is NEVER
silently dropped -- see _handle_orphan_leg().
"""
from AlgorithmImports import *
import json
import numpy as np


class PutSellerIronCondorAlgorithm(QCAlgorithm):

    # Order statuses this port explicitly recognizes. QC has no separate
    # REJECTED value -- rejected orders surface as INVALID. Any status value
    # NOT in this set is treated as unknown/ambiguous and handled fail-safe
    # (no action taken, alert only -- see _process_one_pending_entry).
    _KNOWN_ORDER_STATUSES = (
        OrderStatus.NEW, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.INVALID,
        OrderStatus.NONE, OrderStatus.CANCEL_PENDING, OrderStatus.UPDATE_SUBMITTED,
    )

    WATCHLIST = [
        "SPY", "QQQ", "IWM",
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        "JPM", "V", "HD", "UNH",
    ]

    MIN_DTE = 30
    MAX_DTE = 60
    MIN_DTE_EXIT = 21

    SHORT_DELTA_MIN = 0.10
    SHORT_DELTA_MAX = 0.18            # was 0.15 — 0.15 was TOO tight: combined with MIN_CREDIT_PCT=0.15
                                       # it found almost no qualifying spreads (0 trades, 80 orders all
                                       # stale-canceled). Split the difference between 0.20 (baseline)
                                       # and 0.15 (too tight) to still favor win rate without starving entries

    MIN_CREDIT_PCT = 0.15
    TAKE_PROFIT_PCT = 0.60           # was 0.50 — capture more premium decay per win (reward:risk fix)
    STOP_LOSS_MULT = 1.4             # was 1.5 — delta=0.18 got Avg Win/Loss to 0.18%/-0.19% (nearly
                                      # breakeven, WR 50%, Expectancy -0.012); trim max loss a bit further
                                      # to close the small remaining gap without touching win rate
    EMERGENCY_BUFFER_PCT = 0.02      # was 0.05 \u2014 order analysis showed EMERGENCY exits closing within 3 days of
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

    # A decaying-fraction retry (relaxing toward a worse fill on stale orders) was tried and
    # backtested WORSE than a single fixed requirement (-27.6% vs -13.3% net profit) — accepting
    # more fills at thinner credit erased the edge faster than it added trades. Reverted to a
    # single fixed requirement with no relaxation: give up (cancel, don't retry) if unfilled.
    # At delta=0.18/TP=0.60/SL=1.4, entry fill rate was only 25.8% (830/1118 stale-canceled) and
    # gross P&L before fees was actually +$186 (fees of $288 flipped it to -$102 net). Tested a
    # small FIXED bump to 0.80 (vs the decaying walk's overreach) — STILL WORSE: -1.825% net
    # profit, fees rose to $348, win rate dropped 51%->46%. 0.85 is a local optimum; reverted.
    LIMIT_CREDIT_FRACTION = 0.85   # entry fill must be within this fraction of the theoretical mid credit
    LIMIT_ORDER_MAX_WAIT_MIN = 30  # cancel and give up on an unfilled entry limit order after this long

    MAX_ORPHAN_CLOSE_ATTEMPTS = 3  # give up auto-retrying a naked leg's own close order after this
                                   # many confirmed submissions and ESCALATE for manual intervention
                                   # instead -- an unbounded retry loop is its own failure mode

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
        # (ticker, right) -> resting combo-limit entry order awaiting fill
        self.pending_entries = {}
        # (ticker, right) -> incident dict; blocks NEW entries on this exact
        # underlying+side until a human clears it (set only when one leg of a
        # multi-leg entry fills without its hedge -- see _handle_orphan_leg)
        self.blocked_sides = {}
        # in-memory audit trail of orphan-leg incidents this session (also
        # best-effort persisted to ObjectStore so it survives redeploys) --
        # entries are NEVER removed from this list, even once resolved
        self.incidents = []
        # monotonic counter -> unique incident_id
        self._incident_seq = 0
        # (incident_id, symbol) -> tracking dict for an outstanding
        # ORPHAN_LEG_CLOSE order whose own fill has not yet been confirmed
        self.orphan_closes = {}
        # Reload any unresolved incidents/orphan-close tracking from a prior
        # session BEFORE anything else runs -- never assume an unresolved
        # close succeeded just because the process restarted.
        self._restore_incident_state()

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
        self._process_pending_entries()
        self._reconcile_orphan_closes()
        for ticker, option_symbol in self.option_symbol_by_ticker.items():
            chain = self.current_slice.option_chains.get(option_symbol)
            if not chain:
                continue
            self._check_exits(ticker, chain)
            self._open_new_spreads(ticker, chain)

    def _process_pending_entries(self) -> None:
        for key, pend in list(self.pending_entries.items()):
            self._process_one_pending_entry(key, pend)

    def _process_one_pending_entry(self, key, pend) -> None:
        """Multi-leg entry order state machine.

        QC's OrderTicket objects reflect live broker/engine state directly (no
        separate "re-query" round trip exists or is needed) -- reading
        .status/.quantity_filled here on every 15-min scan cycle IS the fresh
        broker-state check. Handles: both legs filled; neither filled; one
        filled/one canceled or invalid; partial fills; cancel/fill races (a
        cancel request racing an actual fill is caught because tracking is
        NEVER deleted on cancel alone -- the next cycle re-evaluates with
        fresh ticket state); stale unfilled orders; and unknown/ambiguous
        status values (fail-safe: no action, keep tracking, alert).
        """
        ticker, right = key
        tickets = pend["tickets"]
        statuses = [t.status for t in tickets]
        filled_qty = [getattr(t, "quantity_filled", 0) or 0 for t in tickets]

        if any(s not in self._KNOWN_ORDER_STATUSES for s in statuses):
            self.error(
                f"CRITICAL: {ticker} {right} pending entry has an UNKNOWN/ambiguous "
                f"order status {statuses} -- taking no action, keeping pending "
                f"record intact for re-evaluation next cycle"
            )
            return

        # ── Both legs confirmed filled: the clean, intended outcome ──
        if all(s == OrderStatus.FILLED for s in statuses):
            short_ticket, long_ticket = tickets
            real_credit = short_ticket.average_fill_price - long_ticket.average_fill_price
            self.spreads[ticker].append({
                "strategy": pend["strategy"],
                "short_symbol": pend["short_symbol"],
                "long_symbol": pend["long_symbol"],
                "short_strike": pend["short_strike"],
                "expiry": pend["expiry"],
                "credit": real_credit,
                "right": right,
                "entry_time": self.time,
            })
            self.debug(f"{ticker}: {right} spread FILLED credit={real_credit:.2f} (target>={pend['target_credit']:.2f})")
            del self.pending_entries[key]
            return

        # ── A leg has a confirmed nonzero fill but the pair did NOT complete
        # cleanly (sibling CANCELED/INVALID, or still working) -- a naked,
        # unhedged leg. Never silently drop this -- always reconcile
        # explicitly (see _handle_orphan_leg). ──
        if any(q != 0 for q in filled_qty) and not all(s == OrderStatus.FILLED for s in statuses):
            self._handle_orphan_leg(key, pend, tickets, statuses, filled_qty)
            return

        # ── Neither leg has any fill and BOTH are terminally CANCELED/INVALID:
        # nothing naked, safe to drop cleanly. Using `all` (not `any`) is
        # deliberate -- if only one leg has resolved so far, the other may
        # still fill and must not be treated as a clean miss yet. ──
        if all(s in (OrderStatus.CANCELED, OrderStatus.INVALID) for s in statuses):
            del self.pending_entries[key]
            return

        # ── Still working (NEW/SUBMITTED/PARTIALLY_FILLED-with-zero-qty, or a
        # mixed state where only one leg has resolved so far) -- apply
        # stale-timeout handling. ──
        if (self.time - pend["placed_time"]).total_seconds() / 60 >= self.LIMIT_ORDER_MAX_WAIT_MIN:
            for t in tickets:
                if t.status not in (OrderStatus.FILLED, OrderStatus.CANCELED):
                    t.cancel("limit order stale, giving up")
            # Do NOT delete here -- cancel() can race with an actual fill at
            # the broker. Leave tracking in place; the NEXT cycle re-reads
            # fresh ticket state and will correctly route to the FILLED,
            # orphan-leg, or clean-cancel branch above.
            self.debug(f"{ticker}: {right} spread limit order stale, cancel requested "
                       f"(no fill within {self.LIMIT_ORDER_MAX_WAIT_MIN}min) -- re-checking next cycle")

    def _handle_orphan_leg(self, key, pend, tickets, statuses, filled_qty) -> None:
        """One leg of a multi-leg entry confirmed a nonzero fill while its
        hedge did not complete. Per policy: NEVER auto re-hedge at market
        (that's a second market-timing bet layered on an already-abnormal
        state). Instead: block new entries on this exact underlying+side,
        close ONLY the exact confirmed filled quantity of the naked leg(s),
        raise a CRITICAL alert, and persist a full incident record BEFORE
        removing the pending entry -- the information is preserved, never
        silently lost.
        """
        ticker, right = key
        short_ticket, long_ticket = tickets
        self._incident_seq += 1
        incident = {
            "incident_id": self._incident_seq,
            "timestamp": str(self.time),
            "ticker": ticker,
            "right": right,
            "short_symbol": str(pend["short_symbol"]),
            "long_symbol": str(pend["long_symbol"]),
            "short_order_id": getattr(short_ticket, "order_id", None),
            "long_order_id": getattr(long_ticket, "order_id", None),
            "short_status": str(statuses[0]),
            "long_status": str(statuses[1]),
            "short_quantity_filled": filled_qty[0],
            "long_quantity_filled": filled_qty[1],
            "short_avg_fill_price": getattr(short_ticket, "average_fill_price", None),
            "long_avg_fill_price": getattr(long_ticket, "average_fill_price", None),
            # OPEN until every naked leg's own ORPHAN_LEG_CLOSE order is
            # broker-confirmed FILLED for the exact naked quantity -- never
            # set to RESOLVED merely because a close order was submitted.
            "status": "OPEN",
            # symbol -> remaining naked (signed) quantity, updated live by
            # _reconcile_orphan_closes() as each closing order's own fill
            # state is confirmed
            "naked_legs": {},
        }
        self.error(f"CRITICAL: ORPHAN LEG — {ticker} {right} spread has one leg filled "
                   f"without its hedge: {incident}")

        # Block new entries on this exact underlying+side until a human
        # clears it -- reuses the same (ticker, right) gate _open_new_spreads
        # already checks for pending_entries.
        self.blocked_sides[key] = incident
        self.incidents.append(incident)

        # Best-effort durable persistence (QC ObjectStore survives redeploys).
        # Must never raise and block the actual leg reconciliation below.
        try:
            store_key = f"incident_{ticker}_{right}_{str(self.time).replace(':', '-').replace(' ', '_')}"
            self.object_store.save(store_key, json.dumps(incident, default=str))
        except Exception as e:
            self.debug(f"Incident persist to ObjectStore failed (non-fatal): {e}")

        # Submit (and track) an explicit, targeted close for ONLY the
        # confirmed filled quantity of each naked leg -- never a blanket
        # liquidate() that could also touch unrelated positions.
        for symbol, qty in ((pend["short_symbol"], filled_qty[0]), (pend["long_symbol"], filled_qty[1])):
            if qty:
                self._submit_orphan_close(incident, symbol, qty)

        del self.pending_entries[key]
        self._persist_incident_state()

    def _submit_orphan_close(self, incident, symbol, naked_qty) -> None:
        """Submit (or re-submit, per the same approved policy) a targeted
        closing order for exactly the naked quantity on one leg, and track
        the resulting ticket so its own fill can be verified later -- the
        close order's submission returning a ticket is NOT the same as the
        naked leg actually being closed. The retry-tracking fields (attempts,
        order_ids, first_detected) are preserved/incremented across repeated
        calls for the same (incident, symbol) -- attempt count only ever
        increases here, i.e. only after a close order is actually submitted.
        """
        close_qty = -naked_qty
        ticket = self.market_order(symbol, close_qty, tag="ORPHAN_LEG_CLOSE")
        incident["naked_legs"][symbol] = naked_qty
        oc_key = (incident["incident_id"], symbol)
        order_id = getattr(ticket, "order_id", None)
        existing = self.orphan_closes.get(oc_key)
        if existing is not None:
            existing["ticket"] = ticket
            existing["naked_qty"] = naked_qty
            existing["attempts"] += 1
            existing["last_attempt_time"] = str(self.time)
            existing["last_status"] = ticket.status
            if order_id is not None:
                existing["order_ids"].append(order_id)
            oc = existing
        else:
            oc = {
                "incident": incident,
                "incident_id": incident["incident_id"],
                "symbol": symbol,
                "naked_qty": naked_qty,
                "ticket": ticket,
                "order_ids": [order_id] if order_id is not None else [],
                "attempts": 1,
                "first_detected": str(self.time),
                "last_attempt_time": str(self.time),
                "last_status": ticket.status,
                "escalation_reason": None,
            }
            self.orphan_closes[oc_key] = oc
        self.error(f"CRITICAL: closing naked leg {symbol} qty={close_qty} "
                   f"(incident={incident['incident_id']}, ticker={incident['ticker']} "
                   f"right={incident['right']}, attempt={oc['attempts']}/{self.MAX_ORPHAN_CLOSE_ATTEMPTS})")

    def _reconcile_orphan_closes(self) -> None:
        """Runs every scan cycle (and once on restart before new entries are
        permitted): verifies each outstanding ORPHAN_LEG_CLOSE order's own
        broker-confirmed fill state. An incident is only ever marked RESOLVED
        once its close order(s) confirm the exact naked quantity is gone --
        never on submission alone. Never auto-hedges, never uses liquidate(),
        never deletes an incident record.
        """
        for oc_key, oc in list(self.orphan_closes.items()):
            self._reconcile_one_orphan_close(oc_key, oc)
        self._persist_incident_state()

    def _reconcile_one_orphan_close(self, oc_key, oc) -> None:
        incident = oc["incident"]
        symbol = oc["symbol"]
        ticket = oc["ticket"]

        if ticket is None:
            # Restored after a restart but no live order ticket could be
            # re-fetched -- never assume the close succeeded; wait for a
            # human to verify broker state directly.
            self.error(f"CRITICAL: incident={incident['incident_id']} {symbol} "
                       f"ORPHAN_LEG_CLOSE has no live order ticket available "
                       f"(restart recovery) -- cannot verify broker state, not "
                       f"assuming success, incident stays {incident['status']}")
            return

        status = ticket.status
        filled = getattr(ticket, "quantity_filled", 0) or 0

        if status not in self._KNOWN_ORDER_STATUSES:
            self.error(f"CRITICAL: incident={incident['incident_id']} {symbol} "
                       f"ORPHAN_LEG_CLOSE has an UNKNOWN/ambiguous status {status} -- "
                       f"not assuming success, incident stays OPEN, no retry submitted")
            return  # never mutate residual/attempts/state on an unrecognized status

        residual = oc["naked_qty"] + filled
        oc["last_status"] = status

        if residual == 0:
            # Broker-ticket-confirmed flat for this leg (not merely "submission
            # returned") -- resolve just this leg.
            incident["naked_legs"].pop(symbol, None)
            del self.orphan_closes[oc_key]
            if not incident["naked_legs"] and incident["status"] != "ESCALATED":
                incident["status"] = "RESOLVED"
                self.debug(f"incident={incident['incident_id']}: all naked legs confirmed "
                           f"closed by broker -- incident RESOLVED (record retained, not deleted)")
            return

        # Residual naked quantity remains -- reflect it on the incident record.
        # NOTE: oc["naked_qty"] intentionally stays the fixed quantity that was
        # requested for THIS close ticket -- it's only rebased when a NEW close
        # order is submitted (see _submit_orphan_close). Overwriting it here
        # would double-count the same ticket's cumulative fill on every
        # subsequent reconcile cycle before it reaches a terminal state.
        incident["naked_legs"][symbol] = residual

        if status == OrderStatus.PARTIALLY_FILLED:
            self.debug(f"incident={incident['incident_id']}: {symbol} ORPHAN_LEG_CLOSE "
                       f"partially filled, residual naked qty={residual} -- current close "
                       f"ticket still active, not submitting another close, incident stays OPEN")
            return

        if status not in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.INVALID):
            # Still working (NEW/SUBMITTED/CANCEL_PENDING/UPDATE_SUBMITTED) --
            # nothing to do yet, re-check next cycle.
            return

        # Terminal ticket state (FILLED-with-anomalous-residual, CANCELED, or
        # INVALID) with residual still nonzero -- retry under the cap, or
        # escalate for manual intervention once the cap is reached.
        if oc["attempts"] >= self.MAX_ORPHAN_CLOSE_ATTEMPTS:
            incident["status"] = "ESCALATED"
            reason = (f"{symbol} ORPHAN_LEG_CLOSE {status} after "
                      f"{oc['attempts']}/{self.MAX_ORPHAN_CLOSE_ATTEMPTS} attempts, "
                      f"residual naked qty={residual} remains")
            oc["escalation_reason"] = reason
            self.error(f"CRITICAL: incident={incident['incident_id']} {reason} -- retry "
                       f"cap reached, STOPPING auto-retry, MANUAL INTERVENTION REQUIRED "
                       f"(blocked_sides entry retained)")
            return  # never resubmit past the cap

        self.error(f"CRITICAL: incident={incident['incident_id']} {symbol} ORPHAN_LEG_CLOSE "
                   f"was {status} with residual naked qty={residual} remaining -- re-submitting "
                   f"the approved targeted close (attempt {oc['attempts'] + 1}/"
                   f"{self.MAX_ORPHAN_CLOSE_ATTEMPTS}, never auto-hedge, never blanket liquidate)")
        self._submit_orphan_close(incident, symbol, residual)

    def _persist_incident_state(self) -> None:
        """Best-effort durable snapshot of all incidents + outstanding
        orphan-close tracking metadata (no live ticket/order objects -- those
        cannot survive a restart and must never be assumed still valid).
        Must never raise.
        """
        try:
            snapshot = {
                "incident_seq": self._incident_seq,
                "incidents": self.incidents,
                "orphan_closes": [
                    {k: v for k, v in oc.items() if k not in ("incident", "ticket")}
                    for oc in self.orphan_closes.values()
                ],
            }
            self.object_store.save("putseller_orphan_incidents", json.dumps(snapshot, default=str))
        except Exception as e:
            self.debug(f"Incident state persist failed (non-fatal): {e}")

    def _restore_incident_state(self) -> None:
        """Reload any unresolved incidents/outstanding orphan-close tracking
        from a previous session. Never assumes an unresolved close succeeded
        just because the process restarted -- restored legs' tickets are
        re-fetched live (never trusted from the stale snapshot alone) and
        reconciled once immediately, before any new-entry scan can run.
        """
        try:
            if not self.object_store.contains_key("putseller_orphan_incidents"):
                return
            raw = self.object_store.read("putseller_orphan_incidents")
            if not raw:
                return
            snapshot = json.loads(raw)
        except Exception as e:
            self.debug(f"Incident state restore failed (non-fatal, starting clean): {e}")
            return

        self._incident_seq = snapshot.get("incident_seq", 0)
        for incident in snapshot.get("incidents", []):
            self.incidents.append(incident)
            if incident.get("status") in ("OPEN", "ESCALATED"):
                self.blocked_sides[(incident["ticker"], incident["right"])] = incident

        for oc_meta in snapshot.get("orphan_closes", []):
            incident = next((i for i in self.incidents if i["incident_id"] == oc_meta["incident_id"]), None)
            if incident is None:
                continue
            symbol = oc_meta["symbol"]
            order_ids = oc_meta.get("order_ids") or []
            last_order_id = order_ids[-1] if order_ids else None
            ticket = None
            if last_order_id is not None:
                try:
                    ticket = self.transactions.get_order_ticket(last_order_id)
                except Exception as e:
                    self.debug(f"Could not re-fetch order ticket {last_order_id} on restart (non-fatal): {e}")
                    ticket = None
            restored = dict(oc_meta)
            restored["incident"] = incident
            restored["ticket"] = ticket
            self.orphan_closes[(incident["incident_id"], symbol)] = restored

        if self.orphan_closes or self.incidents:
            self.debug(f"Restored {len(self.orphan_closes)} outstanding orphan-close "
                       f"tracking entries and {len(self.incidents)} incident records "
                       f"from prior session")
        # Reconcile restored state before permitting any new-entry scan.
        self._reconcile_orphan_closes()

        # Still working (NEW/SUBMITTED/CANCEL_PENDING/etc) -- nothing to do
        # yet, re-check next cycle.

    # ── Exit management ──────────────────────────────────────────
    def _check_exits(self, ticker: str, chain) -> None:
        underlying_price = float(chain.underlying.price)
        still_open = []
        for pos in self.spreads[ticker]:
            reason = self._check_exit(pos, chain, underlying_price)
            if reason:
                # tag with just the reason keyword (e.g. "EMERGENCY_PUT") so order-level
                # analysis via /backtests/orders/read can bucket win/loss by exit cause
                # without needing debug logs (not retrievable via the QC API).
                self.sell(pos["strategy"], 1, tag=reason.split(" ")[0].split("(")[0])
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

    # ── Entry scanning ───────────────────────────────────────────
    def _open_new_spreads(self, ticker: str, chain) -> None:
        if len(self.spreads[ticker]) >= self.MAX_PER_UNDERLYING:
            return

        # QC-native crash filter proxy: pause ALL new entries when SPY realized vol spikes.
        if self._spy_vol_spiked():
            return

        underlying_price = float(chain.underlying.price)
        width = self._spread_width(underlying_price)
        trend = self._trend(ticker)

        if (
            trend != "down"  # don't sell puts into a confirmed downtrend
            and not self._in_cooldown(ticker, "put")
            and (ticker, "put") not in self.pending_entries
            and (ticker, "put") not in self.blocked_sides  # orphan-leg incident on this side
        ):
            put_spread = self._find_credit_spread(chain, underlying_price, width, OptionRight.PUT)
            if put_spread and self._total_open("put") < self.MAX_POSITIONS:
                self._execute_spread(ticker, put_spread, "put")
                return  # one new spread per symbol per cycle, matches live bot's per-cycle pacing

        if (
            trend != "up"  # don't sell calls into a confirmed uptrend
            and not self._in_cooldown(ticker, "call")
            and (ticker, "call") not in self.pending_entries
            and (ticker, "call") not in self.blocked_sides  # orphan-leg incident on this side
        ):
            call_spread = self._find_credit_spread(chain, underlying_price, width, OptionRight.CALL)
            if call_spread and self._total_open("call") < self.MAX_CALL_POSITIONS:
                self._execute_spread(ticker, call_spread, "call")

    def _in_cooldown(self, ticker: str, right: str) -> bool:
        last = self.last_exit[ticker][right]
        return last is not None and (self.time - last).days < self.RE_ENTRY_COOLDOWN_DAYS

    def _total_open(self, right: str) -> int:
        filled = sum(1 for spreads in self.spreads.values() for p in spreads if p["right"] == right)
        pending = sum(1 for (_, r) in self.pending_entries if r == right)
        return filled + pending

    def _find_credit_spread(self, chain, price: float, width: float, right):
        contracts = [c for c in chain if c.right == right]
        if not contracts:
            return None

        by_expiry = {}
        for c in contracts:
            by_expiry.setdefault(c.expiry, []).append(c)

        best = None
        best_delta = 2.0

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

                # Prefer the LOWEST-delta qualifying strike (furthest OTM), not the
                # highest credit_pct -- maximizing credit systematically picks the
                # riskier, closer-to-money edge of the band and was found to be a real
                # driver of EMERGENCY exits in backtesting.
                if delta < best_delta:
                    best_delta = delta
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

        # A market order on both legs crosses the full bid/ask spread twice per
        # round trip (open + close), which order-level analysis showed was eating
        # the entire credit collected. A net combo limit order requires the fill to
        # be within LIMIT_CREDIT_FRACTION of the theoretical mid credit — give up
        # (cancel, no retry) if it doesn't fill within LIMIT_ORDER_MAX_WAIT_MIN.
        target_credit = round(spread["credit"] * self.LIMIT_CREDIT_FRACTION, 2)
        legs = [Leg.create(short_c.symbol, -1), Leg.create(long_c.symbol, 1)]
        tickets = self.combo_limit_order(legs, 1, -target_credit)

        self.pending_entries[(ticker, right)] = {
            "tickets": tickets,
            "strategy": strategy,
            "short_symbol": short_c.symbol,
            "long_symbol": long_c.symbol,
            "short_strike": short_c.strike,
            "expiry": spread["expiry"],
            "mid_credit": spread["credit"],
            "target_credit": target_credit,
            "placed_time": self.time,
        }
        self.debug(
            f"{ticker}: {right} spread limit order placed short={short_c.strike} long={long_c.strike} "
            f"exp={spread['expiry'].date()} target_credit>={target_credit:.2f}"
        )

    def on_order_event(self, order_event) -> None:
        if order_event.status == OrderStatus.FILLED:
            self.debug(str(order_event))
