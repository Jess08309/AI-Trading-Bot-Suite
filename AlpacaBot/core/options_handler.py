"""
Options Chain Handler
Selects optimal contracts based on DTE, liquidity, strike distance, and signals.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

from core.config import Config
from core.api_client import AlpacaAPI

log = logging.getLogger("alpacabot.options")


class OptionsHandler:
    """Finds and evaluates options contracts for a given underlying."""

    def __init__(self, config: Config, api: AlpacaAPI):
        self.config = config
        self.api = api

    def find_best_contract(self, underlying: str, direction: str,
                           current_price: float) -> Optional[Dict[str, Any]]:
        """
        Find the best options contract for a trade.

        Args:
            underlying: 'SPY', 'QQQ', etc.
            direction: 'bullish' → calls, 'bearish' → puts
            current_price: current underlying price

        Returns:
            Best contract dict with quote data, or None.
        """
        option_type = "call" if direction == "bullish" else "put"

        # Per-symbol DTE window
        target_dte = self.config.get_target_dte(underlying)
        now = datetime.now()
        # NEVER 0DTE: minimum 1 day out
        min_dte = max(1, self.config.MIN_DTE)
        # Window: from min_dte to target_dte + small buffer (or MAX_DTE)
        max_dte = min(target_dte + 3, self.config.MAX_DTE)
        exp_after = (now + timedelta(days=min_dte)).strftime("%Y-%m-%d")
        exp_before = (now + timedelta(days=max_dte)).strftime("%Y-%m-%d")

        # Strike range — ITM preferred (3% ITM target, up to 1% OTM)
        itm_target = current_price * self.config.TARGET_ITM_PCT
        otm_buffer = current_price * self.config.MAX_OTM_PCT
        if option_type == "call":
            strike_gte = current_price - (itm_target * 2)    # up to ~6% ITM
            strike_lte = current_price + otm_buffer           # up to 1% OTM
        else:
            strike_gte = current_price - otm_buffer           # up to 1% OTM
            strike_lte = current_price + (itm_target * 2)    # up to ~6% ITM

        # Fetch chain
        contracts = self.api.get_options_chain(
            underlying=underlying,
            expiration_after=exp_after,
            expiration_before=exp_before,
            option_type=option_type,
            strike_price_gte=strike_gte,
            strike_price_lte=strike_lte,
        )

        if not contracts:
            # Fallback: retry with wider ITM range before giving up
            wide_range = itm_target * 3
            if option_type == "call":
                w_gte = current_price - wide_range         # up to ~9% ITM
                w_lte = current_price + (otm_buffer * 2)  # up to 2% OTM
            else:
                w_gte = current_price - (otm_buffer * 2)  # up to 2% OTM
                w_lte = current_price + wide_range         # up to ~9% ITM
            log.info(f"{underlying}: retrying with wider ITM strike range "
                     f"(${w_gte:.0f}-${w_lte:.0f})")
            contracts = self.api.get_options_chain(
                underlying=underlying,
                expiration_after=exp_after,
                expiration_before=exp_before,
                option_type=option_type,
                strike_price_gte=w_gte,
                strike_price_lte=w_lte,
            )
        if not contracts:
            log.warning(f"No {option_type} contracts found for {underlying} "
                        f"(${strike_gte:.0f}-${strike_lte:.0f}, {exp_after} to {exp_before})")
            return None

        # Score and rank contracts
        scored = []
        for contract in contracts:
            score = self._score_contract(contract, current_price, option_type, underlying)
            if score is not None:
                scored.append((score, contract))

        if not scored:
            log.warning(f"No contracts passed filters for {underlying} {option_type}")
            return None

        # Best score wins
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]

        # Get live quote for the winner
        quote = self.api.get_option_quote(best["symbol"])
        if quote:
            best.update(quote)

        log.info(f"Selected: {best['symbol']} | {best['type'].upper()} "
                 f"${best['strike']} exp {best['expiration']} "
                 f"| bid ${best.get('bid', '?')} ask ${best.get('ask', '?')}")

        return best

    def find_spread_contracts(self, underlying: str, direction: str,
                              current_price: float) -> Optional[Dict[str, Any]]:
        """Find contracts for a vertical spread (Level 3).

        Bull call spread: buy lower strike call + sell higher strike call.
        Bear put spread: buy higher strike put + sell lower strike put.

        Returns dict with long_leg, short_leg, net_debit, spread info, or None.
        """
        option_type = "call" if direction == "call" else "put"

        # Spread width based on stock price
        if current_price < 100:
            spread_width = 2.5
        elif current_price < 300:
            spread_width = 5.0
        else:
            spread_width = 10.0

        target_dte = self.config.get_target_dte(underlying)
        now = datetime.now()
        min_dte = max(1, self.config.MIN_DTE)
        max_dte = min(target_dte + 3, self.config.MAX_DTE)
        exp_after = (now + timedelta(days=min_dte)).strftime("%Y-%m-%d")
        exp_before = (now + timedelta(days=max_dte)).strftime("%Y-%m-%d")

        # Wider strike range to find both legs
        otm_range = current_price * 0.06
        if option_type == "call":
            strike_gte = current_price - (otm_range * 0.3)
            strike_lte = current_price + otm_range
        else:
            strike_gte = current_price - otm_range
            strike_lte = current_price + (otm_range * 0.3)

        contracts = self.api.get_options_chain(
            underlying=underlying,
            expiration_after=exp_after,
            expiration_before=exp_before,
            option_type=option_type,
            strike_price_gte=strike_gte,
            strike_price_lte=strike_lte,
        )

        if not contracts or len(contracts) < 2:
            log.debug(f"Not enough contracts for {underlying} {direction} spread")
            return None

        # Group by expiration, find pairs with desired width
        by_exp: Dict[str, List] = {}
        for c in contracts:
            if c.get("open_interest", 0) >= self.config.MIN_OPEN_INTEREST:
                exp = c["expiration"]
                if exp not in by_exp:
                    by_exp[exp] = []
                by_exp[exp].append(c)

        best_spread = None
        best_score = -1.0

        for exp, exp_contracts in by_exp.items():
            exp_contracts.sort(key=lambda c: c["strike"])

            for i, c1 in enumerate(exp_contracts):
                for c2 in exp_contracts[i + 1:]:
                    width = abs(c2["strike"] - c1["strike"])
                    # Width must be close to target
                    if abs(width - spread_width) > spread_width * 0.6:
                        continue

                    if direction == "call":
                        long_leg = c1   # lower strike (buy)
                        short_leg = c2  # higher strike (sell)
                    else:
                        long_leg = c2   # higher strike (buy)
                        short_leg = c1  # lower strike (sell)

                    # Score: prefer ATM long leg, good OI on both
                    long_dist = abs(long_leg["strike"] - current_price) / current_price
                    oi_both = min(long_leg.get("open_interest", 0), 500) + \
                              min(short_leg.get("open_interest", 0), 500)

                    try:
                        exp_date = datetime.strptime(exp, "%Y-%m-%d")
                        dte = (exp_date - now).days
                        dte_score = max(0, 30 - abs(dte - target_dte) * 2)
                    except ValueError:
                        dte_score = 10

                    score = dte_score + max(0, 30 - long_dist * 600) + oi_both / 50

                    if score > best_score:
                        best_score = score
                        best_spread = {
                            "long_leg": long_leg,
                            "short_leg": short_leg,
                            "spread_width": width,
                            "expiration": exp,
                        }

        if not best_spread:
            log.debug(f"No valid spread for {underlying} {direction}")
            return None

        # Get quotes for both legs
        long_quote = self.api.get_option_quote(best_spread["long_leg"]["symbol"])
        short_quote = self.api.get_option_quote(best_spread["short_leg"]["symbol"])

        if not long_quote or not short_quote:
            log.debug(f"Spread quote fetch failed for {underlying}")
            return None

        best_spread["long_leg"].update(long_quote)
        best_spread["short_leg"].update(short_quote)

        long_ask = long_quote.get("ask", 0)
        short_bid = short_quote.get("bid", 0)

        if long_ask <= 0 or short_bid <= 0:
            log.debug(f"Invalid spread quotes: ask={long_ask}, bid={short_bid}")
            return None

        net_debit = long_ask - short_bid
        if net_debit <= 0:
            log.debug(f"Credit spread detected for {underlying} -- skipping")
            return None

        max_profit_per = (best_spread["spread_width"] - net_debit) * 100
        max_loss_per = net_debit * 100

        if max_profit_per <= 0:
            log.debug(f"No profit potential in {underlying} spread")
            return None

        rr = max_profit_per / max_loss_per if max_loss_per > 0 else 0

        best_spread["net_debit"] = net_debit
        best_spread["max_profit_per_contract"] = max_profit_per
        best_spread["max_loss_per_contract"] = max_loss_per
        best_spread["reward_risk"] = rr

        log.info(f"Spread: {underlying} {direction} "
                 f"${best_spread['long_leg']['strike']}/{best_spread['short_leg']['strike']} "
                 f"exp {best_spread['expiration']} | "
                 f"debit ${net_debit:.2f} | max_profit ${max_profit_per:.0f} | R:R {rr:.1f}")

        return best_spread

    def _score_contract(self, contract: Dict, current_price: float,
                        option_type: str, underlying: str = "") -> Optional[float]:
        """
        Score a contract 0-100. Higher = better choice.
        Returns None if contract fails hard filters.
        """
        # Hard filter: open interest
        if contract.get("open_interest", 0) < self.config.MIN_OPEN_INTEREST:
            return None

        score = 0.0

        # ── DTE Score (0-30 pts) ─────────────────────────
        # Prefer per-symbol TARGET_DTE, penalize too close or too far
        target_dte = self.config.get_target_dte(underlying) if underlying else self.config.TARGET_DTE
        exp_str = contract.get("expiration", "")
        if exp_str:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                dte = (exp_date - datetime.now()).days
                dte_diff = abs(dte - target_dte)
                dte_score = max(0, 30 - dte_diff * 2)
                score += dte_score
            except ValueError:
                score += 10  # neutral if parse fails

        # ── Strike Score (0-35 pts) ──────────────────────
        # ITM preferred: 2-5% ITM = max points, ATM acceptable, OTM penalized
        strike = contract.get("strike", 0)
        if current_price > 0 and strike > 0:
            # Signed moneyness: positive = ITM, negative = OTM
            if option_type == "call":
                itm_pct = (current_price - strike) / current_price
            else:
                itm_pct = (strike - current_price) / current_price

            target_itm = self.config.TARGET_ITM_PCT  # 0.03 = 3%

            if 0.02 <= itm_pct <= 0.05:
                strike_score = 35  # Sweet spot: 2-5% ITM (delta ~0.60-0.80)
            elif 0.01 <= itm_pct < 0.02:
                strike_score = 30  # Slightly ITM — still good
            elif 0.0 <= itm_pct < 0.01:
                strike_score = 25  # ATM — acceptable
            elif itm_pct < 0.0:
                # OTM — penalize, allow up to MAX_OTM_PCT
                strike_score = max(0, 15 - abs(itm_pct) * 500)
            else:
                # Deep ITM (>5%) — lower delta benefit, higher premium
                strike_score = max(10, 30 - (itm_pct - 0.05) * 400)

            score += min(strike_score, 35)

        # ── Liquidity Score (0-25 pts) ───────────────────
        oi = contract.get("open_interest", 0)
        if oi >= 1000:
            score += 25
        elif oi >= 500:
            score += 20
        elif oi >= 200:
            score += 15
        elif oi >= 100:
            score += 10

        # ── Open Interest Score (0-10 pts) ───────────────
        # Higher OI = tighter spreads typically
        oi_score = min(10, oi / 500)
        score += oi_score

        return score

    def check_spread(self, contract: Dict) -> bool:
        """
        Verify bid-ask spread is acceptable.
        Should be called after getting a quote.
        """
        bid = contract.get("bid", 0)
        ask = contract.get("ask", 0)
        if not bid or not ask:
            return False

        mid = (bid + ask) / 2
        if mid <= 0:
            return False

        spread_pct = (ask - bid) / mid
        ok = spread_pct <= self.config.MAX_BID_ASK_SPREAD_PCT

        if not ok:
            log.warning(f"Spread too wide for {contract.get('symbol')}: "
                        f"${bid}/{ask} = {spread_pct:.1%} > {self.config.MAX_BID_ASK_SPREAD_PCT:.0%}")
        return ok

    def calculate_contracts(self, balance: float, premium: float,
                            multiplier: int = 100) -> int:
        """
        How many contracts can we buy within position size limits?
        premium: per-share price (the ask or mid price)
        multiplier: contract size (100 for standard options)
        """
        max_spend = balance * self.config.MAX_POSITION_PCT
        cost_per_contract = premium * multiplier

        if cost_per_contract <= 0:
            return 0

        contracts = int(max_spend / cost_per_contract)

        # Floor: at least 1 if we can afford it
        if contracts == 0 and cost_per_contract <= balance * self.config.MAX_PORTFOLIO_RISK_PCT:
            contracts = 1

        return contracts

    def days_to_expiry(self, contract: Dict) -> int:
        """Calculate DTE for a contract."""
        exp_str = contract.get("expiration", "")
        if not exp_str:
            return 0
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            return max(0, (exp_date - datetime.now()).days)
        except ValueError:
            return 0
