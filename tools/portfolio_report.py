#!/usr/bin/env python3
"""tools/portfolio_report.py — CryptoBot P&L + governance report.

As of the 2026-09-27 portfolio consolidation, this repository contains a
single bot (CryptoBot) — this script was previously fleet-wide across 4
bots (AlpacaBot, CallBuyer, PutSeller, CryptoBot) and has been scoped down
to match. Full multi-bot history is on branch
archive/four-bot-portfolio-2026-09-27 if ever needed again.

  * Performance stats computed from CryptoBot's local trade journal
    (data/trades.csv) — trades, win rate, total P&L, profit factor,
    avg win/loss, max single loss, max drawdown.
  * "The Rule" scorecard (see docs/GO_LIVE_CRITERIA.md): pass/fail
    against the go-live/stay-live bar, printed FIRST.
  * Broker-vs-bot reconciliation: compares CryptoBot's locally-tracked
    open positions (cryptotrades/data/state/positions.json) against the
    live Alpaca account (via the Alpaca API) to catch PHANTOM (bot thinks
    it holds a position the broker does not have) and ORPHAN (broker
    holds a position no local record claims) conditions. A broker row
    reporting qty==0 (settlement lag) is treated as absent, not as a
    match and not as an orphan.

Data sources are read-only; this tool never places orders or modifies any
bot's trade/position state files.

Alpaca credentials are read ONLY from the environment (ALPACA_API_KEY /
ALPACA_API_SECRET). Nothing sensitive is ever printed. If the alpaca-py
package is not installed or credentials are absent, the reconciliation
section is skipped with a clear note (all other sections still run).

Usage:
    python3 tools/portfolio_report.py [--json]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# The Rule (see docs/GO_LIVE_CRITERIA.md — kept in sync with that document)
# ---------------------------------------------------------------------------
RULE_MIN_TRADES = 100
RULE_MIN_PROFIT_FACTOR = 1.3
RULE_MIN_WIN_RATE_PCT = 40.0
RULE_MAX_DRAWDOWN_PCT = 10.0


@dataclass
class PositionClaim:
    """One locally-tracked position, expressed as the set of broker-side
    symbols ("legs") that must ALL be present for the position to be
    considered fully matched. CryptoBot's positions are all single-leg.
    """
    position_id: str
    legs: List[str] = field(default_factory=list)


def _extract_cryptobot_claims(positions: Dict[str, Any]) -> List[PositionClaim]:
    claims = []
    for key, pos in positions.items():
        raw = pos.get("symbol") or key
        if not raw:
            continue
        if raw.startswith("PI_"):
            continue  # futures leg disabled per owner Alpaca-only directive; legacy simulated keys
        # Alpaca crypto symbols are unslashed (e.g. "BTCUSD"); local state
        # uses the "BTC/USD" convention — normalize for comparison.
        claims.append(PositionClaim(position_id=key, legs=[raw.replace("/", "")]))
    return claims


@dataclass
class BotSpec:
    name: str
    root: str  # relative to repo root
    trades_csv: str  # relative to `root`
    positions_json: str  # relative to `root`
    pnl_field: str  # column name in trades_csv holding realized $ P&L
    time_field: str  # column name holding the trade timestamp
    claim_extractor: Any  # Dict[str, Any] -> List[PositionClaim]

    @property
    def trades_path(self) -> str:
        return os.path.join(BASE_DIR, self.root, self.trades_csv)

    @property
    def positions_path(self) -> str:
        return os.path.join(BASE_DIR, self.root, self.positions_json)


BOT_SPECS: List[BotSpec] = [
    # NOTE: CryptoBot's package root (cryptotrades/) nests its own data/
    # dir one level deeper than the repo's top-level CryptoBot/ folder --
    # trading_engine.py resolves state paths relative to cryptotrades/.
    BotSpec("CryptoBot", "CryptoBot", "data/trades.csv", "cryptotrades/data/state/positions.json",
            "pnl_usd", "timestamp", _extract_cryptobot_claims),
]


@dataclass
class BotMetrics:
    name: str
    trades_found: bool
    trade_count: int = 0
    win_rate_pct: float = 0.0
    total_pnl: float = 0.0
    profit_factor: Optional[float] = None
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_single_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    error: Optional[str] = None


def _read_trades(spec: BotSpec) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Read closed trades for a bot. Returns (trades, error_message)."""
    path = spec.trades_path
    if not os.path.exists(path):
        return [], f"trade journal not found at {os.path.relpath(path, BASE_DIR)}"

    trades: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_pnl = row.get(spec.pnl_field)
                if raw_pnl in (None, ""):
                    continue
                try:
                    pnl = float(raw_pnl)
                except (TypeError, ValueError):
                    continue
                trades.append({
                    "time": row.get(spec.time_field, ""),
                    "pnl": pnl,
                })
    except Exception as e:  # noqa: BLE001 - surfaced to the report, not raised
        return [], f"failed to read trade journal: {e}"

    trades.sort(key=lambda t: t["time"])
    return trades, None


def _compute_metrics(spec: BotSpec) -> BotMetrics:
    trades, err = _read_trades(spec)
    if err and not trades:
        return BotMetrics(name=spec.name, trades_found=False, error=err)

    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    n = len(trades)
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = (len(wins) / n * 100.0) if n else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else None
    )
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    max_single_loss = min(losses) if losses else 0.0

    # Equity curve drawdown, expressed as % of the running peak. This is a
    # P&L-based curve (peak/trough of cumulative realized P&L), not a
    # percentage of total account equity — refine with real allocation
    # data before treating this as an exact "% of capital" figure.
    equity = 0.0
    peak = 0.0
    max_dd_pct = 0.0
    for t in trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        if peak > 0:
            dd_pct = (peak - equity) / peak * 100.0
            max_dd_pct = max(max_dd_pct, dd_pct)

    return BotMetrics(
        name=spec.name,
        trades_found=True,
        trade_count=n,
        win_rate_pct=win_rate,
        total_pnl=total_pnl,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_single_loss=max_single_loss,
        max_drawdown_pct=max_dd_pct,
        error=err,
    )


def _rule_verdict(m: BotMetrics) -> Tuple[List[Tuple[str, bool, str]], bool]:
    """Return (criteria list, overall_pass)."""
    if not m.trades_found:
        return [], False

    pf = m.profit_factor if m.profit_factor is not None else 0.0
    criteria = [
        ("trades >= 100", m.trade_count >= RULE_MIN_TRADES,
         f"{m.trade_count} closed trades"),
        ("profit factor > 1.3", pf > RULE_MIN_PROFIT_FACTOR,
         f"profit factor = {pf:.2f}" if m.profit_factor is not None else "profit factor = N/A (no losses/wins)"),
        ("win rate > 40%", m.win_rate_pct > RULE_MIN_WIN_RATE_PCT,
         f"win rate = {m.win_rate_pct:.1f}%"),
        ("max drawdown < 10%", m.max_drawdown_pct < RULE_MAX_DRAWDOWN_PCT,
         f"max drawdown = {m.max_drawdown_pct:.1f}%"),
    ]
    overall = all(c[1] for c in criteria)
    return criteria, overall


def _load_positions(spec: BotSpec) -> Tuple[Dict[str, Any], List[PositionClaim], Optional[str]]:
    path = spec.positions_path
    if not os.path.exists(path):
        return {}, [], f"positions file not found at {os.path.relpath(path, BASE_DIR)}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {}, [], f"failed to read positions file: {e}"
    if not isinstance(data, dict):
        return {}, [], "positions file did not contain a JSON object"
    return data, spec.claim_extractor(data), None


def _fetch_broker_positions() -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Fetch live positions from Alpaca. Returns (positions_or_None, note).

    Credentials are read ONLY from ALPACA_API_KEY / ALPACA_API_SECRET
    (never printed). Returns None when reconciliation cannot run (missing
    package or credentials), along with a human-readable reason.
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        return None, "ALPACA_API_KEY / ALPACA_API_SECRET not set in environment"

    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        return None, "alpaca-py package not installed (pip install alpaca-py)"

    paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() != "false"
    try:
        client = TradingClient(api_key, api_secret, paper=paper)
        positions = client.get_all_positions()
    except Exception as e:  # noqa: BLE001
        return None, f"Alpaca API call failed: {e}"

    out = []
    for p in positions:
        out.append({
            "symbol": getattr(p, "symbol", ""),
            "qty": getattr(p, "qty", None),
            "side": str(getattr(p, "side", "")),
            "market_value": getattr(p, "market_value", None),
            "unrealized_pl": getattr(p, "unrealized_pl", None),
        })
    return out, "ok"


def _broker_symbols_present(broker_positions: List[Dict[str, Any]]) -> Set[str]:
    """Symbols the broker reports with a non-zero quantity.

    A qty==0 row can appear transiently during settlement lag (a leg that
    just closed but hasn't dropped off the positions list yet) — treat it
    as absent so it doesn't cause a false ORPHAN alert or mask a real
    PHANTOM condition.
    """
    present = set()
    for p in broker_positions:
        sym = p.get("symbol")
        if not sym:
            continue
        qty = p.get("qty")
        try:
            qty_val = float(qty) if qty is not None else None
        except (TypeError, ValueError):
            qty_val = None
        if qty_val == 0:
            continue
        present.add(sym)
    return present


def _reconcile(bot_claims: Dict[str, List[PositionClaim]],
               broker_positions: Optional[List[Dict[str, Any]]]
               ) -> Tuple[List[str], List[str], List[str]]:
    """Return (phantom_alerts, orphan_alerts, one_leg_phantom_alerts).

    - A claim whose legs are ALL present at the broker is fully matched
      (no alert).
    - A claim whose legs are ALL absent is a full PHANTOM (bot thinks it
      holds a position the broker has no trace of at all).
    """
    phantoms: List[str] = []
    orphans: List[str] = []
    one_leg_phantoms: List[str] = []
    if broker_positions is None:
        return phantoms, orphans, one_leg_phantoms

    broker_symbols = _broker_symbols_present(broker_positions)
    claimed_symbols: Set[str] = set()

    for bot, claims in bot_claims.items():
        for claim in claims:
            legs = claim.legs
            if not legs:
                continue
            claimed_symbols |= set(legs)
            present = [leg for leg in legs if leg in broker_symbols]
            missing = [leg for leg in legs if leg not in broker_symbols]

            if not missing:
                continue  # fully matched, no alert

            if len(legs) == 1 or not present:
                phantoms.append(
                    f"PHANTOM ALERT (bot {bot}): {claim.position_id} "
                    f"(legs: {', '.join(legs)}) tracked locally but NOT found at broker"
                )
            else:
                one_leg_phantoms.append(
                    f"ONE-LEG PHANTOM ALERT (bot {bot}): {claim.position_id} has "
                    f"leg(s) MISSING at broker: {', '.join(missing)} — while "
                    f"{', '.join(present)} is still open (NAKED LEG risk)"
                )

    for sym in sorted(broker_symbols - claimed_symbols):
        orphans.append(f"ORPHAN ALERT: broker position {sym} is not claimed by any bot")

    return phantoms, orphans, one_leg_phantoms


def _fmt_money(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def build_report(bot_names: Optional[List[str]] = None) -> Dict[str, Any]:
    specs = [s for s in BOT_SPECS if not bot_names or s.name.upper() in bot_names]

    metrics_by_bot: Dict[str, BotMetrics] = {}
    verdicts_by_bot: Dict[str, Tuple[List[Tuple[str, bool, str]], bool]] = {}
    positions_notes: Dict[str, Optional[str]] = {}
    bot_claims: Dict[str, List[PositionClaim]] = {}

    for spec in specs:
        m = _compute_metrics(spec)
        metrics_by_bot[spec.name] = m
        verdicts_by_bot[spec.name] = _rule_verdict(m)
        _, claims, note = _load_positions(spec)
        bot_claims[spec.name] = claims
        positions_notes[spec.name] = note

    broker_positions, broker_note = _fetch_broker_positions()
    phantoms, orphans, one_leg_phantoms = _reconcile(bot_claims, broker_positions)

    return {
        "specs": specs,
        "metrics_by_bot": metrics_by_bot,
        "verdicts_by_bot": verdicts_by_bot,
        "positions_notes": positions_notes,
        "broker_positions": broker_positions,
        "broker_note": broker_note,
        "phantoms": phantoms,
        "orphans": orphans,
        "one_leg_phantoms": one_leg_phantoms,
    }


def print_report(report: Dict[str, Any]) -> None:
    phantoms = report["phantoms"]
    orphans = report["orphans"]
    one_leg_phantoms = report["one_leg_phantoms"]

    if phantoms or one_leg_phantoms:
        print("=" * 70)
        print("CRITICAL: PHANTOM POSITIONS DETECTED")
        print("=" * 70)
        for line in one_leg_phantoms:
            print(f"  !! {line}")
        for line in phantoms:
            print(f"  !! {line}")
        print()

    print("=" * 70)
    print("THE RULE SCORECARD")
    print(f"(>= {RULE_MIN_TRADES} closed trades AND profit factor > {RULE_MIN_PROFIT_FACTOR} "
          f"AND win rate > {RULE_MIN_WIN_RATE_PCT:.0f}% AND max drawdown < {RULE_MAX_DRAWDOWN_PCT:.0f}%)")
    print("=" * 70)
    for spec in report["specs"]:
        m = report["metrics_by_bot"][spec.name]
        criteria, overall = report["verdicts_by_bot"][spec.name]
        print(f"\n{spec.name}:")
        if not m.trades_found:
            print(f"  NO DATA — {m.error}")
            continue
        for label, passed, detail in criteria:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {label:<24} ({detail})")
        print(f"  VERDICT: {'LIVE (passes The Rule)' if overall else 'PAUSE — fails The Rule'}")

    print("\n" + "=" * 70)
    print("PER-BOT PERFORMANCE DETAIL")
    print("=" * 70)
    for spec in report["specs"]:
        m = report["metrics_by_bot"][spec.name]
        print(f"\n{spec.name} ({os.path.relpath(spec.trades_path, BASE_DIR)}):")
        if not m.trades_found:
            print(f"  {m.error}")
            continue
        pf_str = f"{m.profit_factor:.2f}" if m.profit_factor not in (None, float("inf")) else str(m.profit_factor)
        print(f"  Trades:          {m.trade_count}")
        print(f"  Win rate:        {m.win_rate_pct:.1f}%")
        print(f"  Total P&L:       {_fmt_money(m.total_pnl)}")
        print(f"  Profit factor:   {pf_str}")
        print(f"  Avg win:         {_fmt_money(m.avg_win)}")
        print(f"  Avg loss:        {_fmt_money(m.avg_loss)}")
        print(f"  Max single loss: {_fmt_money(m.max_single_loss)}")
        print(f"  Max drawdown:    {m.max_drawdown_pct:.1f}% (of cumulative-P&L peak; see script docstring)")

    print("\n" + "=" * 70)
    print("RECONCILIATION (bot state vs. Alpaca broker)")
    print("=" * 70)
    if report["broker_positions"] is None:
        print(f"  SKIPPED — {report['broker_note']}")
    else:
        print(f"  Broker positions fetched: {len(report['broker_positions'])}")
        if not phantoms and not one_leg_phantoms:
            print("  No PHANTOM positions found.")
        if orphans:
            for line in orphans:
                print(f"  !! {line}")
        else:
            print("  No ORPHAN positions found.")

    for bot, note in report["positions_notes"].items():
        if note:
            print(f"  NOTE ({bot}): {note}")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also print machine-readable JSON summary")
    args = parser.parse_args()

    report = build_report(None)
    print_report(report)

    if args.json:
        summary = {
            "phantoms": report["phantoms"],
            "one_leg_phantoms": report["one_leg_phantoms"],
            "orphans": report["orphans"],
            "bots": {
                spec.name: {
                    "trades_found": report["metrics_by_bot"][spec.name].trades_found,
                    "trade_count": report["metrics_by_bot"][spec.name].trade_count,
                    "win_rate_pct": report["metrics_by_bot"][spec.name].win_rate_pct,
                    "total_pnl": report["metrics_by_bot"][spec.name].total_pnl,
                    "profit_factor": report["metrics_by_bot"][spec.name].profit_factor,
                    "max_drawdown_pct": report["metrics_by_bot"][spec.name].max_drawdown_pct,
                    "rule_pass": report["verdicts_by_bot"][spec.name][1],
                }
                for spec in report["specs"]
            },
        }
        print("--- JSON SUMMARY ---")
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
