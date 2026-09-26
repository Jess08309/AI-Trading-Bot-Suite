#!/usr/bin/env python3
"""tools/portfolio_report.py — Workstream F: fleet-wide P&L + governance report.

Unified reporting across all 4 bots (AlpacaBot, CallBuyer, PutSeller,
CryptoBot):

  * Per-bot performance stats computed from each bot's local trade journal
    (data/trades.csv) — trades, win rate, total P&L, profit factor,
    avg win/loss, max single loss, max drawdown.
  * "The Rule" scorecard (see docs/GO_LIVE_CRITERIA.md): per-bot pass/fail
    against the fleet's official go-live/stay-live bar, printed FIRST.
  * Broker-vs-bot reconciliation: compares each bot's locally-tracked open
    positions (data/state/positions.json) against the live Alpaca account
    (via the Alpaca API) to catch PHANTOM (bot thinks it holds a position
    the broker does not have) and ORPHAN (broker holds a position no bot
    claims) conditions. Any phantom is surfaced in a CRITICAL section at
    the very top of the output — this is the standing regression alarm
    for the NFLX phantom-position bug class.

Data sources are read-only; this tool never places orders or modifies any
bot's trade/position state files.

Alpaca credentials are read ONLY from the environment (ALPACA_API_KEY /
ALPACA_API_SECRET), matching the convention already used across the repo
(e.g. PutSeller/tools/_check_positions.py, CryptoBot/cryptotrades/core/
trading_engine.py). Nothing sensitive is ever printed. If the alpaca-py
package is not installed or credentials are absent, the reconciliation
section is skipped with a clear note (all other sections still run).

Usage:
    python3 tools/portfolio_report.py [--bots ALPACABOT,PUTSELLER,...] [--json]
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


def _extract_alpacabot_symbols(positions: Dict[str, Any]) -> Set[str]:
    out = set()
    for keyed_symbol, pos in positions.items():
        if keyed_symbol:
            out.add(keyed_symbol)
        if isinstance(pos, dict):
            primary_symbol = pos.get("symbol")
            short_leg_symbol = pos.get("short_leg_symbol")
            if primary_symbol:
                out.add(primary_symbol)
            if short_leg_symbol:
                out.add(short_leg_symbol)
    return out


def _extract_callbuyer_symbols(positions: Dict[str, Any]) -> Set[str]:
    out = set()
    for sym, pos in positions.items():
        out.add(pos.get("contract") or sym)
    return {s for s in out if s}


def _extract_putseller_symbols(positions: Dict[str, Any]) -> Set[str]:
    out = set()
    for pos in positions.values():
        for leg in ("short_symbol", "long_symbol"):
            s = pos.get(leg)
            if s:
                out.add(s)
    return out


def _extract_cryptobot_symbols(positions: Dict[str, Any]) -> Set[str]:
    out = set()
    for sym, pos in positions.items():
        raw = pos.get("symbol") or sym
        # Alpaca crypto symbols are unslashed (e.g. "BTCUSD"); local state
        # uses the "BTC/USD" convention — normalize for comparison.
        out.add(raw.replace("/", ""))
    return out


@dataclass
class BotSpec:
    name: str
    root: str  # relative to repo root
    trades_csv: str  # relative to `root`
    positions_json: str  # relative to `root`
    pnl_field: str  # column name in trades_csv holding realized $ P&L
    time_field: str  # column name holding the trade timestamp
    symbol_extractor: Any  # Dict[str, Any] -> Set[str]

    @property
    def trades_path(self) -> str:
        return os.path.join(BASE_DIR, self.root, self.trades_csv)

    @property
    def positions_path(self) -> str:
        return os.path.join(BASE_DIR, self.root, self.positions_json)


BOT_SPECS: List[BotSpec] = [
    BotSpec("AlpacaBot", "AlpacaBot", "data/trades.csv", "data/state/positions.json",
            "pnl", "timestamp", _extract_alpacabot_symbols),
    BotSpec("CallBuyer", "CallBuyer", "data/trades.csv", "data/state/positions.json",
            "pnl_dollar", "timestamp", _extract_callbuyer_symbols),
    BotSpec("PutSeller", "PutSeller", "data/trades.csv", "data/state/positions.json",
            "pnl", "timestamp", _extract_putseller_symbols),
    BotSpec("CryptoBot", "CryptoBot", "data/trades.csv", "data/state/positions.json",
            "pnl_usd", "timestamp", _extract_cryptobot_symbols),
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
    # percentage of total account equity — refine with real per-bot
    # allocation data before treating this as an exact "% of capital" figure.
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


def _load_positions(spec: BotSpec) -> Tuple[Dict[str, Any], Set[str], Optional[str]]:
    path = spec.positions_path
    if not os.path.exists(path):
        return {}, set(), f"positions file not found at {os.path.relpath(path, BASE_DIR)}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {}, set(), f"failed to read positions file: {e}"
    if not isinstance(data, dict):
        return {}, set(), "positions file did not contain a JSON object"
    return data, spec.symbol_extractor(data), None


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


def _has_nonzero_qty(position: Dict[str, Any]) -> bool:
    qty = position.get("qty")
    try:
        return float(qty) != 0.0
    except (TypeError, ValueError):
        return qty not in (None, "", 0, "0")


def _broker_symbol_set(broker_positions: List[Dict[str, Any]]) -> Set[str]:
    return {
        p["symbol"]
        for p in broker_positions
        if p.get("symbol") and _has_nonzero_qty(p)
    }


def _phantom_alert_for_legs(bot: str,
                            position_label: str,
                            expected_legs: List[str],
                            missing_legs: List[str]) -> str:
    if len(expected_legs) == 1:
        return (
            f"PHANTOM ALERT (bot {bot}): {position_label} tracked locally but NOT found at broker"
        )

    if len(missing_legs) == 1:
        return (
            f"PHANTOM ALERT (bot {bot}): {position_label} spread missing broker leg "
            f"{missing_legs[0]} (other leg still present)"
        )

    return (
        f"PHANTOM ALERT (bot {bot}): {position_label} spread tracked locally but "
        "neither leg was found at broker"
    )


def _reconcile_position(bot: str,
                        position_label: str,
                        expected_legs: List[str],
                        broker_symbols: Set[str]) -> Optional[str]:
    normalized_legs = [leg for leg in expected_legs if leg]
    missing_legs = [leg for leg in normalized_legs if leg not in broker_symbols]
    if not missing_legs:
        return None
    return _phantom_alert_for_legs(bot, position_label, normalized_legs, missing_legs)


def _reconcile_local_positions(spec: BotSpec,
                               positions: Dict[str, Any],
                               broker_symbols: Set[str]) -> Tuple[Set[str], List[str]]:
    claimed_symbols = spec.symbol_extractor(positions)
    phantoms: List[str] = []

    if spec.name == "PutSeller":
        for spread_id, pos in positions.items():
            alert = _reconcile_position(
                spec.name,
                spread_id,
                [pos.get("short_symbol", ""), pos.get("long_symbol", "")],
                broker_symbols,
            )
            if alert:
                phantoms.append(alert)
        return claimed_symbols, phantoms

    if spec.name == "AlpacaBot":
        for keyed_symbol, pos in positions.items():
            if not isinstance(pos, dict):
                alert = _reconcile_position(spec.name, keyed_symbol, [keyed_symbol], broker_symbols)
                if alert:
                    phantoms.append(alert)
                continue

            primary_symbol = pos.get("symbol") or keyed_symbol
            alert = _reconcile_position(
                spec.name,
                primary_symbol,
                [primary_symbol, pos.get("short_leg_symbol", "")],
                broker_symbols,
            )
            if alert:
                phantoms.append(alert)
        return claimed_symbols, phantoms

    for sym in sorted(claimed_symbols):
        alert = _reconcile_position(spec.name, sym, [sym], broker_symbols)
        if alert:
            phantoms.append(alert)
    return claimed_symbols, phantoms


def _reconcile(bot_positions: Dict[str, Dict[str, Any]],
               broker_positions: Optional[List[Dict[str, Any]]]
               ) -> Tuple[List[str], List[str]]:
    """Return (phantom_alerts, orphan_alerts)."""
    phantoms: List[str] = []
    orphans: List[str] = []
    if broker_positions is None:
        return phantoms, orphans

    broker_symbols = _broker_symbol_set(broker_positions)
    claimed_by_any: Set[str] = set()
    for spec in BOT_SPECS:
        positions = bot_positions.get(spec.name)
        if positions is None:
            continue
        claimed_symbols, bot_phantoms = _reconcile_local_positions(spec, positions, broker_symbols)
        claimed_by_any |= claimed_symbols
        phantoms.extend(bot_phantoms)

    for sym in sorted(broker_symbols - claimed_by_any):
        orphans.append(f"ORPHAN ALERT: broker position {sym} is not claimed by any bot")

    return phantoms, orphans


def _fmt_money(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def build_report(bot_names: Optional[List[str]] = None) -> Dict[str, Any]:
    specs = [s for s in BOT_SPECS if not bot_names or s.name.upper() in bot_names]

    metrics_by_bot: Dict[str, BotMetrics] = {}
    verdicts_by_bot: Dict[str, Tuple[List[Tuple[str, bool, str]], bool]] = {}
    positions_notes: Dict[str, Optional[str]] = {}
    bot_positions: Dict[str, Dict[str, Any]] = {}

    for spec in specs:
        m = _compute_metrics(spec)
        metrics_by_bot[spec.name] = m
        verdicts_by_bot[spec.name] = _rule_verdict(m)
        data, _, note = _load_positions(spec)
        bot_positions[spec.name] = data
        positions_notes[spec.name] = note

    broker_positions, broker_note = _fetch_broker_positions()
    phantoms, orphans = _reconcile(bot_positions, broker_positions)

    return {
        "specs": specs,
        "metrics_by_bot": metrics_by_bot,
        "verdicts_by_bot": verdicts_by_bot,
        "positions_notes": positions_notes,
        "broker_positions": broker_positions,
        "broker_note": broker_note,
        "phantoms": phantoms,
        "orphans": orphans,
    }


def print_report(report: Dict[str, Any]) -> None:
    phantoms = report["phantoms"]
    orphans = report["orphans"]

    # --- CRITICAL section (phantoms) always printed first, if any ---
    if phantoms:
        print("=" * 70)
        print("CRITICAL: PHANTOM POSITIONS DETECTED")
        print("=" * 70)
        for line in phantoms:
            print(f"  !! {line}")
        print()

    # --- THE RULE SCORECARD (headline output) ---
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

    # --- Per-bot detailed performance ---
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

    # --- Reconciliation ---
    print("\n" + "=" * 70)
    print("RECONCILIATION (bot state vs. Alpaca broker)")
    print("=" * 70)
    if report["broker_positions"] is None:
        print(f"  SKIPPED — {report['broker_note']}")
    else:
        print(f"  Broker positions fetched: {len(report['broker_positions'])}")
        if not phantoms:
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
    parser.add_argument("--bots", help="Comma-separated subset of bot names, e.g. ALPACABOT,PUTSELLER")
    parser.add_argument("--json", action="store_true", help="Also print machine-readable JSON summary")
    args = parser.parse_args()

    bot_names = None
    if args.bots:
        bot_names = [b.strip().upper() for b in args.bots.split(",") if b.strip()]

    report = build_report(bot_names)
    print_report(report)

    if args.json:
        summary = {
            "phantoms": report["phantoms"],
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

    return 1 if report["phantoms"] else 0


if __name__ == "__main__":
    sys.exit(main())
