#!/usr/bin/env python3
"""tools/clear_phantom_positions.py — cleanup for CryptoBot's tracked-position state.

Removes local position state entries that no longer correspond to a real
broker position:

  1. Entries with broker_qty == 0 (or missing) AND no matching non-zero
     position at the broker — these are exactly the phantom positions the
     fill-confirmation fix (fix/cryptobot-fill-confirmation) prevents going
     forward; this tool cleans up any that already exist from before the fix.
  2. PI_*-prefixed keys unconditionally — these are legacy simulated Kraken
     Futures entries left over from before the futures leg was disabled
     (Alpaca-spot-only directive); they were never real broker positions.

Safe by default: runs as a dry-run and only prints what WOULD be removed
unless --apply is passed. Never places orders, never touches the broker,
never restarts any service.

Alpaca credentials are read ONLY from the environment (ALPACA_API_KEY /
ALPACA_API_SECRET), matching the convention used elsewhere in the repo.
Nothing sensitive is ever printed.

Usage:
    python3 tools/clear_phantom_positions.py               # dry-run (default)
    python3 tools/clear_phantom_positions.py --apply        # actually remove entries
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Set

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSITIONS_PATH = os.path.join(
    BASE_DIR, "CryptoBot", "cryptotrades", "data", "state", "positions.json"
)


def _load_positions(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"No positions file found at {path} — nothing to do.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_broker_symbols() -> Optional[Set[str]]:
    """Fetch symbols with a non-zero qty from Alpaca. Returns None if
    reconciliation can't run (missing package or credentials) — in that
    case only the unconditional PI_* purge still applies."""
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ALPACA_API_KEY / ALPACA_API_SECRET not set — skipping broker reconciliation "
              "(only the unconditional PI_* purge will run).")
        return None

    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        print("alpaca-py not installed — skipping broker reconciliation "
              "(only the unconditional PI_* purge will run).")
        return None

    paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() != "false"
    try:
        client = TradingClient(api_key, api_secret, paper=paper)
        positions = client.get_all_positions()
    except Exception as e:  # noqa: BLE001
        print(f"Alpaca API call failed ({e}) — skipping broker reconciliation.")
        return None

    present = set()
    for p in positions:
        sym = getattr(p, "symbol", "")
        qty = getattr(p, "qty", None)
        try:
            qty_val = float(qty) if qty is not None else None
        except (TypeError, ValueError):
            qty_val = None
        if sym and qty_val not in (0, None):
            present.add(sym)
    return present


def find_removals(positions: Dict[str, Any], broker_symbols: Optional[Set[str]]):
    """Return list of (key, reason) tuples for entries to remove."""
    removals = []
    for key, pos in positions.items():
        raw_symbol = pos.get("symbol") or key

        if raw_symbol.startswith("PI_"):
            removals.append((key, f"PI_* legacy simulated futures key (futures leg disabled)"))
            continue

        broker_qty = pos.get("broker_qty", 0) or 0
        try:
            broker_qty = float(broker_qty)
        except (TypeError, ValueError):
            broker_qty = 0.0

        if broker_qty <= 0:
            if broker_symbols is None:
                # Can't confirm against the broker — leave zero-broker_qty
                # spot entries alone unless reconciliation ran.
                continue
            broker_symbol = raw_symbol.replace("/", "")
            if broker_symbol not in broker_symbols:
                removals.append(
                    (key, f"broker_qty={broker_qty} and no matching broker position")
                )
    return removals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                         help="Actually remove the identified entries (default is dry-run).")
    args = parser.parse_args()

    positions = _load_positions(POSITIONS_PATH)
    if not positions:
        return

    broker_symbols = _fetch_broker_symbols()
    removals = find_removals(positions, broker_symbols)

    if not removals:
        print("No phantom or legacy PI_* entries found — nothing to remove.")
        return

    print(f"{'REMOVING' if args.apply else 'WOULD REMOVE'} {len(removals)} entries:")
    for key, reason in removals:
        print(f"  - {key}: {reason}")

    if not args.apply:
        print("\nDry-run only — re-run with --apply to actually remove these entries.")
        return

    for key, _ in removals:
        positions.pop(key, None)

    with open(POSITIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2, default=str)
    print(f"\nRemoved {len(removals)} entries from {POSITIONS_PATH}.")


if __name__ == "__main__":
    sys.exit(main())
