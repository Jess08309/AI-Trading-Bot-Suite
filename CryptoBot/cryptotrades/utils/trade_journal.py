"""
Structured Trade Journal.

Persists one row per closed round trip with enough detail to audit
execution quality after the fact: signal/entry reason, entry/exit prices,
the mid-price at fill (to measure slippage), slippage in bps, and P&L.

Written to both CSV (for quick spreadsheet review) and SQLite (for
querying), matching the "CSV/SQLite" requirement.
"""
from __future__ import annotations
import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

FIELDNAMES = [
    "timestamp", "symbol", "signal", "side",
    "entry_price", "exit_price", "mid_price_at_fill", "slippage_bps",
    "size_usd", "pnl_usd", "pnl_pct", "exit_reason",
]


@dataclass
class TradeJournalEntry:
    symbol: str
    signal: str
    side: str
    entry_price: float
    exit_price: float
    mid_price_at_fill: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    timestamp: Optional[str] = None
    slippage_bps: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.slippage_bps is None:
            self.slippage_bps = compute_slippage_bps(self.mid_price_at_fill, self.entry_price, self.side)


def compute_slippage_bps(mid_price_at_fill: float, fill_price: float, side: str) -> float:
    """Slippage in bps: positive means the fill was worse than mid for that side.

    For a buy, paying more than mid is adverse (positive slippage).
    For a sell, receiving less than mid is adverse (positive slippage).
    """
    if not mid_price_at_fill:
        return 0.0
    normalized = side.strip().lower()
    diff = fill_price - mid_price_at_fill
    if normalized in ("sell", "short"):
        diff = -diff
    return diff / mid_price_at_fill * 10000.0


class TradeJournal:
    def __init__(self, csv_path: str = "data/trade_journal.csv", sqlite_path: str = "data/trade_journal.db"):
        self.csv_path = csv_path
        self.sqlite_path = sqlite_path
        self._init_sqlite()

    def _init_sqlite(self):
        os.makedirs(os.path.dirname(self.sqlite_path) or ".", exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_journal (
                    timestamp TEXT, symbol TEXT, signal TEXT, side TEXT,
                    entry_price REAL, exit_price REAL, mid_price_at_fill REAL,
                    slippage_bps REAL, size_usd REAL, pnl_usd REAL,
                    pnl_pct REAL, exit_reason TEXT
                )
                """
            )
            conn.commit()

    def record(self, entry: TradeJournalEntry) -> None:
        self._write_csv(entry)
        self._write_sqlite(entry)

    def _write_csv(self, entry: TradeJournalEntry) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow({name: getattr(entry, name) for name in FIELDNAMES})

    def _write_sqlite(self, entry: TradeJournalEntry) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                f"INSERT INTO trade_journal ({', '.join(FIELDNAMES)}) "
                f"VALUES ({', '.join(['?'] * len(FIELDNAMES))})",
                tuple(getattr(entry, name) for name in FIELDNAMES),
            )
            conn.commit()
