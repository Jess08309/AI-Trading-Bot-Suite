#!/usr/bin/env python3
"""tools/compute_indicator_correlation.py — Workstream E pre-staging (read-only).

Computes all 14 AlpacaBot indicators (via
AlpacaBot.core.indicators.compute_all_indicators, UNMODIFIED) over a
rolling window of SPY 10-minute bars, then writes the pairwise Pearson
correlation matrix + a short cluster summary to
docs/indicator_correlation.md.

This script does not import, patch, or otherwise modify
AlpacaBot/core/indicators.py — it only calls the existing public
function, the same way AlpacaBot/tools/backtest*.py already do (sliding
`chunk = prices[i - LOOKBACK : i + 1]` window per bar).

Data source (in priority order):
  1. --input <csv>            a local CSV with a "close" column (or
                               "timestamp,close"), e.g. previously
                               downloaded via AlpacaBot/tools/download_5min.py
                               / download_historical.py style tooling.
  2. Alpaca market data API   fetched live for SPY, TimeFrame(10, Minute),
                               trailing ~6 months, using
                               ALPACA_API_KEY / ALPACA_API_SECRET from the
                               environment ONLY (never printed), matching
                               the convention used across this repo.

If neither source is available (no local file and no network/credentials
for the Alpaca API — e.g. when running in a sandboxed/offline
environment), the script exits with a clear error rather than
fabricating data.

Usage:
    python3 tools/compute_indicator_correlation.py [--input path/to/spy_10min.csv]
                                                     [--lookback 30]
                                                     [--corr-threshold 0.8]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "AlpacaBot"))

from core.indicators import compute_all_indicators  # noqa: E402

FEATURE_NAMES: List[str] = [
    "rsi", "macd_hist", "bb_position", "stochastic", "atr_normalized",
    "cci", "roc", "williams_r", "volatility_ratio", "zscore",
    "trend_strength", "price_change_1", "price_change_5", "price_change_20",
]
OUTPUT_PATH = os.path.join(BASE_DIR, "docs", "indicator_correlation.md")


def _load_from_csv(path: str) -> np.ndarray:
    import csv
    closes = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = "close" if "close" in (reader.fieldnames or []) else None
        if col is None:
            raise ValueError(f"{path} has no 'close' column (found: {reader.fieldnames})")
        for row in reader:
            try:
                closes.append(float(row[col]))
            except (TypeError, ValueError):
                continue
    return np.array(closes, dtype=float)


def _load_from_alpaca(months: int = 6) -> np.ndarray:
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set in environment")

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(api_key, api_secret)
    end = datetime.now()
    start = end - timedelta(days=30 * months)

    closes: List[float] = []
    # Chunk into ~30-day windows like AlpacaBot/tools/download_5min.py to
    # avoid single-request timeouts/limits.
    chunk_days = 30
    n_chunks = -(-(30 * months) // chunk_days)  # ceil
    for i in range(n_chunks):
        chunk_end = end - timedelta(days=i * chunk_days)
        chunk_start = max(start, chunk_end - timedelta(days=chunk_days))
        req = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(10, TimeFrameUnit.Minute),
            start=chunk_start,
            end=chunk_end,
        )
        bars = client.get_stock_bars(req)
        for b in bars.data.get("SPY", []):
            closes.append(float(b.close))

    if not closes:
        raise RuntimeError("Alpaca API returned no SPY bars for the requested window")
    return np.array(closes[::-1], dtype=float)  # chronological order


def build_feature_matrix(prices: np.ndarray, lookback: int) -> np.ndarray:
    """Sliding-window feature extraction, same pattern as AlpacaBot/tools/backtest.py."""
    rows = []
    for i in range(lookback, len(prices)):
        chunk = prices[i - lookback:i + 1]
        ind = compute_all_indicators(chunk)
        rows.append([ind.get(f, 0.0) for f in FEATURE_NAMES])
    return np.array(rows, dtype=float)


def pearson_matrix(features: np.ndarray) -> np.ndarray:
    # np.corrcoef expects variables as rows.
    return np.corrcoef(features, rowvar=False)


def cluster_summary(corr: np.ndarray, names: List[str], threshold: float) -> List[str]:
    """Simple connected-components clustering on |corr| >= threshold."""
    n = len(names)
    visited = [False] * n
    clusters: List[List[str]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in range(n):
                if not visited[k] and k != j and abs(corr[j][k]) >= threshold:
                    visited[k] = True
                    stack.append(k)
        clusters.append([names[idx] for idx in comp])
    return [" / ".join(c) for c in clusters if len(c) > 1]


def render_markdown(corr: np.ndarray, names: List[str], clusters: List[str],
                     n_samples: int, source_note: str) -> str:
    lines = []
    lines.append("# Indicator Correlation Analysis (Workstream E pre-staging)\n")
    lines.append(
        "Read-only analysis of the 14 indicators produced by "
        "`AlpacaBot/core/indicators.py::compute_all_indicators` "
        "(unmodified) over SPY 10-minute bars. Generated by "
        "`tools/compute_indicator_correlation.py`.\n"
    )
    lines.append(f"- Samples: {n_samples}\n- Data source: {source_note}\n")
    lines.append("## Pairwise Pearson Correlation Matrix\n")
    header = "|  | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for i, row_name in enumerate(names):
        cells = " | ".join(f"{corr[i][j]:+.2f}" for j in range(len(names)))
        lines.append(f"| **{row_name}** | {cells} |")
    lines.append("")
    lines.append("## Cluster Analysis\n")
    if clusters:
        lines.append(
            "Indicators grouped below have |Pearson r| >= the configured "
            "threshold with at least one other member of the group "
            "(connected-components clustering), i.e. they carry substantially "
            "redundant information and are candidates for feature reduction:\n"
        )
        for c in clusters:
            lines.append(f"- {c}")
    else:
        lines.append("No indicator pairs met the correlation threshold; all 14 "
                      "indicators appear to carry largely independent information "
                      "at this threshold.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Local CSV with a 'close' column (SPY 10-min bars)")
    parser.add_argument("--lookback", type=int, default=30,
                         help="Rolling window length passed to compute_all_indicators")
    parser.add_argument("--corr-threshold", type=float, default=0.8,
                         help="|r| threshold for cluster grouping")
    parser.add_argument("--months", type=int, default=6,
                         help="Trailing months of SPY bars to fetch from Alpaca")
    args = parser.parse_args()

    source_note = ""
    if args.input:
        prices = _load_from_csv(args.input)
        source_note = f"local file `{args.input}`"
    else:
        try:
            prices = _load_from_alpaca(months=args.months)
            source_note = f"Alpaca market data API (trailing {args.months} months, SPY, 10-min bars)"
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: could not obtain SPY 10-min bars: {e}", file=sys.stderr)
            print("Provide --input <csv> with a local bar cache, or run this "
                  "script where ALPACA_API_KEY/ALPACA_API_SECRET and network "
                  "access to the Alpaca market data API are available.",
                  file=sys.stderr)
            return 1

    features = build_feature_matrix(prices, args.lookback)
    if len(features) < 2:
        print("ERROR: not enough bars to compute a feature matrix", file=sys.stderr)
        return 1

    corr = pearson_matrix(features)
    clusters = cluster_summary(corr, FEATURE_NAMES, args.corr_threshold)
    md = render_markdown(corr, FEATURE_NAMES, clusters, len(features), source_note)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {OUTPUT_PATH} ({len(features)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
