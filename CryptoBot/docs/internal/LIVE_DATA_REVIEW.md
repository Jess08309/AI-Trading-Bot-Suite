# Live-data review — 2026-10-03

## Access and limits

Checked `CryptoBot/data/`, `CryptoBot/cryptotrades/data/`, historical/model
directories and filenames for trade histories, journals, state, models and
logs. There were **no production artifacts**; `cryptotrades/data/__init__.py`
is source, not live data. This task's failed downloads made empty historical
directories, and `walk_forward.py --skip-mc` made a config-only ignored JSON.
Neither is live evidence. No droplet SSH session or broker account access
was available/attempted. No credentials, order flags or services were changed.
Unauthenticated public Alpaca **market-data** probe failed DNS resolution;
the paper **account** API was not probed.

Therefore current per-symbol expectancy, realized P&L, drawdown, direction
edge, stop slippage, worst/best trades and live-vs-backtest divergence cannot
be measured. Historical PR claims below are corroborated repository history,
not fresh account results and not OOS evidence.

## Merged live-behavior evidence

Verified each PR is merged via GitHub:

| PR | Historical finding and fix | What to verify on the droplet now |
|---|---|---|
| [#17](https://github.com/Jess08309/AI-Trading-Bot-Suite/pull/17) | ONDO/HYPE phantom buys; AAVE/PEPE/UNI orphan/partial sells. Confirm filled quantity rather than assuming submission succeeds | Broker vs local quantity and zero/partial-fill close counts |
| [#18](https://github.com/Jess08309/AI-Trading-Bot-Suite/pull/18) | Explicit `qty_available=0` fell back to full qty; TRUMP infinite close retries | Repeated broker-zero confirmations, pending orders, remaining real exposure |
| [#19](https://github.com/Jess08309/AI-Trading-Bot-Suite/pull/19) | GRT/TRUMP/XTZ fee-dust positions 18–21h old; not a recurrence of #17 | Persisted retry count/time; reconciliation only after independent broker-zero evidence |
| [#29](https://github.com/Jess08309/AI-Trading-Bot-Suite/pull/29) | LDO/HYPE submitted/cancelled buys without confirmed fills | Strike/cooldown state, max retry count, confirmed fills after cooldown |
| [#31](https://github.com/Jess08309/AI-Trading-Bot-Suite/pull/31) | Market round trips failed to clear spreads; 8 HYPE cancellations in 13m; LINK/LTC losses. Added limit orders, spread/risk caps, ATR exits, expectancy and journal | Spread-gate skips, limit fills, realized costs and exit overshoot; do not infer a new edge merely from lower costs |
| [#35](https://github.com/Jess08309/AI-Trading-Bot-Suite/pull/35) | Dynamic scanner admitted wide-spread meme coins | Scanner exclusions plus actual engine blacklist; scanner exclusion alone does not prove deployed service settings |
| [#37](https://github.com/Jess08309/AI-Trading-Bot-Suite/pull/37) | SHIB/ONDO wash-trade rejects blocked exits, 2000+ attempts reported | Opposing BUY cancellation, recheck, skipped SELL cycles and eventual successful exit |

PR #31's prompt reported approximately 35% WR / +$130 dominated by one FIL
trade over three days. This is **old, owner-supplied anecdotal account
evidence**, not this audit's result. Stop overshoots may arise from spread
gates, resting limits, wash-trade blocking or partial fills; changing stop
percentages alone cannot solve those execution failures.

## What the code actually records

Paths below distinguish **module-relative state** from **cwd-relative**
records; retrieve both data roots, not just one.

- `cryptotrades/core/trading_engine.py:2854-2859,3070-3128`:
  positions, paper balances, symbol-pause state and recent price-history
  JSON are under the **module-relative** `CryptoBot/cryptotrades/data/state/`.
  The saved price arrays are capped at 500 observations, not six months of
  timestamped OHLC. They cannot substitute for historical candles.
- `_log_trade_csv` (`core/trading_engine.py:3311-3335`) writes cwd-relative
  `data/trades.csv`: `timestamp, symbol, direction, entry_price, exit_price,
  size_usd, pnl_usd, pnl_pct, exit_reason, entry_reason`. CSV `pnl_pct` is
  direction-adjusted price movement; the journal uses realized dollar P&L
  divided by size. Do not assume these percent columns are interchangeable.
  No current writer for `trade_history.csv`/`price_history.csv` was found;
  the retrieval commands include those requested legacy names if present.
- `utils/expectancy_tracker.py:21-119`: default cwd-relative
  `data/state/symbol_expectancy.json`; up to 20 P&L-percent outcomes per
  symbol, consecutive losing round trips, disabled-symbol reasons and
  `saved_at`. Disable after three negative outcomes or negative mean over
  a full 20-trade window. Breakeven neither extends nor resets the streak.
  `timestamp` is accepted by `record_round_trip` but not stored per trade.
  There is no dollar expectancy, fill latency, rejection count or direction
  breakdown in this state file.
- `utils/trade_journal.py:19-44,63-104`: cwd-relative
  `data/trade_journal.csv` and `.db` (SQLite table `trade_journal`).
  Fields: `timestamp, symbol, signal, side, entry_price, exit_price,
  mid_price_at_fill, slippage_bps, size_usd, pnl_usd, pnl_pct, exit_reason`.
  `signal` is the entry reason; `side` is the **closing order side**
  (`sell` for LONG, `buy` for SHORT), not a literal position direction.
- Live wiring (`core/trading_engine.py:3857-3884`) saves expectancy and
  journal on close. The midpoint is a quote fetched during close processing,
  not a broker-stamped midpoint at the exact fill time. For futures/no quote
  it falls back to the exit price.
- **Journal slippage caveat:** `TradeJournalEntry.__post_init__`
  (`utils/trade_journal.py:41-44`) computes default slippage with
  **entry_price** against that **exit-time** midpoint; the caller supplies no
  override. This mixes trade movement with execution slippage. Do not use
  existing `slippage_bps` as observed exit-fill quality or claim it proves
  stops were exceeded. Recompute using trustworthy fill-time quotes/order
  records. No live fix is shipped without the required validation.
- Logging (`core/trading_engine.py:398-425`) defaults to module-relative
  `cryptotrades/logs/trading_YYYYMMDD.log`; `cryptotrades/main.py:12-20` also writes
  module-relative `cryptotrades/bot.log`. Runtime fingerprint/history,
  locked profile, shadows and expectancy may be cwd-relative
  `data/state/`; they help establish actual deployed settings.

The [baseline report](BACKTEST_BASELINE_2026-10.md) resolves production model
load/write paths and explains why current lab results cannot establish
production strategy profitability.

## Owner commands: connect and collect without changing the bot

On your own trusted machine (replace the placeholder; do not paste keys into
chat), the same SSH command connects a terminal used by Sonnet:

```bash
ssh -i "$HOME/.ssh/do_bot_key" botuser@DROPLET_HOST
```

These commands run **on the droplet**. No restart, pull, sudo, order/API call,
environment dump or `.env` copy is needed:

```bash
systemctl show cryptobot --property=MainPID --property=WorkingDirectory
pid="$(systemctl show cryptobot --property=MainPID --value)"
test "$pid" -gt 0 || { echo "No running main process"; exit 1; }
botdir="$(readlink -f "/proc/$pid/cwd")"
test -n "$botdir" && test -d "$botdir" || { echo "Cannot read process cwd"; exit 1; }
printf 'Actual process cwd: %s\n' "$botdir"
cd "$botdir"
# Locate the module root whether cwd is CryptoBot or cryptotrades.
if test -f "$botdir/cryptotrades/core/trading_engine.py"; then
  module="$botdir/cryptotrades"
elif test -f "$botdir/core/trading_engine.py"; then
  module="$botdir"
else
  echo "Set module to the deployed cryptotrades directory before continuing"; exit 1
fi

umask 077
out="$(mktemp -d /tmp/cryptobot-review.XXXXXX)"
# Explicit data paths only: never archive the checkout or credential files.
for root in "$botdir/data" "$module/data"; do
  test -d "$root" || continue
  find "$root" -maxdepth 3 -type f \
    \( -path '*/state/*.json' -o -path '*/state/*.jsonl' \
       -o -name 'trade_history.csv' -o -name 'price_history.csv' \
       -o -name 'trades.csv' -o -name 'trade_journal.csv' \) -print
done | sort -u > "$out/artifacts.txt"
tar -czf "$out/runtime-artifacts.tar.gz" -T "$out/artifacts.txt"
journalctl -u cryptobot --since "7 days ago" --no-pager > "$out/service.log"
find "$botdir/logs" "$module/logs" -maxdepth 1 -type f \
  -name 'trading_*.log' -mtime -8 -print 2>/dev/null > "$out/logs.txt"
test ! -f "$module/bot.log" || printf '%s\n' "$module/bot.log" >> "$out/logs.txt"
tar -czf "$out/file-logs.tar.gz" -T "$out/logs.txt"
test ! -f "$botdir/models/trading_model.joblib" || \
  sha256sum "$botdir/models/trading_model.joblib" > "$out/model-sha256.txt"
printf 'Private output directory: %s\n' "$out"
```

Do not copy a live SQLite file with plain `scp`: take a consistent backup
using standard-library SQLite's backup API (still on the droplet):

```bash
python3 - "$botdir/data/trade_journal.db" "$out/trade_journal.db" <<'PY'
import pathlib, sqlite3, sys
source = pathlib.Path(sys.argv[1])
if source.is_file():
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(sys.argv[2]) as dst:
            src.backup(dst)
else:
    print("No trade journal DB at", source)
PY
```

Review these private files locally for secrets/account identifiers in logs or
state **before sharing**; redact any such information. On your trusted local
machine, substitute the exact printed remote directory:

```bash
mkdir -p "$HOME/cryptobot-review"
chmod 700 "$HOME/cryptobot-review"
scp -i "$HOME/.ssh/do_bot_key" -r \
  botuser@DROPLET_HOST:/tmp/cryptobot-review.EXACT_SUFFIX \
  "$HOME/cryptobot-review/"
```

These are best-effort snapshots: CSV/state files may advance during collection;
note collection time and permissions/missing files. If process cwd differs
from both known entrypoint roots, have the owner locate the deployed module
without exposing environment variables. Keep raw artifacts out of git and
public PRs. Analyze reconciled closed trades by symbol/direction and exit
reason, separate dust adjustments from fills, and pair losses with retry/
wash-trade logs before proposing a change.
