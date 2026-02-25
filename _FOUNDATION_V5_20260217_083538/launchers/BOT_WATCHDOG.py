"""
BOT WATCHDOG — Continuous health monitor for the trading bot.
Runs alongside the bot and checks:
  1. Process alive (is python.exe running cryptotrades/main.py?)
  2. Log freshness (has the log been written to in the last N minutes?)
  3. Crash detection (ERROR / Traceback / Exception in log)
  4. Balance drawdown (total balance vs starting capital)
  5. Win-rate degradation (rolling window)
  6. Stale positions (positions held past max_hold without exit)
  7. Auto-restart on crash (optional)

Alerts go to:
  - Console (always)
  - watchdog_alerts.log (always)
  - Windows toast notification (if win10toast installed)
  - Optional: email (configure SMTP in .env)

Usage:
    python BOT_WATCHDOG.py                  # run with defaults
    python BOT_WATCHDOG.py --no-restart     # monitor only, don't auto-restart
    python BOT_WATCHDOG.py --interval 30    # check every 30 seconds
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent            # c:\Master Chess
BOT_DIR = Path(r"D:\042021\CryptoBot")
PYTHON_EXE = BOT_DIR / ".venv" / "Scripts" / "python.exe"
BOT_MAIN = BOT_DIR / "cryptotrades" / "main.py"
STATE_DIR = BOT_DIR / "data" / "state"
LOG_DIR = BOT_DIR / "logs"
WATCHDOG_LOG = SCRIPT_DIR / "logs" / "watchdog_alerts.log"

# Thresholds
DEFAULT_CHECK_INTERVAL_SEC = 60        # how often the watchdog checks
LOG_STALE_MINUTES = 5                  # alert if log not updated in N min
DRAWDOWN_WARN_PCT = -5.0               # warn at -5 %
DRAWDOWN_CRITICAL_PCT = -10.0          # critical at -10 %
STARTING_CAPITAL = 5000.0
WIN_RATE_WARN = 0.40                   # warn if rolling win-rate < 40 %
WIN_RATE_WINDOW = 50                   # last N trades for rolling WR
CONSECUTIVE_LOSS_WARN = 5              # warn after N consecutive losses
MAX_POSITION_AGE_HOURS = 6             # warn if a position older than this
RESTART_BACKOFF_SEC = 120              # wait before restarting after crash
MAX_RESTARTS_PER_HOUR = 3              # don't restart more than this

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
restart_timestamps: list[float] = []
last_alert_times: dict[str, float] = {}
ALERT_COOLDOWN_SEC = 300               # don't repeat same alert within 5 min

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _log(level: str, msg: str) -> None:
    line = f"{_ts()} | {level:8s} | {msg}"
    print(line)
    try:
        WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def _alert(key: str, msg: str, level: str = "ALERT") -> None:
    """Rate-limited alert: same key won't fire more than once per cooldown."""
    now = time.time()
    if key in last_alert_times and (now - last_alert_times[key]) < ALERT_COOLDOWN_SEC:
        return
    last_alert_times[key] = now
    _log(level, msg)
    _try_toast(msg)

def _try_toast(msg: str) -> None:
    """Best-effort Windows toast notification."""
    try:
        from win10toast import ToastNotifier  # type: ignore
        t = ToastNotifier()
        t.show_toast("Bot Watchdog", msg, duration=5, threaded=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_process_alive() -> bool:
    """Return True if the bot process is running."""
    try:
        import ctypes
        # Use tasklist to find python.exe running cryptotrades
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            if "python.exe" in line.lower():
                # Check if this python is running our bot
                parts = line.strip('"').split('","')
                if len(parts) >= 2:
                    pid = parts[1].strip('"')
                    try:
                        cmd_result = subprocess.run(
                            ["wmic", "process", "where", f"ProcessId={pid}",
                             "get", "CommandLine", "/value"],
                            capture_output=True, text=True, timeout=10,
                        )
                        if "cryptotrades" in cmd_result.stdout:
                            return True
                    except Exception:
                        pass
        return False
    except Exception as e:
        _log("WARN", f"Process check failed: {e}")
        return False


def check_process_alive_simple() -> tuple[bool, int | None]:
    """Simpler check: look for python.exe with CryptoBot venv path."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process python -ErrorAction SilentlyContinue | "
             "Where-Object { $_.Path -like '*CryptoBot*.venv*python.exe' } | "
             "Select-Object -First 1 -ExpandProperty Id"],
            capture_output=True, text=True, timeout=15,
        )
        pid_str = result.stdout.strip()
        if pid_str and pid_str.isdigit():
            return True, int(pid_str)
        return False, None
    except Exception as e:
        _log("WARN", f"Process check error: {e}")
        return False, None


def check_log_freshness() -> tuple[bool, float]:
    """Return (is_fresh, minutes_since_last_write)."""
    try:
        log_files = sorted(LOG_DIR.glob("trading_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            return False, 999.0
        latest = log_files[0]
        age_sec = time.time() - latest.stat().st_mtime
        age_min = age_sec / 60.0
        return age_min < LOG_STALE_MINUTES, age_min
    except Exception as e:
        _log("WARN", f"Log freshness check error: {e}")
        return False, 999.0


def check_log_errors() -> list[str]:
    """Scan last 200 lines of log for ERROR/Traceback/Exception."""
    errors: list[str] = []
    try:
        log_files = sorted(LOG_DIR.glob("trading_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            return errors
        latest = log_files[0]
        lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-200:] if len(lines) > 200 else lines
        for line in tail:
            if re.search(r"\| ERROR\b|\bTraceback\b|\bException\b|\bCRASH\b", line, re.IGNORECASE):
                errors.append(line.strip())
    except Exception as e:
        _log("WARN", f"Log error scan failed: {e}")
    return errors


def check_balance() -> tuple[float, float, float]:
    """Return (total, pnl, pnl_pct)."""
    try:
        bal_file = STATE_DIR / "paper_balances.json"
        if not bal_file.exists():
            return 0.0, 0.0, 0.0
        data = json.loads(bal_file.read_text(encoding="utf-8"))
        spot = float(data.get("spot", 0))
        futures = float(data.get("futures", 0))
        total = spot + futures
        pnl = total - STARTING_CAPITAL
        pnl_pct = (pnl / STARTING_CAPITAL) * 100.0 if STARTING_CAPITAL > 0 else 0.0
        return total, pnl, pnl_pct
    except Exception as e:
        _log("WARN", f"Balance check error: {e}")
        return 0.0, 0.0, 0.0


def check_positions() -> tuple[int, list[str]]:
    """Return (count, list_of_stale_position_symbols)."""
    stale: list[str] = []
    count = 0
    try:
        pos_file = STATE_DIR / "positions.json"
        if not pos_file.exists():
            return 0, []
        data = json.loads(pos_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return 0, []
        now = datetime.now()
        count = len(data)
        for symbol, pos in data.items():
            entry_time_str = pos.get("entry_time") or pos.get("timestamp") or ""
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00").replace("+00:00", ""))
                except Exception:
                    try:
                        entry_time = datetime.strptime(entry_time_str[:19], "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        continue
                age_hours = (now - entry_time).total_seconds() / 3600.0
                if age_hours > MAX_POSITION_AGE_HOURS:
                    stale.append(f"{symbol} ({age_hours:.1f}h)")
    except Exception as e:
        _log("WARN", f"Position check error: {e}")
    return count, stale


def check_recent_trades() -> tuple[float, int]:
    """Return (rolling_win_rate, consecutive_losses) from trade history."""
    try:
        trade_file = BOT_DIR / "data" / "history" / "trade_history.csv"
        if not trade_file.exists():
            return 1.0, 0

        trades: list[dict] = []
        with open(trade_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)

        if not trades:
            return 1.0, 0

        recent = trades[-WIN_RATE_WINDOW:]

        wins = 0
        total = 0
        for t in recent:
            pnl_str = t.get("pnl") or t.get("pnl_dollar") or t.get("profit") or ""
            try:
                pnl = float(pnl_str)
                total += 1
                if pnl > 0:
                    wins += 1
            except (ValueError, TypeError):
                continue

        win_rate = wins / total if total > 0 else 1.0

        # Consecutive losses from the end
        consecutive_losses = 0
        for t in reversed(trades):
            pnl_str = t.get("pnl") or t.get("pnl_dollar") or t.get("profit") or ""
            try:
                pnl = float(pnl_str)
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    break
            except (ValueError, TypeError):
                break

        return win_rate, consecutive_losses
    except Exception as e:
        _log("WARN", f"Trade history check error: {e}")
        return 1.0, 0


# ---------------------------------------------------------------------------
# Auto-restart
# ---------------------------------------------------------------------------

def attempt_restart() -> bool:
    """Try to restart the bot. Returns True if launched successfully."""
    global restart_timestamps
    now = time.time()

    # Rate limit
    restart_timestamps = [ts for ts in restart_timestamps if now - ts < 3600]
    if len(restart_timestamps) >= MAX_RESTARTS_PER_HOUR:
        _alert("restart_limit", f"Restart limit reached ({MAX_RESTARTS_PER_HOUR}/hr). Manual intervention needed.", "CRITICAL")
        return False

    _log("ACTION", f"Attempting bot restart (waiting {RESTART_BACKOFF_SEC}s backoff)...")
    time.sleep(RESTART_BACKOFF_SEC)

    try:
        # Use START_BOT_AGGRESSIVE.bat which sets all the right env vars
        launcher = SCRIPT_DIR / "START_BOT_AGGRESSIVE.bat"
        if not launcher.exists():
            launcher = SCRIPT_DIR / "START_BOT.bat"
        if not launcher.exists():
            _log("ERROR", "No launcher .bat found!")
            return False

        subprocess.Popen(
            ["cmd", "/c", str(launcher)],
            cwd=str(SCRIPT_DIR),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        restart_timestamps.append(time.time())
        _log("ACTION", f"Bot restarted via {launcher.name}")
        time.sleep(30)  # give it time to boot

        alive, pid = check_process_alive_simple()
        if alive:
            _log("OK", f"Bot confirmed running after restart (PID {pid})")
            return True
        else:
            _alert("restart_fail", "Bot failed to start after restart attempt!", "CRITICAL")
            return False
    except Exception as e:
        _alert("restart_error", f"Restart error: {e}", "CRITICAL")
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_watchdog(interval: int, auto_restart: bool) -> None:
    _log("START", "="*60)
    _log("START", "BOT WATCHDOG started")
    _log("START", f"  Check interval: {interval}s")
    _log("START", f"  Auto-restart: {auto_restart}")
    _log("START", f"  Bot dir: {BOT_DIR}")
    _log("START", f"  Log dir: {LOG_DIR}")
    _log("START", f"  Drawdown warn/crit: {DRAWDOWN_WARN_PCT}% / {DRAWDOWN_CRITICAL_PCT}%")
    _log("START", "="*60)

    cycle = 0
    while True:
        cycle += 1
        try:
            # --- 1. Process alive ---
            alive, pid = check_process_alive_simple()
            if not alive:
                _alert("process_dead", "BOT PROCESS IS NOT RUNNING!", "CRITICAL")
                if auto_restart:
                    attempt_restart()
            else:
                if cycle % 10 == 1:  # log OK every 10 cycles
                    _log("OK", f"Bot process alive (PID {pid})")

            # --- 2. Log freshness ---
            fresh, age_min = check_log_freshness()
            if not fresh:
                _alert("log_stale", f"Log file stale! Last update {age_min:.1f} min ago (limit: {LOG_STALE_MINUTES} min)")
            elif cycle % 10 == 1:
                _log("OK", f"Log fresh ({age_min:.1f} min ago)")

            # --- 3. Log errors ---
            errors = check_log_errors()
            if errors:
                # Only alert on new errors (last 5)
                for err in errors[-5:]:
                    key = f"log_err_{hash(err) % 10000}"
                    _alert(key, f"Log error: {err[:200]}")

            # --- 4. Balance / drawdown ---
            total, pnl, pnl_pct = check_balance()
            if total > 0:
                if pnl_pct <= DRAWDOWN_CRITICAL_PCT:
                    _alert("drawdown_crit", f"CRITICAL DRAWDOWN: ${pnl:.2f} ({pnl_pct:.1f}%) — Total: ${total:.2f}", "CRITICAL")
                elif pnl_pct <= DRAWDOWN_WARN_PCT:
                    _alert("drawdown_warn", f"Drawdown warning: ${pnl:.2f} ({pnl_pct:.1f}%) — Total: ${total:.2f}", "WARNING")
                elif cycle % 10 == 1:
                    _log("OK", f"Balance: ${total:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

            # --- 5. Positions ---
            pos_count, stale_positions = check_positions()
            if stale_positions:
                names = ", ".join(stale_positions[:5])
                _alert("stale_pos", f"Stale positions ({len(stale_positions)}): {names}")
            elif cycle % 10 == 1:
                _log("OK", f"Positions: {pos_count}")

            # --- 6. Win rate / consecutive losses ---
            win_rate, consec_losses = check_recent_trades()
            if consec_losses >= CONSECUTIVE_LOSS_WARN:
                _alert("consec_loss", f"Consecutive losses: {consec_losses} (threshold: {CONSECUTIVE_LOSS_WARN})", "WARNING")
            if win_rate < WIN_RATE_WARN and win_rate < 1.0:
                _alert("low_winrate", f"Low rolling win rate: {win_rate:.1%} (last {WIN_RATE_WINDOW} trades, threshold: {WIN_RATE_WARN:.0%})", "WARNING")
            elif cycle % 10 == 1 and win_rate < 1.0:
                _log("OK", f"Win rate: {win_rate:.1%} (last {WIN_RATE_WINDOW}) | Consec losses: {consec_losses}")

            # --- Summary every 10 cycles ---
            if cycle % 10 == 0:
                status = "ALIVE" if alive else "DEAD"
                _log("SUMMARY", f"Cycle {cycle} | Bot: {status} | Balance: ${total:.2f} ({pnl_pct:+.1f}%) | Positions: {pos_count} | WR: {win_rate:.0%}")

        except KeyboardInterrupt:
            _log("STOP", "Watchdog stopped by user (Ctrl+C)")
            break
        except Exception as e:
            _log("ERROR", f"Watchdog cycle error: {e}")

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            _log("STOP", "Watchdog stopped by user (Ctrl+C)")
            break


def main():
    parser = argparse.ArgumentParser(description="Bot Watchdog — continuous health monitor")
    parser.add_argument("--interval", type=int, default=DEFAULT_CHECK_INTERVAL_SEC,
                        help=f"Check interval in seconds (default: {DEFAULT_CHECK_INTERVAL_SEC})")
    parser.add_argument("--no-restart", action="store_true",
                        help="Disable auto-restart on crash")
    args = parser.parse_args()

    run_watchdog(
        interval=args.interval,
        auto_restart=not args.no_restart,
    )


if __name__ == "__main__":
    main()
