"""
Bot Scheduler — manages AlpacaBot and CryptoBot start/stop windows.

AlpacaBot  (weekdays only):  START 07:25 MT | STOP 14:30 MT
CryptoBot  (24/7 except):    PAUSE 03:00-04:00 | 08:00-10:00 | 14:00-18:00 MT

Runs forever, checks every 30 seconds.
"""
import subprocess
import time
import logging
import os
import sys
from datetime import datetime
import pytz

MT = pytz.timezone("America/Denver")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] scheduler: %(message)s",
    handlers=[
        logging.FileHandler(r"C:\AlpacaBot\logs\scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("scheduler")

# ── Config ────────────────────────────────────────────────────────────────────
ALPACA_PYTHON  = r"C:\AlpacaBot\.venv\Scripts\python.exe"
ALPACA_SCRIPT  = r"C:\AlpacaBot\main.py"
ALPACA_CWD     = r"C:\AlpacaBot"

CRYPTO_PYTHON  = r"C:\Bot\.venv\Scripts\python.exe"
CRYPTO_SCRIPT  = r"C:\Bot\cryptotrades\main.py"
CRYPTO_CWD     = r"C:\Bot"
CRYPTO_ENV     = {
    **os.environ,
    "BOT_LOCK_SKIP": "1",
    "ENABLE_FUTURES": "true",
    "ENABLE_COINBASE_FUTURES_DATA": "true",
    "ENABLE_KRAKEN_FUTURES_FALLBACK": "true",
    "DIRECTION_BIAS": "short_lean",
    "DIRECTION_BIAS_STRENGTH": "0.06",
    "RL_SHADOW_MODE": "true",
    "RL_LIVE_SIZE_CONTROL": "false",
    "RL_LIVE_SIZE_MIN_MULT": "0.5",
    "RL_LIVE_SIZE_MAX_MULT": "1.5",
    "USE_LOCKED_PROFILE": "true",
    "LOCKED_PROFILE_PATH": r"C:\Bot\data\state\locked_profile.json",
}

# AlpacaBot: window (hour, minute) in MT, weekdays only
ALPACA_START = (7, 25)
ALPACA_STOP  = (14, 30)

# CryptoBot: pause windows (start_hour, end_hour) in MT — exclusive of end
CRYPTO_PAUSE_WINDOWS = [
    (3,  4),   # 3 AM – 4 AM  : 29% WR, -$5/session
    (8,  10),  # 8 AM – 10 AM : US open chaos, -$34/session
    (14, 18),  # 2 PM – 6 PM  : US close drag, -$34/session
]

# ── Process tracking ──────────────────────────────────────────────────────────
alpaca_proc  = None
crypto_proc  = None


def now_mt() -> datetime:
    return datetime.now(MT)


def is_weekday() -> bool:
    return now_mt().weekday() < 5  # Mon=0 … Fri=4


def minutes_since_midnight(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def alpaca_should_run() -> bool:
    if not is_weekday():
        return False
    t = now_mt()
    start = ALPACA_START[0] * 60 + ALPACA_START[1]
    stop  = ALPACA_STOP[0]  * 60 + ALPACA_STOP[1]
    return start <= minutes_since_midnight(t) < stop


def crypto_should_pause() -> bool:
    h = now_mt().hour
    for (start, end) in CRYPTO_PAUSE_WINDOWS:
        if start <= h < end:
            return True
    return False


def is_alive(proc) -> bool:
    return proc is not None and proc.poll() is None


def start_alpaca():
    global alpaca_proc
    if is_alive(alpaca_proc):
        return
    log.info("START AlpacaBot")
    alpaca_proc = subprocess.Popen(
        [ALPACA_PYTHON, ALPACA_SCRIPT],
        cwd=ALPACA_CWD,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    log.info(f"AlpacaBot PID {alpaca_proc.pid}")


def stop_alpaca(reason: str = "scheduled"):
    global alpaca_proc
    if not is_alive(alpaca_proc):
        alpaca_proc = None
        return
    log.info(f"STOP AlpacaBot ({reason}) PID {alpaca_proc.pid}")
    alpaca_proc.terminate()
    try:
        alpaca_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        alpaca_proc.kill()
    alpaca_proc = None


def start_crypto():
    global crypto_proc
    if is_alive(crypto_proc):
        return
    log.info("START CryptoBot")
    crypto_proc = subprocess.Popen(
        [CRYPTO_PYTHON, CRYPTO_SCRIPT],
        cwd=CRYPTO_CWD,
        env=CRYPTO_ENV,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    log.info(f"CryptoBot PID {crypto_proc.pid}")


def stop_crypto(reason: str = "pause window"):
    global crypto_proc
    if not is_alive(crypto_proc):
        crypto_proc = None
        return
    log.info(f"STOP CryptoBot ({reason}) PID {crypto_proc.pid}")
    crypto_proc.terminate()
    try:
        crypto_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        crypto_proc.kill()
    crypto_proc = None


def main():
    log.info("=" * 60)
    log.info("Bot Scheduler starting")
    log.info(f"AlpacaBot window: {ALPACA_START[0]:02d}:{ALPACA_START[1]:02d} – "
             f"{ALPACA_STOP[0]:02d}:{ALPACA_STOP[1]:02d} MT (weekdays)")
    log.info(f"CryptoBot pauses: {CRYPTO_PAUSE_WINDOWS} MT")
    log.info(f"Python: {sys.executable}")
    log.info("=" * 60)

    # On startup, always start crypto if not in a pause window
    if not crypto_should_pause():
        start_crypto()
    else:
        log.info(f"CryptoBot: inside pause window — holding off until {now_mt().hour+1}:00 MT")

    global alpaca_proc, crypto_proc

    tick = 0
    while True:
        tick += 1
        try:
            t = now_mt()
            log.info(f"[tick {tick}] {t.strftime('%a %H:%M:%S')} MT | "
                     f"alpaca={'alive' if is_alive(alpaca_proc) else 'dead'} "
                     f"(should={'Y' if alpaca_should_run() else 'N'}) | "
                     f"crypto={'alive' if is_alive(crypto_proc) else 'dead'} "
                     f"(paused={'Y' if crypto_should_pause() else 'N'})")

            # ── AlpacaBot ──────────────────────────────────────────────────
            if alpaca_should_run():
                if not is_alive(alpaca_proc):
                    start_alpaca()
            else:
                if is_alive(alpaca_proc):
                    if not is_weekday():
                        stop_alpaca("weekend")
                    else:
                        stop_alpaca("outside trading window")

            # ── CryptoBot ──────────────────────────────────────────────────
            if crypto_should_pause():
                if is_alive(crypto_proc):
                    h = t.hour
                    window = next((w for w in CRYPTO_PAUSE_WINDOWS if w[0] <= h < w[1]), None)
                    stop_crypto(f"pause window {window[0]:02d}:00-{window[1]:02d}:00 MT")
            else:
                if not is_alive(crypto_proc):
                    start_crypto()

            # ── Crash restart ───────────────────────────────────────────────
            if alpaca_proc is not None and not is_alive(alpaca_proc):
                if alpaca_should_run():
                    log.warning(f"AlpacaBot crashed (exit {alpaca_proc.poll()}) — restarting")
                    start_alpaca()
                else:
                    alpaca_proc = None  # type: ignore[assignment]

            if crypto_proc is not None and not is_alive(crypto_proc):
                if not crypto_should_pause():
                    log.warning(f"CryptoBot crashed (exit {crypto_proc.poll()}) — restarting")
                    start_crypto()
                else:
                    crypto_proc = None  # type: ignore[assignment]

        except Exception as exc:
            log.error(f"Scheduler loop error (tick {tick}): {exc}", exc_info=True)

        time.sleep(30)


if __name__ == "__main__":
    main()
