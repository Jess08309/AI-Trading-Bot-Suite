"""
CallBuyer — Left Leg Entry Point

Momentum-based call buying bot.
Buys OTM calls on high-momentum stocks using ML + rules ensemble.

Usage:
  python main.py
  START_CALLBUYER.bat
"""
import logging
import os
import signal
import socket
import sys
import time

from core.config import CallBuyerConfig, CFG
from core.call_engine import CallBuyerEngine

# ── Single Instance Lock ─────────────────────────────
LOCK_PORT = 19799  # unique port for CallBuyer

def ensure_single_instance():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", LOCK_PORT))
        s.listen(1)
        return s
    except OSError:
        print("ERROR: CallBuyer is already running!")
        sys.exit(1)

# ── Logging Setup ────────────────────────────────────
def setup_logging(config: CallBuyerConfig):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(config.LOG_DIR, f"callbuyer_{time.strftime('%Y%m%d')}.log")

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-25s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return logging.getLogger("callbuyer.main")

# ── Main ─────────────────────────────────────────────
def main():
    lock = ensure_single_instance()

    config = CFG
    log = setup_logging(config)

    log.info("=" * 70)
    log.info("   CallBuyer — Left Leg — Momentum Call Buying Bot")
    log.info("=" * 70)

    if not config.has_keys:
        log.error("API keys not configured! Set ALPACA_API_KEY and ALPACA_API_SECRET in .env")
        sys.exit(1)

    engine = CallBuyerEngine(config)

    # Graceful shutdown
    def shutdown(signum, frame):
        log.info(f"Signal {signum} received, shutting down...")
        engine.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        engine.run()
    except Exception as e:
        log.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        lock.close()
        log.info("CallBuyer shut down cleanly")

if __name__ == "__main__":
    main()
