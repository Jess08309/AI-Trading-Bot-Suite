"""
IronCondor Entry Point — Credit Put + Call Spread Bot (Right Leg)
Sells bull put spreads AND bear call spreads on quality large-cap stocks.
"""
import logging
import os
import signal
import socket
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import CFG
from core.put_engine import PutSellerEngine


def setup_logging():
    """Configure logging to console + file."""
    os.makedirs(CFG.LOG_DIR, exist_ok=True)

    log_file = os.path.join(
        CFG.LOG_DIR,
        f"putseller_{__import__('datetime').date.today().isoformat()}.log"
    )

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if CFG.VERBOSE else logging.INFO,
        format=fmt,
        handlers=handlers,
    )

    # Silence noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("alpaca").setLevel(logging.WARNING)


def single_instance_lock(port: int = 49720) -> socket.socket:
    """Prevent multiple PutSeller instances. Different port from AlpacaBot (49710)."""
    try:
        lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lock.bind(("127.0.0.1", port))
        lock.listen(1)
        return lock
    except OSError:
        print("ERROR: Another PutSeller instance is already running!")
        print("Kill it first or check port 49720.")
        sys.exit(1)


def main():
    """Main entry point."""
    print("=" * 50)
    print("  IronCondor v2.0 - Credit Put + Call Spreads")
    print("  Strategy: Bull Put + Bear Call Spreads")
    print("  Appendage: Right Leg")
    print("=" * 50)

    # Single instance lock (different port from AlpacaBot)
    lock = single_instance_lock()

    # Setup logging
    setup_logging()
    log = logging.getLogger("putseller")

    # Validate API keys
    if not CFG.has_keys:
        log.error("API keys not configured!")
        log.error("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env file")
        print("\nCreate a .env file with:")
        print("  ALPACA_API_KEY=your_key_here")
        print("  ALPACA_API_SECRET=your_secret_here")
        sys.exit(1)

    # Create engine
    engine = PutSellerEngine(CFG)

    # Graceful shutdown
    def shutdown(signum, frame):
        log.info("Shutdown signal received")
        engine.running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Run
    try:
        engine.run()
    except Exception as e:
        log.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
