"""
AlpacaBot Entry Point - Scalp Options Trading Bot
Starts the scalp engine + web dashboard.
"""
import logging
import os
import signal
import socket
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.trading_engine import ScalpTradingEngine
from dashboard import Dashboard


def setup_logging(config: Config):
    """Configure logging to console + file."""
    os.makedirs(config.LOG_DIR, exist_ok=True)

    log_file = os.path.join(
        config.LOG_DIR,
        f"alpacabot_{__import__('datetime').date.today().isoformat()}.log"
    )

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if config.VERBOSE else logging.INFO,
        format=fmt,
        handlers=handlers,
    )

    # Reduce noise from third-party libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("alpaca").setLevel(logging.WARNING)


def single_instance_lock(port: int = 49710) -> socket.socket:
    """Prevent multiple bot instances using a socket lock."""
    try:
        lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lock.bind(("127.0.0.1", port))
        lock.listen(1)
        return lock
    except OSError:
        print("ERROR: Another AlpacaBot instance is already running!")
        print("Kill it first or check port 49700.")
        sys.exit(1)


def main():
    """Main entry point."""
    print("=" * 50)
    print("  AlpacaBot v2.4 - SCALP Options Trading")
    print("  Strategy: Per-symbol DTE, 10-min bars")
    print("=" * 50)

    # Single instance
    lock = single_instance_lock()

    # Config
    config = Config()

    # Setup logging
    setup_logging(config)
    log = logging.getLogger("alpacabot")

    # Validate API keys
    if not config.has_keys:
        log.error("API keys not configured!")
        log.error("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env file")
        log.error("Or set environment variables ALPACA_API_KEY and ALPACA_API_SECRET")
        print("\nCreate a .env file with:")
        print("  ALPACA_API_KEY=your_key_here")
        print("  ALPACA_API_SECRET=your_secret_here")
        print("\nGet keys from: https://app.alpaca.markets/paper/dashboard/overview")
        sys.exit(1)

    # Create engine
    engine = ScalpTradingEngine(config)

    # Start web dashboard
    dashboard = Dashboard(
        engine,
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
    )
    dashboard.start()
    print(f"\n  Dashboard: http://localhost:{config.DASHBOARD_PORT}")
    wl = ', '.join(config.WATCHLIST)
    print(f"  Strategy:  {wl} | per-symbol DTE | 10-min bars")
    print(f"  Mode:      {'PAPER' if config.PAPER else 'LIVE'}")
    print(f"  SL/TP:     {config.STOP_LOSS_PCT:.0%} / +{config.TAKE_PROFIT_PCT:.0%}\n")

    # Graceful shutdown on Ctrl+C
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
