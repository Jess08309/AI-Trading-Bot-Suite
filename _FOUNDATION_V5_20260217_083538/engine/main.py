import sys, os
import multiprocessing as mp
import socket
from datetime import datetime, timezone
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


_main_lock_socket = None


def _log_main(message: str) -> None:
    try:
        module_dir = os.path.abspath(os.path.dirname(__file__))
        log_path = os.path.abspath(os.path.join(module_dir, "bot.log"))
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{ts} [MAIN] {message}\n")
    except Exception:
        pass


def _acquire_main_lock() -> None:
    global _main_lock_socket
    lock_port = int(os.getenv("BOT_LOCK_PORT", "49632"))
    _main_lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _main_lock_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        _main_lock_socket.bind(("127.0.0.1", lock_port))
    except OSError:
        _log_main(f"Exit: lock port {lock_port} already in use")
        sys.exit(0)

if __name__ == "__main__":
    parent_pid = os.getenv("BOT_PARENT_PID")
    if parent_pid and parent_pid.isdigit() and int(parent_pid) == os.getppid():
        _log_main(f"Exit: BOT_PARENT_PID matches parent pid {parent_pid}")
        sys.exit(0)
    _acquire_main_lock()
    main_pid = os.getenv("BOT_MAIN_PID")
    if main_pid and main_pid.isdigit() and int(main_pid) != os.getpid():
        _log_main(f"Exit: BOT_MAIN_PID {main_pid} does not match current pid {os.getpid()}")
        sys.exit(0)
    os.environ["BOT_MAIN_PID"] = str(os.getpid())
    os.environ["BOT_PARENT_PID"] = str(os.getpid())
    if mp.parent_process() is not None:
        _log_main("Exit: running as a multiprocessing child process")
        sys.exit(0)
    os.environ["BOT_LOCK_SKIP"] = "1"
    from dotenv import load_dotenv
    load_dotenv()
    try:
        from core.trading_engine import main
        main()
    except KeyboardInterrupt:
        pass
