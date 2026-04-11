import argparse
import os
import runpy
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_script(path: str) -> int:
    if not os.path.exists(path):
        print(f"Missing script: {path}")
        return 1
    runpy.run_path(path, run_name="__main__")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Master Chess workspace entrypoint for reports/backtests/monitoring"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        choices=["help", "report", "backtest", "exporter"],
        help="report: quick_report.py, backtest: backtest_harness.py, exporter: monitoring/state_exporter.py",
    )
    args, passthrough = parser.parse_known_args()

    if args.command == "help":
        parser.print_help()
        return 0

    if args.command == "report":
        target = os.path.join(BASE_DIR, "quick_report.py")
    elif args.command == "backtest":
        target = os.path.join(BASE_DIR, "backtest_harness.py")
    else:
        target = os.path.join(BASE_DIR, "monitoring", "state_exporter.py")

    sys.argv = [target] + passthrough
    return _run_script(target)


if __name__ == "__main__":
    raise SystemExit(main())
