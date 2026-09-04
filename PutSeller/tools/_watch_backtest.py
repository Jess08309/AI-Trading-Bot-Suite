"""Poll a QC backtest every 45 min until it completes, then print final stats."""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from qc_deploy import post

PROJECT_ID = 35170958
BACKTEST_ID = "1f2c6c291d1668f3457d61b0d98f6f86"
POLL_SECONDS = 45 * 60


def fetch_creds():
    env = subprocess.run(
        ["ssh", "-i", os.path.expanduser("~/.ssh/bot_deploy_key"), "root@165.227.28.126",
         "systemctl show putseller.service -p Environment"],
        capture_output=True, text=True, check=True,
    ).stdout
    user_id = re.search(r"QC_USER_ID=(\d+)", env).group(1)
    token = re.search(r"QC_API_TOKEN=([a-f0-9]+)", env).group(1)
    return user_id, token


def main():
    user_id, token = fetch_creds()
    while True:
        r = post("/backtests/read", user_id, token, {"projectId": PROJECT_ID, "backtestId": BACKTEST_ID})
        b = r["backtest"]
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] status={b.get('status')} progress={b.get('progress')} completed={b.get('completed')}", flush=True)
        if b.get("completed"):
            print("=== COMPLETED ===", flush=True)
            print("error:", b.get("error"), flush=True)
            stats = b.get("statistics") or {}
            for k, v in stats.items():
                print(f"  {k}: {v}", flush=True)
            break
        if b.get("error"):
            print("=== ERRORED ===", flush=True)
            print(b.get("error"), flush=True)
            print(b.get("stacktrace"), flush=True)
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
