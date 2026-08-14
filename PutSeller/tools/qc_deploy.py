"""Deploy a bot's quantconnect/main.py to QuantConnect cloud and run a backtest.

Creates (or reuses) a QC project per bot, uploads the algorithm file, compiles
it, starts a backtest, waits for completion, and prints the statistics.

Reads credentials exclusively from QC_USER_ID / QC_API_TOKEN env vars.

Usage:
    python qc_deploy.py --bot alpacabot
    python qc_deploy.py --bot callbuyer
    python qc_deploy.py --bot putseller
    python qc_deploy.py --bot all
"""
import argparse
import json
import os
import sys
import time
from base64 import b64encode
from hashlib import sha256
from pathlib import Path

import requests

BASE_URL = "https://www.quantconnect.com/api/v2"
REPO_ROOT = Path(__file__).resolve().parents[2]

BOT_FILES = {
    "putseller": REPO_ROOT / "PutSeller" / "quantconnect" / "main.py",
    "alpacabot": REPO_ROOT / "AlpacaBot" / "quantconnect" / "main.py",
    "callbuyer": REPO_ROOT / "CallBuyer" / "quantconnect" / "main.py",
}
PROJECT_NAMES = {
    "putseller": "PutSeller_IronCondor",
    "alpacabot": "AlpacaBot_Scalp",
    "callbuyer": "CallBuyer_Momentum",
}


def get_headers(user_id: str, api_token: str) -> dict:
    timestamp = str(int(time.time()))
    hashed_token = sha256(f"{api_token}:{timestamp}".encode("utf-8")).hexdigest()
    auth = b64encode(f"{user_id}:{hashed_token}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {auth}", "Timestamp": timestamp}


def post(path: str, user_id: str, api_token: str, payload: dict = None) -> dict:
    resp = requests.post(f"{BASE_URL}{path}", headers=get_headers(user_id, api_token), json=payload or {})
    resp.raise_for_status()
    return resp.json()


def find_or_create_project(name: str, user_id: str, api_token: str) -> int:
    result = post("/projects/read", user_id, api_token)
    for p in result.get("projects", []):
        if p["name"] == name:
            return p["projectId"]
    result = post("/projects/create", user_id, api_token, {"name": name, "language": "Py"})
    if not result.get("success"):
        raise RuntimeError(f"Failed to create project: {result.get('errors')}")
    return result["projects"][0]["projectId"]


def upload_file(project_id: int, content: str, user_id: str, api_token: str) -> None:
    result = post("/files/update", user_id, api_token,
                  {"projectId": project_id, "name": "main.py", "content": content})
    if not result.get("success"):
        result = post("/files/create", user_id, api_token,
                       {"projectId": project_id, "name": "main.py", "content": content})
    if not result.get("success"):
        raise RuntimeError(f"Failed to upload file: {result.get('errors')}")


def compile_project(project_id: int, user_id: str, api_token: str) -> str:
    result = post("/compile/create", user_id, api_token, {"projectId": project_id})
    if not result.get("success"):
        raise RuntimeError(f"Failed to start compile: {result.get('errors')}")
    compile_id = result["compileId"]

    for _ in range(30):
        result = post("/compile/read", user_id, api_token,
                       {"projectId": project_id, "compileId": compile_id})
        state = result.get("state")
        if state == "BuildSuccess":
            return compile_id
        if state == "BuildError":
            raise RuntimeError(f"Compile failed:\n{json.dumps(result, indent=2)}")
        time.sleep(2)
    raise RuntimeError("Compile timed out")


def create_backtest(project_id: int, compile_id: str, name: str, user_id: str, api_token: str) -> str:
    result = post("/backtests/create", user_id, api_token,
                  {"projectId": project_id, "compileId": compile_id, "backtestName": name})
    if not result.get("success"):
        raise RuntimeError(f"Failed to create backtest: {result.get('errors')}")
    return result["backtest"]["backtestId"]


def wait_for_backtest(project_id: int, backtest_id: str, user_id: str, api_token: str, max_wait: int = 900) -> dict:
    start = time.time()
    while time.time() - start < max_wait:
        result = post("/backtests/read", user_id, api_token,
                       {"projectId": project_id, "backtestId": backtest_id})
        backtest = result.get("backtest", {})
        print(f"  progress={backtest.get('progress', 0):.0%} status={backtest.get('status', '?')}")
        if backtest.get("completed"):
            return backtest
        time.sleep(5)
    raise RuntimeError("Backtest timed out")


def deploy_bot(bot: str, user_id: str, api_token: str) -> None:
    file_path = BOT_FILES[bot]
    project_name = PROJECT_NAMES[bot]
    content = file_path.read_text()

    print(f"[{bot}] finding or creating project '{project_name}'...")
    project_id = find_or_create_project(project_name, user_id, api_token)
    print(f"[{bot}] projectId={project_id}")

    print(f"[{bot}] uploading {file_path.name}...")
    upload_file(project_id, content, user_id, api_token)

    print(f"[{bot}] compiling...")
    compile_id = compile_project(project_id, user_id, api_token)
    print(f"[{bot}] compileId={compile_id}")

    backtest_name = f"{project_name}_{int(time.time())}"
    print(f"[{bot}] starting backtest '{backtest_name}'...")
    backtest_id = create_backtest(project_id, compile_id, backtest_name, user_id, api_token)
    print(f"[{bot}] backtestId={backtest_id}, waiting for completion...")

    backtest = wait_for_backtest(project_id, backtest_id, user_id, api_token)

    if backtest.get("hasInitializeError") or backtest.get("error"):
        print(f"[{bot}] BACKTEST ERROR:\n{backtest.get('error')}\n{backtest.get('stacktrace', '')}")
        return

    print(f"[{bot}] Backtest complete!")
    for k, v in backtest.get("statistics", {}).items():
        print(f"  {k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", required=True, choices=list(BOT_FILES.keys()) + ["all"])
    args = parser.parse_args()

    user_id = os.environ.get("QC_USER_ID")
    api_token = os.environ.get("QC_API_TOKEN")
    if not user_id or not api_token:
        print("ERROR: QC_USER_ID and QC_API_TOKEN must be set in the environment.")
        return 1

    bots = list(BOT_FILES.keys()) if args.bot == "all" else [args.bot]
    for bot in bots:
        deploy_bot(bot, user_id, api_token)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
