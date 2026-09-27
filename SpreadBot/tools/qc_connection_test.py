"""Verify QuantConnect API credentials (QC_USER_ID / QC_API_TOKEN env vars) are valid.

Usage: python tools/qc_connection_test.py
"""
import os
import sys
import time
from base64 import b64encode
from hashlib import sha256

import requests

BASE_URL = "https://www.quantconnect.com/api/v2"


def get_headers(user_id: str, api_token: str) -> dict:
    timestamp = str(int(time.time()))
    hashed_token = sha256(f"{api_token}:{timestamp}".encode("utf-8")).hexdigest()
    auth = b64encode(f"{user_id}:{hashed_token}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {auth}", "Timestamp": timestamp}


def main() -> int:
    user_id = os.environ.get("QC_USER_ID")
    api_token = os.environ.get("QC_API_TOKEN")
    if not user_id or not api_token:
        print("ERROR: QC_USER_ID and QC_API_TOKEN must be set in the environment.")
        return 1

    resp = requests.post(f"{BASE_URL}/authenticate", headers=get_headers(user_id, api_token), json={})
    print(f"HTTP {resp.status_code}: {resp.text}")
    return 0 if resp.ok and resp.json().get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
