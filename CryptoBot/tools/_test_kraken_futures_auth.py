"""
Standalone, read-only auth check for Kraken Futures/Derivatives API keys.

Tries a given key/secret against both the demo and live Derivatives REST
endpoints via get_accounts() (no orders are placed). Useful for quickly
validating a newly generated Kraken Pro API key without touching the bot.

Usage:
    KRAKEN_API_KEY=... KRAKEN_API_SECRET=... python _test_kraken_futures_auth.py
    python _test_kraken_futures_auth.py <api_key> <api_secret>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cryptotrades", "core"))

from kraken_futures_client import KrakenFuturesClient, DEMO_BASE_URL, LIVE_BASE_URL  # noqa: E402

if len(sys.argv) >= 3:
    api_key, api_secret = sys.argv[1], sys.argv[2]
else:
    api_key = os.getenv("KRAKEN_API_KEY", "")
    api_secret = os.getenv("KRAKEN_API_SECRET", "")

if not api_key or not api_secret:
    print("Provide KRAKEN_API_KEY/KRAKEN_API_SECRET env vars or as two CLI args.")
    sys.exit(1)

for label, base_url in (("DEMO", DEMO_BASE_URL), ("LIVE", LIVE_BASE_URL)):
    client = KrakenFuturesClient(api_key, api_secret, base_url)
    try:
        result = client.get_accounts()
        result_type = result.get("result") if isinstance(result, dict) else None
        if result_type == "success":
            print(f"{label} ({base_url}): AUTH OK")
            print(result)
        else:
            print(f"{label} ({base_url}): AUTH FAILED -> {result}")
    except Exception as e:
        print(f"{label} ({base_url}): ERROR -> {e}")
