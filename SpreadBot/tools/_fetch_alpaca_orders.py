import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from qc_deploy import post

user_id = os.environ["QC_USER_ID"]
token = os.environ["QC_API_TOKEN"]
pid = 35170934
bt = "bf9ad61bb5bb224f85a1dd833148f7bf"

all_orders = []
start = 0
page_size = 100
total_length = None
while total_length is None or start < total_length:
    resp = None
    for attempt in range(20):
        resp = post("/backtests/orders/read", user_id, token,
                     {"projectId": pid, "backtestId": bt, "start": start, "end": start + page_size})
        if resp.get("status") == "loading":
            print("loading...", attempt)
            time.sleep(3)
            continue
        break
    if resp is None or "orders" not in resp:
        print("FAILED", resp)
        break
    all_orders.extend(resp["orders"])
    total_length = resp.get("length", len(all_orders))
    start += page_size
    print(f"fetched {len(all_orders)}/{total_length}")

print("total", len(all_orders))
with open("/tmp/alpacabot_orders.json", "w") as f:
    json.dump(all_orders, f)
print("saved")
