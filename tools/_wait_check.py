"""Wait for backtest to finish then print results."""
import time, sys

LOG = r"c:\Bot\data\backtest\backtest_6mo_output.txt"
MAX_WAIT = 720  # 12 min max

for i in range(MAX_WAIT // 10):
    time.sleep(10)
    try:
        content = open(LOG).read()
    except:
        continue
    if "BACKTEST RESULTS" in content:
        lines = content.strip().split("\n")
        print(f"\n=== DONE at {time.strftime('%H:%M:%S')} ===")
        for line in lines[-80:]:
            print(line)
        sys.exit(0)
    if i % 6 == 0:
        lines = content.strip().split("\n")
        print(f"{time.strftime('%H:%M:%S')} - {len(lines)} lines...")

print(f"\n=== TIMED OUT at {time.strftime('%H:%M:%S')} ===")
content = open(LOG).read()
lines = content.strip().split("\n")
for line in lines[-30:]:
    print(line)
