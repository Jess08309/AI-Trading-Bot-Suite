#!/usr/bin/env python3
"""Apply audit-driven fixes to CryptoBot, PutSeller, CallBuyer."""
import json, os, re, shutil, sys, time
from datetime import datetime

TS = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
changes = []

def backup(path):
    if os.path.exists(path):
        bak = f"{path}.bak.{TS}"
        shutil.copy2(path, bak)
        return bak
    return None

# ───────────────────────── CRYPTOBOT ─────────────────────────
cb_locked = "/home/botuser/CryptoBot/data/state/locked_profile.json"
backup(cb_locked)
with open(cb_locked) as f:
    prof = json.load(f)
ov = prof["config_overrides"]

# Blacklist the worst performers (lifetime trade analytics 2026-04-21)
blacklist = ["PI_XBTUSD", "ADA/USD", "UNI/USD", "MATIC/USD", "NEAR/USD",
             "MATIC-USD", "NEAR-USD"]
ov["SYMBOL_BLACKLIST"] = blacklist
changes.append(f"[CryptoBot] SYMBOL_BLACKLIST = {blacklist}")

# Remove blacklisted symbols from SPOT/FUTURES universes too (no wasted scans)
before_spot = list(ov.get("SPOT_SYMBOLS", []))
ov["SPOT_SYMBOLS"] = [s for s in before_spot if s not in blacklist]
removed_spot = [s for s in before_spot if s not in ov["SPOT_SYMBOLS"]]
if removed_spot:
    changes.append(f"[CryptoBot] SPOT_SYMBOLS removed: {removed_spot}")

before_fut = list(ov.get("FUTURES_SYMBOLS", []))
ov["FUTURES_SYMBOLS"] = [s for s in before_fut if s not in blacklist]
removed_fut = [s for s in before_fut if s not in ov["FUTURES_SYMBOLS"]]
if removed_fut:
    changes.append(f"[CryptoBot] FUTURES_SYMBOLS removed: {removed_fut}")

# Tighten SIDE regime override: 0.66 → 0.72
# Rationale: SIDE regime = 60% of trades, 30% WR, -$86. Raise the ML bar.
old_side = ov.get("SIDE_MARKET_ML_OVERRIDE", 0.66)
ov["SIDE_MARKET_ML_OVERRIDE"] = 0.72
changes.append(f"[CryptoBot] SIDE_MARKET_ML_OVERRIDE {old_side} -> 0.72")

# Also raise MIN_ML_CONFIDENCE floor slightly (0.60 -> 0.62)
old_min = ov.get("MIN_ML_CONFIDENCE", 0.60)
ov["MIN_ML_CONFIDENCE"] = 0.62
changes.append(f"[CryptoBot] MIN_ML_CONFIDENCE {old_min} -> 0.62")

with open(cb_locked, "w") as f:
    json.dump(prof, f, indent=2)

# ───────────────────────── PUTSELLER ─────────────────────────
ps_cfg = "/home/botuser/PutSeller/core/config.py"
backup(ps_cfg)
with open(ps_cfg) as f:
    src = f.read()

# EMERGENCY_BUFFER_PCT: 0.05 -> 0.08 (audit: 1W/16L, -$2893)
new_src, n = re.subn(
    r"(EMERGENCY_BUFFER_PCT:\s*float\s*=\s*)0\.05",
    r"\g<1>0.08",
    src, count=1)
if n != 1:
    print("ERROR: EMERGENCY_BUFFER_PCT not found or already patched", file=sys.stderr)
    sys.exit(2)
src = new_src
changes.append("[PutSeller] EMERGENCY_BUFFER_PCT 0.05 -> 0.08 (5% -> 8% buffer)")

# Update the inline comment to reflect the change
src = src.replace(
    "close if underlying within 5% of short strike (was 2%",
    "close if underlying within 8% of short strike (was 5% -- audit 2026-04-21 showed 1W/16L, was firing too late; was 2%",
    1)

with open(ps_cfg, "w") as f:
    f.write(src)

# ───────────────────────── CALLBUYER ─────────────────────────
cb_grades = "/home/botuser/CallBuyer/data/state/backtest_grades.json"
backup(cb_grades)
with open(cb_grades) as f:
    grades = json.load(f)

cb = grades["bots"]["callbuyer"]
blocked = cb.get("blocked", [])
for sym in ("BP", "BITO"):
    if sym not in blocked:
        blocked.append(sym)
        changes.append(f"[CallBuyer] blocked += {sym}")
cb["blocked"] = sorted(blocked)

# Also remove from qualified if present (shouldn't be, but safe)
cb["qualified"] = [s for s in cb.get("qualified", []) if s not in ("BP", "BITO")]

grades["audit_patch_2026_04_21"] = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "note": "Added BP, BITO to callbuyer.blocked after 0W/3L in audit"
}

with open(cb_grades, "w") as f:
    json.dump(grades, f, indent=2)

# ───────────────────────── SUMMARY ─────────────────────────
print("=" * 60)
print(f"PATCH APPLIED {TS}")
print("=" * 60)
for c in changes:
    print("  " + c)
print()
print("Backups created with suffix .bak." + TS)
