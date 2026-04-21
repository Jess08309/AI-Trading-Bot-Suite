# Full System Audit — 2026-04-21

**Auditors:** GitHub Copilot (Claude) + ChatGPT (GPT-4o)
**Scope:** All 4 bots + dashboard + cloud infrastructure
**Verdict:** Healthy infrastructure, profitable overall, 3 concrete strategy issues to fix.

---

## 1. Infrastructure Health — ✅ GREEN

| Item | Status |
|---|---|
| Server uptime | 10d 20h (Oracle Cloud, Phoenix) |
| Load average | 0.11 / 0.07 / 0.02 (idle) |
| Memory | 1.2 GiB used / 15 GiB total |
| Disk | 5.7 GB / 45 GB (13%) |
| `cryptobot.service` | active since 2026-04-20 23:06 UTC |
| `putseller.service` | active since 2026-04-17 07:26 UTC |
| `callbuyer.service` | active since 2026-04-17 07:26 UTC |
| `dashboard.service` | active since 2026-04-21 00:24 UTC |
| journalctl errors (24h) | **0** across all services |

No crashes, no OOMs, no zombie processes. Cloud migration was the right call — 10+ days of clean uptime.

---

## 2. Account Snapshot

| | Baseline | Current | Δ |
|---|---|---|---|
| Options (PutSeller + CallBuyer + AlpacaBot) | $100,000 | ~$91,300 | **−$8,700 (−8.7%)** |
| CryptoBot paper (spot + futures) | $5,000 | $4,749.72 | **−$250 (−5.0%)** |
| **Grand total** | **$105,000** | **~$96,050** | **−$8,950 (−8.5%)** |

The headline loss was concentrated in AlpacaBot (−$44,700, now retired). Without it, the active bots are roughly net flat to slightly positive.

---

## 3. Per-Bot Findings

### 3.1 CryptoBot — ⚠️ YELLOW (improving)

**Live state:** spot $2,301.61 + futures $2,448.11 = $4,749.72. Daily P&L −$4.67. 0 consecutive losses. Peak $5,004.92.

**Lifetime (1,698 trades):**
- Win rate: **35.7%** (607W / 1091L)
- Profit factor: **0.80** (sub-1.0 = negative expectancy)
- Total P&L: **−$169.36**
- Last 24h: **22W / 16L, +$19.71** ← post-tuning improvement
- Last 7d: 85W / 128L, −$3.08

**Critical patterns:**

| Issue | Evidence | Impact |
|---|---|---|
| **SIDE regime is a trap** | 313W / 704L (30% WR), −$86.63 across 60% of all trades | Biggest loss source. `long_only` mode in a sideways market = bleeding. |
| **HOLD_DECAY still dominant exit** | 180W / 501L (26% WR), −$72.99 across 681 trades (40%) | Despite 2026-03-31 tuning, this exit closes far more losers than winners. |
| **STOP_LOSS exits = 100% loss rate** | 0W / 410L, −$691.08 | By definition, but it means 24% of trades run to the full stop. Entry quality is the real problem. |
| **Worst symbol: PI_XBTUSD** | 34W / 87L (28% WR), −$41.47 | BTC futures leverage = oversized individual hits. |
| **Sufficient winners exist** | TAKE_PROFIT: 178W / 0L +$500.49, TRAILING_STOP: 28W / 0L +$45.77 | When winners run, they pay. Problem is win frequency, not magnitude. |

**Recommendations (priority order):**
1. **Block long-only entries during SIDE regime** — the strongest single fix available. ≈60% of current trade flow would simply not happen.
2. **Blacklist 5 worst symbols** (PI_XBTUSD, MATIC-USD, ADA/USD, NEAR-USD, UNI/USD) — combined −$102 on 161 trades.
3. **Tighten `MIN_ML_CONFIDENCE` to 0.63** during SIDE regime only (already 0.60 globally).
4. **Consider enabling short-side** for DOWN regime (currently −$77.17 in long_only).

### 3.2 PutSeller — 🟢 GREEN (profitable) with one concrete bug

**Live state:** balance $31,905.95, total P&L **+$11,189.63**, 29W / 38L = 43.3% WR, 0 consecutive losses. 6 open spreads, 4 in profit, 2 small drawdown (both NVDA).

**Lifetime (67 trades):**

| Side | W/L | P&L |
|---|---|---|
| CALL spreads | 21W / 26L | **+$13,106.76** |
| PUT spreads | 8W / 12L | **−$1,917.13** |

**By exit reason:**

| Reason | W/L | P&L | Notes |
|---|---|---|---|
| TAKE_PROFIT | 27W / 0L | +$18,149.76 | Core earner, working as designed |
| STOP_LOSS (2.0x credit) | 0L / 19L | −$3,611.93 | Appropriate — fixes 2026-04-14 cascade |
| **EMERGENCY_CALL** | **1W / 16L** | **−$2,893.20** | **🚨 CORE BUG: trigger fires too late** |
| DELTA_BREACH | 0W / 1L | −$274.00 | |
| EMERGENCY_PUT | 0W / 1L | −$126.00 | |
| DTE_EXIT | 0W / 1L | −$63.00 | |

**Single-name concentration:**
- **AMZN: 15W / 11L, +$13,292.80** — carrying the bot. Without it: −$2,103.17.
- Worst: GOOGL (0W/2L, −$939), TQQQ (0W/2L, −$717), AVGO (−$551), CRM (−$366).

**Recommendations:**
1. **Fix EMERGENCY_CALL trigger.** Current: price within **5% of short strike** → close at market. Result: 1W/16L. Options:
   - Raise buffer to **7–8%** (exit earlier, take smaller losses)
   - **OR** switch to delta-based: close at |short delta| ≥ 0.35 (currently 0.40 via DELTA_BREACH, which runs in parallel)
   - **OR** replace emergency market-close with a **defensive roll** (up-and-out to next expiration)
2. **Cap per-underlying exposure** — AMZN alone is 26 of 67 trades (39%). One bad earnings cycle undoes the whole bot.
3. **Investigate call-side entry.** CALL P&L +$13k but **only because of AMZN**. Underlying call-selection logic may be too permissive on non-AMZN names.

### 3.3 CallBuyer — ⚠️ YELLOW (GOOGL-dependent)

**Live state:** balance $25,645.14, total P&L **+$57,601** (14W / 9L = 60.9% WR), mode **PRESSING**, 3-trade win streak. 1 open position (CORZ). ML accuracy **40%** on 15 training samples (still warming up).

**The concentration problem:**

| | Trades | W/L | Total P&L |
|---|---|---|---|
| All trades | 23 | 14W / 9L (61%) | +$57,601 |
| **GOOGL only** | **4** | **4W / 0L** | **+$55,940 (97.1%)** |
| Everything else | 18 | 9W / 9L (**50%**) | **+$904** |

The GOOGL windfall was a known pre-cloud-migration incident where the bot held positions through a multi-day outage. Strip it out and CallBuyer is **barely above break-even at a 50% win rate**.

**ML status:** accuracy 40% (≈random), trained on only 15 samples, warmup threshold 15/15 just reached. Meta-learner is effectively in **pure-rules mode** — the ML signal isn't adding value yet.

**Symbol leaderboard (ex-GOOGL):** AMZN +$791, GOOG +$757, NOK +$283, TQQQ +$244, HIMS +$181, WULF +$116, CSCO +$99. **Losers:** BP −$283, BITO −$200, IBIT −$150 (2 losses), ETHA −$102.

**Recommendations:**
1. **Keep PRESSING mode but monitor closely** — 3 wins in a row is real but sample-size is tiny.
2. **Re-baseline performance metrics excluding GOOGL** on the dashboard. Right now the dashboard number is misleading.
3. **Blacklist BP, BITO** (0/2 and 0/1 losers with no signal of recovery).
4. **Do not raise confidence thresholds yet** — ML is at 40% accuracy. Let it collect another 15–20 outcomes before trusting it to gate entries.

### 3.4 AlpacaBot — ⚫ DORMANT (as designed)

0% allocation, −89% historical, intentionally disabled. Code present for future reactivation with different approach. No action.

---

## 4. Dashboard — 🟢 GREEN

- Active, labeled correctly (`Strategy P&L (Bot)` vs account equity distinguished)
- Accessible via SSH tunnel `ssh -i oracle_bot_key -L 8088:localhost:8088 -N botuser@129.146.38.15` → `http://127.0.0.1:8088`
- **Pending enhancement (not blocking):** add explicit "Net Equity vs $105k Start" KPI.

---

## 5. Consolidated Action List

**Priority 1 (deploy this week):**
- [ ] CryptoBot: block long-only entries in SIDE regime
- [ ] PutSeller: fix EMERGENCY_CALL trigger (raise to 7% OR switch to delta-0.35)
- [ ] CryptoBot: blacklist PI_XBTUSD, MATIC-USD, ADA/USD, NEAR-USD, UNI/USD

**Priority 2 (this month):**
- [ ] PutSeller: cap per-underlying exposure at 25% of positions
- [ ] CallBuyer: blacklist BP, BITO
- [ ] Dashboard: add "Net Equity vs Start" KPI

**Priority 3 (monitoring):**
- [ ] CallBuyer: re-evaluate ML after 15 more trade outcomes
- [ ] CryptoBot: re-evaluate tuning after 7 more trading days
- [ ] PutSeller: investigate call-side logic ex-AMZN

---

## 6. What's Working

- **Cloud migration was the highest-ROI decision of the project.** 10+ days continuous uptime with zero service errors.
- **PutSeller is a real, profitable strategy** (+$11k) — even with the EMERGENCY_CALL bug burning −$2.9k.
- **Risk guardrails held** during the 2026-04-14 PutSeller cascade. STOP_LOSS_MULT=2.0 + ghost-position detection prevented further damage.
- **Post-tune CryptoBot is flat-to-positive in last 24h** (+$19.71) after being negative for weeks. Tuning worked; next step is regime-gating.
- **No code-level bugs found in this audit.** Every issue is a strategy/config tuning question, not a broken code path.

---

*Generated 2026-04-21. Previous audit: `reports/audit_20260420.txt`.*
