# Go-Live Criteria (DRAFT — for owner review)

Status: **DRAFT**. This document defines the governance bar for the bot
fleet (AlpacaBot, CallBuyer, PutSeller, CryptoBot). It is produced by
Workstream F (portfolio/governance layer) and is intended for owner
review and sign-off before any of the protocols below are enforced
automatically.

## The Rule

> A bot stays live only if:
> **>= 100 closed trades** AND **profit factor > 1.3** AND
> **win rate > 40%** AND **max drawdown < 10%**.
> Bots failing get paused until they pass in the lab with margin.

The Rule is computed per bot by `tools/portfolio_report.py`, which is the
canonical, authoritative implementation of these four checks. Any change
to the thresholds above must be mirrored in that script's `RULE_MIN_*` /
`RULE_MAX_*` constants.

## Probation Protocol

- A bot currently **passing** The Rule enters **probation monitoring**:
  it is re-checked **weekly** against The Rule using the trailing trade
  history available at that time.
- **2 consecutive weekly failures** of The Rule while on probation
  triggers an **auto-pause proposal** to the owner (not an automatic
  pause) — the owner makes the final call to pause or grant a documented
  exception.
- A bot that clears a weekly re-check resets its consecutive-failure
  counter to zero.

## Go-Live Requirements (paper → real capital)

A bot may move from paper trading to real capital only once **all** of
the following hold simultaneously:

1. The Rule has been passed on paper for **4 consecutive weeks** (not
   just at a single point in time).
2. **Zero** phantom/orphan alerts from `tools/portfolio_report.py`
   reconciliation for the trailing **2 weeks**.
3. The fleet-wide kill switch (`/home/botuser/KILL_ALL`) has been
   **tested live** against that bot (touch file → confirm no new
   entries + warning log → remove file → confirm normal operation
   resumes) with the test evidence attached to the approval record.

## Kill Criteria (immediate halt + human review)

Either of the following triggers an **immediate halt** of the affected
bot (via the kill switch) **and** a mandatory human review before it may
resume:

- **-15% drawdown** on any single bot (measured against that bot's
  allocated capital / peak equity), OR
- **Any** phantom position detected by the reconciliation report
  (a phantom means the bot's internal state has drifted from the
  broker's ground truth — this is exactly the regression class that
  caused the historical NFLX bug, and is treated as a hard stop
  regardless of P&L impact).

## Capital Plan

- A bot that meets the go-live requirements above starts real-money
  trading at **its current paper allocation percentage** of a **small**
  real account (i.e. the same `ALLOCATION_PCT` used in paper is applied
  to a deliberately small live account, not the full target book).
- Capital is scaled up only after **4 more profitable weeks** live,
  re-evaluated against The Rule using live (not paper) trade data.
- Any kill-criteria event (above) resets the capital plan: the bot
  returns to paper and must re-clear the full go-live bar before
  real capital is reconsidered.

## Relationship to Other Workstream F Deliverables

- `tools/portfolio_report.py` — computes The Rule scorecard and runs
  reconciliation; intended to run daily via
  `deploy/portfolio-report.service` + `.timer` (21:30 UTC, after US
  market close). These units are included for review only and are
  **not** installed/enabled by this change.
- Fleet-wide kill switch — implemented as a `/home/botuser/KILL_ALL`
  file check in each bot's main loop; see the kill-switch section of the
  Workstream F PR description for per-bot latency estimates and the
  manual test evidence.
