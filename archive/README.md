# Archive

This directory holds retired-bot artifacts that were left behind in the
active codebase after the 2026-09-27 portfolio consolidation (PutSeller,
CallBuyer, and AlpacaBot retired; CryptoBot is now the repository's only
active bot — see the top-level [README.md](../README.md)).

- **`retired-oracle-services/`** — the systemd unit files
  (`putseller.service`, `callbuyer.service`, `alpacabot.service`) that used
  to be installed on the Oracle Cloud droplet alongside `cryptobot.service`.
  They are kept here only as a reference for what to look for/remove if a
  droplet still has these units installed (see "Retiring a droplet's old
  units" below); `CryptoBot/deploy/oracle/setup_server.sh` no longer
  installs or enables them.

The full pre-consolidation source for PutSeller, CallBuyer, and AlpacaBot
(bot code, tests, LEAN projects, backtest_lab/) is **not** duplicated here —
it is preserved in full on the `archive/four-bot-portfolio-2026-09-27`
branch and in `main`'s git history, so nothing was permanently lost.

## Retiring a droplet's old units

If a droplet was provisioned before the consolidation, it may still have
`putseller`, `callbuyer`, and/or `alpacabot` systemd units installed and
enabled. Running the updated `CryptoBot/deploy/oracle/setup_server.sh`
stops, disables, and removes any of these units it finds (see the
"Retiring old bot units" step in that script). To check/clean up manually
instead:

```bash
for s in putseller callbuyer alpacabot spreadbot; do
    sudo systemctl stop "$s" 2>/dev/null
    sudo systemctl disable "$s" 2>/dev/null
    sudo rm -f "/etc/systemd/system/${s}.service"
done
sudo systemctl daemon-reload
```
