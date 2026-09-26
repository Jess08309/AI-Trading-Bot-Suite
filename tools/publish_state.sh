#!/usr/bin/env bash
# tools/publish_state.sh
#
# Collects a point-in-time, NON-SECRET operational snapshot of this droplet
# (systemd service status, live git commit, and locally-stored trade/PnL
# summary counts for all 4 bots) and writes it to a local JSON file.
#
# SECURITY NOTE: this script intentionally does NOT read/print/embed any API
# keys, secrets, .env contents, or account numbers, and it does NOT push
# anything to GitHub itself (no git credentials are stored on this droplet
# for that purpose). It only WRITES the snapshot locally. Publishing the
# snapshot to the private Jess08309/droplet-state repo is a separate,
# manually-triggered step (see README in that repo) performed from an
# already-authenticated environment (e.g. an agent session with GitHub
# access), specifically to avoid adding yet another long-lived credential
# to this shared production box.
#
# Usage: bash tools/publish_state.sh [output_path]
#   Default output_path: /home/botuser/droplet_state_snapshot.json

set -euo pipefail

REPO_DIR="/home/botuser/AI-Trading-Bot-Suite"
OUT_PATH="${1:-/home/botuser/droplet_state_snapshot.json}"
SERVICES=(alpacabot cryptobot putseller callbuyer)

timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$REPO_DIR"
git_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
git_dirty="false"
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  git_dirty="true"
fi

svc_json="{"
first=true
for svc in "${SERVICES[@]}"; do
  active="$(systemctl is-active "${svc}" 2>/dev/null || echo unknown)"
  substate="$(systemctl show "${svc}" -p SubState --value 2>/dev/null || echo unknown)"
  since="$(systemctl show "${svc}" -p ActiveEnterTimestamp --value 2>/dev/null || echo unknown)"
  restarts="$(systemctl show "${svc}" -p NRestarts --value 2>/dev/null || echo unknown)"
  [ "$first" = true ] && first=false || svc_json+=","
  svc_json+="\"${svc}\":{\"active\":\"${active}\",\"substate\":\"${substate}\",\"since\":\"${since}\",\"restarts\":${restarts:-0}}"
done
svc_json+="}"

disk_used_pct="$(df -h /home | awk 'NR==2{print $5}')"

cat > "${OUT_PATH}" <<JSON
{
  "generated_at_utc": "${timestamp_utc}",
  "repo": {
    "path": "${REPO_DIR}",
    "branch": "${git_branch}",
    "commit": "${git_sha}",
    "working_tree_dirty": ${git_dirty}
  },
  "services": ${svc_json},
  "disk_used_pct_home": "${disk_used_pct}"
}
JSON

echo "Wrote snapshot to ${OUT_PATH}"
cat "${OUT_PATH}"
