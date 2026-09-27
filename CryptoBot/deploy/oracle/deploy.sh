#!/bin/bash
# ============================================================================
#  Deploy CryptoBot to Oracle Cloud server from Linux/macOS/Codespace
#  Usage: ./deploy.sh <SERVER_IP> [--first-deploy] [--sync-state] [--key <path>]
#  Run from anywhere inside the repo (path is resolved relative to this script)
# ============================================================================
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <SERVER_IP> [--first-deploy] [--sync-state] [--key <path>]"
    exit 1
fi

SERVER_IP="$1"; shift
KEY_FILE="${HOME}/.ssh/oracle_bot_key"
FIRST_DEPLOY=false
SYNC_STATE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --first-deploy) FIRST_DEPLOY=true ;;
        --sync-state)   SYNC_STATE=true ;;
        --key)          KEY_FILE="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

BOT_USER="botuser"
SSH="ssh -i ${KEY_FILE} -o StrictHostKeyChecking=accept-new"
SCP="scp -i ${KEY_FILE} -o StrictHostKeyChecking=accept-new"

log() { echo -e "\033[36m[deploy]\033[0m $1"; }

# Resolve repo root (this script lives at <repo>/CryptoBot/deploy/oracle)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------------------------------------------------------------------------
#  1. Push latest code via git (single monorepo)
#     NOTE: this commits and pushes local changes to GitHub. Review `git status`
#     first if you're unsure what will be pushed.
# ---------------------------------------------------------------------------
log "Pushing code updates to GitHub..."
cd "$REPO_ROOT"
git add -A
if ! git diff --cached --quiet; then
    git commit -m "deploy: $(date '+%Y-%m-%d %H:%M')"
fi
git push origin main

log "Pulling latest on server..."
$SSH "${BOT_USER}@${SERVER_IP}" 'cd ~/AI-Trading-Bot-Suite && git pull --ff-only'

# ---------------------------------------------------------------------------
#  2. Sync .env files (secrets — NOT in git)
# ---------------------------------------------------------------------------
log "Syncing .env files..."

declare -A ENV_FILES=(
    ["${REPO_ROOT}/CryptoBot/cryptotrades/.env"]="AI-Trading-Bot-Suite/CryptoBot/cryptotrades/.env"
)

for src in "${!ENV_FILES[@]}"; do
    dst="${ENV_FILES[$src]}"
    if [ -f "$src" ]; then
        log "  $src -> ~/$dst"
        $SCP "$src" "${BOT_USER}@${SERVER_IP}:~/${dst}"
    fi
done

# ---------------------------------------------------------------------------
#  3. Optionally sync state files (positions, balances, models)
# ---------------------------------------------------------------------------
if [ "$SYNC_STATE" = true ]; then
    log "Syncing state files..."

    declare -A STATE_FILES=(
        ["${REPO_ROOT}/CryptoBot/cryptotrades/data/state/paper_balances.json"]="AI-Trading-Bot-Suite/CryptoBot/cryptotrades/data/state/"
        ["${REPO_ROOT}/CryptoBot/cryptotrades/data/state/locked_profile.json"]="AI-Trading-Bot-Suite/CryptoBot/cryptotrades/data/state/"
    )

    for src in "${!STATE_FILES[@]}"; do
        dst="${STATE_FILES[$src]}"
        if [ -f "$src" ]; then
            log "  $src -> ~/$dst"
            $SCP "$src" "${BOT_USER}@${SERVER_IP}:~/${dst}"
        fi
    done
fi

# ---------------------------------------------------------------------------
#  4. First deploy: copy setup files & run setup
# ---------------------------------------------------------------------------
if [ "$FIRST_DEPLOY" = true ]; then
    log "First deploy — uploading setup files..."
    $SSH "${BOT_USER}@${SERVER_IP}" 'mkdir -p ~/deploy'

    for f in "${SCRIPT_DIR}"/*; do
        log "  $(basename "$f")"
        $SCP "$f" "${BOT_USER}@${SERVER_IP}:~/deploy/"
    done

    log "Running setup_server.sh..."
    $SSH "${BOT_USER}@${SERVER_IP}" 'chmod +x ~/deploy/*.sh && sudo ~/deploy/setup_server.sh'
fi

# ---------------------------------------------------------------------------
#  5. Enforce retired service cleanup + restart active services
# ---------------------------------------------------------------------------
log "Stopping/disabling retired services (if present)..."
$SSH "${BOT_USER}@${SERVER_IP}" 'for unit in putseller.service putseller.timer spreadbot.service spreadbot.timer alpacabot.service alpacabot.timer callbuyer.service callbuyer.timer; do sudo systemctl stop "$unit" 2>/dev/null || true; sudo systemctl disable "$unit" 2>/dev/null || true; sudo rm -f /etc/systemd/system/"$unit" /lib/systemd/system/"$unit" /usr/lib/systemd/system/"$unit"; for link_dir in /etc/systemd/system/*.wants /etc/systemd/system/*.requires; do [ -d "$link_dir" ] || continue; sudo rm -f "$link_dir/$unit"; done; sudo systemctl reset-failed "$unit" 2>/dev/null || true; done; sudo systemctl daemon-reload'

log "Restarting active services..."
$SSH "${BOT_USER}@${SERVER_IP}" 'sudo systemctl daemon-reload; sudo systemctl restart cryptobot; sudo systemctl restart bot-watchdog.timer'

log "Checking status..."
$SSH "${BOT_USER}@${SERVER_IP}" 'printf "%-12s %s\n" "cryptobot" "$(systemctl is-active cryptobot 2>/dev/null || echo inactive)"; printf "%-12s %s\n" "watchdog" "$(systemctl is-active bot-watchdog.timer 2>/dev/null || echo inactive)"; for s in putseller spreadbot alpacabot callbuyer; do printf "%-12s %s\n" "$s" "$(systemctl is-enabled "$s" 2>/dev/null || echo not-installed)"; done'

log "Deploy complete!"
