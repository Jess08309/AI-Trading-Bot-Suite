#!/bin/bash
# ============================================================================
#  Bot Watchdog — checks all bot services are alive, restarts if dead
#  Called by bot-watchdog.timer every 2 minutes
# ============================================================================
set -euo pipefail

LOG_TAG="bot-watchdog"
MAX_MEM_PCT=85  # Restart bot if >85% system RAM used

log() { logger -t "$LOG_TAG" "$1"; }

check_and_restart() {
    local svc="$1"
    local enabled
    enabled=$(systemctl is-enabled "$svc" 2>/dev/null || echo "disabled")
    
    if [ "$enabled" != "enabled" ]; then
        return 0  # Skip disabled services (e.g. AlpacaBot at 0%)
    fi

    if ! systemctl is-active --quiet "$svc"; then
        log "WARNING: $svc is down — restarting..."
        systemctl restart "$svc"
        log "$svc restarted"
    fi
}

# Check each bot
for svc in cryptobot putseller callbuyer alpacabot; do
    check_and_restart "$svc"
done

# Memory pressure check — log warning if high
MEM_USED=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')
if [ "$MEM_USED" -gt "$MAX_MEM_PCT" ]; then
    log "WARNING: Memory usage at ${MEM_USED}% — consider restarting a bot"
fi
