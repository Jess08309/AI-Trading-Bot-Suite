#!/bin/bash
# ============================================================================
#  Quick update — pull latest code and restart CryptoBot
#  Run from the server: ~/deploy/update_bots.sh
# ============================================================================
set -euo pipefail

echo "=== Pulling latest code ==="
cd "/home/botuser/AI-Trading-Bot-Suite"
git pull --ff-only
CHANGED_FILES=$(git diff HEAD@{1} --name-only 2>/dev/null || true)

if echo "$CHANGED_FILES" | grep -q "^CryptoBot/requirements.txt$"; then
    echo "  CryptoBot/requirements.txt changed — reinstalling..."
    cd "/home/botuser/AI-Trading-Bot-Suite/CryptoBot"
    source .venv/bin/activate
    pip install -r requirements.txt -q
    deactivate
    cd "/home/botuser/AI-Trading-Bot-Suite"
fi

echo ""
echo "=== Retired service cleanup ==="
for s in putseller spreadbot alpacabot callbuyer; do
    sudo systemctl stop "$s" 2>/dev/null || true
    sudo systemctl disable "$s" 2>/dev/null || true
done

echo ""
echo "=== Restarting active services ==="
sudo systemctl daemon-reload
sudo systemctl restart cryptobot
sudo systemctl restart bot-watchdog.timer
sudo systemctl start bot-watchdog.service

echo ""
echo "=== Status ==="
printf "%-12s %s\n" "cryptobot" "$(systemctl is-active cryptobot 2>/dev/null || echo inactive)"
printf "%-12s %s\n" "watchdog" "$(systemctl is-active bot-watchdog.timer 2>/dev/null || echo inactive)"
for s in putseller spreadbot alpacabot callbuyer; do
    printf "%-12s %s\n" "$s" "$(systemctl is-enabled "$s" 2>/dev/null || echo not-installed)"
done
