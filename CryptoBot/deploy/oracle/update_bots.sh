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
sudo /home/botuser/AI-Trading-Bot-Suite/CryptoBot/deploy/oracle/cleanup_retired_units.sh

echo ""
echo "=== Restarting active services ==="
sudo systemctl restart cryptobot
sudo systemctl restart bot-watchdog.timer

echo ""
echo "=== Status ==="
printf "%-12s %s\n" "cryptobot" "$(systemctl show -p ActiveState --value cryptobot 2>/dev/null || echo not-installed)"
printf "%-12s %s\n" "watchdog" "$(systemctl show -p ActiveState --value bot-watchdog.timer 2>/dev/null || echo not-installed)"
for s in putseller spreadbot alpacabot callbuyer; do
    printf "%-12s %s\n" "$s" "$(systemctl is-enabled "$s" 2>/dev/null || echo not-installed)"
done
