#!/bin/bash
# ============================================================================
#  Quick update — pull latest code and restart bots
#  Run from the server: ~/deploy/update_bots.sh
# ============================================================================
set -euo pipefail

echo "=== Pulling latest code ==="
cd "/home/botuser/AI-Trading-Bot-Suite"
git pull --ff-only
CHANGED_FILES=$(git diff HEAD@{1} --name-only 2>/dev/null || true)

for BOT in CryptoBot PutSeller CallBuyer AlpacaBot; do
    if echo "$CHANGED_FILES" | grep -q "^${BOT}/requirements.txt$"; then
        echo "  ${BOT}/requirements.txt changed — reinstalling..."
        cd "/home/botuser/AI-Trading-Bot-Suite/${BOT}"
        source .venv/bin/activate
        pip install -r requirements.txt -q
        deactivate
        cd "/home/botuser/AI-Trading-Bot-Suite"
    fi
done

echo ""
echo "=== Restarting services ==="
sudo systemctl restart cryptobot putseller callbuyer alpacabot

echo ""
echo "=== Status ==="
for s in cryptobot putseller callbuyer alpacabot; do
    printf "%-12s %s\n" "$s" "$(systemctl is-active $s 2>/dev/null)"
done
