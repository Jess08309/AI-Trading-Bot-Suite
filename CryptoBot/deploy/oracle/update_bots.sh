#!/bin/bash
# ============================================================================
#  Quick update — pull latest code and restart bots
#  Run from the server: ~/deploy/update_bots.sh
# ============================================================================
set -euo pipefail

echo "=== Pulling latest code ==="
for BOT in CryptoBot PutSeller CallBuyer AlpacaBot; do
    echo "  $BOT..."
    cd "/home/botuser/$BOT"
    git pull --ff-only
    
    # Reinstall deps if requirements.txt changed
    if git diff HEAD~1 --name-only 2>/dev/null | grep -q requirements.txt; then
        echo "    requirements.txt changed — reinstalling..."
        source .venv/bin/activate
        pip install -r requirements.txt -q
        deactivate
    fi
    cd ~
done

echo ""
echo "=== Restarting services ==="
sudo systemctl restart cryptobot putseller callbuyer

echo ""
echo "=== Status ==="
for s in cryptobot putseller callbuyer alpacabot; do
    printf "%-12s %s\n" "$s" "$(systemctl is-active $s 2>/dev/null)"
done
