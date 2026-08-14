#!/bin/bash
# ============================================================================
#  Oracle Cloud Free Tier — Bot Server Setup
#  Run once after creating your ARM instance.
#  Usage: chmod +x setup_server.sh && sudo ./setup_server.sh
# ============================================================================
set -euo pipefail

BOT_USER="botuser"
BOT_HOME="/home/${BOT_USER}"

echo "========================================="
echo "  Trading Bot Server Setup (ARM/aarch64)"
echo "========================================="

# --- System packages ---
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-venv python3-dev python3-full \
    python3-pip git curl htop tmux jq ufw

PYTHON=$(command -v python3)
echo "  Using Python: $PYTHON ($($PYTHON --version))"

# --- Create bot user ---
echo "[2/7] Creating bot user..."
if ! id "$BOT_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$BOT_USER"
    echo "  Created user: $BOT_USER"
else
    echo "  User $BOT_USER already exists"
fi

# --- Firewall (no inbound ports needed — bots only make outbound calls) ---
echo "[3/7] Configuring firewall..."
ufw default deny incoming > /dev/null
ufw default allow outgoing > /dev/null
ufw allow ssh > /dev/null
ufw --force enable > /dev/null
echo "  UFW: deny incoming, allow outgoing, allow SSH"

# --- Clone repository ---
echo "[4/7] Cloning bot repository (monorepo)..."
sudo -u "$BOT_USER" bash <<'CLONE_SCRIPT'
cd ~
REPO_URL="https://github.com/Jess08309/AI-Trading-Bot-Suite.git"
REPO_DIR="AI-Trading-Bot-Suite"

if [ -d "$REPO_DIR" ]; then
    echo "  $REPO_DIR: already cloned, pulling latest..."
    cd "$REPO_DIR" && git pull --ff-only && cd ~
else
    echo "  $REPO_DIR: cloning..."
    git clone "$REPO_URL" "$REPO_DIR"
fi
CLONE_SCRIPT

# --- Create virtual environments & install deps ---
echo "[5/7] Setting up Python virtual environments..."
PYTHON_PATH=$PYTHON
sudo -u "$BOT_USER" bash <<VENV_SCRIPT
cd ~/AI-Trading-Bot-Suite
for BOT in CryptoBot PutSeller CallBuyer AlpacaBot; do
    echo "  \$BOT: creating venv..."
    cd ~/AI-Trading-Bot-Suite/\$BOT
    $PYTHON_PATH -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt -q
    fi
    deactivate
    cd ~/AI-Trading-Bot-Suite
done
VENV_SCRIPT

# --- Create required directories ---
echo "[6/7] Creating data directories..."
sudo -u "$BOT_USER" bash <<'DIR_SCRIPT'
cd ~/AI-Trading-Bot-Suite

# CryptoBot directories (runtime lives under cryptotrades/)
mkdir -p ~/AI-Trading-Bot-Suite/CryptoBot/cryptotrades/logs
mkdir -p ~/AI-Trading-Bot-Suite/CryptoBot/cryptotrades/data/state
mkdir -p ~/AI-Trading-Bot-Suite/CryptoBot/cryptotrades/data/models
mkdir -p ~/AI-Trading-Bot-Suite/CryptoBot/data/state
mkdir -p ~/AI-Trading-Bot-Suite/CryptoBot/reports

# PutSeller directories
mkdir -p ~/AI-Trading-Bot-Suite/PutSeller/logs
mkdir -p ~/AI-Trading-Bot-Suite/PutSeller/data/state
mkdir -p ~/AI-Trading-Bot-Suite/PutSeller/reports

# CallBuyer directories
mkdir -p ~/AI-Trading-Bot-Suite/CallBuyer/logs
mkdir -p ~/AI-Trading-Bot-Suite/CallBuyer/data/state
mkdir -p ~/AI-Trading-Bot-Suite/CallBuyer/reports

# AlpacaBot directories
mkdir -p ~/AI-Trading-Bot-Suite/AlpacaBot/logs
mkdir -p ~/AI-Trading-Bot-Suite/AlpacaBot/data/state
mkdir -p ~/AI-Trading-Bot-Suite/AlpacaBot/reports
DIR_SCRIPT

# --- Install systemd services ---
echo "[7/7] Installing systemd services..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for SVC in cryptobot putseller callbuyer alpacabot bot-watchdog; do
    if [ -f "${SCRIPT_DIR}/${SVC}.service" ]; then
        cp "${SCRIPT_DIR}/${SVC}.service" /etc/systemd/system/
        echo "  Installed ${SVC}.service"
    fi
done

if [ -f "${SCRIPT_DIR}/bot-watchdog.timer" ]; then
    cp "${SCRIPT_DIR}/bot-watchdog.timer" /etc/systemd/system/
    echo "  Installed bot-watchdog.timer"
fi

systemctl daemon-reload

# Enable all services
systemctl enable cryptobot putseller callbuyer alpacabot bot-watchdog.timer

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "NEXT STEPS:"
echo "  1. Copy .env files to the server (run ./deploy.sh from the repo, or manually):"
echo "     scp AlpacaBot/.env                   botuser@<IP>:~/AI-Trading-Bot-Suite/AlpacaBot/.env"
echo "     scp PutSeller/.env                   botuser@<IP>:~/AI-Trading-Bot-Suite/PutSeller/.env"
echo "     scp CallBuyer/.env                   botuser@<IP>:~/AI-Trading-Bot-Suite/CallBuyer/.env"
echo "     scp CryptoBot/cryptotrades/.env      botuser@<IP>:~/AI-Trading-Bot-Suite/CryptoBot/cryptotrades/.env"
echo ""
echo "  2. Copy state files (paper balances, positions, etc.) — use deploy.sh -SyncState"
echo ""
echo "  3. Start the bots:"
echo "     sudo systemctl start cryptobot putseller callbuyer alpacabot"
echo "     sudo systemctl start bot-watchdog.timer"
echo ""
echo "  4. Check status:"
echo "     sudo systemctl status cryptobot putseller callbuyer alpacabot"
echo "     journalctl -u cryptobot -f   (live logs)"
echo ""
