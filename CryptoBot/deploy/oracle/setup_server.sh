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
    python3.11 python3.11-venv python3.11-dev \
    python3-pip git curl htop tmux jq ufw \
    > /dev/null 2>&1

# Use python3.11 as default if available
if command -v python3.11 &>/dev/null; then
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 2>/dev/null || true
fi

PYTHON=$(command -v python3.11 || command -v python3)
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

# --- Clone repositories ---
echo "[4/7] Cloning bot repositories..."
sudo -u "$BOT_USER" bash <<'CLONE_SCRIPT'
cd ~

declare -A REPOS
REPOS[CryptoBot]="https://github.com/Jess08309/CryptoBot-Updated-20260321.git"
REPOS[PutSeller]="https://github.com/Jess08309/PutSeller.git"
REPOS[CallBuyer]="https://github.com/Jess08309/CallBuyer.git"
REPOS[AlpacaBot]="https://github.com/Jess08309/AlpacaBot.git"

for BOT in "${!REPOS[@]}"; do
    if [ -d "$BOT" ]; then
        echo "  $BOT: already cloned, pulling latest..."
        cd "$BOT" && git pull --ff-only && cd ~
    else
        echo "  $BOT: cloning..."
        git clone "${REPOS[$BOT]}" "$BOT"
    fi
done
CLONE_SCRIPT

# --- Create virtual environments & install deps ---
echo "[5/7] Setting up Python virtual environments..."
PYTHON_PATH=$PYTHON
sudo -u "$BOT_USER" bash <<VENV_SCRIPT
cd ~
for BOT in CryptoBot PutSeller CallBuyer AlpacaBot; do
    echo "  \$BOT: creating venv..."
    cd ~/\$BOT
    $PYTHON_PATH -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt -q
    fi
    deactivate
    cd ~
done
VENV_SCRIPT

# --- Create required directories ---
echo "[6/7] Creating data directories..."
sudo -u "$BOT_USER" bash <<'DIR_SCRIPT'
cd ~

# CryptoBot directories
mkdir -p ~/CryptoBot/logs
mkdir -p ~/CryptoBot/data/state
mkdir -p ~/CryptoBot/data/models
mkdir -p ~/CryptoBot/reports

# PutSeller directories
mkdir -p ~/PutSeller/logs
mkdir -p ~/PutSeller/data/state
mkdir -p ~/PutSeller/reports

# CallBuyer directories
mkdir -p ~/CallBuyer/logs
mkdir -p ~/CallBuyer/data/state
mkdir -p ~/CallBuyer/reports

# AlpacaBot directories
mkdir -p ~/AlpacaBot/logs
mkdir -p ~/AlpacaBot/data/state
mkdir -p ~/AlpacaBot/reports
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
echo "  1. Copy .env files to the server:"
echo "     scp C:\\Bot\\.env         botuser@<IP>:~/CryptoBot/.env"
echo "     scp C:\\Bot\\cryptotrades\\.env botuser@<IP>:~/CryptoBot/cryptotrades/.env"
echo "     scp C:\\PutSeller\\.env   botuser@<IP>:~/PutSeller/.env"
echo "     scp C:\\CallBuyer\\.env   botuser@<IP>:~/CallBuyer/.env"
echo "     scp C:\\AlpacaBot\\.env   botuser@<IP>:~/AlpacaBot/.env"
echo ""
echo "  2. Copy state files (paper balances, positions, etc.):"
echo "     Use deploy.ps1 from your Windows machine"
echo ""
echo "  3. Start the bots:"
echo "     sudo systemctl start cryptobot putseller callbuyer"
echo "     sudo systemctl start bot-watchdog.timer"
echo ""
echo "  4. Check status:"
echo "     sudo systemctl status cryptobot putseller callbuyer alpacabot"
echo "     journalctl -u cryptobot -f   (live logs)"
echo ""
