# Oracle Cloud Free Tier — CryptoBot Deployment Guide

## What You Get (Free Forever)
- **ARM instance**: 4 OCPU (cores), 24 GB RAM, 200 GB disk
- More than enough for CryptoBot + watchdog
- No credit card charges — truly free tier (card required for signup verification only)

---

## Step 1: Create Oracle Cloud Account

1. Go to **https://cloud.oracle.com/sign-up**
2. Sign up with email, set a password
3. **Home Region**: Choose `us-phoenix-1` or `us-ashburn-1`
4. Add a credit card for verification (you will NOT be charged)
5. Wait for account activation (~5 minutes)

---

## Step 2: Generate SSH Key Pair (on your Windows machine)

Open PowerShell and run:
```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\oracle_bot_key" -N '""'
```

This creates:
- Private key: `~\.ssh\oracle_bot_key` (keep secret)
- Public key: `~\.ssh\oracle_bot_key.pub` (paste into Oracle)

Copy the public key to clipboard:
```powershell
Get-Content "$env:USERPROFILE\.ssh\oracle_bot_key.pub" | Set-Clipboard
```

---

## Step 3: Create ARM Compute Instance

1. Log into **https://cloud.oracle.com**
2. Click **"Create a VM instance"**
3. **Name**: `trading-bot`
4. Use shape **VM.Standard.A1.Flex** with **4 OCPUs** and **24 GB RAM**
5. Use Ubuntu 22.04 or 24.04 image
6. Assign a public IPv4 address
7. Paste your SSH public key
8. Click **Create**, wait until running, copy the public IP

---

## Step 4: First Connection Test

```powershell
ssh -i "$env:USERPROFILE\.ssh\oracle_bot_key" ubuntu@<YOUR_SERVER_IP>
```

---

## Step 5: Create botuser

```powershell
ssh -i "$env:USERPROFILE\.ssh\oracle_bot_key" ubuntu@<YOUR_SERVER_IP>
```

On the server:
```bash
sudo useradd -m -s /bin/bash botuser
sudo mkdir -p /home/botuser/.ssh
sudo cp ~/.ssh/authorized_keys /home/botuser/.ssh/
sudo chown -R botuser:botuser /home/botuser/.ssh
sudo chmod 700 /home/botuser/.ssh
sudo chmod 600 /home/botuser/.ssh/authorized_keys
exit
```

---

## Step 6: Run Deploy Script

From **Windows** (PowerShell):
```powershell
cd C:\path\to\AI-Trading-Bot-Suite\CryptoBot\deploy\oracle
.\deploy.ps1 -ServerIP <YOUR_SERVER_IP> -FirstDeploy -SyncState
```

From **Linux/macOS/Codespace**:
```bash
cd /workspaces/AI-Trading-Bot-Suite/CryptoBot/deploy/oracle
./deploy.sh <YOUR_SERVER_IP> --first-deploy --sync-state
```

Deploy scripts:
1. Push latest code to GitHub
2. Pull latest on server
3. Copy CryptoBot `.env`
4. Copy CryptoBot state files (optional)
5. Run `setup_server.sh`
6. Restart `cryptobot` and `bot-watchdog.timer`

---

## Step 7: Verify Runtime Services

```powershell
ssh -i "$env:USERPROFILE\.ssh\oracle_bot_key" botuser@<YOUR_SERVER_IP>
```

On the server:
```bash
sudo systemctl status cryptobot
sudo systemctl status bot-watchdog.timer
journalctl -u cryptobot -f
```

To verify retired services are gone:
```bash
for s in putseller spreadbot alpacabot callbuyer; do
  echo "$s: $(systemctl is-enabled "$s" 2>/dev/null || echo not-installed)"
done
```

---

## Day-to-Day Operations

Push updates:
```powershell
.\deploy.ps1 -ServerIP <YOUR_SERVER_IP>
```

Update directly on server:
```bash
~/deploy/update_bots.sh
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Bot crashes immediately | `journalctl -u cryptobot -n 50` |
| .env not found | Re-run deploy script to sync `CryptoBot/cryptotrades/.env` |
| SSH connection refused | Confirm Oracle security list allows port 22 |
