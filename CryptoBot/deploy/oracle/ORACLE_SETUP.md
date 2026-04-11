# Oracle Cloud Free Tier — Bot Deployment Guide

## What You Get (Free Forever)
- **ARM instance**: 4 OCPU (cores), 24 GB RAM, 200 GB disk
- More than enough for all 4 bots (they use ~200MB RAM each)
- No credit card charges — truly free tier (card required for signup verification only)

---

## Step 1: Create Oracle Cloud Account

1. Go to **https://cloud.oracle.com/sign-up**
2. Sign up with email, set a password
3. **Home Region**: Choose `us-phoenix-1` or `us-ashburn-1` (closest to you in Sparks, NV — Phoenix is ~700 miles, Ashburn ~2,500)
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
2. Click **"Create a VM instance"** (or: Menu → Compute → Instances → Create Instance)

3. **Name**: `trading-bots`

4. **Image and shape**:
   - Click **"Edit"** next to Shape
   - Select **Ampere** tab → **VM.Standard.A1.Flex**
   - Set: **4 OCPUs**, **24 GB RAM** (max free tier)
   - Image: **Canonical Ubuntu 22.04** (or 24.04 if available)

5. **Networking**:
   - Use default VCN or create new
   - **Assign a public IPv4 address**: YES
   - Subnet: public subnet

6. **Add SSH keys**:
   - Select **"Paste public keys"**
   - Paste the key you copied in Step 2

7. Click **Create**

8. Wait 2-5 minutes for RUNNING status
9. **Copy the Public IP Address** shown on the instance page

> **Note**: ARM free tier instances are in high demand. If you get "Out of capacity", 
> try again in a few hours or try a different availability domain.

---

## Step 4: First Connection Test

```powershell
ssh -i "$env:USERPROFILE\.ssh\oracle_bot_key" ubuntu@<YOUR_SERVER_IP>
```

You should get a shell. Type `exit` to disconnect.

---

## Step 5: Create botuser & Deploy

From your Windows machine, SSH in and create the bot user:

```powershell
# SSH as ubuntu (the default user has sudo)
ssh -i "$env:USERPROFILE\.ssh\oracle_bot_key" ubuntu@<YOUR_SERVER_IP>
```

On the server:
```bash
# Create botuser and allow SSH
sudo useradd -m -s /bin/bash botuser
sudo mkdir -p /home/botuser/.ssh
sudo cp ~/.ssh/authorized_keys /home/botuser/.ssh/
sudo chown -R botuser:botuser /home/botuser/.ssh
sudo chmod 700 /home/botuser/.ssh
sudo chmod 600 /home/botuser/.ssh/authorized_keys
exit
```

Test botuser login:
```powershell
ssh -i "$env:USERPROFILE\.ssh\oracle_bot_key" botuser@<YOUR_SERVER_IP>
```

---

## Step 6: Run the Deploy Script

On your Windows machine:

```powershell
cd C:\Bot\deploy\oracle
.\deploy.ps1 -ServerIP <YOUR_SERVER_IP> -FirstDeploy -SyncState
```

This will:
1. Push your latest code to GitHub
2. Pull it on the server
3. Copy .env files (API keys)
4. Copy state files (positions, balances)
5. Run setup_server.sh (install Python, create venvs, install deps, enable services)
6. Start all bots

---

## Step 7: Verify Everything is Running

```powershell
ssh -i "$env:USERPROFILE\.ssh\oracle_bot_key" botuser@<YOUR_SERVER_IP>
```

On the server:
```bash
# Check service status
sudo systemctl status cryptobot putseller callbuyer

# Watch live logs
journalctl -u cryptobot -f           # CryptoBot logs
journalctl -u putseller -f           # PutSeller logs
journalctl -u callbuyer -f           # CallBuyer logs

# Check watchdog timer
sudo systemctl status bot-watchdog.timer

# Check resource usage
htop
```

---

## Day-to-Day Operations

### Push code updates:
```powershell
.\deploy.ps1 -ServerIP <YOUR_SERVER_IP>
```

### Pull code updates on server:
```bash
~/deploy/update_bots.sh
```

### Stop/start individual bot:
```bash
sudo systemctl stop putseller
sudo systemctl start putseller
```

### View recent logs:
```bash
journalctl -u cryptobot --since "1 hour ago"
journalctl -u putseller --since today
```

### Check if bots survived a reboot:
```bash
# systemd auto-starts them on boot — just verify:
for s in cryptobot putseller callbuyer; do
    printf "%-12s %s\n" "$s" "$(systemctl is-active $s)"
done
```

---

## Oracle Cloud Security Group (already configured)

The bots only make **outbound** HTTPS connections (to Kraken, Alpaca APIs).
No inbound ports are needed except SSH (port 22).

The default Oracle security list allows:
- **Inbound**: SSH (22) only
- **Outbound**: All traffic

This is exactly what we need. Don't open additional ports.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Out of capacity" on instance creation | Try different Availability Domain, or wait a few hours |
| Bot crashes immediately | `journalctl -u cryptobot -n 50` to see error |
| .env not found | Re-run `.\deploy.ps1 -ServerIP <IP>` (syncs .env files) |
| pip install fails on ARM | All deps are pure Python — should work. Check `journalctl` for errors |
| SSH connection refused | Check Oracle Security List allows port 22 from your IP |
| Instance stopped by Oracle | Only if you have a paid account and run out of credits. Free tier instances run forever |
