# ============================================================================
#  Deploy CryptoBot to Oracle Cloud server from Windows
#  Usage: .\deploy.ps1 -ServerIP <IP> [-KeyFile <path>] [-SyncState] [-FirstDeploy]
#  Run from anywhere inside the repo (path is resolved relative to this script)
# ============================================================================
param(
    [Parameter(Mandatory=$true)]
    [string]$ServerIP,

    [string]$KeyFile = "$env:USERPROFILE\.ssh\oracle_bot_key",

    [switch]$SyncState,   # Also sync state/positions files
    [switch]$FirstDeploy  # Run full setup_server.sh
)

$ErrorActionPreference = "Stop"
$BOT_USER = "botuser"
$SSH = "ssh -i `"$KeyFile`" -o StrictHostKeyChecking=accept-new"
$SCP = "scp -i `"$KeyFile`" -o StrictHostKeyChecking=accept-new"

function Log($msg) { Write-Host "[deploy] $msg" -ForegroundColor Cyan }

# Resolve repo root (this script lives at <repo>\CryptoBot\deploy\oracle)
$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path

# ---------------------------------------------------------------------------
#  1. Push latest code via git (single monorepo)
# ---------------------------------------------------------------------------
Log "Pushing code updates to GitHub..."
Push-Location $RepoRoot
git add -A 2>$null
git diff --cached --quiet 2>$null
if ($LASTEXITCODE -ne 0) {
    git commit -m "deploy: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" 2>$null
}
git push origin main
Pop-Location

Log "Pulling latest on server..."
Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'cd ~/AI-Trading-Bot-Suite && git pull --ff-only'"

# ---------------------------------------------------------------------------
#  2. Sync .env files (secrets — NOT in git)
# ---------------------------------------------------------------------------
Log "Syncing .env files..."

$envFiles = @(
    @("$RepoRoot\CryptoBot\cryptotrades\.env", "AI-Trading-Bot-Suite/CryptoBot/cryptotrades/.env")
)

foreach ($pair in $envFiles) {
    $src = $pair[0]
    $dst = $pair[1]
    if (Test-Path $src) {
        Log "  $src -> ~/$dst"
        Invoke-Expression "$SCP `"$src`" ${BOT_USER}@${ServerIP}:~/$dst"
    }
}

# ---------------------------------------------------------------------------
#  3. Optionally sync state files (positions, balances, models)
# ---------------------------------------------------------------------------
if ($SyncState) {
    Log "Syncing state files..."

    $stateFiles = @(
        @("$RepoRoot\CryptoBot\cryptotrades\data\state\paper_balances.json", "AI-Trading-Bot-Suite/CryptoBot/cryptotrades/data/state/"),
        @("$RepoRoot\CryptoBot\cryptotrades\data\state\locked_profile.json", "AI-Trading-Bot-Suite/CryptoBot/cryptotrades/data/state/")
    )

    foreach ($pair in $stateFiles) {
        $src = $pair[0]
        $dst = $pair[1]
        if (Test-Path $src) {
            Log "  $src -> ~/$dst"
            Invoke-Expression "$SCP `"$src`" ${BOT_USER}@${ServerIP}:~/$dst"
        }
    }
}

# ---------------------------------------------------------------------------
#  4. First deploy: copy setup files & run setup
# ---------------------------------------------------------------------------
if ($FirstDeploy) {
    Log "First deploy — uploading setup files..."
    Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'mkdir -p ~/deploy'"

    $deployDir = "$RepoRoot\CryptoBot\deploy\oracle"
    Get-ChildItem "$deployDir\*" | ForEach-Object {
        Log "  $($_.Name)"
        Invoke-Expression "$SCP `"$($_.FullName)`" ${BOT_USER}@${ServerIP}:~/deploy/"
    }

    Log "Running setup_server.sh..."
    Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'chmod +x ~/deploy/*.sh && sudo ~/deploy/setup_server.sh'"
}

# ---------------------------------------------------------------------------
#  5. Enforce retired service cleanup + restart active services
# ---------------------------------------------------------------------------
Log "Stopping/disabling retired services (if present)..."
Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'for s in putseller spreadbot alpacabot callbuyer; do sudo systemctl stop `$s 2>/dev/null || true; sudo systemctl disable `$s 2>/dev/null || true; done'"

Log "Restarting active services..."
Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'sudo systemctl restart cryptobot 2>&1; sudo systemctl start bot-watchdog.timer 2>&1'"

Log "Checking status..."
Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'printf ""%-12s %s\n"" ""cryptobot"" ""`$(systemctl is-active cryptobot)""; printf ""%-12s %s\n"" ""watchdog"" ""`$(systemctl is-active bot-watchdog.timer)""; for s in putseller spreadbot alpacabot callbuyer; do printf ""%-12s %s\n"" `$s ""`$(systemctl is-enabled `$s 2>/dev/null || echo not-installed)""; done'"

Log "Deploy complete!"
