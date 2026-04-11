# ============================================================================
#  Deploy bots to Oracle Cloud server from Windows
#  Usage: .\deploy.ps1 -ServerIP <IP> [-KeyFile <path>] [-SyncState]
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

# ---------------------------------------------------------------------------
#  1. Push latest code via git (repos are already cloned on server)
# ---------------------------------------------------------------------------
Log "Pushing code updates to GitHub..."

$repos = @{
    "C:\Bot"        = "CryptoBot"
    "C:\PutSeller"  = "PutSeller"
    "C:\CallBuyer"  = "CallBuyer"
    "C:\AlpacaBot"  = "AlpacaBot"
}

foreach ($local in $repos.Keys) {
    $remote = $repos[$local]
    Log "  $remote : git push..."
    Push-Location $local
    git add -A 2>$null
    $hasChanges = git diff --cached --quiet 2>$null; $LASTEXITCODE -ne 0
    if ($hasChanges) {
        git commit -m "deploy: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" 2>$null
    }
    git push origin main 2>$null
    Pop-Location
}

Log "Pulling latest on server..."
Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'for d in CryptoBot PutSeller CallBuyer AlpacaBot; do cd ~/`$d && git pull --ff-only 2>&1; cd ~; done'"

# ---------------------------------------------------------------------------
#  2. Sync .env files (secrets — NOT in git)
# ---------------------------------------------------------------------------
Log "Syncing .env files..."

$envFiles = @(
    @("C:\Bot\.env",              "CryptoBot/.env"),
    @("C:\Bot\cryptotrades\.env", "CryptoBot/cryptotrades/.env"),
    @("C:\PutSeller\.env",        "PutSeller/.env"),
    @("C:\CallBuyer\.env",        "CallBuyer/.env"),
    @("C:\AlpacaBot\.env",        "AlpacaBot/.env")
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
        @("C:\Bot\data\state\paper_balances.json",    "CryptoBot/data/state/"),
        @("C:\Bot\data\state\locked_profile.json",    "CryptoBot/data/state/"),
        @("C:\PutSeller\data\state\positions.json",   "PutSeller/data/state/"),
        @("C:\PutSeller\data\state\bot_state.json",   "PutSeller/data/state/"),
        @("C:\CallBuyer\data\state\bot_state.json",   "CallBuyer/data/state/"),
        @("C:\AlpacaBot\data\state\bot_state.json",   "AlpacaBot/data/state/")
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

    $deployDir = "C:\Bot\deploy\oracle"
    Get-ChildItem "$deployDir\*" | ForEach-Object {
        Log "  $($_.Name)"
        Invoke-Expression "$SCP `"$($_.FullName)`" ${BOT_USER}@${ServerIP}:~/deploy/"
    }

    Log "Running setup_server.sh..."
    Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'chmod +x ~/deploy/*.sh && sudo ~/deploy/setup_server.sh'"
}

# ---------------------------------------------------------------------------
#  5. Restart services
# ---------------------------------------------------------------------------
Log "Restarting bot services..."
Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'sudo systemctl restart cryptobot putseller callbuyer 2>&1; sudo systemctl start bot-watchdog.timer 2>&1'"

Log "Checking status..."
Invoke-Expression "$SSH ${BOT_USER}@${ServerIP} 'for s in cryptobot putseller callbuyer alpacabot; do printf ""%-12s %s\n"" `$s `$(systemctl is-active `$s); done'"

Log "Deploy complete!"
