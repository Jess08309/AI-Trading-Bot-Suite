$ErrorActionPreference = "SilentlyContinue"

function Resolve-BotRoot {
    if ($env:BOT_ROOT -and (Test-Path $env:BOT_ROOT)) {
        return $env:BOT_ROOT
    }
    return "D:\042021\CryptoBot"
}

function Resolve-StateDir {
    param([string]$BotRoot)

    $external = Join-Path $BotRoot "data\state"
    if (Test-Path (Join-Path $external "positions.json")) {
        return $external
    }

    $local = Join-Path $PSScriptRoot "data\state"
    return $local
}

function Resolve-LogFile {
    param([string]$BotRoot)

    $logDir = Join-Path $BotRoot "logs"
    if (Test-Path $logDir) {
        $latest = Get-ChildItem $logDir -Filter "trading_*.log" -File |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($latest) { return $latest.FullName }
    }

    $fallback = Join-Path $PSScriptRoot "logs\bot_output.log"
    return $fallback
}

function Get-BotProcess {
    Get-Process python -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -like "*CryptoBot*.venv*python.exe" } |
        Select-Object -First 1
}

function Show-OpenPositions {
    param([string]$StateDir)

    $positionsPath = Join-Path $StateDir "positions.json"
    if (-not (Test-Path $positionsPath)) {
        Write-Host "Open Positions: no positions file" -ForegroundColor DarkGray
        return
    }

    $positions = Get-Content $positionsPath | ConvertFrom-Json
    $count = ($positions.PSObject.Properties | Measure-Object).Count
    Write-Host ("Open Positions: {0}" -f $count) -ForegroundColor Yellow

    if ($count -eq 0) {
        Write-Host "  None" -ForegroundColor Gray
        return
    }

    foreach ($prop in $positions.PSObject.Properties) {
        $symbol = $prop.Name
        $pos = $prop.Value
        $reason = if ($pos.entry_reason) { $pos.entry_reason } else { "n/a" }
        $line = "  {0} {1} | entry={2:N4} | reason={3}" -f $pos.direction, $symbol, [double]$pos.entry_price, $reason
        Write-Host $line -ForegroundColor White
    }
}

function Show-DecisionTrail {
    param([string]$LogFile)

    if (-not (Test-Path $LogFile)) {
        Write-Host "No decision log found: $LogFile" -ForegroundColor DarkGray
        return
    }

    $patterns = @(
        "\[TRADE\]",
        "\[RISK\]",
        "Skip ",
        "Sentiment status:",
        "Futures history points:",
        "Model not ready",
        "Model training trigger",
        "Model is now ready",
        "Total:",
        "Shutdown signal received"
    )

    $regex = [string]::Join("|", $patterns)
    $lines = Select-String -Path $LogFile -Pattern $regex | Select-Object -Last 35

    foreach ($match in $lines) {
        $line = $match.Line
        $color = "Gray"
        if ($line -match "\[TRADE\]") { $color = "Cyan" }
        elseif ($line -match "Skip ") { $color = "Red" }
        elseif ($line -match "Sentiment status:") { $color = "Magenta" }
        elseif ($line -match "Total:") { $color = "Green" }
        elseif ($line -match "Shutdown signal received") { $color = "Yellow" }
        Write-Host $line -ForegroundColor $color
    }
}

$botRoot = Resolve-BotRoot
$stateDir = Resolve-StateDir -BotRoot $botRoot
$logFile = Resolve-LogFile -BotRoot $botRoot

while ($true) {
    Clear-Host
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $proc = Get-BotProcess

    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host "        MASTER CHESS BOT - DECISION TRACE DASHBOARD" -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host "Updated: $ts" -ForegroundColor DarkGray
    Write-Host "Bot root: $botRoot" -ForegroundColor DarkGray
    Write-Host "State dir: $stateDir" -ForegroundColor DarkGray
    Write-Host "Log file: $logFile" -ForegroundColor DarkGray

    if ($proc) {
        Write-Host ("Process: RUNNING (PID {0}, started {1})" -f $proc.Id, $proc.StartTime) -ForegroundColor Green
    }
    else {
        Write-Host "Process: STOPPED" -ForegroundColor Red
    }

    Write-Host ""
    Show-OpenPositions -StateDir $stateDir
    Write-Host ""
    Write-Host "Recent Decision Trail:" -ForegroundColor Yellow
    Show-DecisionTrail -LogFile $logFile

    Write-Host ""
    Write-Host "Refreshing every 5 seconds... Press Ctrl+C to stop" -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}
