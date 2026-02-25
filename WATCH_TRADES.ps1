# Live Trading Monitor - Shows active trades, exits, and P/L updates
# Run this in a separate PowerShell window while bot is running

function Resolve-LogPath {
    $candidates = @()

    $externalLogs = "D:\042021\CryptoBot\logs"
    if (Test-Path $externalLogs) {
        $latestExternalTrading = Get-ChildItem $externalLogs -Filter "trading_*.log" -File |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($latestExternalTrading) { $candidates += $latestExternalTrading.FullName }
        $candidates += (Join-Path $externalLogs "bot_output.log")
    }

    $localLogs = Join-Path $PSScriptRoot "logs"
    if (Test-Path $localLogs) {
        $latestLocalTrading = Get-ChildItem $localLogs -Filter "trading_*.log" -File |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($latestLocalTrading) { $candidates += $latestLocalTrading.FullName }
        $candidates += (Join-Path $localLogs "bot_output.log")
    }

    $candidates += (Join-Path $PSScriptRoot "bot_output.log")

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-Path $candidate) { return $candidate }
    }

    return $null
}

$logPath = Resolve-LogPath
$colors = @{
    'BUY' = 'Green'
    'SELL' = 'Yellow'
    'CLOSE' = 'Cyan'
    'PROFIT' = 'Green'
    'STOP_LOSS' = 'Red'
    'PAPER:' = 'White'
    'P/L:' = 'Magenta'
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LIVE TRADING MONITOR - Master Chess  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Watching for:" -ForegroundColor Yellow
Write-Host "  - Balance updates (every 60s)" -ForegroundColor Gray
Write-Host "  - Trade entries (BUY/LONG/SHORT)" -ForegroundColor Gray
Write-Host "  - Trade exits (SELL/CLOSE)" -ForegroundColor Gray
Write-Host "  - P/L notifications" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop watching" -ForegroundColor Yellow
Write-Host ""

# Monitor log file in real-time
if (-not (Test-Path $logPath)) {
    Write-Host "Log file not found: $logPath" -ForegroundColor Yellow
    Write-Host "If the bot is running elsewhere, ensure logs/trading_YYYYMMDD.log exists." -ForegroundColor Gray
    exit
}

Write-Host "Using log file: $logPath" -ForegroundColor DarkGray

Get-Content $logPath -Wait -Tail 0 | ForEach-Object {
    $line = $_
    
    # Filter for important events
    if ($line -match 'PAPER: Spot|BUY |SELL |<< |>> |LONG |SHORT |CLOSE |PROFIT|STOP_LOSS|P/L:') {
        
        # Color code based on content
        $color = 'White'
        foreach ($key in $colors.Keys) {
            if ($line -match $key) {
                $color = $colors[$key]
                break
            }
        }
        
        # Extract timestamp and message
        if ($line -match '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*? - (?:INFO|WARNING) - (.+)') {
            $timestamp = $matches[1]
            $message = $matches[2]
            
            # Format output
            Write-Host "[$timestamp] " -NoNewline -ForegroundColor DarkGray
            Write-Host $message -ForegroundColor $color
        } else {
            Write-Host $line -ForegroundColor $color
        }
    }
}
