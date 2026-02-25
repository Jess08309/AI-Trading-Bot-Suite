# TRADE EXIT ANALYZER - Shows why trades were closed

function Resolve-StateDir {
    $local = Join-Path $PSScriptRoot "data\state"
    $external = "D:\042021\CryptoBot\data\state"

    if (Test-Path (Join-Path $external "trade_transparency.json")) {
        return $external
    }
    return $local
}

$StateDir = Resolve-StateDir
$tradesFile = Join-Path $StateDir "trade_transparency.json"

if (-not (Test-Path $tradesFile)) {
    Write-Host "[!] No trade history found. Bot hasn't closed any trades yet." -ForegroundColor Yellow
    exit
}

$allTrades = Get-Content $tradesFile | ConvertFrom-Json
$closedTrades = $allTrades | Where-Object { $_.status -eq "CLOSED" } | Sort-Object exit_time -Descending

if ($closedTrades.Count -eq 0) {
    Write-Host "[!] No closed trades found" -ForegroundColor Yellow
    exit
}

Write-Host "" -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta
Write-Host " EXIT REASON BREAKDOWN" -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta

# Count exit reasons
$exitReasons = @{}
foreach ($trade in $closedTrades) {
    $reason = $trade.exit_reason
    if (-not $exitReasons.ContainsKey($reason)) {
        $exitReasons[$reason] = @{ Count = 0; WinCount = 0; TotalPnL = 0.0 }
    }
    $exitReasons[$reason].Count++
    if ($trade.pnl_usd -gt 0) { $exitReasons[$reason].WinCount++ }
    $exitReasons[$reason].TotalPnL += $trade.pnl_usd
}

Write-Host "`nExit Reason Summary:" -ForegroundColor Cyan
foreach ($reason in ($exitReasons.Keys | Sort-Object)) {
    $stats = $exitReasons[$reason]
    $winRate = if ($stats.Count -gt 0) { ($stats.WinCount / $stats.Count) * 100 } else { 0 }
    
    $color = if ($stats.TotalPnL -gt 0) { "Green" } elseif ($stats.TotalPnL -lt 0) { "Red" } else { "Gray" }
    
    Write-Host "`n$reason" -ForegroundColor Yellow
    $winRateText = $winRate.ToString("N1") + "% win rate"
    Write-Host "  Trades: $($stats.Count) | Wins: $($stats.WinCount) ($winRateText)" -ForegroundColor White
    $pnlText = '$' + $stats.TotalPnL.ToString("N2")
    Write-Host "  Total P/L: $pnlText" -ForegroundColor $color
}

Write-Host "`n`n============================================" -ForegroundColor Magenta
Write-Host " RECENT EXITS (Last 15)" -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta

$recentExits = $closedTrades | Select-Object -First 15

foreach ($trade in $recentExits) {
    $entryTime = [datetime]::Parse($trade.entry_time)
    $exitTime = [datetime]::Parse($trade.exit_time)
    
    $pnlColor = if ($trade.pnl_usd -gt 0) { "Green" } elseif ($trade.pnl_usd -lt 0) { "Red" } else { "Gray" }
    
    $reasonColor = switch ($trade.exit_reason) {
        "TAKE_PROFIT" { "Green" }
        "STOP_LOSS" { "Red" }
        "TRAILING_STOP" { "Yellow" }
        "ML_SIGNAL" { "Cyan" }
        default { "White" }
    }
    
    Write-Host "`n-------------------------------------------" -ForegroundColor DarkGray
    Write-Host "$($trade.symbol) $($trade.direction)" -ForegroundColor Cyan -NoNewline
    $pnlUsd = '$' + $trade.pnl_usd.ToString("N2")
    $pnlPct = $trade.pnl_pct.ToString("N2") + '%'
    Write-Host " | P/L: $pnlUsd ($pnlPct)" -ForegroundColor $pnlColor
    
    Write-Host "Exit Reason: $($trade.exit_reason)" -ForegroundColor $reasonColor
    
    $entryPrice = '$' + $trade.entry_price.ToString("N2")
    $exitPrice = '$' + $trade.exit_price.ToString("N2")
    $entryTimeStr = $entryTime.ToString("HH:mm:ss")
    $exitTimeStr = $exitTime.ToString("HH:mm:ss")
    $holdMin = $trade.hold_minutes.ToString("N0")
    
    Write-Host "Entry: $entryPrice at $entryTimeStr" -ForegroundColor Gray
    Write-Host " Exit: $exitPrice at $exitTimeStr" -ForegroundColor Gray
    Write-Host "  Hold: $holdMin minutes" -ForegroundColor Gray
    
    # Entry signal details
    Write-Host "Entry Signal:" -ForegroundColor White
    $conf = ($trade.signal.confidence * 100).ToString("N1") + '%'
    $vol = $trade.signal.volatility.ToString("N3")
    $rsi = $trade.signal.rsi.ToString("N0")
    Write-Host "  Confidence: $conf | Volatility: $vol | RSI: $rsi" -ForegroundColor Gray
    
    if ($trade.signal.regime) {
        $corr = $trade.signal.correlation.ToString("N2")
        Write-Host "  Regime: $($trade.signal.regime) | Correlation: $corr" -ForegroundColor Gray
    }
    
    if ($trade.signal.multiplier -gt 1) {
        Write-Host "  > SCALED POSITION: $($trade.signal.multiplier)x size (Quality Score: $($trade.signal.score)/6)" -ForegroundColor Magenta
    }
}

Write-Host "`n============================================" -ForegroundColor Gray
Write-Host "`nTip: Run TRANSPARENCY_DASHBOARD.ps1 for full performance analysis" -ForegroundColor Yellow
Write-Host ""
