# TRANSPARENCY DASHBOARD - View trade history, exit reasons, and performance stats
# Shows complete transparency into bot's decision-making

$ErrorActionPreference = "SilentlyContinue"

function Resolve-StateDir {
    $local = Join-Path $PSScriptRoot "data\state"
    $external = "D:\042021\CryptoBot\data\state"

    if (Test-Path (Join-Path $external "trade_transparency.json")) {
        return $external
    }
    return $local
}

$StateDir = Resolve-StateDir

function Get-ColorForPnL {
    param([double]$value)
    if ($value > 0) { return "Green" }
    elseif ($value < 0) { return "Red" }
    else { return "Gray" }
}

function Show-DailySummary {
    $summaryFile = Join-Path $StateDir "daily_summary.json"
    if (-not (Test-Path $summaryFile)) {
        Write-Host "`n[!] No daily summary data found" -ForegroundColor Yellow
        return
    }
    
    $summary = Get-Content $summaryFile | ConvertFrom-Json
    
    Write-Host "`n╔═══════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host " DAILY SUMMARY - $($summary.date)" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════╝" -ForegroundColor Cyan
    
    Write-Host "`nOverall Performance:" -ForegroundColor Yellow
    Write-Host "  Total Trades: $($summary.total_trades)" -ForegroundColor White
    Write-Host "  Wins: $($summary.wins) | Losses: $($summary.losses)" -ForegroundColor White
    Write-Host ("  Win Rate: {0:N1}%" -f $summary.win_rate) -ForegroundColor White
    
    $pnlColor = Get-ColorForPnL $summary.total_pnl_usd
    Write-Host ("  Total P/L: `${0:N2} ({1:N2}%)" -f $summary.total_pnl_usd, $summary.total_pnl_pct) -ForegroundColor $pnlColor
    
    if ($summary.wins -gt 0) {
        Write-Host ("  Avg Win: {0:N2}%" -f $summary.avg_win) -ForegroundColor Green
    }
    if ($summary.losses -gt 0) {
        Write-Host ("  Avg Loss: {0:N2}%" -f $summary.avg_loss) -ForegroundColor Red
    }
    
    Write-Host "`nBest Trade:" -ForegroundColor Green
    if ($summary.best_trade) {
        Write-Host ("  {0}: `${1:N2} ({2:N2}%) - {3}" -f `
            $summary.best_trade.symbol, `
            $summary.best_trade.pnl_usd, `
            $summary.best_trade.pnl_pct, `
            $summary.best_trade.exit_reason) -ForegroundColor Green
    }
    
    Write-Host "`nWorst Trade:" -ForegroundColor Red
    if ($summary.worst_trade) {
        Write-Host ("  {0}: `${1:N2} ({2:N2}%) - {3}" -f `
            $summary.worst_trade.symbol, `
            $summary.worst_trade.pnl_usd, `
            $summary.worst_trade.pnl_pct, `
            $summary.worst_trade.exit_reason) -ForegroundColor Red
    }
    
    Write-Host "`nExit Reasons:" -ForegroundColor Yellow
    $summary.exit_reasons.PSObject.Properties | ForEach-Object {
        Write-Host ("  {0}: {1} trades" -f $_.Name, $_.Value) -ForegroundColor White
    }
}

function Show-SymbolPerformance {
    $symbolFile = Join-Path $StateDir "symbol_performance.json"
    if (-not (Test-Path $symbolFile)) {
        Write-Host "`n[!] No symbol performance data found" -ForegroundColor Yellow
        return
    }
    
    $symbols = Get-Content $symbolFile | ConvertFrom-Json
    
    Write-Host "`n╔═══════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host " PER-SYMBOL PERFORMANCE" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════╝" -ForegroundColor Cyan
    
    $symbols.PSObject.Properties | Sort-Object {$_.Value.total_pnl_usd} -Descending | ForEach-Object {
        $symbol = $_.Name
        $stats = $_.Value
        
        $pnlColor = Get-ColorForPnL $stats.total_pnl_usd
        
        Write-Host "`n$symbol" -ForegroundColor Cyan
        Write-Host ("  Trades: {0} (W:{1} L:{2}) | Win Rate: {3:N1}%" -f `
            $stats.total_trades, $stats.wins, $stats.losses, $stats.win_rate) -ForegroundColor White
        Write-Host ("  Total P/L: `${0:N2} ({1:N2}%)" -f $stats.total_pnl_usd, $stats.total_pnl_pct) -ForegroundColor $pnlColor
        Write-Host ("  Avg Win: {0:N2}% | Avg Loss: {1:N2}%" -f $stats.avg_win_pct, $stats.avg_loss_pct) -ForegroundColor White
        Write-Host ("  Best: {0:N2}% | Worst: {1:N2}%" -f $stats.best_trade_pct, $stats.worst_trade_pct) -ForegroundColor White
        Write-Host ("  Avg Hold: {0:N0} minutes" -f $stats.avg_hold_minutes) -ForegroundColor White
        
        if ($stats.exit_reasons) {
            Write-Host "  Exit Reasons: " -NoNewline -ForegroundColor Gray
            $reasons = $stats.exit_reasons.PSObject.Properties | ForEach-Object { "$($_.Name)($($_.Value))" }
            Write-Host ($reasons -join ", ") -ForegroundColor Gray
        }
    }
}

function Show-RecentTrades {
    param([int]$count = 10)
    
    $tradesFile = Join-Path $StateDir "trade_transparency.json"
    if (-not (Test-Path $tradesFile)) {
        Write-Host "`n[!] No trade history found" -ForegroundColor Yellow
        return
    }
    
    $allTrades = Get-Content $tradesFile | ConvertFrom-Json
    $closedTrades = $allTrades | Where-Object { $_.status -eq "CLOSED" }
    $recentTrades = $closedTrades | Select-Object -Last $count | Sort-Object exit_time -Descending
    
    Write-Host "`n╔═══════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host " RECENT TRADES (Last $count)" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════╝" -ForegroundColor Cyan
    
    foreach ($trade in $recentTrades) {
        $entryTime = [datetime]::Parse($trade.entry_time)
        $exitTime = [datetime]::Parse($trade.exit_time)
        $duration = $exitTime - $entryTime
        
        $pnlColor = Get-ColorForPnL $trade.pnl_usd
        
        Write-Host "`n$($trade.symbol) $($trade.direction)" -ForegroundColor Cyan
        Write-Host ("  Entry: `${0:N2} @ {1:HH:mm}" -f $trade.entry_price, $entryTime) -ForegroundColor White
        Write-Host ("  Exit: `${0:N2} @ {1:HH:mm}" -f $trade.exit_price, $exitTime) -ForegroundColor White
        Write-Host ("  P/L: `${0:N2} ({1:N2}%)" -f $trade.pnl_usd, $trade.pnl_pct) -ForegroundColor $pnlColor
        Write-Host ("  Duration: {0:N0} minutes" -f $trade.hold_minutes) -ForegroundColor White
        Write-Host ("  Exit Reason: {0}" -f $trade.exit_reason) -ForegroundColor Yellow
        
        # Show signal strength that triggered entry
        Write-Host "  Entry Signal: " -NoNewline -ForegroundColor Gray
        Write-Host ("Conf:{0:N1}% Vol:{1:N3} RSI:{2:N0}" -f `
            ($trade.signal.confidence * 100), `
            $trade.signal.volatility, `
            $trade.signal.rsi) -ForegroundColor Gray
        
        if ($trade.signal.multiplier -gt 1) {
            Write-Host ("  Position Size: {0}x (Score: {1}/6)" -f `
                $trade.signal.multiplier, $trade.signal.score) -ForegroundColor Magenta
        }
    }
}

function Show-OpenPositions {
    $tradesFile = Join-Path $StateDir "trade_transparency.json"
    if (-not (Test-Path $tradesFile)) {
        Write-Host "`n[!] No trade history found" -ForegroundColor Yellow
        return
    }
    
    $allTrades = Get-Content $tradesFile | ConvertFrom-Json
    $openTrades = $allTrades | Where-Object { $_.status -eq "OPEN" }
    
    if ($openTrades.Count -eq 0) {
        Write-Host "`n[!] No open positions" -ForegroundColor Yellow
        return
    }
    
    Write-Host "`n╔═══════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host " OPEN POSITIONS" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════╝" -ForegroundColor Cyan
    
    foreach ($trade in $openTrades) {
        $entryTime = [datetime]::Parse($trade.entry_time)
        $holdTime = (Get-Date) - $entryTime
        
        Write-Host "`n$($trade.symbol) $($trade.direction)" -ForegroundColor Cyan
        Write-Host ("  Entry: `${0:N2} @ {1:HH:mm}" -f $trade.entry_price, $entryTime) -ForegroundColor White
        Write-Host ("  Size: `${0:N2}" -f $trade.size) -ForegroundColor White
        Write-Host ("  Holding: {0:N0} minutes" -f $holdTime.TotalMinutes) -ForegroundColor White
        
        # Show entry signal strength
        Write-Host "  Entry Signal: " -NoNewline -ForegroundColor Gray
        Write-Host ("Conf:{0:N1}% Vol:{1:N3} RSI:{2:N0} Regime:{3}" -f `
            ($trade.signal.confidence * 100), `
            $trade.signal.volatility, `
            $trade.signal.rsi, `
            $trade.signal.regime) -ForegroundColor Gray
        
        if ($trade.signal.multiplier -gt 1) {
            Write-Host ("  Position Size: {0}x (Score: {1}/6)" -f `
                $trade.signal.multiplier, $trade.signal.score) -ForegroundColor Magenta
        }
    }
}

# ============================================================
# MAIN MENU
# ============================================================

Clear-Host
Write-Host "╔═══════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host " CRYPTO BOT TRANSPARENCY DASHBOARD" -ForegroundColor Magenta
Write-Host "╚═══════════════════════════════════════════════╝" -ForegroundColor Magenta

while ($true) {
    Write-Host "`n" -NoNewline
    Write-Host "[1]" -ForegroundColor Yellow -NoNewline
    Write-Host " Daily Summary  " -NoNewline
    Write-Host "[2]" -ForegroundColor Yellow -NoNewline
    Write-Host " Per-Symbol Stats  " -NoNewline
    Write-Host "[3]" -ForegroundColor Yellow -NoNewline
    Write-Host " Recent Trades" -NoNewline
    Write-Host "`n" -NoNewline
    Write-Host "[4]" -ForegroundColor Yellow -NoNewline
    Write-Host " Open Positions  " -NoNewline
    Write-Host "[5]" -ForegroundColor Yellow -NoNewline
    Write-Host " All (Full Report)  " -NoNewline
    Write-Host "[Q]" -ForegroundColor Yellow -NoNewline
    Write-Host " Quit"
    Write-Host "`nSelect: " -NoNewline -ForegroundColor Cyan
    
    $choice = Read-Host
    
    switch ($choice) {
        "1" { Show-DailySummary }
        "2" { Show-SymbolPerformance }
        "3" { Show-RecentTrades }
        "4" { Show-OpenPositions }
        "5" { 
            Show-DailySummary
            Show-SymbolPerformance
            Show-RecentTrades
            Show-OpenPositions
        }
        "Q" { 
            Write-Host "`nExiting..." -ForegroundColor Yellow
            exit 
        }
        default { 
            Write-Host "`n[!] Invalid choice" -ForegroundColor Red 
        }
    }
}
