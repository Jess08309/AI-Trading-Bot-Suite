# Crypto Bot Dashboard
Clear-Host

function Show-Dashboard {
    Clear-Host
    
    $trades = Import-Csv "D:\042021\CryptoBot\trade_history.csv" -Header @('timestamp','pair','action','price','amount','balance','position') | Select-Object -Skip 1
    
    $startBalance = 2500
    $currentBalance = [decimal]$trades[-1].balance
    $pnl = $currentBalance - $startBalance
    $returnPct = ($pnl / $startBalance) * 100
    
    $totalTrades = $trades.Count
    $buyTrades = ($trades | Where-Object {$_.action -eq 'BUY'}).Count
    $sellTrades = ($trades | Where-Object {$_.action -eq 'SELL'}).Count
    
    # Get open positions
    $positions = @{}
    foreach ($trade in $trades) {
        if ($trade.position) {
            $positions[$trade.pair] = [decimal]$trade.position
        }
    }
    $openPositions = $positions.GetEnumerator() | Where-Object { [Math]::Abs($_.Value) -gt 0.001 }
    
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "    CRYPTO TRADING BOT - LIVE DASHBOARD" -ForegroundColor Cyan  
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "BALANCE:      " -NoNewline -ForegroundColor White
    Write-Host "`$$($currentBalance.ToString('N2'))" -ForegroundColor Green
    
    $pnlColor = if ($pnl -ge 0) { "Green" } else { "Red" }
    Write-Host "P&L:          " -NoNewline -ForegroundColor White
    Write-Host "`$$($pnl.ToString('N2'))" -ForegroundColor $pnlColor
    
    Write-Host "RETURN:       " -NoNewline -ForegroundColor White
    Write-Host "$($returnPct.ToString('N2'))%" -ForegroundColor $pnlColor
    
    Write-Host "TOTAL TRADES: " -NoNewline -ForegroundColor White
    Write-Host "$totalTrades (BUY: $buyTrades | SELL: $sellTrades)" -ForegroundColor Yellow
    
    Write-Host ""
    Write-Host "=== OPEN POSITIONS ===" -ForegroundColor Yellow
    if ($openPositions.Count -eq 0) {
        Write-Host "No open positions" -ForegroundColor Gray
    } else {
        foreach ($pos in $openPositions) {
            $type = if ($pos.Value -gt 0) { "LONG" } else { "SHORT" }
            $typeColor = if ($pos.Value -gt 0) { "Green" } else { "Red" }
            Write-Host "$($pos.Key): " -NoNewline -ForegroundColor White
            Write-Host "$type " -NoNewline -ForegroundColor $typeColor
            Write-Host "$([Math]::Abs($pos.Value).ToString('N4'))" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "=== RECENT TRADES (Last 10) ===" -ForegroundColor Yellow
    $recentTrades = $trades | Select-Object -Last 10
    foreach ($trade in $recentTrades) {
        $time = ([datetime]$trade.timestamp).ToString("HH:mm:ss")
        $actionColor = if ($trade.action -eq 'BUY') { "Cyan" } else { "Magenta" }
        
        Write-Host "$time  " -NoNewline -ForegroundColor Gray
        Write-Host "$($trade.pair.PadRight(10)) " -NoNewline -ForegroundColor White
        Write-Host "$($trade.action.PadRight(5)) " -NoNewline -ForegroundColor $actionColor
        Write-Host "`$$([decimal]$trade.price) " -NoNewline -ForegroundColor Yellow
        Write-Host "-> `$$([math]::Round([decimal]$trade.balance, 2))" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Last updated: $(Get-Date -Format 'HH:mm:ss')  |  Press Ctrl+C to exit" -ForegroundColor Gray
}

Write-Host "Starting dashboard..." -ForegroundColor Green
Start-Sleep 2

while ($true) {
    try {
        Show-Dashboard
        Start-Sleep 10
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        Start-Sleep 5
    }
}
