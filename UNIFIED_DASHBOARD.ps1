# COMPREHENSIVE TRADING DASHBOARD
# Live positions + Recent exits + Daily summary + Performance stats
# All in one unified view that auto-refreshes

$ErrorActionPreference = "SilentlyContinue"

function Resolve-StateDir {
    $envOverride = $env:BOT_STATE_DIR
    if ($envOverride -and (Test-Path (Join-Path $envOverride "paper_balances.json"))) {
        return $envOverride
    }

    $external = "D:\042021\CryptoBot\data\state"
    if (Test-Path (Join-Path $external "paper_balances.json")) {
        return $external
    }

    $local = Join-Path $PSScriptRoot "data\state"
    if (Test-Path (Join-Path $local "paper_balances.json")) {
        return $local
    }

    throw "No valid state directory found. Checked BOT_STATE_DIR, D:\042021\CryptoBot\data\state, and local data\state."
}

$StateDir = Resolve-StateDir
Write-Host "Using state dir: $StateDir" -ForegroundColor DarkGray

function Get-PositionSets {
    $spot = @()
    $futures = @()

    $posPath = Join-Path $StateDir "positions.json"
    if (Test-Path $posPath) {
        $positions = Get-Content $posPath | ConvertFrom-Json
        foreach ($prop in $positions.PSObject.Properties) {
            if ($prop.Name -like "PI_*") {
                $futures += [PSCustomObject]@{ Symbol = $prop.Name; Data = $prop.Value; Source = "positions" }
            } else {
                $spot += [PSCustomObject]@{ Symbol = $prop.Name; Data = $prop.Value; Source = "positions" }
            }
        }
    }

    if ($futures.Count -eq 0) {
        $futPath = Join-Path $StateDir "futures_positions.json"
        if (Test-Path $futPath) {
            $futPositions = Get-Content $futPath | ConvertFrom-Json
            foreach ($prop in $futPositions.PSObject.Properties) {
                $futures += [PSCustomObject]@{ Symbol = $prop.Name; Data = $prop.Value; Source = "futures_positions" }
            }
        }
    }

    return [PSCustomObject]@{ Spot = $spot; Futures = $futures }
}

function Get-CoinbasePrice {
    param([string]$symbol)
    try {
        $response = Invoke-RestMethod -Uri "https://api.coinbase.com/v2/prices/$symbol/spot" -Method Get -TimeoutSec 3
        return [decimal]$response.data.amount
    } catch {
        return $null
    }
}

function Get-CoinbaseFuturesPrice {
    param([string]$symbol)

    $asset = ($symbol -replace '^PI_', '') -replace 'USD$', ''
    if (-not $asset) {
        return $null
    }

    $products = @(
        "$asset-USDC-PERP",
        "$asset-USD-PERP",
        "$asset-PERP"
    )

    foreach ($product in $products) {
        try {
            $response = Invoke-RestMethod -Uri "https://api.exchange.coinbase.com/products/$product/ticker" -Method Get -TimeoutSec 3
            if ($null -ne $response.price) {
                return [decimal]$response.price
            }
        } catch {
            continue
        }
    }

    return $null
}

function Get-KrakenFuturesPrice {
    param([string]$symbol)

    $price = $null

    try {
        $response = Invoke-RestMethod -Uri "https://futures.kraken.com/derivatives/api/v3/tickers" -Method Get -TimeoutSec 3
        $ticker = $response.tickers | Where-Object { $_.symbol -eq $symbol }
        if ($ticker) {
            $price = [decimal]$ticker.last
        }
    } catch {
        $price = $null
    }

    if ($null -eq $price) {
        $price = Get-CoinbaseFuturesPrice $symbol
    }

    if ($null -eq $price) {
        $spotSymbol = (($symbol -replace '^PI_', '') -replace 'USD$', '') + '-USD'
        $price = Get-CoinbasePrice $spotSymbol
    }

    return $price
}

function Show-Snapshot {
    Clear-Host
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "          TRADING BOT TRANSPARENCY DASHBOARD               " -ForegroundColor Cyan
    Write-Host "          Last Update: $timestamp                          " -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    
    # ========== BALANCES ==========
    $balPath = Join-Path $StateDir "paper_balances.json"
    if (Test-Path $balPath) {
        $bal = Get-Content $balPath | ConvertFrom-Json
        $spot = $bal.spot
        $futures = $bal.futures
        $total = $spot + $futures
        $pnl = $total - 5000
        $pnlPct = ($pnl / 5000) * 100
        
        Write-Host "BALANCES:" -ForegroundColor Yellow
        Write-Host ("  Spot:    {0,12:N2}" -f $spot) -ForegroundColor White
        Write-Host ("  Futures: {0,12:N2}" -f $futures) -ForegroundColor White
        Write-Host ("  Total:   {0,12:N2}" -f $total) -ForegroundColor White
        $pnlColor = if ($pnl -gt 0) { "Green" } else { "Red" }
        Write-Host ("  P/L:     `${0,12:N2} ({1:N2}%)" -f $pnl, $pnlPct) -ForegroundColor $pnlColor
        Write-Host ""
    }
    
    # ========== OPEN POSITIONS ==========
    $positionSets = Get-PositionSets
    $spotPositions = $positionSets.Spot
    $futuresPositions = $positionSets.Futures
    $openCount = $spotPositions.Count + $futuresPositions.Count

    $favoredCount = 0
    $againstCount = 0
    $flatCount = 0
    $pricedCount = 0
    $netOpenPnl = 0.0
    $spotPriceCache = @{}
    $futuresPriceCache = @{}

    Write-Host ("OPEN POSITIONS: {0}" -f $openCount) -ForegroundColor Yellow

    foreach ($position in $spotPositions) {
        $symbol = $position.Symbol
        $pos = $position.Data
        $direction = if ($null -ne $pos.direction -and [string]$pos.direction -ne "") { ([string]$pos.direction).ToUpper() } else { "LONG" }
        $entryPrice = [double]$pos.entry_price
        $amount = if ($null -ne $pos.amount) { [double]$pos.amount } else { 0.0 }
        $positionCost = if ($null -ne $pos.size -and [double]$pos.size -gt 0) { [double]$pos.size } else { $entryPrice * $amount }
        if ($amount -le 0 -and $entryPrice -gt 0 -and $positionCost -gt 0) {
            $amount = $positionCost / $entryPrice
        }

        $currentPrice = Get-CoinbasePrice $symbol
        $spotPriceCache[$symbol] = $currentPrice
        if ($currentPrice) {
            $pnlPct = if ($direction -eq "SHORT") { (($entryPrice - $currentPrice) / $entryPrice) * 100 } else { (($currentPrice - $entryPrice) / $entryPrice) * 100 }
            $pnlUsd = ($pnlPct / 100.0) * $positionCost

            $pricedCount += 1
            $netOpenPnl += $pnlUsd
            if ($pnlUsd -gt 0) { $favoredCount += 1 } elseif ($pnlUsd -lt 0) { $againstCount += 1 } else { $flatCount += 1 }
        }
    }

    foreach ($position in $futuresPositions) {
        $symbol = $position.Symbol
        $pos = $position.Data
        $direction = [string]$pos.direction
        $entryPrice = [double]$pos.entry_price
        $contractValue = if ($null -ne $pos.contract_value) { [double]$pos.contract_value } else { [double]$pos.size }
        $leverage = if ($null -ne $pos.leverage) { [double]$pos.leverage } else { 1.0 }

        $currentPrice = Get-KrakenFuturesPrice $symbol
        $futuresPriceCache[$symbol] = $currentPrice
        if ($currentPrice) {
            if ($direction -eq "LONG") {
                $pnlUsd = (($currentPrice - $entryPrice) / $entryPrice) * $contractValue
            }
            else {
                $pnlUsd = (($entryPrice - $currentPrice) / $entryPrice) * $contractValue
            }

            $pricedCount += 1
            $netOpenPnl += $pnlUsd
            if ($pnlUsd -gt 0) { $favoredCount += 1 } elseif ($pnlUsd -lt 0) { $againstCount += 1 } else { $flatCount += 1 }
        }
    }

    $inFavorPct = if ($pricedCount -gt 0) { ($favoredCount / $pricedCount) * 100.0 } else { 0.0 }
    $summaryColor = if ($netOpenPnl -gt 0) { "Green" } elseif ($netOpenPnl -lt 0) { "Red" } else { "Yellow" }
    Write-Host "POSITION HEALTH:" -ForegroundColor Yellow
    Write-Host ("  In Favor: {0} | Against: {1} | Flat: {2} | Net Open P/L: `${3:N2} | In Favor %: {4:N1}%" -f `
        $favoredCount, $againstCount, $flatCount, $netOpenPnl, $inFavorPct) -ForegroundColor $summaryColor

    # Spot
    if ($spotPositions.Count -gt 0) {
        Write-Host ("SPOT POSITIONS ({0}):" -f $spotPositions.Count) -ForegroundColor Yellow
        $spotPositions | ForEach-Object {
            $symbol = $_.Symbol
            $pos = $_.Data
            $direction = if ($null -ne $pos.direction -and [string]$pos.direction -ne "") { ([string]$pos.direction).ToUpper() } else { "LONG" }
            $entryPrice = [double]$pos.entry_price
            $amount = if ($null -ne $pos.amount) { [double]$pos.amount } else { 0.0 }
            $positionCost = if ($null -ne $pos.size -and [double]$pos.size -gt 0) { [double]$pos.size } else { $entryPrice * $amount }
            if ($amount -le 0 -and $entryPrice -gt 0 -and $positionCost -gt 0) {
                $amount = $positionCost / $entryPrice
            }

            $currentPrice = $spotPriceCache[$symbol]
            if ($currentPrice) {
                if ($direction -eq "SHORT") {
                    $pnlPct = (($entryPrice - $currentPrice) / $entryPrice) * 100
                }
                else {
                    $pnlPct = (($currentPrice - $entryPrice) / $entryPrice) * 100
                }

                $pnlUsd = ($pnlPct / 100.0) * $positionCost
                $pnlColor = if ($pnlUsd -gt 0) { "Green" } elseif ($pnlUsd -lt 0) { "Red" } else { "Yellow" }

                Write-Host ("  {0} {1} | Qty: {2:N4} | Size: `${3:N2} | Entry: `${4:N4} | Now: `${5:N4} | P/L: `${6:N2} ({7:N2}%)" -f `
                    $direction, $symbol, $amount, $positionCost, $entryPrice, $currentPrice, $pnlUsd, $pnlPct) -ForegroundColor $pnlColor
            }
            else {
                Write-Host ("  {0} {1} | Qty: {2:N4} | Size: `${3:N2} | Entry: `${4:N4}" -f `
                    $direction, $symbol, $amount, $positionCost, $entryPrice) -ForegroundColor Gray
            }
        }
    }
    else {
        Write-Host "SPOT POSITIONS: None" -ForegroundColor Gray
    }

    Write-Host ""

    # Futures
    if ($futuresPositions.Count -gt 0) {
        Write-Host ("FUTURES POSITIONS ({0}):" -f $futuresPositions.Count) -ForegroundColor Yellow
        $futuresPositions | ForEach-Object {
            $symbol = $_.Symbol
            $pos = $_.Data
            $direction = [string]$pos.direction
            $entryPrice = [double]$pos.entry_price
            $contractValue = if ($null -ne $pos.contract_value) { [double]$pos.contract_value } else { [double]$pos.size }
            $leverage = if ($null -ne $pos.leverage) { [double]$pos.leverage } else { 1.0 }

            $currentPrice = $futuresPriceCache[$symbol]
            if ($currentPrice) {
                if ($direction -eq "LONG") {
                    $pnlPct = (($currentPrice - $entryPrice) / $entryPrice) * 100 * $leverage
                    $pnlUsd = (($currentPrice - $entryPrice) / $entryPrice) * $contractValue
                }
                else {
                    $pnlPct = (($entryPrice - $currentPrice) / $entryPrice) * 100 * $leverage
                    $pnlUsd = (($entryPrice - $currentPrice) / $entryPrice) * $contractValue
                }

                $pnlColor = if ($pnlUsd -gt 0) { "Green" } elseif ($pnlUsd -lt 0) { "Red" } else { "Yellow" }

                Write-Host ("  {0} {1} | Entry: `${2:N2} | Notional: `${3:N2} | Now: `${4:N2} | P/L: `${5:N2} ({6:N2}%)" -f `
                    $direction, $symbol, $entryPrice, $contractValue, $currentPrice, $pnlUsd, $pnlPct) -ForegroundColor $pnlColor
            }
            else {
                Write-Host ("  {0} {1} | Entry: `${2:N2} | Notional: `${3:N2}" -f `
                    $direction, $symbol, $entryPrice, $contractValue) -ForegroundColor Gray
            }
        }
    }
    else {
        Write-Host "FUTURES POSITIONS: None" -ForegroundColor Gray
    }

    Write-Host ""
    
    # ========== RECENT CLOSED TRADES ==========
    $tradesPath = Join-Path $StateDir "trade_transparency.json"
    if (Test-Path $tradesPath) {
        $allTrades = Get-Content $tradesPath | ConvertFrom-Json
        $closedTrades = $allTrades | Where-Object { $_.status -eq "CLOSED" } | Sort-Object exit_time -Descending | Select-Object -First 5
        
        if ($closedTrades.Count -gt 0) {
            Write-Host "RECENT EXITS (Last 5):" -ForegroundColor Yellow
            foreach ($trade in $closedTrades) {
                $pnlColor = if ($trade.pnl_usd -gt 0) { "Green" } else { "Red" }
                $reasonColor = switch ($trade.exit_reason) {
                    "TAKE_PROFIT" { "Green" }
                    "STOP_LOSS" { "Red" }
                    "TRAILING_STOP" { "Yellow" }
                    default { "Cyan" }
                }
                
                $exitTime = [datetime]::Parse($trade.exit_time)
                $timeStr = $exitTime.ToString("HH:mm")
                
                Write-Host ("  {0} {1} | " -f $trade.symbol, $trade.direction) -NoNewline -ForegroundColor Cyan
                Write-Host ("{0} " -f $trade.exit_reason) -NoNewline -ForegroundColor $reasonColor
                Write-Host ("| P/L: `${0:N2} ({1:N2}%) | {2}" -f $trade.pnl_usd, $trade.pnl_pct, $timeStr) -ForegroundColor $pnlColor
            }
            Write-Host ""
        }
    }
    
    # ========== DAILY SUMMARY ==========
    $summaryPath = Join-Path $StateDir "daily_summary.json"
    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath | ConvertFrom-Json
        
        Write-Host "TODAY'S PERFORMANCE:" -ForegroundColor Yellow
        Write-Host ("  Trades: {0} | Wins: {1} | Losses: {2} | Win Rate: {3:N1}%" -f `
            $summary.total_trades, $summary.wins, $summary.losses, $summary.win_rate) -ForegroundColor White
        
        if ($summary.total_trades -gt 0) {
            $pnlColor = if ($summary.total_pnl_usd -gt 0) { "Green" } else { "Red" }
            Write-Host ("  Total P/L: `${0:N2} ({1:N2}%)" -f $summary.total_pnl_usd, $summary.total_pnl_pct) -ForegroundColor $pnlColor
            
            if ($summary.best_trade) {
                Write-Host ("  Best: {0} `${1:N2} ({2:N2}%)" -f `
                    $summary.best_trade.symbol, $summary.best_trade.pnl_usd, $summary.best_trade.pnl_pct) -ForegroundColor Green
            }
        }
        Write-Host ""
    }
    
    # ========== TOP PERFORMERS ==========
    $symbolStatsPath = Join-Path $StateDir "symbol_performance.json"
    if (Test-Path $symbolStatsPath) {
        $symbolStats = Get-Content $symbolStatsPath | ConvertFrom-Json
        $topPerformers = $symbolStats.PSObject.Properties | 
            Sort-Object { $_.Value.total_pnl_usd } -Descending | 
            Select-Object -First 3
        
        if ($topPerformers.Count -gt 0) {
            Write-Host "TOP PERFORMING SYMBOLS:" -ForegroundColor Yellow
            foreach ($perf in $topPerformers) {
                $stats = $perf.Value
                $pnlColor = if ($stats.total_pnl_usd -gt 0) { "Green" } else { "Red" }
                Write-Host ("  {0}: {1} trades | Win Rate: {2:N1}% | P/L: `${3:N2}" -f `
                    $perf.Name, $stats.total_trades, $stats.win_rate, $stats.total_pnl_usd) -ForegroundColor $pnlColor
            }
            Write-Host ""
        }
    }
    
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Refreshing every 5 seconds... Press Ctrl+C to stop" -ForegroundColor Gray
    Write-Host ""
}

# Main loop
while ($true) {
    Show-Snapshot
    Start-Sleep -Seconds 5
}
