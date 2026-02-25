# Quick Status Display - Shows current balances and positions with live P&L
# Run this anytime to see current bot status

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

function Resolve-StateDir {
    $envOverride = $env:BOT_STATE_DIR
    if ($envOverride -and (Test-Path (Join-Path $envOverride "positions.json"))) {
        return $envOverride
    }

    $local = Join-Path $PSScriptRoot "data\state"
    $external = "D:\042021\CryptoBot\data\state"

    if (Test-Path (Join-Path $external "positions.json")) {
        return $external
    }
    return $local
}

$StateDir = Resolve-StateDir

function Get-PositionSets {
    $spot = @()
    $futures = @()

    $posPath = Join-Path $StateDir "positions.json"
    if (Test-Path $posPath) {
        $positions = Get-Content $posPath | ConvertFrom-Json
        foreach ($prop in $positions.PSObject.Properties) {
            if ($prop.Name -like "PI_*") {
                $futures += [PSCustomObject]@{
                    Symbol = $prop.Name
                    Data = $prop.Value
                    Source = "positions"
                }
            } else {
                $spot += [PSCustomObject]@{
                    Symbol = $prop.Name
                    Data = $prop.Value
                    Source = "positions"
                }
            }
        }
    }

    if ($futures.Count -eq 0) {
        $futPath = Join-Path $StateDir "futures_positions.json"
        if (Test-Path $futPath) {
            $futPositions = Get-Content $futPath | ConvertFrom-Json
            foreach ($prop in $futPositions.PSObject.Properties) {
                $futures += [PSCustomObject]@{
                    Symbol = $prop.Name
                    Data = $prop.Value
                    Source = "futures_positions"
                }
            }
        }
    }

    return [PSCustomObject]@{ Spot = $spot; Futures = $futures }
}

function Show-Status {
    Clear-Host
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "     TRADING BOT STATUS SNAPSHOT        " -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Data source: $StateDir" -ForegroundColor DarkGray
    Write-Host ""
    
    # Read balances
    $balPath = Join-Path $StateDir "paper_balances.json"
    if (Test-Path $balPath) {
        $bal = Get-Content $balPath | ConvertFrom-Json
        $spot = $bal.spot
        $futures = $bal.futures
        $total = $spot + $futures
        $pnl = $total - 5000
        $pnlPct = ($pnl / 5000) * 100
        
        Write-Host "BALANCES:" -ForegroundColor Yellow
        Write-Host "  Spot:    " -NoNewline
        Write-Host ("{0,12:N2}" -f $spot) -ForegroundColor White
        Write-Host "  Futures: " -NoNewline
        Write-Host ("{0,12:N2}" -f $futures) -ForegroundColor White
        Write-Host "  Total:   " -NoNewline
        Write-Host ("{0,12:N2}" -f $total) -ForegroundColor Cyan
        
        $pnlColor = if ($pnl -ge 0) { 'Green' } else { 'Red' }
        Write-Host "  P/L:     " -NoNewline
        Write-Host ("{0,12:N2} ({1:N2}%)" -f $pnl, $pnlPct) -ForegroundColor $pnlColor
        Write-Host ""
    }
    
    # Read spot positions
    $positionSets = Get-PositionSets
    $spotPositions = $positionSets.Spot
    $futuresPositions = $positionSets.Futures

    if ($spotPositions.Count -gt 0) {
        $posCount = $spotPositions.Count
        
        if ($posCount -gt 0) {
            Write-Host "SPOT POSITIONS ($posCount):" -ForegroundColor Yellow
            foreach ($position in $spotPositions) {
                $pair = $position.Symbol
                $pos = $position.Data
                $direction = if ($null -ne $pos.direction -and [string]$pos.direction -ne '') { ([string]$pos.direction).ToUpper() } else { 'LONG' }
                $entryPrice = [double]$pos.entry_price
                $amount = if ($null -ne $pos.amount) { [double]$pos.amount } else { 0.0 }
                $positionCost = if ($null -ne $pos.size -and [double]$pos.size -gt 0) { [double]$pos.size } else { $entryPrice * $amount }
                if ($amount -le 0 -and $entryPrice -gt 0 -and $positionCost -gt 0) {
                    $amount = $positionCost / $entryPrice
                }
                $currentPrice = Get-CoinbasePrice $pair
                
                Write-Host "  $direction $pair" -ForegroundColor White -NoNewline
                Write-Host " | Entry: $" -NoNewline -ForegroundColor Gray
                Write-Host ("{0:N4}" -f $entryPrice) -NoNewline -ForegroundColor Cyan
                
                if ($currentPrice) {
                    Write-Host " | Now: $" -NoNewline -ForegroundColor Gray
                    Write-Host ("{0:N4}" -f $currentPrice) -NoNewline -ForegroundColor Cyan
                    
                    if ($direction -eq 'SHORT') {
                        $pnlPct = (($entryPrice - $currentPrice) / $entryPrice) * 100
                    } else {
                        $pnlPct = (($currentPrice - $entryPrice) / $entryPrice) * 100
                    }
                    $pnlDollar = ($pnlPct / 100.0) * $positionCost
                    $pnlColor = if ($pnlDollar -gt 0) { 'Green' } elseif ($pnlDollar -lt 0) { 'Red' } else { 'Yellow' }
                    
                    Write-Host " | P/L: " -NoNewline -ForegroundColor Gray
                    Write-Host ("{0:N2} ({1:+0.00;-0.00}%)" -f $pnlDollar, $pnlPct) -ForegroundColor $pnlColor
                } else {
                    Write-Host " | Price: N/A" -ForegroundColor DarkGray
                }
            }
            Write-Host ""
        }
    } else {
        Write-Host "SPOT POSITIONS: None" -ForegroundColor Gray
        Write-Host ""
    }

    if ($futuresPositions.Count -gt 0) {
        $futCount = $futuresPositions.Count
            Write-Host "FUTURES POSITIONS ($futCount):" -ForegroundColor Yellow
            foreach ($position in $futuresPositions) {
                $symbol = $position.Symbol
                $pos = $position.Data
                $sideColor = if ($pos.direction -eq 'LONG') { 'Green' } else { 'Red' }
                $currentPrice = Get-KrakenFuturesPrice $symbol
                
                Write-Host "  $($pos.direction) $symbol" -ForegroundColor $sideColor -NoNewline
                Write-Host " | Entry: $" -NoNewline -ForegroundColor Gray
                Write-Host ("{0:N2}" -f $pos.entry_price) -NoNewline -ForegroundColor Cyan
                Write-Host " | Size: $" -NoNewline -ForegroundColor Gray
                $contractValue = if ($null -ne $pos.contract_value) { [double]$pos.contract_value } else { [double]$pos.size }
                $leverage = if ($null -ne $pos.leverage) { [double]$pos.leverage } else { 1.0 }
                Write-Host ("{0:N2}" -f $contractValue) -NoNewline -ForegroundColor Cyan
                
                if ($currentPrice) {
                    Write-Host " | Now: $" -NoNewline -ForegroundColor Gray
                    Write-Host ("{0:N2}" -f $currentPrice) -NoNewline -ForegroundColor Cyan
                    
                    # Calculate P/L based on direction
                    if ($pos.direction -eq 'LONG') {
                        # LONG: profit when price goes up
                        $priceDiff = $currentPrice - $pos.entry_price
                    } else {
                        # SHORT: profit when price goes down
                        $priceDiff = $pos.entry_price - $currentPrice
                    }
                    
                    $return = ($priceDiff / $pos.entry_price)
                    # Dollar P/L is based on notional exposure; % shown is on margin (leveraged)
                    $pnlDollar = $return * $contractValue
                    $pnlPct = $return * 100 * $leverage
                    $pnlColor = if ($pnlDollar -ge 0) { 'Green' } else { 'Red' }
                    
                    Write-Host " | P/L: " -NoNewline -ForegroundColor Gray
                    Write-Host ("{0:+0.00;-0.00} ({1:+0.00;-0.00}%)" -f $pnlDollar, $pnlPct) -ForegroundColor $pnlColor
                } else {
                    Write-Host " | Price: N/A" -ForegroundColor DarkGray
                }
            }
            Write-Host ""
    } else {
        Write-Host "FUTURES POSITIONS: None" -ForegroundColor Gray
        Write-Host ""
    }
    
    Write-Host "Last updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray
    Write-Host ""
}

# Auto-refresh every 5 seconds
Write-Host "Auto-refreshing every 5 seconds... Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

while ($true) {
    Show-Status
    Start-Sleep -Seconds 5
}
