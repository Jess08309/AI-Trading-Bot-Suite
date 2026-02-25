# Monitor the bot process console output by tailing the actual running log
# Since FileHandler isn't working, we'll watch for new state file changes instead

function Resolve-StateDir {
    $local = Join-Path $PSScriptRoot "data\state"
    $external = "D:\042021\CryptoBot\data\state"

    if (Test-Path (Join-Path $external "paper_balances.json")) {
        return $external
    }
    return $local
}

$StateDir = Resolve-StateDir
$statusFile = Join-Path $StateDir "paper_balances.json"
$positionsFile = Join-Path $StateDir "positions.json"
$futuresFile = Join-Path $StateDir "futures_positions.json"

Write-Host "=== LIVE TRADE MONITOR (State File Watcher) ===" -ForegroundColor Cyan
Write-Host "Watching for balance and position changes..." -ForegroundColor Yellow
Write-Host "Press CTRL-C to stop.`n" -ForegroundColor Gray

$lastBalanceHash = ""
$lastPosHash = ""
$lastFutHash = ""

while ($true) {
    Start-Sleep -Milliseconds 500
    
    if (Test-Path $statusFile) {
        $currentBalHash = (Get-FileHash $statusFile -Algorithm MD5).Hash
        if ($currentBalHash -ne $lastBalanceHash) {
            $lastBalanceHash = $currentBalHash
            $bal = Get-Content $statusFile | ConvertFrom-Json
            $total = [math]::Round($bal.spot + $bal.futures, 2)
            $spot = [math]::Round($bal.spot, 2)
            $fut = [math]::Round($bal.futures, 2)
            
            Write-Host "$(Get-Date -Format 'HH:mm:ss') " -NoNewline -ForegroundColor DarkGray
            Write-Host "BALANCE UPDATE: " -NoNewline -ForegroundColor White
            Write-Host "TOTAL: `$$total " -NoNewline -ForegroundColor Cyan
            Write-Host "(Spot: `$$spot, Futures: `$$fut)" -ForegroundColor Gray
        }
    }
    
    if (Test-Path $positionsFile) {
        $currentPosHash = (Get-FileHash $positionsFile -Algorithm MD5).Hash
        if ($currentPosHash -ne $lastPosHash -and $lastPosHash -ne "") {
            $lastPosHash = $currentPosHash
            $positions = Get-Content $positionsFile | ConvertFrom-Json
            $posCount = ($positions.PSObject.Properties | Measure-Object).Count
            
            if ($posCount -eq 0) {
                Write-Host "$(Get-Date -Format 'HH:mm:ss') " -NoNewline -ForegroundColor DarkGray
                Write-Host "SPOT POSITION CLOSED" -ForegroundColor Yellow
            } else {
                foreach ($prop in $positions.PSObject.Properties) {
                    $symbol = $prop.Name
                    $pos = $prop.Value
                    Write-Host "$(Get-Date -Format 'HH:mm:ss') " -NoNewline -ForegroundColor DarkGray
                    Write-Host "SPOT: " -NoNewline -ForegroundColor Green
                    Write-Host "$symbol @ `$$([math]::Round($pos.entry_price, 4))" -ForegroundColor White
                }
            }
        } elseif ($lastPosHash -eq "") {
            $lastPosHash = $currentPosHash
        }
    }
    
    if (Test-Path $futuresFile) {
        $currentFutHash = (Get-FileHash $futuresFile -Algorithm MD5).Hash
        if ($currentFutHash -ne $lastFutHash -and $lastFutHash -ne "") {
            $lastFutHash = $currentFutHash
            $futPositions = Get-Content $futuresFile | ConvertFrom-Json
            $futCount = ($futPositions.PSObject.Properties | Measure-Object).Count
            
            if ($futCount -eq 0) {
                Write-Host "$(Get-Date -Format 'HH:mm:ss') " -NoNewline -ForegroundColor DarkGray
                Write-Host "FUTURES POSITION(S) CLOSED" -ForegroundColor Yellow  
            } else {
                foreach ($prop in $futPositions.PSObject.Properties) {
                    $symbol = $prop.Name
                    $pos = $prop.Value
                    $side = $pos.direction
                    $color = if ($side -eq "LONG") { "Green" } else { "Red" }
                    Write-Host "$(Get-Date -Format 'HH:mm:ss') " -NoNewline -ForegroundColor DarkGray
                    Write-Host "FUTURES $side: " -NoNewline -ForegroundColor $color
                    Write-Host "$symbol @ `$$([math]::Round($pos.entry_price, 2)) (Size: `$$([math]::Round($pos.contract_value, 2)))" -ForegroundColor White
                }
            }
        } elseif ($lastFutHash -eq "") {
            $lastFutHash = $currentFutHash
        }
    }
}
