$ErrorActionPreference = 'SilentlyContinue'

$botDir = 'D:\042021\CryptoBot'
$reportPath = Join-Path $botDir 'data\state\rl_shadow_report.json'
$eventsPath = Join-Path $botDir 'data\state\rl_shadow_events.jsonl'

function Get-JsonSafe([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    try {
        return (Get-Content $path -Raw | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Get-RecentEvents([string]$path, [int]$count = 18) {
    if (-not (Test-Path $path)) { return @() }
    $lines = Get-Content $path -Tail $count
    $items = @()
    foreach ($line in $lines) {
        try {
            $obj = $line | ConvertFrom-Json
            $items += $obj
        } catch {}
    }
    return $items
}

while ($true) {
    Clear-Host
    $now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host '===============================================' -ForegroundColor Cyan
    Write-Host '      SHADOW RL PERFORMANCE DASHBOARD' -ForegroundColor Cyan
    Write-Host '===============================================' -ForegroundColor Cyan
    Write-Host "Now: $now"
    Write-Host "Report: $reportPath"
    Write-Host ''

    $p = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*CryptoBot*.venv*python.exe' } | Select-Object -First 1
    if ($null -ne $p) {
        Write-Host ("Bot: RUNNING (PID {0})" -f $p.Id) -ForegroundColor Green
    } else {
        Write-Host 'Bot: STOPPED' -ForegroundColor Red
    }

    $report = Get-JsonSafe $reportPath
    if ($null -eq $report) {
        Write-Host ''
        Write-Host 'Shadow report not found yet.' -ForegroundColor Yellow
        Write-Host 'It will appear after the bot writes first shadow snapshot.' -ForegroundColor Yellow
    } else {
        Write-Host ''
        Write-Host ("Updated: {0} | Cycle: {1}" -f $report.updated_at, $report.cycle)
        Write-Host ("Mode: {0} | RL Min Mult: {1}" -f $report.mode, $report.rl_shadow_min_multiplier)

        $b = $report.books.baseline
        $r = $report.books.rl
        $d = $report.delta

        Write-Host ''
        Write-Host '--- Baseline ---' -ForegroundColor Gray
        Write-Host ("Equity: ${0} | Realized: ${1} | DD: {2}% | Trades: {3} | WinRate: {4}%" -f $b.equity, $b.realized_pnl, $b.max_drawdown_pct, $b.trades, $b.win_rate)

        Write-Host '--- RL Shadow ---' -ForegroundColor Gray
        Write-Host ("Equity: ${0} | Realized: ${1} | DD: {2}% | Trades: {3} | WinRate: {4}%" -f $r.equity, $r.realized_pnl, $r.max_drawdown_pct, $r.trades, $r.win_rate)

        Write-Host ''
        $eqColor = if ([double]$d.equity -ge 0) { 'Green' } else { 'Red' }
        $pnlColor = if ([double]$d.realized_pnl -ge 0) { 'Green' } else { 'Red' }
        $ddColor = if ([double]$d.max_drawdown_pct -ge 0) { 'Red' } else { 'Green' }

        Write-Host '--- RL minus Baseline ---' -ForegroundColor Cyan
        Write-Host ("Equity Delta: ${0}" -f $d.equity) -ForegroundColor $eqColor
        Write-Host ("Realized Delta: ${0}" -f $d.realized_pnl) -ForegroundColor $pnlColor
        Write-Host ("Drawdown Delta: {0}%" -f $d.max_drawdown_pct) -ForegroundColor $ddColor
    }

    Write-Host ''
    Write-Host '--- Recent Shadow Events ---' -ForegroundColor Cyan
    $events = Get-RecentEvents $eventsPath 14
    if ($events.Count -eq 0) {
        Write-Host 'No shadow events yet.' -ForegroundColor Yellow
    } else {
        foreach ($e in $events) {
            $line = "[{0}] {1} {2} {3} {4}" -f $e.ts, $e.strategy, $e.type, $e.symbol, ($e.reason)
            Write-Host $line
        }
    }

    Start-Sleep -Seconds 5
}
