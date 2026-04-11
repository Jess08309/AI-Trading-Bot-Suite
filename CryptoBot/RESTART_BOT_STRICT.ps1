param(
    [string]$BotRoot = "C:\Bot",
    [ValidateSet("off", "normal", "strict")]
    [string]$RealismProfile = "strict"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Stopping existing bot/python processes..." -ForegroundColor Yellow
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "cryptotrades\\main.py|cryptotrades\.main" } |
    ForEach-Object {
        try {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
            Write-Host "  Stopped PID $($_.ProcessId)" -ForegroundColor DarkYellow
        } catch {
            Write-Host "  Could not stop PID $($_.ProcessId): $($_.Exception.Message)" -ForegroundColor Red
        }
    }

Write-Host "[2/4] Validating bot paths..." -ForegroundColor Yellow
$pythonExe = Join-Path $BotRoot ".venv\Scripts\python.exe"
$mainPy = Join-Path $BotRoot "cryptotrades\main.py"
if (-not (Test-Path $pythonExe)) { throw "Python executable not found: $pythonExe" }
if (-not (Test-Path $mainPy)) { throw "Bot entrypoint not found: $mainPy" }

Write-Host "[3/4] Setting simulation realism profile..." -ForegroundColor Yellow
$env:SIM_REALISM_PROFILE = $RealismProfile
Write-Host "  SIM_REALISM_PROFILE=$env:SIM_REALISM_PROFILE" -ForegroundColor Cyan

Write-Host "[3.5/4] Enforcing Coinbase spot+futures runtime flags..." -ForegroundColor Yellow
foreach ($name in @("ENABLE_SPOT", "ENABLE_FUTURES", "ENABLE_COINBASE", "ENABLE_KRAKEN", "ENABLE_COINBASE_FUTURES_DATA")) {
    if (Test-Path ("Env:" + $name)) {
        Remove-Item ("Env:" + $name) -ErrorAction SilentlyContinue
    }
}
$env:ENABLE_SPOT = "true"
$env:ENABLE_FUTURES = "true"
$env:ENABLE_COINBASE = "true"
$env:ENABLE_KRAKEN = "false"
$env:ENABLE_COINBASE_FUTURES_DATA = "true"
Write-Host "  ENABLE_SPOT=$env:ENABLE_SPOT | ENABLE_FUTURES=$env:ENABLE_FUTURES | ENABLE_COINBASE=$env:ENABLE_COINBASE | ENABLE_KRAKEN=$env:ENABLE_KRAKEN | ENABLE_COINBASE_FUTURES_DATA=$env:ENABLE_COINBASE_FUTURES_DATA" -ForegroundColor Cyan

Write-Host "[REMINDER] Restart safety: prefer START_BOT.bat so locked profile protection is applied." -ForegroundColor Magenta
Write-Host "[REMINDER] If config intent changes, refresh lock with LOCK_CURRENT_PROFILE.ps1." -ForegroundColor Magenta

Write-Host "[4/4] Starting bot..." -ForegroundColor Yellow
Set-Location $BotRoot
& $pythonExe $mainPy
