<# 
  PERFORMANCE TUNER — Automatically optimizes system for trading bots.
  Run after starting bots, or schedule with Task Scheduler.
  
  What it does:
    1. Sets power plan to High Performance (no CPU throttling)
    2. Sets all bot processes to AboveNormal priority
    3. Disables Nagle's algorithm for lower network latency
    4. Reports system resource usage
#>

Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  TRADING BOT PERFORMANCE TUNER" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# 1. Power Plan → High Performance
$currentPlan = powercfg /getactivescheme 2>$null
if ($currentPlan -notmatch "High performance") {
    powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
    Write-Host "[+] Power plan set to High Performance" -ForegroundColor Green
} else {
    Write-Host "[=] Power plan already High Performance" -ForegroundColor Gray
}

# 2. Set bot process priority
$bots = Get-Process pythonw -ErrorAction SilentlyContinue
$boosted = 0
foreach ($proc in $bots) {
    if ($proc.PriorityClass -ne 'AboveNormal') {
        $proc.PriorityClass = 'AboveNormal'
        $boosted++
    }
}
Write-Host "[+] Process priority: $($bots.Count) processes, $boosted boosted to AboveNormal" -ForegroundColor Green

# 3. Network optimization — Disable Nagle's algorithm for lower latency
$nagleKey = "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters"
try {
    $current = Get-ItemProperty -Path $nagleKey -Name "TcpNoDelay" -ErrorAction SilentlyContinue
    if ($null -eq $current -or $current.TcpNoDelay -ne 1) {
        Set-ItemProperty -Path $nagleKey -Name "TcpNoDelay" -Value 1 -Type DWord -ErrorAction Stop
        Write-Host "[+] Nagle's algorithm disabled (lower network latency)" -ForegroundColor Green
    } else {
        Write-Host "[=] Nagle's algorithm already disabled" -ForegroundColor Gray
    }
} catch {
    Write-Host "[!] Nagle tweak requires admin — run as Administrator for network optimization" -ForegroundColor Yellow
}

# 4. System report
Write-Host ""
Write-Host "═══ SYSTEM STATUS ═══" -ForegroundColor Cyan

$cpu = Get-CimInstance Win32_Processor
Write-Host "CPU: $($cpu.Name)" -ForegroundColor White
Write-Host "  Cores: $($cpu.NumberOfCores) / Threads: $($cpu.NumberOfLogicalProcessors) / Load: $($cpu.LoadPercentage)%"

$os = Get-CimInstance Win32_OperatingSystem
$ramUsed = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory)/1MB, 1)
$ramFree = [math]::Round($os.FreePhysicalMemory/1MB, 1)
$ramTotal = [math]::Round($os.TotalVisibleMemorySize/1MB, 1)
Write-Host "RAM: ${ramUsed}GB / ${ramTotal}GB used (${ramFree}GB free)" -ForegroundColor White

# GPU check
try {
    $gpuInfo = nvidia-smi --query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null
    if ($gpuInfo) {
        $parts = $gpuInfo.Split(",").Trim()
        Write-Host "GPU: $($parts[0])" -ForegroundColor White
        Write-Host "  Temp: $($parts[1])°C | Power: $($parts[2])W | VRAM: $($parts[3])/$($parts[4]) MB | Util: $($parts[5])%"
    }
} catch {}

# Bot process details
Write-Host ""
Write-Host "═══ BOT PROCESSES ═══" -ForegroundColor Cyan
$bots | ForEach-Object {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
    $ramMB = [math]::Round($_.WorkingSet64/1MB)
    $cpuSec = [math]::Round($_.CPU, 1)
    Write-Host "  PID $($_.Id): ${ramMB}MB RAM, ${cpuSec}s CPU — $($_.PriorityClass)" -ForegroundColor White
}

$totalRAM = [math]::Round(($bots | Measure-Object WorkingSet64 -Sum).Sum / 1MB)
Write-Host ""
Write-Host "Total bot RAM: ${totalRAM}MB across $($bots.Count) processes" -ForegroundColor Green
Write-Host ""
Write-Host "Performance tuning complete." -ForegroundColor Cyan
