$botDir = 'D:\042021\CryptoBot'
$fingerprintPath = Join-Path $botDir 'data\state\runtime_fingerprint_latest.json'
$lockedPath = Join-Path $botDir 'data\state\locked_profile.json'

if (-not (Test-Path $fingerprintPath)) {
    Write-Host "Fingerprint not found at $fingerprintPath" -ForegroundColor Yellow
    Write-Host "Start the bot once to generate a fingerprint, then run this again." -ForegroundColor Yellow
    exit 1
}

try {
    $fp = Get-Content $fingerprintPath -Raw | ConvertFrom-Json
    if (-not $fp.config) {
        Write-Host "Fingerprint missing config block." -ForegroundColor Red
        exit 1
    }

    $payload = [ordered]@{
        created_at = (Get-Date).ToString('s')
        source_fingerprint = $fingerprintPath
        source_engine_sha256 = $fp.engine_sha256
        config_overrides = $fp.config
    }

    $dir = Split-Path -Parent $lockedPath
    if (-not (Test-Path $dir)) {
        New-Item -Path $dir -ItemType Directory -Force | Out-Null
    }

    $json = $payload | ConvertTo-Json -Depth 12
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($lockedPath, $json, $utf8NoBom)
    Write-Host "Locked profile saved:" -ForegroundColor Green
    Write-Host $lockedPath -ForegroundColor Green
}
catch {
    Write-Host "Failed to create locked profile: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
