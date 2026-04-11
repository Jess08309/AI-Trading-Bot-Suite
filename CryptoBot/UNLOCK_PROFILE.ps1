$lockedPath = 'C:\Bot\data\state\locked_profile.json'

if (Test-Path $lockedPath) {
    Remove-Item $lockedPath -Force
    Write-Host "Removed locked profile: $lockedPath" -ForegroundColor Green
}
else {
    Write-Host "No locked profile found at: $lockedPath" -ForegroundColor Yellow
}
