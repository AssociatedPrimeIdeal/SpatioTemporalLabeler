$ErrorActionPreference = "Stop"

$TargetDir = Join-Path $env:LOCALAPPDATA "SpatioTemporalLabeler"
$Shortcut = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\SpatioTemporal Labeler.lnk"

if (Test-Path $TargetDir) {
    Remove-Item $TargetDir -Recurse -Force
}
if (Test-Path $Shortcut) {
    Remove-Item $Shortcut -Force
}

Write-Host "SpatioTemporal Labeler has been uninstalled."
