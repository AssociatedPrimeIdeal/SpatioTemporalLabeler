$ErrorActionPreference = "Stop"

$SourceDir = Join-Path $PSScriptRoot "SpatioTemporalLabeler"
$TargetDir = Join-Path $env:LOCALAPPDATA "SpatioTemporalLabeler"
$Executable = Join-Path $TargetDir "SpatioTemporalLabeler.exe"

if (-not (Test-Path (Join-Path $SourceDir "SpatioTemporalLabeler.exe"))) {
    throw "Portable application directory not found: $SourceDir"
}

if (Test-Path $TargetDir) {
    Remove-Item $TargetDir -Recurse -Force
}
Copy-Item $SourceDir $TargetDir -Recurse

$Shell = New-Object -ComObject WScript.Shell
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$Shortcut = $Shell.CreateShortcut((Join-Path $StartMenu "SpatioTemporal Labeler.lnk"))
$Shortcut.TargetPath = $Executable
$Shortcut.WorkingDirectory = $TargetDir
$Shortcut.IconLocation = "$Executable,0"
$Shortcut.Save()

Write-Host "Installed to $TargetDir"
Write-Host "Created the SpatioTemporal Labeler Start menu shortcut."
