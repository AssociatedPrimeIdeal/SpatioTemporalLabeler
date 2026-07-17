$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$Version = if ($env:STL_VERSION) { $env:STL_VERSION } else { "0.2.5" }
$PackageName = "SpatioTemporalLabeler-$Version-windows-x64"
$ReleaseRoot = Join-Path $ProjectRoot "release"
$PackageRoot = Join-Path $ReleaseRoot $PackageName

Set-Location $ProjectRoot
& $PythonBin -c "import PyInstaller, PySide6, vtk, pyqtgraph, nrrd, numpy"
if ($LASTEXITCODE -ne 0) { throw "Missing build dependencies. Run: pip install '.[portable]'" }
& $PythonBin -m PyInstaller --noconfirm --clean packaging/SpatioTemporalLabeler.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

if (Test-Path $PackageRoot) {
    Remove-Item $PackageRoot -Recurse -Force
}
New-Item $PackageRoot -ItemType Directory -Force | Out-Null
Copy-Item (Join-Path $ProjectRoot "dist\SpatioTemporalLabeler") $PackageRoot -Recurse
Copy-Item (Join-Path $ProjectRoot "src\spatiotemporal_labeler\assets\app-icon.png") (Join-Path $PackageRoot "SpatioTemporalLabeler\app-icon.png")
Copy-Item (Join-Path $PSScriptRoot "install.ps1") $PackageRoot
Copy-Item (Join-Path $PSScriptRoot "uninstall.ps1") $PackageRoot
Copy-Item (Join-Path $ProjectRoot "packaging\PORTABLE-README.zh-CN.txt") (Join-Path $PackageRoot "README.txt")

$Archive = Join-Path $ReleaseRoot "$PackageName.zip"
if (Test-Path $Archive) {
    Remove-Item $Archive -Force
}
Compress-Archive -Path $PackageRoot -DestinationPath $Archive -CompressionLevel Optimal
$Hash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
"$Hash  $PackageName.zip" | Set-Content "$Archive.sha256" -Encoding ascii
Write-Host "Portable directory: $PackageRoot"
Write-Host "Release archive: $Archive"
Write-Host "Checksum file: $Archive.sha256"
