#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="${STL_VERSION:-0.2.1}"
PACKAGE_NAME="SpatioTemporalLabeler-${VERSION}-linux-x86_64"
RELEASE_ROOT="$PROJECT_ROOT/release"
PACKAGE_ROOT="$RELEASE_ROOT/$PACKAGE_NAME"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -c "import PyInstaller, PySide6, vtk, pyqtgraph, nrrd, numpy"
"$PYTHON_BIN" -m PyInstaller --noconfirm --clean packaging/SpatioTemporalLabeler.spec

rm -rf "$PACKAGE_ROOT"
mkdir -p "$PACKAGE_ROOT"
cp -a "$PROJECT_ROOT/dist/SpatioTemporalLabeler" "$PACKAGE_ROOT/"
cp "$PROJECT_ROOT/src/spatiotemporal_labeler/assets/app-icon.png" "$PACKAGE_ROOT/SpatioTemporalLabeler/app-icon.png"
cp "$SCRIPT_DIR/install.sh" "$PACKAGE_ROOT/install.sh"
cp "$SCRIPT_DIR/uninstall.sh" "$PACKAGE_ROOT/uninstall.sh"
cp "$PROJECT_ROOT/packaging/PORTABLE-README.zh-CN.txt" "$PACKAGE_ROOT/README.txt"
chmod +x "$PACKAGE_ROOT/install.sh" "$PACKAGE_ROOT/uninstall.sh"

tar -C "$RELEASE_ROOT" -czf "$RELEASE_ROOT/$PACKAGE_NAME.tar.gz" "$PACKAGE_NAME"
(
    cd "$RELEASE_ROOT"
    sha256sum "$PACKAGE_NAME.tar.gz" > "$PACKAGE_NAME.tar.gz.sha256"
)
echo "绿色目录: $PACKAGE_ROOT"
echo "发行压缩包: $RELEASE_ROOT/$PACKAGE_NAME.tar.gz"
echo "校验文件: $RELEASE_ROOT/$PACKAGE_NAME.tar.gz.sha256"
