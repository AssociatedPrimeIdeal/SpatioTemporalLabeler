#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/SpatioTemporalLabeler"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN_HOME="${XDG_BIN_HOME:-$HOME/.local/bin}"
APPLICATIONS_HOME="$DATA_HOME/applications"
TARGET_DIR="$DATA_HOME/SpatioTemporalLabeler"

if [[ ! -x "$SOURCE_DIR/SpatioTemporalLabeler" ]]; then
    echo "未找到绿色程序目录: $SOURCE_DIR" >&2
    exit 1
fi

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR" "$BIN_HOME" "$APPLICATIONS_HOME"
cp -a "$SOURCE_DIR/." "$TARGET_DIR/"
ln -sfn "$TARGET_DIR/SpatioTemporalLabeler" "$BIN_HOME/spatiotemporal-labeler"

cat > "$APPLICATIONS_HOME/spatiotemporal-labeler.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SpatioTemporal Labeler
Comment=3D+t medical image segmentation editor
Exec=$TARGET_DIR/SpatioTemporalLabeler %F
Icon=$TARGET_DIR/app-icon.png
Terminal=false
Categories=Graphics;Science;MedicalSoftware;
StartupNotify=true
EOF

chmod +x "$TARGET_DIR/SpatioTemporalLabeler"
echo "已安装到 $TARGET_DIR"
echo "命令行入口: $BIN_HOME/spatiotemporal-labeler"
