#!/usr/bin/env bash
set -euo pipefail

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN_HOME="${XDG_BIN_HOME:-$HOME/.local/bin}"

rm -rf "$DATA_HOME/SpatioTemporalLabeler"
rm -f "$BIN_HOME/spatiotemporal-labeler"
rm -f "$DATA_HOME/applications/spatiotemporal-labeler.desktop"
echo "SpatioTemporal Labeler 已从当前用户环境卸载。"
