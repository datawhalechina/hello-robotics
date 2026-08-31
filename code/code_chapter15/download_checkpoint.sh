#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$ROOT/checkpoints/pi05_base}"
mkdir -p "$(dirname "$DEST")"
if command -v gsutil >/dev/null 2>&1; then
  gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base "$DEST"
elif command -v gcloud >/dev/null 2>&1; then
  gcloud storage cp --recursive gs://openpi-assets/checkpoints/pi05_base "$DEST"
else
  echo "需要安装 gsutil/gcloud，或设置 CHAPTER15_PI05_BASE=/absolute/path/to/pi05_base" >&2
  exit 1
fi
printf 'pi0.5 base checkpoint: %s\n' "$DEST"
