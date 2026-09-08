#!/usr/bin/env bash
set -euo pipefail

source_uri="gs://openpi-assets/checkpoints/pi05_base"
checkpoint_name="pi05_base"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/acotvla/.venv/bin/python"
final_dir="$root/checkpoints/base/$checkpoint_name"

if [[ -d "$final_dir/params" ]]; then
  echo "[已存在] $final_dir"
  exit 0
fi
if [[ ! -x "$python_bin" ]]; then
  echo "找不到 ACoT-VLA 训练环境，请先按照教程完成环境配置。" >&2
  exit 1
fi

downloaded="$("$python_bin" - "$source_uri" <<'PY'
import sys
from openpi.shared import download

print(download.maybe_download(sys.argv[1], gs={"token": "anon"}))
PY
)"
if [[ ! -d "$downloaded/params" ]]; then
  echo "下载结果缺少 params：$downloaded" >&2
  exit 1
fi
mkdir -p "$final_dir"
cp -a "$downloaded/." "$final_dir/"
test -d "$final_dir/params"
echo "[完成] $final_dir"
