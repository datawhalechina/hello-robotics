#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OPENPI="$ROOT/third_party/openpi"
command -v uv >/dev/null 2>&1 || { echo "请先安装 uv" >&2; exit 1; }
cd "$OPENPI"
uv sync
# RTX 50-series (Blackwell, sm_120) requires the CUDA 12.8 PyTorch wheels.
# OpenPI pins torch 2.7.1, so keep that version and replace only its CUDA build.
uv pip install \
  --python "$OPENPI/.venv/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --reinstall-package torch \
  --reinstall-package torchvision \
  "torch==2.7.1" \
  "torchvision==0.22.1"
uv pip install --python "$OPENPI/.venv/bin/python" pytest
"$OPENPI/.venv/bin/python" - <<'PYTORCH_CHECK'
import torch

arch_flags = torch._C._cuda_getArchFlags()
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, arch={arch_flags}")
if "sm_120" not in arch_flags.split():
    raise RuntimeError("PyTorch wheel does not contain RTX 50-series sm_120 kernels")
PYTORCH_CHECK
cat > "$ROOT/.env.example" <<EOF
export CHAPTER15_OPENPI_ROOT="$OPENPI"
export CHAPTER15_ISAAC_PYTHON="/home/robot/isaac-sim/python.sh"
export CHAPTER15_PI05_BASE="$ROOT/checkpoints/pi05_base"
export CHAPTER15_OUTPUT_ROOT="$ROOT"
EOF
echo "环境完成。Isaac Sim 脚本仍请使用本机 Isaac Sim 的 python.sh 运行。"
