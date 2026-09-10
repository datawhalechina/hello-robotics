#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-vision}"
[[ "$MODE" == vision || "$MODE" == graspgenx ]] || { echo 'usage: bash setup_env.sh [vision|graspgenx]'; exit 2; }
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
ENV="$ROOT/.envs/$MODE"
[[ -x "$ENV/bin/python" ]] || "$CONDA" create -y --prefix "$ENV" python=3.11 pip
PY="$ENV/bin/python"
mkdir -p "$ROOT/outputs" "$ROOT/weights/ultralytics_config"
export PIP_DEFAULT_TIMEOUT=90
# cu128 支持本机 5090；不安装/替换系统 CUDA toolkit。SAM2 CUDA 后处理扩展不必编译。
"$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if [[ "$MODE" == vision ]]; then
  "$PY" -m pip install -r "$ROOT/requirements.txt"
  SAM2_BUILD_CUDA=0 "$PY" -m pip install --no-build-isolation --no-deps -e "$ROOT/third_party/sam2"
  "$PY" -m pip install -e "$ROOT/third_party/CLIP"
else
  # 上游限制 torch<2.7；本章为5090采用已实测的cu128组合，以no-deps隔离差异。
  # 保留上游源码未改动，依赖列表在独立文件；pip check 会报告上游metadata版本冲突，详见README。
  "$PY" -m pip install -r "$ROOT/requirements-graspgenx.txt"
  "$PY" -m pip install --no-build-isolation --no-deps -e "$ROOT/third_party/GraspGenX"
fi
"$PY" -m pip freeze > "$ROOT/outputs/environment-$MODE.txt"
echo "环境就绪：conda activate $ENV"
