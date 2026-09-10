#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ISAAC_SIM="${ISAAC_SIM:-/home/robot/isaac-sim}"
[[ -x "$ISAAC_SIM/python.sh" ]] || { echo '请设置 ISAAC_SIM 为 Isaac Sim 独立安装目录' >&2; exit 1; }
# 仅清理这个子进程的环境，不修改当前 shell/现有 conda。
unset PYTHONHOME PYTHONPATH PYTHONEXE LD_PRELOAD CONDA_PREFIX
export PYTHONNOUSERSITE=1
# 6x6 IK 小矩阵不应启动大量 BLAS 线程。
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
exec "$ISAAC_SIM/python.sh" "$ROOT/demo_grasp.py" "$@"
