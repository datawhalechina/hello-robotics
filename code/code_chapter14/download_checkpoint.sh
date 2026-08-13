#!/usr/bin/env bash
# 可选工具：下载公开的 G2 π0.5 baseline 到本章 checkpoints/。
# 程序不会自动执行此脚本；运行前需自行安装 ModelScope CLI。
set -euo pipefail

DATASET_ID="agibot_world/GenieSim3.0-Dataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_NAME="${1:-instruction_and_robust_pi05}"
CHECKPOINT_ROOT="${SCRIPT_DIR}/checkpoints"

case "${CHECKPOINT_NAME}" in
  instruction_and_robust_pi05|manipulation_pi05|spatial_pi05) ;;
  -h|--help)
    echo "用法: bash download_checkpoint.sh [NAME]"
    echo "NAME: instruction_and_robust_pi05 | manipulation_pi05 | spatial_pi05"
    echo "权重固定保存到: ${CHECKPOINT_ROOT}"
    exit 0
    ;;
  *)
    echo "不支持的权重名称: ${CHECKPOINT_NAME}" >&2
    exit 2
    ;;
esac

if ! command -v modelscope >/dev/null 2>&1; then
  echo "找不到 modelscope，请先在你选择的 Python 环境中安装 ModelScope CLI。" >&2
  exit 1
fi

REMOTE_PATH="checkpoints/${CHECKPOINT_NAME}"
FINAL_DIR="${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT
mkdir -p "${FINAL_DIR}"

echo "下载 ${DATASET_ID}/${REMOTE_PATH}"
echo "保存到 ${FINAL_DIR}"
modelscope download \
  --dataset "${DATASET_ID}" \
  --include "${REMOTE_PATH}/**" \
  --local_dir "${TEMP_DIR}"
cp -a "${TEMP_DIR}/${REMOTE_PATH}/." "${FINAL_DIR}/"
echo "[完成] ${FINAL_DIR}"
