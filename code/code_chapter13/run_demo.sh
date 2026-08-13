#!/usr/bin/env bash
# 终端 1：Isaac Sim、G2 传感器、YOLO-World、本地 Qwen3-VL 和 Nav2 bridge。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/robot/isaac-sim}"
INTERNAL_ROS_LIB="$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/humble/lib"
if [[ -z "${VLM_PYTHON:-}" ]]; then
  # 这是系统运行环境，不是章节模型路径；可通过 VLM_PYTHON 覆盖。
  VLM_PYTHON="/home/robot/miniconda3/envs/navigation/bin/python"
fi
export VLM_PYTHON

if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
  echo "找不到 Isaac Sim：$ISAAC_SIM_ROOT/python.sh" >&2
  exit 1
fi
if [[ ! -x "$VLM_PYTHON" ]]; then
  echo "找不到本地 VLM Python：$VLM_PYTHON" >&2
  echo "请先激活 VLM Conda 环境，或设置 VLM_PYTHON=/你的环境/bin/python。" >&2
  exit 1
fi

unset PYTHONPATH OLD_PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
export ROS_DISTRO=humble
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3
export AMENT_PREFIX_PATH=/opt/ros/humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$INTERNAL_ROS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$CHAPTER_DIR/ros2_runtime_log}"
mkdir -p "$ROS_LOG_DIR" "$CHAPTER_DIR/outputs"

exec "$ISAAC_SIM_ROOT/python.sh" "$CHAPTER_DIR/demo_vlm_navigation.py" "$@"
