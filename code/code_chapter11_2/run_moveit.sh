#!/usr/bin/env bash
# 终端 2：move_group、robot_state_publisher 与 RViz。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROS 2 Humble 使用系统 Python 3.10；避免当前 Conda 环境污染 CMake/插件加载。
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/" | paste -sd: -)"
  export PATH
fi
unset PYTHONHOME
source /opt/ros/humble/setup.bash
if [[ ! -f "$CHAPTER_DIR/ros2_install/setup.bash" ]]; then
  echo "尚未构建，请先运行：bash $CHAPTER_DIR/build_moveit.sh" >&2
  exit 1
fi
source "$CHAPTER_DIR/ros2_install/setup.bash"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$CHAPTER_DIR/ros2_runtime_log}"
mkdir -p "$ROS_LOG_DIR"
exec ros2 launch g2_chapter11_moveit move_group.launch.py "$@"
