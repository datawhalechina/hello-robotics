#!/usr/bin/env bash
# 终端 1：运行 Isaac Sim、G2 底盘、传感器和 ROS 2 桥接器。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/robot/isaac-sim}"
INTERNAL_ROS_LIB="$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/humble/lib"

if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
  echo "找不到 Isaac Sim：$ISAAC_SIM_ROOT/python.sh" >&2
  echo "请设置 ISAAC_SIM_ROOT 后重试。" >&2
  exit 1
fi

# 系统 ROS Humble rclpy 使用 Python 3.10；Isaac Sim 使用 Python 3.11。
# 这里使用 Isaac ROS 2 Bridge 自带库，避免 Python ABI 冲突。
unset PYTHONPATH OLD_PYTHONPATH
export ROS_DISTRO=humble
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3
export AMENT_PREFIX_PATH=/opt/ros/humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$INTERNAL_ROS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$CHAPTER_DIR/ros2_runtime_log}"
mkdir -p "$ROS_LOG_DIR"

exec "$ISAAC_SIM_ROOT/python.sh" "$CHAPTER_DIR/nav2_bridge.py" "$@"
