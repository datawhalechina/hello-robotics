#!/usr/bin/env bash
# 运行“不使用 Nav2”的教学导航，并使用 Isaac Sim 自带的 Python 3.11 rclpy。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/robot/isaac-sim}"
INTERNAL_ROS_LIB="$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/humble/lib"

if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
  echo "找不到 Isaac Sim：$ISAAC_SIM_ROOT/python.sh" >&2
  echo "请设置 ISAAC_SIM_ROOT 后重试。" >&2
  exit 1
fi

# 系统 ROS Humble 的 rclpy 是 Python 3.10，不能被 Isaac Sim Python 3.11 导入。
unset PYTHONPATH OLD_PYTHONPATH
export ROS_DISTRO=humble
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3
export AMENT_PREFIX_PATH=/opt/ros/humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$INTERNAL_ROS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$CHAPTER_DIR/ros2_runtime_log}"
mkdir -p "$ROS_LOG_DIR"

exec "$ISAAC_SIM_ROOT/python.sh" "$CHAPTER_DIR/demo_navigation.py" "$@"
