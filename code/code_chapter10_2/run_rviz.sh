#!/usr/bin/env bash
# Nav2 已使用 use_rviz:=false 启动时，可单独运行本脚本。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
export ROS_LOG_DIR="${ROS_LOG_DIR:-$CHAPTER_DIR/ros2_runtime_log}"
mkdir -p "$ROS_LOG_DIR"
exec rviz2 -d "$CHAPTER_DIR/config/chapter10_2_navigation.rviz"
