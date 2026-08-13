#!/usr/bin/env bash
# 终端 2：静态地图、Nav2 服务器与 RViz。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$CHAPTER_DIR/ros2_runtime_log}"
mkdir -p "$ROS_LOG_DIR"
source /opt/ros/humble/setup.bash
exec /usr/bin/python3 "$CHAPTER_DIR/launch/nav2_navigation.launch.py" "$@"
