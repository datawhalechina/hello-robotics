#!/usr/bin/env bash
# 终端 2：启动 map_server、Nav2 全部服务器和 RViz。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$CHAPTER_DIR/ros2_runtime_log}"
mkdir -p "$ROS_LOG_DIR"
source /opt/ros/humble/setup.bash
if [[ ! -f "$CHAPTER_DIR/ros2_install/setup.bash" ]]; then
  echo "尚未构建，请先运行：bash $CHAPTER_DIR/build_nav2.sh" >&2
  exit 1
fi
source "$CHAPTER_DIR/ros2_install/setup.bash"
exec ros2 launch g2_chapter10_2_nav nav2_navigation.launch.py "$@"
