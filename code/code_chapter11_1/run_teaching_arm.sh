#!/usr/bin/env bash
# 终端 1：运行不使用 MoveIt 2 的机械臂规划、优化、跟踪与抓取示例。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CHAPTER_DIR/../.." && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/robot/isaac-sim}"
unset PYTHONPATH
unset PYTHONHOME
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
cd "$PROJECT_ROOT"
exec "$ISAAC_SIM_ROOT/python.sh" "$CHAPTER_DIR/demo_pick_red.py" "$@"
