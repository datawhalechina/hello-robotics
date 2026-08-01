#!/usr/bin/env bash
# 终端 1：Isaac Sim、/joint_states 与 FollowJointTrajectory action server。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CHAPTER_DIR/../.." && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/robot/isaac-sim}"
unset PYTHONPATH
unset PYTHONHOME
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$CHAPTER_DIR/ros2_runtime_log}"
mkdir -p "$ROS_LOG_DIR"
cd "$PROJECT_ROOT"
exec "$ISAAC_SIM_ROOT/python.sh" "$CHAPTER_DIR/isaac_moveit_bridge.py" "$@"
