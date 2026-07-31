#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec rviz2 -d "$SCRIPT_DIR/rviz/chapter8_mapping.rviz" --ros-args -p use_sim_time:=true
