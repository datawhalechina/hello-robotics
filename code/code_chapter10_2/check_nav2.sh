#!/usr/bin/env bash
# 快速检查 Nav2 所需的关键话题、TF 和 action 是否存在。
set -eo pipefail
source /opt/ros/humble/setup.bash
printf '%s\n' '=== 关键话题 ==='
ros2 topic list | grep -E '^/(clock|cmd_vel|map|odom|scan|plan|local_plan)$' || true
printf '%s\n' '=== NavigateToPose action ==='
ros2 action list | grep '/navigate_to_pose' || true
printf '%s\n' '=== TF: map -> base_link ==='
timeout 5 ros2 run tf2_ros tf2_echo map base_link || true
