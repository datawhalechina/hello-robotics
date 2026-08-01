#!/usr/bin/env bash
# 启动非 MoveIt 教学示例的 RViz 可视化。
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash

# 与 Isaac Sim 教学程序使用相同的 DDS 实现和 Domain，避免 RViz 能启动但
# 看不到 /chapter11/* 话题。
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

# 教学 Marker 已在发布端从 G2 arm_base_link 转换到 ROS Z-up world，
# 因此 RViz 不依赖系统中可能重复存在的 robot_state_publisher/TF 树。
exec rviz2 -d "$CHAPTER_DIR/config/chapter11_1_planning.rviz"
