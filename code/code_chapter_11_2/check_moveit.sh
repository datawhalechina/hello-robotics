#!/usr/bin/env bash
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROS 2 Humble 使用系统 Python 3.10；避免当前 Conda 环境污染 CMake/插件加载。
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/" | paste -sd: -)"
  export PATH
fi
unset PYTHONHOME
source /opt/ros/humble/setup.bash
source "$CHAPTER_DIR/ros2_install/setup.bash"
echo '--- nodes ---'; ros2 node list
echo '--- important actions ---'; ros2 action list | grep -E 'move_action|follow_joint_trajectory' || true
echo '--- one joint state ---'; timeout 5 ros2 topic echo /joint_states --once || true
echo '--- static TF: world -> arm_base_link ---'; timeout 5 ros2 run tf2_ros tf2_echo world arm_base_link || true
echo '--- arm TF: arm_base_link -> gripper_r_center_link ---'; timeout 5 ros2 run tf2_ros tf2_echo arm_base_link gripper_r_center_link || true
echo '--- planning scene ---'; timeout 5 ros2 topic echo /monitored_planning_scene --once --field is_diff || true
