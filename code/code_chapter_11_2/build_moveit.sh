#!/usr/bin/env bash
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[构建] 这里只编译 MoveIt 2 包，不会启动 Isaac Sim。"
# ROS 2 Humble 使用系统 Python 3.10；避免当前 Conda 环境污染 CMake/插件加载。
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/" | paste -sd: -)"
  export PATH
fi
unset PYTHONHOME
source /opt/ros/humble/setup.bash
rm -rf "$CHAPTER_DIR/ros2_build" "$CHAPTER_DIR/ros2_install" "$CHAPTER_DIR/ros2_log"
colcon --log-base "$CHAPTER_DIR/ros2_log" build \
  --base-paths "$CHAPTER_DIR/g2_chapter11_moveit" \
  --build-base "$CHAPTER_DIR/ros2_build" \
  --install-base "$CHAPTER_DIR/ros2_install" \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
echo "构建完成：$CHAPTER_DIR/ros2_install"
