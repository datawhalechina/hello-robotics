#!/usr/bin/env bash
set -eo pipefail
CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$CHAPTER_DIR/g2_chapter10_2_nav"

# 先生成地图，再把便于阅读的顶层配置同步到标准 ROS 2 包。
python3 "$CHAPTER_DIR/build_map.py"
cp "$CHAPTER_DIR/maps/chapter10_2_map.yaml" \
   "$CHAPTER_DIR/maps/chapter10_2_map.pgm" "$PACKAGE_DIR/maps/"
cp "$CHAPTER_DIR/config/nav2_params.yaml" \
   "$CHAPTER_DIR/config/chapter10_2_navigation.rviz" "$PACKAGE_DIR/config/"
cp "$CHAPTER_DIR/launch/nav2_navigation.launch.py" "$PACKAGE_DIR/launch/"

source /opt/ros/humble/setup.bash
colcon --log-base "$CHAPTER_DIR/ros2_log" build \
  --base-paths "$PACKAGE_DIR" \
  --build-base "$CHAPTER_DIR/ros2_build" \
  --install-base "$CHAPTER_DIR/ros2_install" \
  --symlink-install
printf '\n构建完成。下一步：bash %s/run_nav2.sh\n' "$CHAPTER_DIR"
