#!/usr/bin/env bash
# 用 Isaac Sim 自带的 Python 3.11 ROS2 bridge 运行第八章示例。
# 系统 ROS Humble 通常是 Python 3.10，不能把它的 PYTHONPATH 直接交给 Isaac Python。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/robot/isaac-sim}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
BRIDGE_ROOT="$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/$ROS_DISTRO"
BRIDGE_PYTHON="$BRIDGE_ROOT/rclpy"
BRIDGE_LIB="$BRIDGE_ROOT/lib"

if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
    echo "错误：找不到 $ISAAC_SIM_ROOT/python.sh" >&2
    echo "可先设置 ISAAC_SIM_ROOT=/path/to/isaac-sim" >&2
    exit 1
fi
if [[ ! -d "$BRIDGE_PYTHON" || ! -d "$BRIDGE_LIB" ]]; then
    echo "错误：找不到 Isaac Sim 内置 ROS2 bridge：$BRIDGE_ROOT" >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    set -- "$SCRIPT_DIR/demo_3d_mapping.py"
elif [[ "$1" != /* && -f "$REPO_ROOT/$1" ]]; then
    set -- "$REPO_ROOT/$1" "${@:2}"
elif [[ "$1" != /* && -f "$SCRIPT_DIR/$1" ]]; then
    set -- "$SCRIPT_DIR/$1" "${@:2}"
fi

# 不让系统 /opt/ros 的 Python 3.10 rclpy 抢先进入 Isaac Python 3.11。
# ROS bridge 扩展启动后会自动把它自带的 rclpy 加入 sys.path。
export ROS_DISTRO
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
unset PYTHONPATH
export LD_LIBRARY_PATH="$BRIDGE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ROS 默认日志目录在受限/容器环境中可能不可写，改到本章输出目录。
export ROS_LOG_DIR="${ROS_LOG_DIR:-$REPO_ROOT/outputs/chapter8/ros_logs}"
mkdir -p "$ROS_LOG_DIR"

# 避免 Isaac 的 python.sh 将当前 Conda 环境误认为目标运行环境。
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONHOME

cd "$REPO_ROOT"
exec "$ISAAC_SIM_ROOT/python.sh" "$@"
