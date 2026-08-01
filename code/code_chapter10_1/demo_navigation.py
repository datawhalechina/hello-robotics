"""示例 1：不使用 Nav2，运行完整导航教学管线。

运行后会自动前往默认目标；启动 RViz 后也可使用“2D Goal Pose”随时发送新目标。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys
import traceback

# 允许直接运行本文件，并复用第四章底盘控制代码。
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from code_chapter4.base_controller import G2BaseController
from code_chapter4.config import ControlLimits, RobotGeometry
from code_chapter4.kinematics import SwerveKinematics

try:
    from .build_map import build_navigation_map
    from .config import MapConfig, PlannerConfig, SafetyConfig, SimulationConfig, TrackerConfig
    from .geometry import Pose2D
    from .global_planner import AStarPlanner
    from .local_planner import LocalPlanner
    from .navigator import NavigationState, TeachingNavigator
    from .path_tracker import HolonomicPathTracker
    from .ros2_interface import NavigationRos2Interface
    from .simulation import G2NavigationSimulation
    from .trajectory import TrajectoryOptimizer
except ImportError:
    from build_map import build_navigation_map
    from config import MapConfig, PlannerConfig, SafetyConfig, SimulationConfig, TrackerConfig
    from geometry import Pose2D
    from global_planner import AStarPlanner
    from local_planner import LocalPlanner
    from navigator import NavigationState, TeachingNavigator
    from path_tracker import HolonomicPathTracker
    from ros2_interface import NavigationRos2Interface
    from simulation import G2NavigationSimulation
    from trajectory import TrajectoryOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无 Isaac Sim 窗口运行")
    parser.add_argument("--no-rviz", action="store_true", help="不初始化 ROS 2/RViz 接口")
    parser.add_argument("--no-dynamic-obstacle", action="store_true", help="关闭移动障碍")
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "YAW"), help="覆盖默认目标")
    parser.add_argument("--timeout", type=float, default=90.0, help="最大仿真时间，秒")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_config = replace(
        SimulationConfig(),
        headless=args.headless,
        dynamic_obstacle=not args.no_dynamic_obstacle,
    )
    if args.goal:
        sim_config = replace(sim_config, default_goal=Pose2D(*args.goal))

    map_config = MapConfig()
    planner_config = PlannerConfig()
    tracker_config = TrackerConfig()
    safety_config = SafetyConfig()
    raw_map = build_navigation_map(map_config)
    planning_map = raw_map.inflated(map_config.inflation_radius)

    planner = AStarPlanner(
        planning_map,
        allow_diagonal=planner_config.allow_diagonal,
        heuristic_weight=planner_config.heuristic_weight,
    )
    optimizer = TrajectoryOptimizer(planning_map, planner_config.path_spacing)
    local_planner = LocalPlanner(
        planning_map,
        lookahead_distance=planner_config.local_lookahead_distance,
        window_radius=planner_config.local_window_radius,
        path_spacing=planner_config.path_spacing,
        dynamic_padding=map_config.inflation_radius,
        detour_hold_cycles=planner_config.detour_hold_cycles,
    )
    tracker = HolonomicPathTracker(**tracker_config.__dict__, costmap=planning_map)
    navigator = TeachingNavigator(
        planning_map,
        planner,
        optimizer,
        local_planner,
        tracker,
        collision_horizon=safety_config.collision_horizon,
        collision_step=safety_config.collision_step,
        progress_timeout=safety_config.progress_timeout,
        minimum_progress=safety_config.minimum_progress,
        max_recovery_attempts=safety_config.max_recovery_attempts,
        local_replan_period=planner_config.local_replan_period,
        safety_costmap=raw_map.inflated(map_config.robot_radius),
    )

    simulation = G2NavigationSimulation(sim_config, enable_ros2=not args.no_rviz)
    geometry = RobotGeometry()
    kinematics = SwerveKinematics(geometry.wheel_positions, geometry.wheel_radius)
    # 导航比单独速度演示更强调稳定性：降低加速度，避免局部路径变化时冲击底盘。
    navigation_limits = ControlLimits(
        max_linear_speed=tracker_config.max_linear_speed,
        max_angular_speed=tracker_config.max_angular_speed,
        max_linear_acceleration=0.40,
        max_angular_acceleration=0.90,
    )
    base = G2BaseController(simulation.robot, kinematics, navigation_limits)
    ros2 = NavigationRos2Interface() if not args.no_rviz else None

    initial_pose = simulation.get_pose2d()
    goal = sim_config.default_goal
    navigator.set_goal(goal, initial_pose, now=simulation.sim_time)
    if ros2:
        ros2.publish_map(raw_map, simulation.sim_time)

    control_period_steps = max(1, sim_config.physics_hz // 20)
    visualization_period_steps = max(1, sim_config.physics_hz // 10)
    scan_angles = [(-math.pi + index * 2.0 * math.pi / 359.0) for index in range(360)]
    last_state = None
    last_message = ""
    command = None
    output = None

    print("\n=== 第十章教学版：不使用 Nav2 的完整导航 ===", flush=True)
    print(f"起点：({initial_pose.x:.2f}, {initial_pose.y:.2f})", flush=True)
    print(f"目标：({goal.x:.2f}, {goal.y:.2f}, {goal.yaw:.2f})", flush=True)
    print("算法链：A* → 路径优化 → 局部重规划 → 全向前视跟踪 → 碰撞监控", flush=True)

    try:
        max_steps = int(args.timeout / sim_config.physics_dt)
        for step in range(max_steps):
            # 无窗口模式下 Kit 可能不维护窗口运行标志，但物理世界仍可正常推进。
            if not sim_config.headless and not simulation.is_running():
                break
            pose = simulation.get_pose2d()
            tilt_angle = simulation.get_tilt_angle()
            if tilt_angle > safety_config.max_tilt_angle:
                raise RuntimeError(
                    f"底盘倾斜达到 {math.degrees(tilt_angle):.1f}°，已触发防倾倒停车"
                )
            dynamic = simulation.dynamic_obstacles()

            if ros2:
                ros2.spin_once()
                requested_goal = ros2.take_goal()
                if requested_goal is not None:
                    goal = requested_goal
                    navigator.set_goal(goal, pose, now=simulation.sim_time)

            if step % control_period_steps == 0:
                output = navigator.update(pose, dynamic, now=simulation.sim_time)
                command = output.command
                if output.state != last_state or output.message != last_message:
                    print(f"[{output.state.value}] {output.message}", flush=True)
                    last_state, last_message = output.state, output.message
            # 非控制周期保持上一条命令，底层仍以物理频率更新限速和车轮关节。
            assert output is not None and command is not None

            base.set_velocity(command.vx, command.vy, command.wz)
            base.update(sim_config.physics_dt)

            # TF/里程计以 20 Hz 先发布；同时间戳的激光必须在 TF 之后发布。
            if ros2 and step % control_period_steps == 0:
                ros2.publish_pose(simulation.sim_time, pose, command)
                ros2.publish_command(command)

            if ros2 and step % visualization_period_steps == 0:
                ros2.publish_visualization(
                    simulation.sim_time,
                    pose,
                    output.global_path,
                    output.local_path,
                    output.target,
                    goal,
                    dynamic,
                    f"{output.state.value}: {output.message}",
                )
                sensor_map = raw_map.copy()
                sensor_map.add_circles(dynamic)
                ranges = sensor_map.ray_cast(pose, scan_angles, 0.12, 8.0)
                ros2.publish_scan(simulation.sim_time, ranges, -math.pi, math.pi, 0.12, 8.0)

            simulation.step()
            if output.state == NavigationState.SUCCEEDED:
                print(
                    f"[完成] 到达目标，最终位姿：({pose.x:.2f}, {pose.y:.2f}, {pose.yaw:.2f})",
                    flush=True,
                )
                break
            if output.state == NavigationState.FAILED:
                raise RuntimeError(output.message)
        else:
            raise TimeoutError(f"导航在 {args.timeout:.1f} 秒内未完成")
    except Exception:
        # Isaac Sim close() 可能在进程退出前吞掉未刷新的 traceback，因此先明确打印。
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        base.stop()
        if ros2:
            ros2.close()
        simulation.close()


if __name__ == "__main__":
    main()
