"""完整案例：自编规划器避开障挡物，抓起桌面红色物体。"""

import argparse
import time
import traceback

try:
    from .arm_model import G2PlanningModel
    from .collision import ArmCollisionChecker
    from .config import (
        HOME_JOINT_POSITIONS,
        LIFT_POSITION,
        PRE_GRASP_POSITION,
        GRASP_POSITION,
        RED_OBJECT_POSITION,
        ROBOT_BODY_BOX,
        SCENE_BOXES,
        PlannerConfig,
        PerceptionConfig,
        SimulationConfig,
        TrajectoryConfig,
    )
    from .gripper import G2GripperController
    from .local_avoidance import LocalPathRepair
    from .motion_planner import RRTConnectPlanner
    from .perception import G2HeadDepthPerception
    from .rviz_visualizer import PlanningRvizPublisher
    from .simulation import G2ManipulationSimulation
    from .trajectory_optimizer import TrajectoryOptimizer
    from .trajectory_tracker import G2TrajectoryTracker
except ImportError:
    from arm_model import G2PlanningModel
    from collision import ArmCollisionChecker
    from config import HOME_JOINT_POSITIONS, LIFT_POSITION, PRE_GRASP_POSITION, GRASP_POSITION, RED_OBJECT_POSITION, ROBOT_BODY_BOX, SCENE_BOXES, PlannerConfig, PerceptionConfig, SimulationConfig, TrajectoryConfig
    from gripper import G2GripperController
    from local_avoidance import LocalPathRepair
    from motion_planner import RRTConnectPlanner
    from perception import G2HeadDepthPerception
    from rviz_visualizer import PlanningRvizPublisher
    from simulation import G2ManipulationSimulation
    from trajectory_optimizer import TrajectoryOptimizer
    from trajectory_tracker import G2TrajectoryTracker


def parse_args():
    parser = argparse.ArgumentParser(description="不使用 MoveIt 2 的 G2 完整机械臂规划")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def solve_target(model, label, position, seed):
    result = model.inverse(position, seed)
    print(
        f"[IK] {label}: success={result.success}, iterations={result.iterations}, "
        f"error={result.position_error * 1000:.2f} mm",
        flush=True,
    )
    if not result.success:
        raise RuntimeError(f"{label} IK 未收敛")
    return result.joint_positions


def plan_and_execute(label, start, goal, planner, repair, optimizer, tracker, rviz, sim):
    print(f"\n[{label}] 1/4 RRT-Connect 全局规划", flush=True)
    result = planner.plan(start, goal)
    if not result.success:
        print(f"[{label}] 规划失败：{result.message}", flush=True)
        raise RuntimeError(f"{label} 规划失败：{result.message}")
    print(f"[{label}] 原始路径 {len(result.path)} 点，迭代 {result.iterations} 次", flush=True)

    print(f"[{label}] 2/4 局部碰撞监测与路径修复", flush=True)
    repaired = repair.repair(result.path)

    print(f"[{label}] 3/4 碰撞安全捷径优化与时间参数化", flush=True)
    optimized = optimizer.optimize(repaired)
    trajectory = optimizer.time_parameterize(optimized)
    print(
        f"[{label}] 路径长度 {optimizer.length(result.path):.3f} -> "
        f"{optimizer.length(optimized):.3f} rad，轨迹 {trajectory[-1].time_from_start:.2f} s",
        flush=True,
    )
    rviz.publish_paths(result.path, optimized)

    print(f"[{label}] 4/4 关节位置闭环轨迹跟踪", flush=True)
    max_error = tracker.execute(trajectory, sim.step, rviz.publish_arm)
    print(f"[{label}] 完成，最大跟踪误差 {max_error:.3f} rad", flush=True)
    return tracker.get_positions()


def main() -> None:
    args = parse_args()
    planner_cfg = PlannerConfig(random_seed=args.seed)
    trajectory_cfg = TrajectoryConfig()
    model = G2PlanningModel()
    checker = ArmCollisionChecker(model, [ROBOT_BODY_BOX], planner_cfg)
    planner = RRTConnectPlanner(checker, planner_cfg)
    repair = LocalPathRepair(checker)
    optimizer = TrajectoryOptimizer(checker, trajectory_cfg)

    # 先在纯 NumPy 层求解全部目标，可在启动大型仿真前尽早发现不可达目标。
    pre_grasp_q = solve_target(model, "预抓取", PRE_GRASP_POSITION, HOME_JOINT_POSITIONS)
    grasp_q = solve_target(model, "抓取", GRASP_POSITION, pre_grasp_q)
    lift_q = solve_target(model, "抬升", LIFT_POSITION, grasp_q)

    sim = G2ManipulationSimulation(
        SimulationConfig(headless=args.headless, enable_rviz=not args.no_rviz)
    )
    rviz = PlanningRvizPublisher(model, enabled=not args.no_rviz)
    try:
        tracker = G2TrajectoryTracker(sim.robot, trajectory_cfg)
        gripper = G2GripperController(sim.robot)
        perception = G2HeadDepthPerception(sim, PerceptionConfig())
        voxel_centers, sensed_obstacles = perception.capture_head_obstacles()
        checker.set_obstacles([ROBOT_BODY_BOX, *sensed_obstacles])
        # 已知场景和头部深度 OctoMap 分开显示，便于理解“模型”和“感知地图”。
        rviz.publish_scene(SCENE_BOXES)
        rviz.publish_octomap(voxel_centers, perception.config.map_voxel_size)

        print("\n[准备] 张开夹爪并回到 HOME", flush=True)
        gripper.hold(True, 60, sim.step)
        current = tracker.get_positions()
        current = plan_and_execute(
            "头部深度地图避障到预抓取位姿", current, pre_grasp_q,
            planner, repair, optimizer, tracker, rviz, sim,
        )

        print(
            f"\n[目标] 红色物块已知中心 = {RED_OBJECT_POSITION.tolist()} "
            "(arm_base_link)，不使用右夹爪相机",
            flush=True,
        )
        current = plan_and_execute(
            "按已知物块位姿运动到抓取点", current, grasp_q,
            planner, repair, optimizer, tracker, rviz, sim,
        )

        print("\n[抓取] 关闭夹爪并建立教学用刚性附着", flush=True)
        gripper.hold(False, 90, sim.step)
        sim.attach_red_object()

        current = plan_and_execute(
            "抓取后抬升", current, lift_q,
            planner, repair, optimizer, tracker, rviz, sim,
        )
        print("\n[完成] 红色物体已被夹起；黄色阻挡物未发生碰撞。", flush=True)

        if args.headless:
            for _ in range(120):
                sim.step()
        else:
            print("关闭 Isaac Sim 窗口结束示例。", flush=True)
            while sim.is_running():
                sim.step()
                time.sleep(0.0)
    except Exception as exc:
        # SimulationApp.close() 可能使尚未刷新的 traceback 不可见，先明确打印根因。
        print(f"[失败] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        rviz.close()
        sim.close()


if __name__ == "__main__":
    main()
