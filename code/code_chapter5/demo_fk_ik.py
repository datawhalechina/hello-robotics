"""示例 2：自编 FK、Jacobian 和 DLS-IK，支持左右臂同步运动。"""

import argparse
import math

import numpy as np

try:
    from .arm_controller import G2ArmController, G2DualArmController
    from .config import (
        DEMO_TARGET_JOINT_POSITIONS,
        HOME_JOINT_POSITIONS,
        SimulationConfig,
    )
    from .kinematics import orientation_error
    from .simulation import G2Simulation
except ImportError:
    from arm_controller import G2ArmController, G2DualArmController
    from config import DEMO_TARGET_JOINT_POSITIONS, HOME_JOINT_POSITIONS, SimulationConfig
    from kinematics import orientation_error
    from simulation import G2Simulation


def parse_args():
    parser = argparse.ArgumentParser(description="G2 机械臂 FK/IK 演示")
    parser.add_argument(
        "--arm", choices=("left", "right", "both"), default="right"
    )
    parser.add_argument("--position-only", action="store_true", help="IK 只约束末端位置")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def print_ik_result(side, target_pose, result) -> None:
    print(f"\n[{side} FK] 目标位置（arm_base_link，m）：", np.round(target_pose.position, 4), flush=True)
    print(f"[{side} FK] 目标四元数 [w, x, y, z]：", np.round(target_pose.quaternion, 4), flush=True)
    print(f"[{side} IK] success={result.success}, iterations={result.iterations}", flush=True)
    print(f"[{side} IK] 位置残差={result.position_error * 1000.0:.3f} mm", flush=True)
    print(f"[{side} IK] 姿态残差={math.degrees(result.orientation_error):.3f} deg", flush=True)
    print(f"[{side} IK] 求解关节角（deg）：", np.round(np.rad2deg(result.joint_positions), 2), flush=True)


def report_final_pose(sim, arm, side, target_pose) -> None:
    predicted = arm.forward_kinematics()
    position_error = np.linalg.norm(target_pose.position - predicted.position)
    rotation_error = np.linalg.norm(
        orientation_error(target_pose.rotation, predicted.rotation)
    )
    print(
        f"[{side} 结果] 位置误差={position_error * 1000:.3f} mm，"
        f"姿态误差={math.degrees(rotation_error):.3f} deg",
        flush=True,
    )

    sim_position, sim_rotation = sim.get_end_effector_pose(side)
    model_position_error = np.linalg.norm(sim_position - predicted.position)
    model_rotation_error = np.linalg.norm(
        orientation_error(sim_rotation, predicted.rotation)
    )
    print(
        f"[{side} 模型核对] 自编 FK 与 USD：位置差="
        f"{model_position_error * 1000:.3f} mm，姿态差="
        f"{math.degrees(model_rotation_error):.3f} deg",
        flush=True,
    )


def run_single(sim, side: str, position_only: bool) -> None:
    arm = G2ArmController(sim.robot, arm=side)
    dt = sim.config.physics_dt
    arm.move_to(HOME_JOINT_POSITIONS, 2.5, dt, sim.step)

    target_pose = arm.kinematics.forward(DEMO_TARGET_JOINT_POSITIONS)
    target_rotation = None if position_only else target_pose.rotation
    result = arm.solve_ik(
        target_pose.position, target_rotation, HOME_JOINT_POSITIONS
    )
    print_ik_result(side, target_pose, result)
    if not result.success:
        raise RuntimeError(f"{side} IK 未收敛")

    arm.move_to(result.joint_positions, 3.5, dt, sim.step)
    report_final_pose(sim, arm, side, target_pose)


def run_both(sim, position_only: bool) -> None:
    arms = G2DualArmController(sim.robot)
    dt = sim.config.physics_dt
    arms.move_to(
        HOME_JOINT_POSITIONS, HOME_JOINT_POSITIONS, 2.5, dt, sim.step
    )

    # 左右臂是镜像安装的。如果两侧都用同一组关节角生成 FK 目标，
    # J1/J3/J5/J7 会在画面中表现为反向。这里先生成视觉同向的
    # 左右关节目标，再分别通过各自的 FK 得到真实末端位姿。
    target_joint_positions = arms.mirrored_targets(
        HOME_JOINT_POSITIONS,
        DEMO_TARGET_JOINT_POSITIONS - HOME_JOINT_POSITIONS,
    )
    target_poses = {
        side: getattr(arms, side).kinematics.forward(target_joint_positions[side])
        for side in ("left", "right")
    }
    print("\n[双臂 FK/IK] 用于生成目标位姿的镜像关节角（deg）：", flush=True)
    for side in ("left", "right"):
        print(
            f"  {side}: {np.round(np.rad2deg(target_joint_positions[side]), 2)}",
            flush=True,
        )
    results = arms.solve_ik(
        left_position=target_poses["left"].position,
        right_position=target_poses["right"].position,
        left_rotation=None if position_only else target_poses["left"].rotation,
        right_rotation=None if position_only else target_poses["right"].rotation,
    )

    for side in ("left", "right"):
        print_ik_result(side, target_poses[side], results[side])
    failed = [side for side, result in results.items() if not result.success]
    if failed:
        raise RuntimeError(f"IK 未收敛：{failed}")

    print("\n[双臂 FK/IK] 两侧 IK 均成功，开始同步运动", flush=True)
    arms.move_to(
        results["left"].joint_positions,
        results["right"].joint_positions,
        3.5,
        dt,
        sim.step,
    )
    report_final_pose(sim, arms.left, "left", target_poses["left"])
    report_final_pose(sim, arms.right, "right", target_poses["right"])


def main() -> None:
    args = parse_args()
    sim = G2Simulation(SimulationConfig(headless=args.headless))
    try:
        if args.arm == "both":
            run_both(sim, args.position_only)
        else:
            run_single(sim, args.arm, args.position_only)

        if args.headless:
            for _ in range(sim.config.physics_hz):
                sim.step()
        else:
            print("[FK/IK] 演示完成，关闭窗口即可退出", flush=True)
            while sim.is_running():
                sim.step()
    finally:
        sim.close()


if __name__ == "__main__":
    main()
