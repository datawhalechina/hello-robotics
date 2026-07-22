"""示例 1：逐个关节运动，并支持左右机械臂同步控制。"""

import argparse
import math

import numpy as np

try:
    from .arm_controller import G2ArmController, G2DualArmController
    from .config import HOME_JOINT_POSITIONS, SimulationConfig
    from .simulation import G2Simulation
except ImportError:
    from arm_controller import G2ArmController, G2DualArmController
    from config import HOME_JOINT_POSITIONS, SimulationConfig
    from simulation import G2Simulation


def parse_args():
    parser = argparse.ArgumentParser(description="G2 机械臂关节位置控制演示")
    parser.add_argument(
        "--arm", choices=("left", "right", "both"), default="right"
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def run_single_arm(sim: G2Simulation, side: str) -> None:
    arm = G2ArmController(sim.robot, arm=side)
    dt = sim.config.physics_dt
    arm.move_to(HOME_JOINT_POSITIONS, 2.5, dt, sim.step)
    print(f"\n[关节控制] {side} arm 已到达 HOME 姿态", flush=True)

    for joint_id in range(7):
        target = HOME_JOINT_POSITIONS.copy()
        delta = 0.30 if joint_id != 3 else 0.35
        target[joint_id] += delta
        print(
            f"[关节控制] {side} J{joint_id + 1}：+{math.degrees(delta):.1f} deg",
            flush=True,
        )
        arm.move_to(target, 1.2, dt, sim.step)
        arm.move_to(HOME_JOINT_POSITIONS, 1.0, dt, sim.step)

    combined = HOME_JOINT_POSITIONS + np.array(
        [0.45, -0.20, 0.30, -0.25, 0.20, 0.15, -0.25]
    )
    actual = arm.move_to(combined, 3.0, dt, sim.step)
    error = np.rad2deg(arm.clamp_positions(combined) - actual)
    print("[关节控制] 最终关节误差（deg）：", np.round(error, 3), flush=True)


def run_both_arms(sim: G2Simulation) -> None:
    arms = G2DualArmController(sim.robot)
    dt = sim.config.physics_dt
    arms.move_to(
        HOME_JOINT_POSITIONS, HOME_JOINT_POSITIONS, 2.5, dt, sim.step
    )
    print("\n[双臂关节控制] 左右臂已同步到达 HOME 姿态", flush=True)

    for joint_id in range(7):
        offsets = np.zeros(7, dtype=np.float64)
        delta = 0.30 if joint_id != 3 else 0.35
        offsets[joint_id] = delta
        right_sign = "-" if joint_id % 2 == 0 else "+"
        print(
            f"[双臂关节控制] J{joint_id + 1} 视觉同向："
            f"左 +{math.degrees(delta):.1f} deg，"
            f"右 {right_sign}{math.degrees(delta):.1f} deg",
            flush=True,
        )
        arms.move_mirrored(
            HOME_JOINT_POSITIONS, offsets, 1.2, dt, sim.step
        )
        arms.move_mirrored(
            HOME_JOINT_POSITIONS, np.zeros(7), 1.0, dt, sim.step
        )

    combined_offsets = np.array(
        [0.45, -0.20, 0.30, -0.25, 0.20, 0.15, -0.25]
    )
    targets = arms.mirrored_targets(HOME_JOINT_POSITIONS, combined_offsets)
    actual = arms.move_mirrored(
        HOME_JOINT_POSITIONS, combined_offsets, 3.0, dt, sim.step
    )
    left_error = np.rad2deg(targets["left"] - actual["left"])
    right_error = np.rad2deg(targets["right"] - actual["right"])
    print("[双臂关节控制] 左臂误差（deg）：", np.round(left_error, 3), flush=True)
    print("[双臂关节控制] 右臂误差（deg）：", np.round(right_error, 3), flush=True)


def main() -> None:
    args = parse_args()
    sim = G2Simulation(SimulationConfig(headless=args.headless))
    try:
        if args.arm == "both":
            run_both_arms(sim)
        else:
            run_single_arm(sim, args.arm)

        if args.headless:
            for _ in range(sim.config.physics_hz):
                sim.step()
        else:
            print("[关节控制] 演示完成，关闭窗口即可退出", flush=True)
            while sim.is_running():
                sim.step()
    finally:
        sim.close()


if __name__ == "__main__":
    main()
