"""示例 3：直接给定末端位置，支持左右机械臂同步运动到各自目标。"""

import argparse

import numpy as np

try:
    from .arm_controller import G2ArmController, G2DualArmController
    from .config import (
        HOME_JOINT_POSITIONS,
        MIRRORED_END_EFFECTOR_POSITION_SIGNS,
        SimulationConfig,
    )
    from .simulation import G2Simulation
except ImportError:
    from arm_controller import G2ArmController, G2DualArmController
    from config import (
        HOME_JOINT_POSITIONS,
        MIRRORED_END_EFFECTOR_POSITION_SIGNS,
        SimulationConfig,
    )
    from simulation import G2Simulation


# 目标位置使用 arm_base_link 坐标系，单位为米。
# 对视觉同向的双臂动作，右臂目标应为 [left_x, left_y, -left_z]，
# 而不是把 y、z 同时反号。
DEFAULT_LEFT_TARGET_POSITION = np.array(
    [0.50, 0.35, 0.45], dtype=np.float64
)
DEFAULT_TARGET_POSITIONS = {
    "left": DEFAULT_LEFT_TARGET_POSITION.copy(),
    "right": (
        MIRRORED_END_EFFECTOR_POSITION_SIGNS * DEFAULT_LEFT_TARGET_POSITION
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="输入末端位置并控制 G2 机械臂到达")
    parser.add_argument(
        "--arm", choices=("left", "right", "both"), default="right"
    )
    parser.add_argument("--x", type=float, help="单臂目标 x，单位 m")
    parser.add_argument("--y", type=float, help="单臂目标 y，单位 m")
    parser.add_argument("--z", type=float, help="单臂目标 z，单位 m")
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-x", type=float, help=f"{side} arm 目标 x")
        parser.add_argument(f"--{side}-y", type=float, help=f"{side} arm 目标 y")
        parser.add_argument(f"--{side}-z", type=float, help=f"{side} arm 目标 z")
    parser.add_argument("--duration", type=float, default=3.5, help="运动时间，单位 s")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def override_target(target: np.ndarray, values) -> np.ndarray:
    result = target.copy()
    for index, value in enumerate(values):
        if value is not None:
            result[index] = value
    return result


def single_target(args) -> np.ndarray:
    side_values = (
        getattr(args, f"{args.arm}_x"),
        getattr(args, f"{args.arm}_y"),
        getattr(args, f"{args.arm}_z"),
    )
    target = override_target(DEFAULT_TARGET_POSITIONS[args.arm], side_values)
    return override_target(target, (args.x, args.y, args.z))


def dual_targets(args) -> dict[str, np.ndarray]:
    if any(value is not None for value in (args.x, args.y, args.z)):
        raise ValueError(
            "--arm both 时请使用 --left-x/y/z 和 --right-x/y/z 分别设置目标"
        )
    return {
        side: override_target(
            DEFAULT_TARGET_POSITIONS[side],
            (
                getattr(args, f"{side}_x"),
                getattr(args, f"{side}_y"),
                getattr(args, f"{side}_z"),
            ),
        )
        for side in ("left", "right")
    }


def run_single(sim, side: str, target: np.ndarray, duration: float) -> None:
    arm = G2ArmController(sim.robot, arm=side)
    dt = sim.config.physics_dt
    arm.move_to(HOME_JOINT_POSITIONS, 2.5, dt, sim.step)

    print(f"\n[{side} 位置控制] 目标（arm_base_link，m）：", np.round(target, 4), flush=True)
    result = arm.solve_ik(target, target_rotation=None)
    print(
        f"[{side} 位置控制] IK success={result.success}, "
        f"iterations={result.iterations}, residual={result.position_error * 1000:.3f} mm",
        flush=True,
    )
    if not result.success:
        raise RuntimeError(f"{side} IK 未收敛，目标可能超出工作空间")

    arm.move_to(result.joint_positions, duration, dt, sim.step)
    actual = arm.forward_kinematics().position
    error = np.linalg.norm(target - actual)
    print(f"[{side} 位置控制] 实际位置：", np.round(actual, 4), flush=True)
    print(f"[{side} 位置控制] 最终误差：{error * 1000:.3f} mm", flush=True)


def run_both(sim, targets: dict[str, np.ndarray], duration: float) -> None:
    arms = G2DualArmController(sim.robot)
    dt = sim.config.physics_dt
    arms.move_to(
        HOME_JOINT_POSITIONS, HOME_JOINT_POSITIONS, 2.5, dt, sim.step
    )

    print("\n[双臂位置控制] 左右臂目标：", flush=True)
    for side in ("left", "right"):
        print(f"  {side}: {np.round(targets[side], 4)} m", flush=True)
    expected_right = MIRRORED_END_EFFECTOR_POSITION_SIGNS * targets["left"]
    if not np.allclose(targets["right"], expected_right, atol=1e-9):
        print(
            "[双臂位置控制] 提示：当前是两组独立目标，不是视觉同向的镜像目标；"
            f"左臂目标对应的镜像右臂目标应为 {np.round(expected_right, 4)} m",
            flush=True,
        )

    results = arms.solve_ik(targets["left"], targets["right"])
    for side in ("left", "right"):
        result = results[side]
        print(
            f"[{side} IK] success={result.success}, iterations={result.iterations}, "
            f"residual={result.position_error * 1000:.3f} mm",
            flush=True,
        )
        print(
            f"[{side} IK] 关节角（deg）："
            f"{np.round(np.rad2deg(result.joint_positions), 2)}",
            flush=True,
        )
    failed = [side for side, result in results.items() if not result.success]
    if failed:
        raise RuntimeError(f"以下机械臂 IK 未收敛：{failed}")

    print("[双臂位置控制] 两侧 IK 均成功，开始同步运动", flush=True)
    arms.move_to(
        results["left"].joint_positions,
        results["right"].joint_positions,
        duration,
        dt,
        sim.step,
    )
    actual_poses = arms.forward_kinematics()
    for side in ("left", "right"):
        actual = actual_poses[side].position
        error = np.linalg.norm(targets[side] - actual)
        print(f"[{side} 结果] 实际位置：{np.round(actual, 4)} m", flush=True)
        print(f"[{side} 结果] 最终误差：{error * 1000:.3f} mm", flush=True)


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration 必须大于 0")

    sim = G2Simulation(SimulationConfig(headless=args.headless))
    try:
        if args.arm == "both":
            run_both(sim, dual_targets(args), args.duration)
        else:
            run_single(sim, args.arm, single_target(args), args.duration)

        if args.headless:
            for _ in range(sim.config.physics_hz):
                sim.step()
        else:
            print("[位置控制] 运动完成，关闭窗口即可退出", flush=True)
            while sim.is_running():
                sim.step()
    finally:
        sim.close()


if __name__ == "__main__":
    main()
