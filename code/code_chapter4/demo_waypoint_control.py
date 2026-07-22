"""示例 2：使用自编写位姿反馈控制器依次到达多个航点。"""

import argparse
from dataclasses import replace
import math

try:
    from .base_controller import G2BaseController, PoseController
    from .config import ControlLimits, RobotGeometry, SimulationConfig
    from .kinematics import Pose2D, SwerveKinematics
    from .simulation import G2Simulation
except ImportError:
    from base_controller import G2BaseController, PoseController
    from config import ControlLimits, RobotGeometry, SimulationConfig
    from kinematics import Pose2D, SwerveKinematics
    from simulation import G2Simulation


WAYPOINTS = (
    Pose2D(1.0, 0.0, 0.0),
    Pose2D(1.0, 1.0, math.pi / 2.0),
    Pose2D(0.0, 1.0, math.pi),
    Pose2D(0.0, 0.0, 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument(
        "--timeout", type=float, default=20.0, help="每个航点的最大控制时间"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_config = replace(
        SimulationConfig(),
        headless=args.headless,
        physics_hz=args.physics_hz,
    )
    geometry = RobotGeometry()
    limits = ControlLimits()

    simulation = G2Simulation(sim_config)
    kinematics = SwerveKinematics(geometry.wheel_positions, geometry.wheel_radius)
    base = G2BaseController(simulation.robot, kinematics, limits)
    pose_controller = PoseController()

    try:
        print("\n=== 第四章示例 2：二维航点闭环控制 ===")
        for index, target in enumerate(WAYPOINTS, start=1):
            print(
                f"\n[航点 {index}/{len(WAYPOINTS)}] "
                f"x={target.x:.2f}, y={target.y:.2f}, yaw={target.yaw:.2f}"
            )
            max_steps = int(args.timeout / sim_config.physics_dt)

            for step in range(max_steps):
                if not simulation.is_running():
                    return

                current = simulation.get_pose2d()
                result = pose_controller.compute(current, target)
                if result.reached:
                    base.stop()
                    print(
                        f"[到达] 位置误差={result.distance_error:.3f} m, "
                        f"角度误差={result.yaw_error:.3f} rad"
                    )
                    break

                command = result.command
                base.set_velocity(command.vx, command.vy, command.wz)
                base.update(sim_config.physics_dt)
                simulation.step()

                if step % sim_config.physics_hz == 0:
                    print(
                        f"  当前=({current.x:+.2f}, {current.y:+.2f}, {current.yaw:+.2f}) "
                        f"误差=({result.distance_error:.2f} m, {result.yaw_error:+.2f} rad)"
                    )
            else:
                raise RuntimeError(
                    f"航点 {index} 在 {args.timeout:.1f} 秒内未到达；"
                    "请检查轮子方向、地面摩擦或控制参数"
                )

        final_pose = simulation.get_pose2d()
        print(
            "\n[完成] 所有航点已到达，最终位姿："
            f"x={final_pose.x:.3f}, y={final_pose.y:.3f}, yaw={final_pose.yaw:.3f}"
        )
    finally:
        base.stop()
        simulation.close()


if __name__ == "__main__":
    main()
