"""示例 1：直接控制 G2 底盘的 vx、vy、wz。"""

import argparse
from dataclasses import replace

try:
    from .base_controller import G2BaseController
    from .config import ControlLimits, RobotGeometry, SimulationConfig
    from .kinematics import SwerveKinematics
    from .simulation import G2Simulation
except ImportError:
    from base_controller import G2BaseController
    from config import ControlLimits, RobotGeometry, SimulationConfig
    from kinematics import SwerveKinematics
    from simulation import G2Simulation


# (名称, vx, vy, wz, 持续时间)
MOTION_SEQUENCE = (
    ("前进",       0.40,  0.00,  0.00, 2.5),
    ("向左横移",   0.00,  0.35,  0.00, 2.5),
    ("原地左转",   0.00,  0.00,  0.65, 3.0),
    ("斜向并旋转", 0.30, -0.18, -0.35, 3.5),
    ("平滑停车",   0.00,  0.00,  0.00, 1.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--physics-hz", type=int, default=120)
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
    controller = G2BaseController(simulation.robot, kinematics, limits)

    try:
        print("\n=== 第四章示例 1：底盘速度控制 ===")
        print("坐标：vx 向前，vy 向左，wz 逆时针为正\n")

        for name, vx, vy, wz, duration in MOTION_SEQUENCE:
            print(
                f"[动作] {name:<10} "
                f"vx={vx:+.2f} m/s, vy={vy:+.2f} m/s, "
                f"wz={wz:+.2f} rad/s"
            )
            steps = int(duration / sim_config.physics_dt)
            for _ in range(steps):
                if not simulation.is_running():
                    return
                # 持续发送命令，模拟真实机器人周期性的速度指令。
                controller.set_velocity(vx, vy, wz)
                controller.update(sim_config.physics_dt)
                simulation.step()

        controller.stop()
        pose = simulation.get_pose2d()
        measured = controller.measured_chassis_velocity()
        print(
            "\n[完成] 最终仿真位姿："
            f"x={pose.x:.3f} m, y={pose.y:.3f} m, yaw={pose.yaw:.3f} rad"
        )
        print(
            "[完成] 轮速正运动学估计："
            f"vx={measured.vx:.3f}, vy={measured.vy:.3f}, wz={measured.wz:.3f}"
        )
    finally:
        controller.stop()
        simulation.close()


if __name__ == "__main__":
    main()
