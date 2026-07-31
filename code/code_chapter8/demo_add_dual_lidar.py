"""示例 1：给 G2 添加双 RTX LiDAR，读取 intensity 并发布到 RViz。"""

import argparse
from dataclasses import replace

try:
    from .config import ROBOT_PRIM_PATH, LidarConfig, SimulationConfig
    from .lidar import DualRtxLidar
    from .ros2_visualization import MappingRos2Publisher
    from .simulation import G2MappingSimulation, SquarePatrol
except ImportError:
    from config import ROBOT_PRIM_PATH, LidarConfig, SimulationConfig
    from lidar import DualRtxLidar
    from ros2_visualization import MappingRos2Publisher
    from simulation import G2MappingSimulation, SquarePatrol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=600)
    lidar_visual = parser.add_mutually_exclusive_group()
    lidar_visual.add_argument("--show-lidar", dest="show_lidar", action="store_true")
    lidar_visual.add_argument("--hide-lidar", dest="show_lidar", action="store_false")
    parser.set_defaults(show_lidar=None)
    parser.add_argument("--no-ros", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation = G2MappingSimulation(
        replace(SimulationConfig(), headless=args.headless),
        enable_ros2=not args.no_ros,
    )
    ros = None
    try:
        patrol = SquarePatrol()
        simulation.set_pose(patrol.pose(0))
        show_lidar = (not args.headless) if args.show_lidar is None else args.show_lidar
        lidar = DualRtxLidar(
            LidarConfig(),
            show_visual=show_lidar,
            publish_raw_ros=not args.no_ros,
        )
        ros = None if args.no_ros else MappingRos2Publisher()
        if ros is not None:
            ros.register_robot_tf_tree(simulation.stage, ROBOT_PRIM_PATH)
            ros.publish_lidar_static_tf(simulation.sim_time)
        print("双雷达已添加：/genie/base_link/chapter8_lidar_left 和 chapter8_lidar_right")
        for frame in range(args.max_frames):
            if not simulation.is_running():
                break
            pose = patrol.pose(frame)
            simulation.step(pose)
            if ros is not None:
                ros.publish_dynamic_tree(pose, simulation.sim_time)
            if frame % 6 != 0:
                continue
            fused, individual = lidar.capture()
            if ros is not None:
                ros.publish_fused_cloud(fused.xyzi(), simulation.sim_time)
            if frame % 60 == 0:
                counts = ", ".join(f"{name}={len(cloud)}" for name, cloud in individual.items())
                intensity_max = float(fused.intensity.max()) if len(fused) else 0.0
                print(f"[点云] frame={frame:04d}, {counts}, fused={len(fused)}, intensity_max={intensity_max:.1f}")
    finally:
        if ros is not None:
            ros.close()
        simulation.close()


if __name__ == "__main__":
    main()
