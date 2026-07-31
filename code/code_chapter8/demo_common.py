"""3D/2D 建图示例共享的运行循环。"""

from dataclasses import replace
from pathlib import Path
import argparse

try:
    from .config import (
        OUTPUT_DIR,
        ROBOT_PRIM_PATH,
        LidarConfig,
        Map2DConfig,
        Map3DConfig,
        SimulationConfig,
        SlamConfig,
    )
    from .geometry import between, compose
    from .lidar import DualRtxLidar
    from .ros2_visualization import MappingRos2Publisher
    from .simulation import G2MappingSimulation, SquarePatrol
    from .slam import DriftedOdometry, MappingBackend
except ImportError:
    from config import OUTPUT_DIR, ROBOT_PRIM_PATH, LidarConfig, Map2DConfig, Map3DConfig, SimulationConfig, SlamConfig
    from geometry import between, compose
    from lidar import DualRtxLidar
    from ros2_visualization import MappingRos2Publisher
    from simulation import G2MappingSimulation, SquarePatrol
    from slam import DriftedOdometry, MappingBackend


def add_common_arguments(parser: argparse.ArgumentParser, default_output: str) -> None:
    parser.add_argument("--headless", action="store_true", help="无窗口运行 Isaac Sim")
    parser.add_argument("--max-frames", type=int, default=1560, help="仿真总帧数；默认完整巡航一圈并经过起点")
    parser.add_argument("--map-every", type=int, default=4, help="每 N 帧处理一次双雷达点云")
    parser.add_argument("--publish-every", type=int, default=30, help="每 N 帧更新一次 RViz 地图")
    lidar_visual = parser.add_mutually_exclusive_group()
    lidar_visual.add_argument(
        "--show-lidar", dest="show_lidar", action="store_true",
        help="在 Isaac Sim 中显示双雷达扫描点",
    )
    lidar_visual.add_argument(
        "--hide-lidar", dest="show_lidar", action="store_false",
        help="关闭 Isaac Sim 中的双雷达扫描点显示",
    )
    parser.set_defaults(show_lidar=None)
    parser.add_argument("--no-ros", action="store_true", help="关闭 ROS2/RViz 发布，仅保存文件")
    parser.add_argument("--perfect-odometry", action="store_true", help="关闭人为里程计漂移，用作对照")
    parser.add_argument("--disable-loop", action="store_true", help="关闭回环检测和位姿图优化")
    parser.add_argument("--lidar-profile", default=LidarConfig().profile, help="RTX LiDAR 配置名称")
    parser.add_argument("--scan-voxel", type=float, default=LidarConfig().scan_voxel_size)
    parser.add_argument("--output", default=str(OUTPUT_DIR / default_output))


def run_mapping(args, build_3d: bool, build_2d: bool) -> None:
    sim_config = replace(SimulationConfig(), headless=args.headless)
    lidar_config = replace(LidarConfig(), profile=args.lidar_profile, scan_voxel_size=args.scan_voxel)
    slam_config = SlamConfig()
    if args.perfect_odometry:
        slam_config = replace(
            slam_config,
            odom_linear_scale=1.0,
            odom_lateral_scale=1.0,
            odom_yaw_scale=1.0,
            odom_yaw_bias_per_meter=0.0,
            odom_noise_std=0.0,
        )
    if args.disable_loop:
        slam_config = replace(slam_config, loop_min_keyframe_gap=10**9)

    simulation = G2MappingSimulation(sim_config, enable_ros2=not args.no_ros)
    lidar = None
    ros = None
    backend = MappingBackend(
        slam_config=slam_config,
        map3d_config=Map3DConfig(),
        map2d_config=Map2DConfig(),
        build_3d=build_3d,
        build_2d=build_2d,
    )
    odometry = DriftedOdometry(slam_config)
    patrol = SquarePatrol()
    output = Path(args.output)

    try:
        first_pose = patrol.pose(0)
        simulation.set_pose(first_pose)
        show_lidar = (not args.headless) if args.show_lidar is None else args.show_lidar
        lidar = DualRtxLidar(
            lidar_config,
            show_visual=show_lidar,
            publish_raw_ros=not args.no_ros,
        )
        ros = None if args.no_ros else MappingRos2Publisher()
        if ros is not None:
            ros.register_robot_tf_tree(simulation.stage, ROBOT_PRIM_PATH)
            ros.publish_lidar_static_tf(simulation.sim_time)
        odometry.reset(first_pose)

        # RTX LiDAR 创建后需要若干渲染帧才能产生第一包有效点云。
        for _ in range(12):
            simulation.step(first_pose)

        print("=" * 72)
        print("第八章：G2 双 OS1 点云建图")
        print(f"模式：{'3D XYZI 体素地图' if build_3d else '2D 占据栅格地图'}")
        print(f"里程计：{'理想对照' if args.perfect_odometry else '带系统漂移'}")
        print(f"回环优化：{'关闭' if args.disable_loop else '开启'}")
        print(f"输出：{output}")
        print("=" * 72)

        for frame in range(args.max_frames):
            if not simulation.is_running():
                break
            true_pose = patrol.pose(frame)
            simulation.step(true_pose)
            odom_pose = odometry.update(true_pose)

            if frame % max(1, args.map_every) != 0:
                if ros is not None:
                    ros.publish_dynamic_tree(
                        corrected_current_pose(backend, odom_pose), simulation.sim_time
                    )
                continue

            fused, _ = lidar.capture()
            added, loop = backend.add_scan(fused, odom_pose)
            if ros is not None:
                ros.publish_fused_cloud(fused.xyzi(), simulation.sim_time)
                ros.publish_dynamic_tree(
                    corrected_current_pose(backend, odom_pose), simulation.sim_time
                )

            if loop is not None:
                print(
                    f"[回环] keyframe {loop.source} -> {loop.target}, "
                    f"rmse={loop.rmse:.3f} m, inlier={loop.inlier_ratio:.2f}, "
                    f"descriptor={loop.descriptor_score:.3f}"
                )
            if added and len(backend.slam.keyframes) % 10 == 0:
                map_size = len(backend.map3d.points()) if backend.map3d is not None else len(backend.map2d._log_odds)
                print(
                    f"[建图] frame={frame:04d}, keyframes={len(backend.slam.keyframes)}, "
                    f"loops={len(backend.slam.loop_results)}, map_size={map_size}, scan_points={len(fused)}"
                )

            if ros is not None and frame % max(1, args.publish_every) == 0:
                publish_maps(ros, backend, simulation.sim_time)

    finally:
        save_results(output, backend, build_3d, build_2d)
        if ros is not None:
            publish_maps(ros, backend, simulation.sim_time)
            ros.close()
        simulation.close()


def corrected_current_pose(backend: MappingBackend, odom_pose):
    """把最新回环修正平滑延伸到当前里程计位姿。"""
    if not backend.slam.keyframes:
        return odom_pose
    anchor = backend.slam.keyframes[-1]
    return compose(anchor.optimized_pose, between(anchor.odom_pose, odom_pose))


def publish_maps(
    ros: MappingRos2Publisher,
    backend: MappingBackend,
    sim_time: float | None = None,
) -> None:
    if backend.map3d is not None:
        ros.publish_map_cloud(backend.map3d.points(), sim_time)
    if backend.map2d is not None:
        ros.publish_grid(backend.map2d, sim_time)
    ros.publish_paths(backend.slam.raw_path(), backend.slam.optimized_path(), sim_time)


def save_results(output: Path, backend: MappingBackend, build_3d: bool, build_2d: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if build_3d and backend.map3d is not None:
        point_count = backend.map3d.save_ply(output.with_suffix(".ply"))
        print(f"[保存] 3D 地图：{output.with_suffix('.ply')}，{point_count} 点")
    if build_2d and backend.map2d is not None:
        pgm, yaml, png = backend.map2d.save(output)
        print(f"[保存] 2D 地图：{pgm} / {yaml}" + (f" / {png}" if png.exists() else ""))
    trajectory = output.with_name(output.stem + "_trajectory.csv")
    backend.slam.save_trajectory(trajectory)
    print(f"[保存] 漂移/优化轨迹：{trajectory}")
    print(
        f"[完成] keyframes={len(backend.slam.keyframes)}, "
        f"loop_closures={len(backend.slam.loop_results)}"
    )
