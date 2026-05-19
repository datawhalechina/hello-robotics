"""
G2 Robot Standalone Sensor Publisher

Loads the G2 robot in Isaac Sim, stands it in a room, and publishes
ALL sensor data over ROS2 for RViz visualization.

Usage:
    python -m g2_robot.examples.run_sensors [--headless] [--config path.yaml]

Then, in another terminal:
    ros2 topic list
    rviz2 -d g2_robot/config/g2_rviz.rviz
"""

import os
import sys
import time
import argparse

import numpy as np
import yaml

# Ensure g2_robot package is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap
from g2_robot.core.constants import FIXED_JOINT_TARGETS


def parse_args():
    p = argparse.ArgumentParser(description="G2 standalone sensor publisher")
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    p.add_argument("--config", default=os.path.join(config_dir, "sensor_publisher_config.yaml"))
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.headless:
        cfg["sim"]["headless"] = True

    print(f"[SensorPub] Loaded config: {args.config}")

    # ── Bootstrap ───────────────────────────────────────────────────
    boot = SimBootstrap(cfg)
    boot.setup_physics_scene()
    boot.init_ros2()

    # ── Load robot + scene ──────────────────────────────────────────
    robot_cfg = cfg["robot"]
    boot.load_robot(
        robot_cfg["robot_usd"],
        robot_cfg["base_prim_path"],
        position=robot_cfg["init_position"],
        orientation=robot_cfg["init_rotation"],
    )
    boot.load_scene(cfg["scene"]["scene_usd"])
    boot.set_viewport_camera(position=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.8])
    boot.play_and_warmup()

    art = boot.init_articulation(robot_cfg["base_prim_path"])

    # ── Set body posture ────────────────────────────────────────────
    dof_names = art.dof_names
    print("[SensorPub] Setting body/head joints to standard posture...")
    for joint_name, target_val in FIXED_JOINT_TARGETS.items():
        idx = next(i for i, d in enumerate(dof_names) if d == joint_name)
        art.set_joint_positions(
            positions=np.array([target_val]),
            joint_indices=np.array([idx]),
        )
    for _ in range(60):
        boot.world.step(render=True)

    # ── Create RTX Lidar ────────────────────────────────────────────
    import omni
    import omni.kit.commands
    from pxr import Gf

    lidar = None
    if cfg["lidar"]["enabled"]:
        lcfg = cfg["lidar"]
        lidar_full_path = lcfg["parent"] + lcfg["child_path"]
        print(f"[SensorPub] Creating RTX lidar '{lcfg['model']}' at {lidar_full_path}")

        _result, lidar_prim = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=lcfg["child_path"],
            parent=lcfg["parent"],
            config=lcfg["model"],
            translation=Gf.Vec3d(*lcfg["translation"]),
            orientation=Gf.Quatd(
                lcfg["orientation"][0],
                lcfg["orientation"][1],
                lcfg["orientation"][2],
                lcfg["orientation"][3],
            ),
        )
        if lidar_prim is None or not lidar_prim.IsValid():
            raise RuntimeError(f"IsaacSensorCreateRtxLidar failed for config={lcfg['model']}")

        lidar_prim_path = str(lidar_prim.GetPath())
        print(f"[SensorPub] Lidar prim created at: {lidar_prim_path}")

        from omni.isaac.sensor import LidarRtx
        lidar = LidarRtx(prim_path=lidar_prim_path)
        lidar.initialize()
        boot.world.step(render=True)

    # ── RobotInterface (joint_states, TF, RGB cameras) ──────────────
    from g2_robot.ros.robot_interface import RobotInterface

    robot_interface = RobotInterface()
    robot_interface.register_joint_state(art)
    robot_interface.register_robot_tf(boot.stage, robot_cfg["base_prim_path"])

    cam_list = []
    if cfg["cameras"]["enabled"]:
        for name in ("head", "left", "right"):
            c = cfg["cameras"][name]
            cam_list.append((c["prim"], c["resolution"], c["hz"]))
            every_n_frame = max(1, cfg["sim"]["physics_step"] // c["hz"])
            robot_interface.register_camera(c["prim"], c["resolution"], every_n_frame)

    print(f"[SensorPub] RobotInterface ready ({len(cam_list)} cameras)")

    # ── Writer-based camera publishers (depth + info + pointcloud) ──
    from isaacsim.sensors.camera import Camera
    from g2_robot.ros.publishers.camera import publish_depth, publish_camera_info, publish_pointcloud_from_depth

    _cam_writers = []
    for prim_path, (w, h), hz in cam_list:
        camera = Camera(prim_path=prim_path, frequency=hz, resolution=(w, h))
        camera.initialize()
        step_size = max(1, cfg["sim"]["rendering_step"] // hz)
        publish_depth(camera, step_size, "")
        publish_camera_info(camera, step_size, "")
        publish_pointcloud_from_depth(camera, step_size, "")
        _cam_writers.append(camera)
        print(f"[SensorPub]   {prim_path}: depth + camera_info + pointcloud (step={step_size})")

    # ── Lidar publishers ────────────────────────────────────────────
    if lidar is not None:
        from g2_robot.ros.publishers.lidar import publish_lidar_pointcloud, publish_lidar_scan
        lcfg = cfg["lidar"]
        if lcfg.get("publish_pointcloud", True):
            publish_lidar_pointcloud(lidar, lcfg["frequency"], lcfg.get("pointcloud_topic", ""))
        if lcfg.get("publish_scan", False):
            publish_lidar_scan(lidar, lcfg["frequency"], lcfg.get("scan_topic", ""))

    # ── Clock publisher ─────────────────────────────────────────────
    from g2_robot.ros.publishers.clock import publish_clock
    publish_clock()
    print("[SensorPub] /clock publisher created")

    # ── Odometry publisher ──────────────────────────────────────────
    from nav_msgs.msg import Odometry
    import rclpy

    odom_pub = robot_interface.create_publisher(Odometry, "/odom", 10)

    def publish_odom(sim_time):
        bp, bo = art.get_world_pose()
        msg = Odometry()
        sec = int(sim_time)
        nsec = int((sim_time - sec) * 1e9)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(bp[0])
        msg.pose.pose.position.y = float(bp[1])
        msg.pose.pose.position.z = float(bp[2])
        msg.pose.pose.orientation.w = float(bo[0])
        msg.pose.pose.orientation.x = float(bo[1])
        msg.pose.pose.orientation.y = float(bo[2])
        msg.pose.pose.orientation.z = float(bo[3])
        odom_pub.publish(msg)

    # ── Physics callback ────────────────────────────────────────────
    _frame_count = 0
    _last_hz_time = time.time()

    def on_physics(step_size):
        nonlocal _frame_count, _last_hz_time
        _frame_count += 1
        now = time.time()
        if now - _last_hz_time >= 5.0:
            hz = _frame_count / (now - _last_hz_time)
            print(f"[Physics] {hz:.1f} Hz")
            _frame_count = 0
            _last_hz_time = now

        sim_time = boot.world.current_time
        step_idx = boot.world.current_time_step_index
        robot_interface.tick(sim_time, step_idx)
        publish_odom(sim_time)
        rclpy.spin_once(robot_interface, timeout_sec=0.0)

    boot.world.add_physics_callback("sensor_pub", callback_fn=on_physics)

    # ── Main loop ───────────────────────────────────────────────────
    print("=" * 60)
    print("  G2 Standalone Sensor Publisher")
    print(f"  Lidar: {cfg['lidar']['enabled']} | Cameras: {len(cam_list)}")
    print("  Run: ros2 topic list")
    print("=" * 60)

    try:
        while boot.app.is_running():
            boot.world.step(render=True)
    finally:
        try:
            robot_interface.destroy_node()
        except Exception:
            pass
        boot.cleanup()


if __name__ == "__main__":
    main()
