"""
G2 Robot ROS2 Demo - Main Entry Point

Demonstrates three capabilities on the G2 humanoid robot in Isaac Sim:
  1. Navigation: Move robot base to a target position
  2. Manipulation: Pick up an object from a table using right arm
  3. Depth Camera: Capture depth images and generate point clouds

Usage:
    python -m g2_robot.examples.run_demo [--headless] [--config path.yaml] [--no-auto-run]

    # In another terminal, trigger actions manually:
    ros2 service call /demo/navigate_to std_srvs/srv/Trigger
    ros2 service call /demo/pick_object std_srvs/srv/Trigger
    ros2 service call /demo/capture_depth std_srvs/srv/Trigger
"""

import os
import sys
import time
import argparse

import yaml

# Ensure g2_robot package is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


# ── Load config ──────────────────────────────────────────────────────
def load_config(config_path=None):
    if config_path is None:
        config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
        config_path = os.path.join(config_dir, "demo_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="G2 Robot ROS2 Demo")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--no-auto-run", action="store_true", help="Disable auto-run demo")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.headless:
        config["sim"]["headless"] = True
    if args.no_auto_run:
        config["demo"]["auto_run"] = False

    # ── Bootstrap ───────────────────────────────────────────────────
    boot = SimBootstrap(config)
    boot.setup_physics_scene()
    boot.init_ros2()

    # ── Load robot + scene ──────────────────────────────────────────
    robot_cfg = config.get("robot", {})
    boot.load_robot(
        robot_cfg.get("robot_usd", "robot/G2_omnipicker/robot_fix.usda"),
        robot_cfg.get("base_prim_path", "/genie"),
        position=robot_cfg.get("init_position", [-1.5, 0.0, -0.01]),
        orientation=robot_cfg.get("init_rotation", [1.0, 0.0, 0.0, 0.0]),
    )
    scene_cfg = config.get("scene", {})
    boot.load_scene(scene_cfg.get("scene_usd", "background/room/room_1/background.usda"))

    boot.set_viewport_camera(position=[-0.5, 2.0, 1.5], target=[-1.5, 0.0, 0.8])
    boot.play_and_warmup()

    art = boot.init_articulation(robot_cfg.get("base_prim_path", "/genie"))

    # Import after SimulationApp is running (isaacsim requires this)
    from g2_robot.core.sim_setup import SimSetup

    # ── SimSetup: body posture, IK, gripper, Ruckig ─────────────────
    sim_setup = SimSetup(boot.world, art, config)
    sim_setup.setup_all()

    # ── ROS2 bridge ─────────────────────────────────────────────────
    from g2_robot.ros.bridge import ROSBridge
    ros_bridge = ROSBridge()
    ros_bridge.setup(art)

    # ── Demo node ───────────────────────────────────────────────────
    from g2_robot.ros.demo_node import DemoNode
    demo_node = DemoNode(sim_setup, ros_bridge, config)

    # ── Physics callback ────────────────────────────────────────────
    _frame_count = 0
    _last_time = time.time()

    def callback_physics(step_size):
        nonlocal _frame_count, _last_time
        _frame_count += 1
        now = time.time()
        if now - _last_time >= 5.0:
            hz = _frame_count / (now - _last_time)
            print(f"[Physics] {hz:.1f} Hz")
            _frame_count = 0
            _last_time = now

        demo_node.physics_tick(step_size)

    boot.world.add_physics_callback("demo_physics", callback_fn=callback_physics)

    # ── Main loop ───────────────────────────────────────────────────
    print("=" * 60)
    print("  G2 Robot ROS2 Demo")
    print(f"  Auto-run: {config['demo']['auto_run']}")
    print("  Services: /demo/navigate_to, /demo/pick_object, /demo/capture_depth")
    print("=" * 60)

    while boot.app.is_running():
        boot.world.step(render=True)
        if not boot.world.is_playing():
            time.sleep(0.1)

    demo_node.destroy_node()
    sim_setup.cleanup()
    boot.cleanup()


if __name__ == "__main__":
    main()
