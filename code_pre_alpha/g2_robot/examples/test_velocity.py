"""
Minimal test: G2 robot velocity-based wheel control.
No ROS2. Drives the chassis using DifferentialController via apply_action.

Uses robot.usda (floating base) so wheel velocities physically move the robot.
The robot drives in a circle using constant linear + angular velocity.

Usage:
    python -m g2_robot.examples.test_velocity
"""

import math
import os
import sys
import time

import numpy as np

# Ensure g2_robot package is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


def main():
    # ── Setup ───────────────────────────────────────────────────────
    config = {
        "sim": {"headless": False, "physics_step": 120, "rendering_step": 60},
    }
    boot = SimBootstrap(config)
    boot.setup_physics_scene()
    boot.load_robot("robot/G2_omnipicker/robot.usda", "/genie",
                     position=[-1.5, 0.0, -0.01], orientation=[1.0, 0.0, 0.0, 0.0])
    boot.load_scene("background/room/room_1/background.usda")
    boot.set_viewport_camera(position=[1.0, 4.0, 1.5], target=[-2.0, 0.0, 0.8])

    boot.play_and_warmup()
    art = boot.init_articulation("/genie")

    # Import after SimulationApp is running
    from g2_robot.controllers.base import BaseController

    # ── Base controller with velocity mode ──────────────────────────
    base = BaseController(art, physics_dt=1.0 / 120)
    base.init_velocity_mode(
        max_linear_speed=1.0,
        max_angular_speed=2.0,
        max_wheel_speed=20.0,
        cmd_timeout=2.0,  # longer timeout for scripted test
    )

    # ── Spin-in-place parameters ──────────────────────────────────
    # vx=0, omega=0.5 → pure rotation, one full turn = 2*pi/0.5 ≈ 12.6s
    spin_vx = 0.0
    spin_omega = 0.5      # rad/s turning left
    spin_duration = 2.0 * math.pi / spin_omega  # one full revolution

    state = {"step": 0, "warmup": 0, "done": False}
    physics_dt = 1.0 / 120
    spin_steps = int(spin_duration / physics_dt)

    def physics_callback(step_size):
        s = state

        # Warmup
        if s["warmup"] < 120:
            s["warmup"] += 1
            if s["warmup"] == 120:
                pos, _ = art.get_world_pose()
                print(f"[Test] Spin in place: omega={spin_omega} rad/s, "
                      f"duration={spin_duration:.1f}s")
                print(f"[Test] Start pos: ({pos[0]:.2f}, {pos[1]:.2f})")
            return

        if s["done"]:
            return

        # Continuously send command
        base.set_velocity_command(spin_vx, spin_omega)
        base.velocity_step()
        s["step"] += 1

        if s["step"] >= spin_steps:
            base.stop()
            pos, _ = art.get_world_pose()
            print(f"[Test] Spin complete! Final pos: ({pos[0]:.2f}, {pos[1]:.2f})")
            s["done"] = True

    boot.world.add_physics_callback("test_velocity", callback_fn=physics_callback)

    # ── Main loop ───────────────────────────────────────────────────
    print("=" * 50)
    print("  G2 Robot: Spin in place (omega=0.5 rad/s)")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    boot.cleanup()


if __name__ == "__main__":
    main()
