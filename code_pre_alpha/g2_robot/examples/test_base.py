"""
Minimal test: G2 robot base movement.
No ROS2, no arm, no depth camera. Pure Isaac Sim.
The robot will walk a square path using set_world_pose navigation.

Usage:
    python -m g2_robot.examples.test_base
"""

import math
import os
import sys
import time

import numpy as np

# Ensure g2_robot package is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


# Square path waypoints (x, y, yaw)
WAYPOINTS = [
    (-1.5,  1.5, 0.0),
    (-1.5,  1.5, math.pi / 2),
    (-3.0,  1.5, math.pi / 2),
    (-3.0,  1.5, math.pi),
    (-3.0,  0.0, math.pi),
    (-3.0,  0.0, -math.pi / 2),
    (-1.5,  0.0, -math.pi / 2),
    (-1.5,  0.0, 0.0),
]


def main():
    # ── Setup ───────────────────────────────────────────────────────
    config = {
        "sim": {"headless": False, "physics_step": 120, "rendering_step": 60},
    }
    boot = SimBootstrap(config)
    boot.setup_physics_scene()
    boot.load_robot("robot/G2_omnipicker/robot_fix.usda", "/genie",
                     position=[-1.5, 0.0, -0.01], orientation=[1.0, 0.0, 0.0, 0.0])
    boot.load_scene("background/room/room_1/background.usda")
    boot.set_viewport_camera(position=[1.0, 4.0, 1.5], target=[-2.0, 0.0, 0.8])
    boot.play_and_warmup()
    art = boot.init_articulation("/genie")

    # Import after SimulationApp is running (isaacsim requires this)
    from g2_robot.controllers.base import BaseController

    # ── Base controller ─────────────────────────────────────────────
    base = BaseController(art, physics_dt=1.0 / 120)

    nav_state = {"wp_idx": 0, "warmup": 0, "done": False}

    def physics_callback(step_size):
        s = nav_state
        if s["done"]:
            return

        # Warmup
        if s["warmup"] < 120:
            s["warmup"] += 1
            if s["warmup"] == 120:
                tx, ty, tyaw = WAYPOINTS[0]
                base.navigate_to(tx, ty, tyaw, linear_speed=0.5, angular_speed=1.0)
                print(f"[Test] Starting square path with {len(WAYPOINTS)} waypoints")
            return

        if not base.active:
            # Current waypoint reached
            print(f"[Test] Reached waypoint {s['wp_idx']}")
            s["wp_idx"] += 1
            if s["wp_idx"] >= len(WAYPOINTS):
                print("[Test] Square path complete!")
                s["done"] = True
                return
            tx, ty, tyaw = WAYPOINTS[s["wp_idx"]]
            base.navigate_to(tx, ty, tyaw, linear_speed=0.5, angular_speed=1.0)
            return

        base.navigation_step()

    boot.world.add_physics_callback("test_base", callback_fn=physics_callback)

    # ── Main loop ───────────────────────────────────────────────────
    print("=" * 50)
    print("  G2 Robot: Walking a SQUARE path")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    boot.cleanup()


if __name__ == "__main__":
    main()
