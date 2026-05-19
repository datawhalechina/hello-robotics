"""
Minimal test: G2 robot makes a HEART shape with both arms.
No ROS2, no navigation, no depth camera. Pure Isaac Sim.

Usage:
    python -m g2_robot.examples.test_arm
"""

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
    boot.load_robot("robot/G2_omnipicker/robot_fix.usda", "/genie",
                     position=[-1.5, 0.0, -0.01], orientation=[1.0, 0.0, 0.0, 0.0])
    boot.load_scene("background/room/room_1/background.usda")
    boot.set_viewport_camera(position=[-0.5, 2.0, 1.5], target=[-1.5, 0.0, 0.8])
    boot.play_and_warmup()
    art = boot.init_articulation("/genie")

    # Import after SimulationApp is running (isaacsim requires this)
    from g2_robot.controllers.arm import ArmController

    # ── Create arm controllers ──────────────────────────────────────
    left_arm = ArmController(art, arm="left")
    right_arm = ArmController(art, arm="right")

    cur_left = left_arm.get_joint_positions()
    cur_right = right_arm.get_joint_positions()
    print(f"[Test] Current left  arm: {[f'{v:.3f}' for v in cur_left]}")
    print(f"[Test] Current right arm: {[f'{v:.3f}' for v in cur_right]}")

    # ── Heart pose targets ──────────────────────────────────────────
    tgt_left = cur_left.copy()
    tgt_left[0] -= 1.5   # raise arm up high
    tgt_left[1] += 0.4   # slight outward
    tgt_left[2] -= 0.6   # rotate shoulder inward
    tgt_left[3] -= 1.6   # big elbow bend
    tgt_left[4] += 0.3   # forearm rotation
    tgt_left[5] -= 0.4   # wrist bend inward
    tgt_left[6] += 0.3   # wrist twist

    tgt_right = cur_right.copy()
    tgt_right[0] += 1.5
    tgt_right[1] -= 0.4
    tgt_right[2] += 0.6
    tgt_right[3] += 1.6
    tgt_right[4] -= 0.3
    tgt_right[5] += 0.4
    tgt_right[6] -= 0.3

    # ── Generate trajectories ───────────────────────────────────────
    traj_left = left_arm.interpolate_to(tgt_left, steps=360)
    traj_right = right_arm.interpolate_to(tgt_right, steps=360)
    print(f"[Test] Heart trajectory: {len(traj_left)} waypoints")

    # ── Execute via physics callback ────────────────────────────────
    traj_idx = [0]

    def physics_callback(step_size):
        idx = traj_idx[0]
        if idx < len(traj_left):
            left_arm.set_joint_positions(traj_left[idx])
            right_arm.set_joint_positions(traj_right[idx])
            traj_idx[0] += 1
            if idx % 60 == 0:
                print(f"[Test] Step {idx}/{len(traj_left)}")
        elif idx == len(traj_left):
            print("[Test] Heart pose reached! <3")
            traj_idx[0] += 1

    boot.world.add_physics_callback("test_arm", callback_fn=physics_callback)

    # ── Main loop ───────────────────────────────────────────────────
    print("=" * 50)
    print("  G2 Robot: Making a HEART with both arms!")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    boot.cleanup()


if __name__ == "__main__":
    main()
