"""
Minimal test: G2 robot forms a heart pose with both arms and holds it.
No walking, no spinning, no ROS2. Pure Isaac Sim.

Usage:
    python -m g2_robot.examples.test_heart_only
"""

import os
import sys

import numpy as np

# Ensure g2_robot package is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


HEART_STEPS = 360
HOLD_STEPS = 1200


def main():
    config = {
        "sim": {"headless": False, "physics_step": 120, "rendering_step": 60},
    }
    boot = SimBootstrap(config)
    boot.setup_physics_scene()
    boot.load_robot(
        "robot/G2_omnipicker/robot_fix.usda",
        "/genie",
        position=[-1.5, 0.0, -0.01],
        orientation=[1.0, 0.0, 0.0, 0.0],
    )
    boot.load_scene("background/room/room_1/background.usda")
    boot.set_viewport_camera(position=[-0.6, 2.2, 1.55], target=[-1.5, 0.0, 0.9])
    boot.play_and_warmup()
    art = boot.init_articulation("/genie")

    from g2_robot.controllers.arm import ArmController

    left_arm = ArmController(art, arm="left")
    right_arm = ArmController(art, arm="right")

    init_left = left_arm.get_joint_positions()
    init_right = right_arm.get_joint_positions()

    # Manually tuned heart pose that visually forms a better heart on this robot.
    left_offsets = [1.0447, 1.2537, -2.2135, -1.4284, 1.1156, 0.3318, 0.5634]
    right_offsets = [0.3622, -0.9747, 1.2524, -1.2781, -1.5122, -0.5283, -1.2120]

    target_left = [i + o for i, o in zip(init_left, left_offsets)]
    target_right = [i + o for i, o in zip(init_right, right_offsets)]

    traj_left = left_arm.interpolate_to(target_left, steps=HEART_STEPS)
    traj_right = right_arm.interpolate_to(target_right, steps=HEART_STEPS)

    print(f"[HeartOnly] init_left  = {[round(v, 4) for v in init_left]}")
    print(f"[HeartOnly] init_right = {[round(v, 4) for v in init_right]}")
    print(f"[HeartOnly] left_offsets  = {[round(v, 4) for v in left_offsets]}")
    print(f"[HeartOnly] right_offsets = {[round(v, 4) for v in right_offsets]}")

    state = {"traj_idx": 0, "hold_count": 0, "announced_hold": False}

    def physics_callback(step_size):
        idx = state["traj_idx"]
        if idx < HEART_STEPS:
            left_arm.set_joint_positions(traj_left[idx])
            right_arm.set_joint_positions(traj_right[idx])
            state["traj_idx"] += 1
            if idx % 60 == 0:
                print(f"[HeartOnly] Step {idx}/{HEART_STEPS}")
            return

        if not state["announced_hold"]:
            print("[HeartOnly] Heart pose reached. Holding...")
            state["announced_hold"] = True

        left_arm.set_joint_positions(target_left)
        right_arm.set_joint_positions(target_right)
        state["hold_count"] += 1

        if state["hold_count"] == HOLD_STEPS:
            print("[HeartOnly] Hold complete.")

    boot.world.add_physics_callback("test_heart_only", callback_fn=physics_callback)

    print("=" * 50)
    print("  G2: Heart Pose Only")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    boot.cleanup()


if __name__ == "__main__":
    main()
