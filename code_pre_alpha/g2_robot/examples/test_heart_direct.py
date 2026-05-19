"""
Test: G2 robot walks forward while raising arms into a heart pose
using direct joint targets, then spins a full 360-degree turn in place.

Phases:
  1. Walk forward + transition arms into heart pose
  2. Hold heart pose briefly
  3. Spin 360 degrees in place
  4. Done

Usage:
    python -m g2_robot.examples.test_walk_heart_spin_direct
"""

import math
import os
import sys

import numpy as np

# Ensure g2_robot package is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


# ── Parameters ──────────────────────────────────────────────────────
WALK_TARGET = (-3.5, 0.0)
WALK_SPEED = 0.3
TURN_SPEED = 1.0
HEART_TRANSITION_STEPS = 300   # steps to form heart pose while walking
HOLD_STEPS = 360               # hold heart pose after arriving
SPIN_SPEED = 1.2               # rad/s for the 360 spin
DT = 1.0 / 120


def main():
    # ── Setup ───────────────────────────────────────────────────────
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
    boot.set_viewport_camera(position=[0.5, 4.0, 1.8], target=[-2.5, 0.0, 0.8])
    boot.play_and_warmup()
    art = boot.init_articulation("/genie")

    # Import after SimulationApp is running (isaacsim requires this)
    from isaacsim.core.utils.types import ArticulationAction
    from g2_robot.controllers.arm import ArmController
    from g2_robot.controllers.base import quat_to_yaw, yaw_to_quat, normalize_angle
    from g2_robot.core.constants import find_joint_indices, LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

    # ── Arm controllers ─────────────────────────────────────────────
    left_arm = ArmController(art, arm="left")
    right_arm = ArmController(art, arm="right")

    init_left = left_arm.get_joint_positions()
    init_right = right_arm.get_joint_positions()

    left_idx = find_joint_indices(art.dof_names, LEFT_ARM_JOINT_NAMES)
    right_idx = find_joint_indices(art.dof_names, RIGHT_ARM_JOINT_NAMES)
    both_idx = left_idx + right_idx
#     [Test] left_offsets = [1.0447, 1.2537, -2.2135, -1.4284, 1.1156, 0.3318, 0.5634]
# [Test] right_offsets = [0.3622, -0.9747, 1.2524, -1.2781, -1.5122, -0.5283, -1.212]


    # Direct joint-space heart pose targets.
    # For this robot, left/right symmetry is not a simple full sign flip.
    # A better mirrored mapping is:
    #   joint1 same sign
    #   joint2 opposite sign
    #   joint3 opposite sign
    #   joint4 same sign
    #   joint5 opposite sign
    #   joint6 opposite sign
    #   joint7 opposite sign
    # left_offsets = [1.0447, 1.2537, -1.2135, -1.4284, 1.1156, 0.3318, 0.5634]
    # right_offsets = [1.0447, -1.2537, 1.2135, 1.4284, -1.1156, -0.3318, -0.5634]
    left_offsets = [1.57, 1.0537, 0, 1.0472, 0.5318, 1.0472, 0]
    right_offsets = [-1.57, 1.0537, 0, 1.0472, -0.5318, 1.0472, 0]
    # right_offsets = [sign * offset for sign, offset in zip(mirror_signs, left_offsets)]

    hl = [i + o for i, o in zip(init_left, left_offsets)]
    hr = [i + o for i, o in zip(init_right, right_offsets)]

    print(f"[Test] left_offsets = {[round(v, 4) for v in left_offsets]}")
    print(f"[Test] right_offsets = {[round(v, 4) for v in right_offsets]}")

    # ── State ───────────────────────────────────────────────────────
    state = {"phase": "warmup", "count": 0, "spin_start_yaw": 0.0, "spin_accum": 0.0}

    def physics_callback(step_size):
        s = state
        s["count"] += 1

        # Phase 0: warmup
        if s["phase"] == "warmup":
            if s["count"] >= 60:
                s["phase"] = "walk_heart"
                s["count"] = 0
                print("[Test] Phase 1: Walking forward + forming heart pose")

        # Phase 1: walk forward while transitioning arms to heart pose
        elif s["phase"] == "walk_heart":
            pos, quat = art.get_world_pose()
            cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
            cyaw = quat_to_yaw(quat)

            tx, ty = WALK_TARGET
            dx, dy = tx - cx, ty - cy
            dist = math.sqrt(dx * dx + dy * dy)

            # Base movement
            if dist > 0.03:
                desired_yaw = math.atan2(dy, dx)
                yaw_err = normalize_angle(desired_yaw - cyaw)
                if abs(yaw_err) > 0.05:
                    step = min(TURN_SPEED * DT, abs(yaw_err))
                    new_yaw = cyaw + math.copysign(step, yaw_err)
                    art.set_world_pose(np.array([cx, cy, cz]), yaw_to_quat(new_yaw))
                else:
                    step = min(WALK_SPEED * DT, dist)
                    r = step / dist
                    art.set_world_pose(np.array([cx + dx * r, cy + dy * r, cz]), yaw_to_quat(cyaw))
            else:
                print("[Test] Reached walk target")
                s["phase"] = "hold_heart"
                s["count"] = 0
                print("[Test] Phase 2: Holding heart pose")
                return

            alpha = min(s["count"], HEART_TRANSITION_STEPS) / HEART_TRANSITION_STEPS
            alpha = 0.5 * (1.0 - math.cos(math.pi * alpha))

            targets_l = [i + alpha * (h - i) for i, h in zip(init_left, hl)]
            targets_r = [i + alpha * (h - i) for i, h in zip(init_right, hr)]

            art.apply_action(
                ArticulationAction(
                    joint_positions=np.array(targets_l + targets_r, dtype=np.float64),
                    joint_indices=both_idx,
                )
            )

        # Phase 2: hold heart pose at destination
        elif s["phase"] == "hold_heart":
            art.apply_action(
                ArticulationAction(
                    joint_positions=np.array(hl + hr, dtype=np.float64),
                    joint_indices=both_idx,
                )
            )
            if s["count"] >= HOLD_STEPS:
                _, quat = art.get_world_pose()
                s["spin_start_yaw"] = quat_to_yaw(quat)
                s["spin_accum"] = 0.0
                s["phase"] = "spin"
                s["count"] = 0
                print("[Test] Phase 3: Spinning 360 degrees")

        # Phase 3: spin a full 360 degrees in place
        elif s["phase"] == "spin":
            pos, quat = art.get_world_pose()
            cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])

            yaw_step = SPIN_SPEED * DT
            s["spin_accum"] += yaw_step
            new_yaw = s["spin_start_yaw"] + s["spin_accum"]
            art.set_world_pose(np.array([cx, cy, cz]), yaw_to_quat(new_yaw))

            art.apply_action(
                ArticulationAction(
                    joint_positions=np.array(hl + hr, dtype=np.float64),
                    joint_indices=both_idx,
                )
            )

            if s["spin_accum"] >= 2 * math.pi:
                print("[Test] Spin complete!")
                s["phase"] = "done"
                print("[Test] Done! Walk + heart + spin finished.")

    boot.world.add_physics_callback("test_walk_heart_spin_direct", callback_fn=physics_callback)

    # ── Main loop ───────────────────────────────────────────────────
    print("=" * 50)
    print("  G2: Walk forward + direct heart pose + 360 spin")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    boot.cleanup()


if __name__ == "__main__":
    main()
