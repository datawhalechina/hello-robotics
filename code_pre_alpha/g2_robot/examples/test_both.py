"""
Test: G2 robot moves base AND both arms simultaneously.
Robot walks forward while waving arms, then stops and does the heart pose.

WARNING: set_world_pose navigation causes arm jitter during base movement.
This test demonstrates the issue (see BaseController docstring for details).

Usage:
    python -m g2_robot.examples.test_both
"""

import math
import os
import sys
import time

import numpy as np

# Ensure g2_robot package is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


WALK_TARGET = (-3.5, 0.0)
WALK_SPEED = 0.4
TURN_SPEED = 1.0
WAVE_FREQ = 2.0
WAVE_AMP = 0.6
HEART_STEPS = 360
HOLD_STEPS = 600
DT = 1.0 / 120


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
    from isaacsim.core.utils.types import ArticulationAction
    from g2_robot.controllers.arm import ArmController
    from g2_robot.controllers.base import quat_to_yaw, yaw_to_quat, normalize_angle
    from g2_robot.core.constants import find_joint_indices, LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

    # ── Controllers ─────────────────────────────────────────────────
    left_arm = ArmController(art, arm="left")
    right_arm = ArmController(art, arm="right")

    init_left = left_arm.get_joint_positions()
    init_right = right_arm.get_joint_positions()

    # Combined indices for both arms
    left_idx = find_joint_indices(art.dof_names, LEFT_ARM_JOINT_NAMES)
    right_idx = find_joint_indices(art.dof_names, RIGHT_ARM_JOINT_NAMES)
    both_idx = left_idx + right_idx

    # Heart pose targets
    hl = init_left.copy()
    hl[0] -= 1.5; hl[1] += 0.4; hl[2] -= 0.6
    hl[3] -= 1.6; hl[4] += 0.3; hl[5] -= 0.4; hl[6] += 0.3

    hr = init_right.copy()
    hr[0] += 1.5; hr[1] -= 0.4; hr[2] += 0.6
    hr[3] += 1.6; hr[4] -= 0.3; hr[5] += 0.4; hr[6] -= 0.3

    state = {"phase": "warmup", "count": 0}

    def physics_callback(step_size):
        s = state
        s["count"] += 1

        if s["phase"] == "warmup":
            if s["count"] >= 60:
                s["phase"] = "walk"
                s["count"] = 0
                print("[Test] Phase 1: Walking forward + waving arms")

        elif s["phase"] == "walk":
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
                s["phase"] = "heart_transition"
                s["count"] = 0
                print("[Test] Phase 2: Forming heart pose")
                return

            # Arm waving (sinusoidal)
            t = s["count"] * DT
            wave = math.sin(2 * math.pi * WAVE_FREQ * t) * WAVE_AMP

            wl = init_left.copy()
            wr = init_right.copy()
            wl[0] = init_left[0] - 0.8 + wave * 0.5
            wl[3] = init_left[3] - 0.5 - abs(wave)
            wr[0] = init_right[0] + 0.8 - wave * 0.5
            wr[3] = init_right[3] + 0.5 + abs(wave)

            art.apply_action(ArticulationAction(
                joint_positions=np.array(wl + wr, dtype=np.float64),
                joint_indices=both_idx,
            ))

        elif s["phase"] == "heart_transition":
            alpha = 0.5 * (1.0 - math.cos(math.pi * min(s["count"], HEART_STEPS) / HEART_STEPS))

            wl_end = init_left.copy()
            wl_end[0] -= 0.8; wl_end[3] -= 0.5
            wr_end = init_right.copy()
            wr_end[0] += 0.8; wr_end[3] += 0.5

            targets_l = [w + alpha * (h - w) for w, h in zip(wl_end, hl)]
            targets_r = [w + alpha * (h - w) for w, h in zip(wr_end, hr)]

            art.apply_action(ArticulationAction(
                joint_positions=np.array(targets_l + targets_r, dtype=np.float64),
                joint_indices=both_idx,
            ))

            if s["count"] >= HEART_STEPS:
                print("[Test] Phase 3: Holding heart pose <3")
                s["phase"] = "hold"
                s["count"] = 0

        elif s["phase"] == "hold":
            art.apply_action(ArticulationAction(
                joint_positions=np.array(hl + hr, dtype=np.float64),
                joint_indices=both_idx,
            ))
            if s["count"] >= HOLD_STEPS:
                print("[Test] Done! Robot walked and made a heart.")
                s["phase"] = "done"

    boot.world.add_physics_callback("test_both", callback_fn=physics_callback)

    # ── Main loop ───────────────────────────────────────────────────
    print("=" * 50)
    print("  G2: Walk forward + wave arms, then heart pose")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    boot.cleanup()


if __name__ == "__main__":
    main()
