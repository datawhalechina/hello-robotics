"""
Test: G2 robot walks forward while raising arms into a heart pose,
then spins a full 360-degree turn in place.

Phases:
  1. Walk forward + transition arms into heart pose
  2. Hold heart pose briefly
  3. Spin 360 degrees in place
  4. Done

Usage:
    python -m g2_robot.examples.test_walk_heart_spin
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
HEART_CENTER_OFFSET_LOCAL = np.array([0.10, 0.0, 0.52], dtype=np.float64)
HEART_HAND_SEPARATION_Y = 0.09


def world_to_base_local(world_pos, base_pos, base_yaw):
    """Convert a world-space point to the robot base's yaw-aligned local frame."""
    dx = float(world_pos[0]) - float(base_pos[0])
    dy = float(world_pos[1]) - float(base_pos[1])
    dz = float(world_pos[2]) - float(base_pos[2])
    c = math.cos(base_yaw)
    s = math.sin(base_yaw)
    return np.array([
        c * dx + s * dy,
        -s * dx + c * dy,
        dz,
    ], dtype=np.float64)


def base_local_to_world(local_pos, base_pos, base_yaw):
    """Convert a yaw-aligned base-local point back into world coordinates."""
    c = math.cos(base_yaw)
    s = math.sin(base_yaw)
    return np.array([
        float(base_pos[0]) + c * float(local_pos[0]) - s * float(local_pos[1]),
        float(base_pos[1]) + s * float(local_pos[0]) + c * float(local_pos[1]),
        float(base_pos[2]) + float(local_pos[2]),
    ], dtype=np.float64)


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
    boot.set_viewport_camera(position=[0.5, 4.0, 1.8], target=[-2.5, 0.0, 0.8])
    boot.play_and_warmup()
    art = boot.init_articulation("/genie")

    # Import after SimulationApp is running (isaacsim requires this)
    from isaacsim.core.utils.types import ArticulationAction
    from g2_robot.core.sim_setup import SimSetup
    from g2_robot.controllers.arm import ArmController
    from g2_robot.controllers.base import quat_to_yaw, yaw_to_quat, normalize_angle

    # Initialize body posture + IK solvers before planning the heart pose.
    sim_setup = SimSetup(boot.world, art, config)
    sim_setup.setup_all()

    # ── Arm controllers ─────────────────────────────────────────────
    left_arm = ArmController(art, arm="left", ik_solver=sim_setup.ik_solvers["left"])
    right_arm = ArmController(art, arm="right", ik_solver=sim_setup.ik_solvers["right"])

    init_left = left_arm.get_joint_positions()
    init_right = right_arm.get_joint_positions()

    left_idx = left_arm.joint_indices
    right_idx = right_arm.joint_indices
    both_idx = left_idx + right_idx

    def get_end_effector_world_pos(arm_ctrl):
        base_pos, base_quat = art.get_world_pose()
        arm_ctrl.ik_solver["lula"].set_robot_base_pose(base_pos, base_quat)
        ee_pos, _ = arm_ctrl.ik_solver["art"].compute_end_effector_pose()
        return np.array(ee_pos, dtype=np.float64).reshape(-1)

    base_pos0, base_quat0 = art.get_world_pose()
    base_yaw0 = quat_to_yaw(base_quat0)
    left_ee_local0 = world_to_base_local(get_end_effector_world_pos(left_arm), base_pos0, base_yaw0)
    right_ee_local0 = world_to_base_local(get_end_effector_world_pos(right_arm), base_pos0, base_yaw0)

    relaxed_center_local = 0.5 * (left_ee_local0 + right_ee_local0)
    heart_center_local = relaxed_center_local + HEART_CENTER_OFFSET_LOCAL
    left_heart_local = heart_center_local + np.array([0.0, HEART_HAND_SEPARATION_Y, 0.0], dtype=np.float64)
    right_heart_local = heart_center_local + np.array([0.0, -HEART_HAND_SEPARATION_Y, 0.0], dtype=np.float64)

    # ── State ───────────────────────────────────────────────────────
    state = {
        "phase": "warmup",
        "count": 0,
        "spin_start_yaw": 0.0,
        "spin_accum": 0.0,
        "last_left": init_left.copy(),
        "last_right": init_right.copy(),
        "warned_ik": False,
        "printed_offsets": False,
    }

    def apply_heart_pose(alpha):
        base_pos, base_quat = art.get_world_pose()
        base_yaw = quat_to_yaw(base_quat)

        left_target_local = left_ee_local0 + alpha * (left_heart_local - left_ee_local0)
        right_target_local = right_ee_local0 + alpha * (right_heart_local - right_ee_local0)

        left_target_world = base_local_to_world(left_target_local, base_pos, base_yaw)
        right_target_world = base_local_to_world(right_target_local, base_pos, base_yaw)

        targets_l, ok_l = left_arm.solve_ik(left_target_world)
        targets_r, ok_r = right_arm.solve_ik(right_target_world)

        if ok_l:
            state["last_left"] = targets_l
        if ok_r:
            state["last_right"] = targets_r
        if not (ok_l and ok_r) and not state["warned_ik"]:
            print(f"[Test] IK fallback active (left={ok_l}, right={ok_r})")
            state["warned_ik"] = True

        art.apply_action(ArticulationAction(
            joint_positions=np.array(state["last_left"] + state["last_right"], dtype=np.float64),
            joint_indices=both_idx,
        ))

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

            # Raise both hands toward a symmetric, head-adjacent heart shape
            # in task space so the elbows bend naturally instead of mirroring
            # joint offsets that do not match this robot's arm kinematics.
            alpha = min(s["count"], HEART_TRANSITION_STEPS) / HEART_TRANSITION_STEPS
            alpha = 0.5 * (1.0 - math.cos(math.pi * alpha))  # smooth ease-in-out
            apply_heart_pose(alpha)

        # Phase 2: hold heart pose at destination
        elif s["phase"] == "hold_heart":
            apply_heart_pose(1.0)
            if not s["printed_offsets"]:
                left_offsets = (np.array(s["last_left"]) - np.array(init_left)).tolist()
                right_offsets = (np.array(s["last_right"]) - np.array(init_right)).tolist()
                print(f"[Test] left_offsets = {[round(v, 4) for v in left_offsets]}")
                print(f"[Test] right_offsets = {[round(v, 4) for v in right_offsets]}")
                s["printed_offsets"] = True
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
            cyaw = quat_to_yaw(quat)

            yaw_step = SPIN_SPEED * DT
            s["spin_accum"] += yaw_step
            new_yaw = s["spin_start_yaw"] + s["spin_accum"]
            art.set_world_pose(np.array([cx, cy, cz]), yaw_to_quat(new_yaw))

            # Keep heart pose while spinning
            apply_heart_pose(1.0)

            if s["spin_accum"] >= 2 * math.pi:
                print("[Test] Spin complete!")
                s["phase"] = "done"
                print("[Test] Done! Walk + heart + spin finished.")

    boot.world.add_physics_callback("test_walk_heart_spin", callback_fn=physics_callback)

    # ── Main loop ───────────────────────────────────────────────────
    print("=" * 50)
    print("  G2: Walk forward + heart pose + 360 spin")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    sim_setup.cleanup()
    boot.cleanup()


if __name__ == "__main__":
    main()
