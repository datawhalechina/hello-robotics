"""
Minimal test: G2 robot forms a heart pose using IK and holds it.
No walking, no spinning, no ROS2. Pure Isaac Sim.

Usage:
    python -m g2_robot.examples.test_heart_only_ik
"""

import math
import os
import sys

import numpy as np

# Ensure g2_robot package is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


HEART_STEPS = 360
HOLD_STEPS = 1200
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

    from isaacsim.core.utils.types import ArticulationAction
    from g2_robot.core.sim_setup import SimSetup
    from g2_robot.controllers.arm import ArmController
    from g2_robot.controllers.base import quat_to_yaw

    sim_setup = SimSetup(boot.world, art, config)
    sim_setup.setup_all()

    left_arm = ArmController(art, arm="left", ik_solver=sim_setup.ik_solvers["left"])
    right_arm = ArmController(art, arm="right", ik_solver=sim_setup.ik_solvers["right"])

    init_left = left_arm.get_joint_positions()
    init_right = right_arm.get_joint_positions()
    both_idx = left_arm.joint_indices + right_arm.joint_indices

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

    print(f"[HeartIK] init_left  = {[round(v, 4) for v in init_left]}")
    print(f"[HeartIK] init_right = {[round(v, 4) for v in init_right]}")
    print(f"[HeartIK] left_target_local  = {[round(v, 4) for v in left_heart_local.tolist()]}")
    print(f"[HeartIK] right_target_local = {[round(v, 4) for v in right_heart_local.tolist()]}")

    state = {
        "traj_idx": 0,
        "hold_count": 0,
        "last_left": init_left.copy(),
        "last_right": init_right.copy(),
        "announced_hold": False,
        "warned_ik": False,
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
            print(f"[HeartIK] IK fallback active (left={ok_l}, right={ok_r})")
            state["warned_ik"] = True

        art.apply_action(
            ArticulationAction(
                joint_positions=np.array(state["last_left"] + state["last_right"], dtype=np.float64),
                joint_indices=both_idx,
            )
        )

    def physics_callback(step_size):
        if state["traj_idx"] < HEART_STEPS:
            alpha = 0.5 * (1.0 - math.cos(math.pi * (state["traj_idx"] + 1) / HEART_STEPS))
            apply_heart_pose(alpha)
            if state["traj_idx"] % 60 == 0:
                print(f"[HeartIK] Step {state['traj_idx']}/{HEART_STEPS}")
            state["traj_idx"] += 1
            return

        if not state["announced_hold"]:
            left_offsets = (np.array(state["last_left"]) - np.array(init_left)).tolist()
            right_offsets = (np.array(state["last_right"]) - np.array(init_right)).tolist()
            print(f"[HeartIK] left_offsets  = {[round(v, 4) for v in left_offsets]}")
            print(f"[HeartIK] right_offsets = {[round(v, 4) for v in right_offsets]}")
            print("[HeartIK] Heart pose reached. Holding...")
            state["announced_hold"] = True

        apply_heart_pose(1.0)
        state["hold_count"] += 1

        if state["hold_count"] == HOLD_STEPS:
            print("[HeartIK] Hold complete.")

    boot.world.add_physics_callback("test_heart_only_ik", callback_fn=physics_callback)

    print("=" * 50)
    print("  G2: Heart Pose Only (IK)")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    sim_setup.cleanup()
    boot.cleanup()


if __name__ == "__main__":
    main()
