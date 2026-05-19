"""
Test: G2 robot picks up an object using IK + Ruckig + Gripper.
No ROS2. Pure Isaac Sim.

Usage:
    python -m g2_robot.examples.test_pick
"""

import os
import sys
import time

import numpy as np

# Ensure g2_robot package is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


# Pick parameters
OBJECT_POS = np.array([-4.03, -0.032, 0.81])
PRE_GRASP_OFFSET = np.array([0.0, 0.0, 0.10])
GRASP_OFFSET = np.array([0.0, 0.0, 0.015])
LIFT_HEIGHT = 0.15
GRASP_ORIENT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # wxyz, gripper down


def main():
    # ── Setup ───────────────────────────────────────────────────────
    config = {
        "sim": {"headless": False, "physics_step": 120, "rendering_step": 60},
        "robot": {
            "robot_usd": "robot/G2_omnipicker/robot_fix.usda",
            "base_prim_path": "/genie",
        },
    }
    boot = SimBootstrap(config)
    boot.setup_physics_scene()
    boot.load_robot("robot/G2_omnipicker/robot_fix.usda", "/genie",
                     position=[-3.55, -0.032, -0.01],
                     orientation=[0.0, 0.0, 0.0, 1.0])  # facing -X
    boot.load_scene("background/room/room_1/background.usda")
    boot.set_viewport_camera(position=[-2.5, 1.5, 1.5], target=[-3.8, -0.03, 0.8])
    boot.play_and_warmup()
    art = boot.init_articulation("/genie")

    # Import after SimulationApp is running (isaacsim requires this)
    from g2_robot.core.sim_setup import SimSetup
    from g2_robot.controllers.arm import ArmController, PickController, PickState

    # ── SimSetup: body posture, IK solvers, gripper, Ruckig ─────────
    sim_setup = SimSetup(boot.world, art, config)
    sim_setup.setup_all()

    # ── Create arm controller with all subsystems ───────────────────
    arm_ctrl = ArmController(
        articulation=art,
        arm="right",
        ik_solver=sim_setup.ik_solvers["right"],
        ruckig=sim_setup.ruckig,
        gripper=sim_setup.gripper,
    )

    # ── IK diagnostics ──────────────────────────────────────────────
    pre_grasp_pos = OBJECT_POS + PRE_GRASP_OFFSET
    _, ok_orient = arm_ctrl.solve_ik(pre_grasp_pos, GRASP_ORIENT)
    _, ok_pos = arm_ctrl.solve_ik(pre_grasp_pos)
    print(f"[Pick] IK test - pre_grasp + orient: success={ok_orient}")
    print(f"[Pick] IK test - pre_grasp pos-only: success={ok_pos}")

    grasp_orient = GRASP_ORIENT if ok_orient else None

    # ── Create pick controller ──────────────────────────────────────
    pick = PickController(arm_ctrl)

    warmup_count = [0]
    started = [False]

    def physics_callback(step_size):
        if not started[0]:
            warmup_count[0] += 1
            if warmup_count[0] >= 120:
                started[0] = True
                print("[Pick] === Starting pick sequence ===")
                pick.start_pick(
                    object_position=OBJECT_POS,
                    grasp_orientation=grasp_orient,
                    pre_grasp_offset=PRE_GRASP_OFFSET.tolist(),
                    grasp_offset=GRASP_OFFSET.tolist(),
                    lift_height=LIFT_HEIGHT,
                )
            return

        state, done = pick.step()
        if done:
            if state == PickState.DONE:
                print("[Pick] === Pick sequence complete! ===")
            else:
                print("[Pick] === Pick sequence FAILED ===")

    boot.world.add_physics_callback("test_pick", callback_fn=physics_callback)

    # ── Main loop ───────────────────────────────────────────────────
    print("=" * 50)
    print("  G2: Pick test (IK + Ruckig + Gripper)")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    sim_setup.cleanup()
    boot.cleanup()


if __name__ == "__main__":
    main()
