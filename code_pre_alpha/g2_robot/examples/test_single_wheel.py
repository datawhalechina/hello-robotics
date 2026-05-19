"""
Test: verify individual wheel steering and spinning.

Cycles through each of the 4 swerve wheels, steering it to 45° then -45°,
while spinning it forward. Other wheels stay still for contrast.

Usage:
    python -m g2_robot.examples.test_single_wheel
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


def main():
    config = {
        "sim": {"headless": False, "physics_step": 120, "rendering_step": 60},
    }
    boot = SimBootstrap(config)
    boot.setup_physics_scene()
    boot.load_robot("robot/G2_omnipicker/robot_fix.usda", "/genie",
                     position=[-1.5, 0.0, -0.01], orientation=[1.0, 0.0, 0.0, 0.0])
    boot.load_scene("background/room/room_1/background.usda")
    # Close-up camera to see wheels clearly
    boot.set_viewport_camera(position=[-0.5, 1.5, 0.5], target=[-1.5, 0.0, 0.1])
    boot.play_and_warmup()
    art = boot.init_articulation("/genie")

    from isaacsim.core.utils.types import ArticulationAction
    from g2_robot.core.constants import (
        WHEEL_STEERING_JOINT_NAMES, WHEEL_SPIN_JOINT_NAMES, find_joint_indices,
    )

    dof_names = art.dof_names
    steer_indices = find_joint_indices(dof_names, WHEEL_STEERING_JOINT_NAMES)
    spin_indices = find_joint_indices(dof_names, WHEEL_SPIN_JOINT_NAMES)

    wheel_labels = ["Left-Front", "Left-Rear", "Right-Front", "Right-Rear"]

    print(f"Steering indices: {steer_indices}")
    print(f"Spin indices:     {spin_indices}")
    for i, label in enumerate(wheel_labels):
        print(f"  {label}: steer DOF={steer_indices[i]}, spin DOF={spin_indices[i]}")

    # Test sequence: for each wheel, steer +90° → -45° → 0°, while spinning
    PHASE_DURATION = 1.5  # seconds per sub-phase
    physics_dt = 1.0 / 120
    steps_per_phase = int(PHASE_DURATION / physics_dt)

    # (wheel_index, steer_angle, spin_speed, description)
    phases = []
    for wi in range(4):
        label = wheel_labels[wi]
        phases.append((wi, math.radians(90), 5.0, f"{label}: steer +90°, spin"))
        phases.append((wi, math.radians(-90), 5.0, f"{label}: steer -90°, spin"))
        phases.append((wi, 0.0, 0.0, f"{label}: back to center, stop"))

    state = {"warmup": 0, "phase_idx": 0, "step": 0}

    def physics_callback(step_size):
        s = state

        if s["warmup"] < 120:
            s["warmup"] += 1
            if s["warmup"] == 120:
                wi, angle, speed, desc = phases[0]
                print(f"[Test] Phase 0: {desc}")
            return

        if s["phase_idx"] >= len(phases):
            return

        wi, angle, speed, desc = phases[s["phase_idx"]]

        # Zero all wheels first
        steer_targets = [0.0] * 4
        spin_targets = [0.0] * 4

        # Set only the active wheel
        steer_targets[wi] = angle
        spin_targets[wi] = speed

        art.apply_action(ArticulationAction(
            joint_positions=np.array(steer_targets),
            joint_indices=steer_indices,
        ))
        art.apply_action(ArticulationAction(
            joint_velocities=np.array(spin_targets),
            joint_indices=spin_indices,
        ))

        s["step"] += 1
        if s["step"] >= steps_per_phase:
            s["phase_idx"] += 1
            s["step"] = 0
            if s["phase_idx"] < len(phases):
                wi, angle, speed, desc = phases[s["phase_idx"]]
                print(f"[Test] Phase {s['phase_idx']}: {desc}")
            else:
                # Stop everything
                art.apply_action(ArticulationAction(
                    joint_positions=np.array([0.0] * 4),
                    joint_indices=steer_indices,
                ))
                art.apply_action(ArticulationAction(
                    joint_velocities=np.array([0.0] * 4),
                    joint_indices=spin_indices,
                ))
                print("[Test] All wheels tested!")

    boot.world.add_physics_callback("test_wheel", callback_fn=physics_callback)

    print("=" * 50)
    print("  G2 Robot: Single wheel steering + spin test")
    print("=" * 50)

    while boot.app.is_running():
        boot.world.step(render=True)

    boot.cleanup()


if __name__ == "__main__":
    main()
