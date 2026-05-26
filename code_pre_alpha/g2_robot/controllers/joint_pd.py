"""Joint-space PD controller for G2 arm demos.

The controller converts target joint positions into velocity commands with a
proportional-derivative law. It is intentionally small so it can be used in
tutorials without hiding the control loop behind a planner.
"""

import numpy as np
from isaacsim.core.utils.types import ArticulationAction


class JointPDController:
    """PD controller for a selected set of articulation joints.

    Args:
        articulation: Isaac Sim articulation object.
        joint_indices: DOF indices controlled by this instance.
        kp: proportional gain, scalar or one value per joint.
        kd: derivative gain, scalar or one value per joint.
        max_velocity: optional scalar or per-joint velocity clamp.
        position_tolerance: norm threshold used by :meth:`is_reached`.
    """

    def __init__(
        self,
        articulation,
        joint_indices,
        kp=8.0,
        kd=1.2,
        max_velocity=1.5,
        position_tolerance=0.01,
    ):
        self.articulation = articulation
        self.joint_indices = np.asarray(joint_indices, dtype=np.int64)
        self.kp = self._as_gain(kp)
        self.kd = self._as_gain(kd)
        self.max_velocity = self._as_gain(max_velocity)
        self.position_tolerance = float(position_tolerance)

    def _as_gain(self, value):
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(len(self.joint_indices), float(arr))
        if arr.shape != (len(self.joint_indices),):
            raise ValueError(f"expected {len(self.joint_indices)} gains, got {arr.shape}")
        return arr

    def current_state(self):
        """Return current joint positions and velocities for controlled joints."""
        positions = np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)
        velocities = np.asarray(self.articulation.get_joint_velocities(), dtype=np.float64)
        return positions[self.joint_indices], velocities[self.joint_indices]

    def compute_velocity(self, target_positions):
        """Compute PD joint velocities toward target positions."""
        target = np.asarray(target_positions, dtype=np.float64)
        if target.shape != (len(self.joint_indices),):
            raise ValueError(f"expected {len(self.joint_indices)} targets, got {target.shape}")

        current_pos, current_vel = self.current_state()
        error = target - current_pos
        command = self.kp * error - self.kd * current_vel
        return np.clip(command, -self.max_velocity, self.max_velocity)

    def step(self, target_positions):
        """Apply one PD velocity-control step and return the command."""
        velocity_command = self.compute_velocity(target_positions)
        self.articulation.apply_action(
            ArticulationAction(
                joint_velocities=velocity_command,
                joint_indices=self.joint_indices,
            )
        )
        return velocity_command

    def is_reached(self, target_positions):
        """Check whether the target is reached within the configured tolerance."""
        target = np.asarray(target_positions, dtype=np.float64)
        current_pos, _ = self.current_state()
        return np.linalg.norm(target - current_pos) < self.position_tolerance
