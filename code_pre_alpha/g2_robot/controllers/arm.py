"""
Unified arm controller for the G2 robot.

Provides three control modes:
  1. Direct Joint Control - set joint positions directly or interpolate
  2. IK-Based Control - solve inverse kinematics with fallback chain
  3. Trajectory Execution - smooth Ruckig trajectory following

Also includes PickController state machine for pick-and-place operations.
"""

import enum
import math

import numpy as np
from isaacsim.core.utils.types import ArticulationAction

from g2_robot.core.constants import (
    RIGHT_ARM_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    find_joint_indices,
)


class ArmController:
    """Unified arm controller with multiple control modes.

    Args:
        articulation: SingleArticulation instance
        arm: "right" or "left"
        ik_solver: dict with 'lula' and 'art' keys (optional, for IK mode)
        ruckig: RuckigTrajectory instance (optional, for trajectory mode)
        gripper: ParallelGripper instance (optional, for pick operations)
    """

    def __init__(self, articulation, arm="right", ik_solver=None, ruckig=None, gripper=None):
        self.articulation = articulation
        self.arm = arm
        self.joint_names = RIGHT_ARM_JOINT_NAMES if arm == "right" else LEFT_ARM_JOINT_NAMES
        self.joint_indices = find_joint_indices(articulation.dof_names, self.joint_names)
        self.ik_solver = ik_solver
        self.ruckig = ruckig
        self.gripper = gripper

    # ── Mode 1: Direct Joint Control ────────────────────────────────

    def get_joint_positions(self):
        """Read current arm joint positions.

        Returns:
            list of float joint positions for this arm.
        """
        all_pos = self.articulation.get_joint_positions()
        return [float(all_pos[i]) for i in self.joint_indices]

    def set_joint_positions(self, positions):
        """Directly set arm joint positions.

        Args:
            positions: list/array of joint positions (len = 7 for one arm)
        """
        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=np.array(positions, dtype=np.float64),
                joint_indices=self.joint_indices,
            )
        )

    def interpolate_to(self, target_positions, steps=360):
        """Generate cosine-interpolated trajectory from current to target.

        Uses smooth ease-in-out cosine interpolation for visually natural motion.

        Args:
            target_positions: list of target joint positions (7 values)
            steps: number of interpolation steps

        Returns:
            list of waypoints, each a list of joint positions
        """
        current = self.get_joint_positions()
        trajectory = []
        for i in range(steps):
            alpha = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / steps))
            wp = [c + alpha * (t - c) for c, t in zip(current, target_positions)]
            trajectory.append(wp)
        return trajectory

    # ── Mode 2: IK-Based Control ────────────────────────────────────

    def solve_ik(self, target_pos, target_orient=None, tolerance=None):
        """Solve inverse kinematics with fallback chain.

        Fallback order:
          1. IK with orientation constraint
          2. Position-only IK (drops orientation)
          3. Relaxed position tolerance (5cm)

        Args:
            target_pos: [x, y, z] world target position
            target_orient: [w, x, y, z] wxyz quaternion (None for position-only)
            tolerance: position tolerance override

        Returns:
            (joint_positions, success) where joint_positions is a list of
            arm joint values if success=True, else (None, False).
        """
        if self.ik_solver is None:
            raise RuntimeError("IK solver not configured. Pass ik_solver to ArmController.")

        lula_solver = self.ik_solver["lula"]
        art_solver = self.ik_solver["art"]

        # Update robot base pose for IK
        base_pos, base_orient = self.articulation.get_world_pose()
        lula_solver.set_robot_base_pose(base_pos, base_orient)

        target_pos = np.array(target_pos, dtype=np.float64)

        # Try 1: with orientation constraint
        actions, success = art_solver.compute_inverse_kinematics(
            target_pos, target_orient
        )

        # Fallback 1: position-only
        if not success and target_orient is not None:
            actions, success = art_solver.compute_inverse_kinematics(target_pos)

        # Fallback 2: relaxed tolerance
        if not success:
            tol = tolerance or 0.05
            actions, success = art_solver.compute_inverse_kinematics(
                target_pos, position_tolerance=tol
            )

        if not success:
            return None, False

        ik_joint_positions = [float(v) for v in actions.joint_positions]
        return ik_joint_positions, True

    def move_to_pose(self, target_pos, target_orient=None):
        """Solve IK and generate Ruckig trajectory to target pose.

        Args:
            target_pos: [x, y, z] world target position
            target_orient: [w, x, y, z] wxyz quaternion or None

        Returns:
            list of waypoints or None if IK failed.
        """
        if self.ruckig is None:
            raise RuntimeError("Ruckig controller not configured. Pass ruckig to ArmController.")

        target_joints, success = self.solve_ik(target_pos, target_orient)
        if not success:
            return None

        current_arm = self.get_joint_positions()
        trajectory = self.ruckig.calculate_trajectory(current_arm, target_joints)
        return trajectory

    # ── Mode 3: Trajectory Execution ────────────────────────────────

    def execute_trajectory_step(self, trajectory, index):
        """Apply one waypoint from a trajectory.

        Args:
            trajectory: list of waypoints
            index: current waypoint index

        Returns:
            next index (index + 1), or len(trajectory) if done.
        """
        if index < len(trajectory):
            self.set_joint_positions(trajectory[index])
        return min(index + 1, len(trajectory))


# ── Pick State Machine ──────────────────────────────────────────────


class PickState(enum.Enum):
    IDLE = 0
    OPEN_GRIPPER = 1
    MOVE_PRE_GRASP = 2
    MOVE_GRASP = 3
    CLOSE_GRIPPER = 4
    LIFT = 5
    DONE = 6
    FAILED = 7


class PickController:
    """State-machine controller for picking up an object.

    Uses ArmController for IK solving and trajectory execution.

    Sequence: OPEN_GRIPPER -> MOVE_PRE_GRASP -> MOVE_GRASP -> CLOSE_GRIPPER -> LIFT -> DONE
    """

    def __init__(self, arm_controller: ArmController):
        """
        Args:
            arm_controller: ArmController instance (must have ik_solver, ruckig, gripper configured)
        """
        self.arm = arm_controller

        # State machine
        self.state = PickState.IDLE
        self.wait_counter = 0

        # Trajectory execution
        self.trajectory = []
        self.traj_index = 0

        # Target poses
        self.pre_grasp_pos = None
        self.pre_grasp_orient = None
        self.grasp_pos = None
        self.grasp_orient = None
        self.lift_pos = None
        self.lift_orient = None

    def start_pick(self, object_position, grasp_orientation,
                   pre_grasp_offset, grasp_offset, lift_height):
        """Start a pick operation.

        Args:
            object_position: [x, y, z] world position of the object
            grasp_orientation: [w, x, y, z] wxyz quaternion or None
            pre_grasp_offset: [x, y, z] offset above object for approach
            grasp_offset: [x, y, z] offset for final grasp position
            lift_height: height to lift after grasping
        """
        obj_pos = np.array(object_position, dtype=np.float64)
        orient = np.array(grasp_orientation, dtype=np.float64) if grasp_orientation is not None else None

        self.pre_grasp_pos = obj_pos + np.array(pre_grasp_offset)
        self.pre_grasp_orient = orient

        self.grasp_pos = obj_pos + np.array(grasp_offset)
        self.grasp_orient = orient

        self.lift_pos = self.grasp_pos.copy()
        self.lift_pos[2] += lift_height
        self.lift_orient = orient

        self.state = PickState.OPEN_GRIPPER
        self.wait_counter = 0
        print(f"[PickController] Starting pick at object pos={obj_pos}")

    def step(self):
        """Execute one step of the pick state machine. Call each physics tick.

        Returns:
            (state: PickState, done: bool)
        """
        if self.state == PickState.IDLE:
            return self.state, False

        elif self.state == PickState.OPEN_GRIPPER:
            return self._step_open_gripper()

        elif self.state == PickState.MOVE_PRE_GRASP:
            return self._step_move(self.pre_grasp_pos, self.pre_grasp_orient,
                                   PickState.MOVE_GRASP, "pre-grasp")

        elif self.state == PickState.MOVE_GRASP:
            return self._step_move(self.grasp_pos, self.grasp_orient,
                                   PickState.CLOSE_GRIPPER, "grasp")

        elif self.state == PickState.CLOSE_GRIPPER:
            return self._step_close_gripper()

        elif self.state == PickState.LIFT:
            return self._step_move(self.lift_pos, self.lift_orient,
                                   PickState.DONE, "lift")

        elif self.state == PickState.DONE:
            print("[PickController] Pick sequence completed!")
            return self.state, True

        elif self.state == PickState.FAILED:
            return self.state, True

        return self.state, False

    def _step_open_gripper(self):
        if self.wait_counter == 0:
            print("[PickController] Opening gripper...")
            self.arm.gripper.open()
        self.wait_counter += 1
        if self.wait_counter >= 60:
            self.wait_counter = 0
            self.trajectory = []
            self.traj_index = 0
            self.state = PickState.MOVE_PRE_GRASP
        return self.state, False

    def _step_close_gripper(self):
        if self.wait_counter == 0:
            print("[PickController] Closing gripper...")
            self.arm.gripper.close()
        self.wait_counter += 1
        if self.wait_counter >= 120:
            self.wait_counter = 0
            self.trajectory = []
            self.traj_index = 0
            self.state = PickState.LIFT
        return self.state, False

    def _step_move(self, target_pos, target_orient, next_state, label):
        # Plan trajectory on first entry
        if not self.trajectory:
            self.trajectory = self.arm.move_to_pose(target_pos, target_orient)
            if self.trajectory is None:
                print(f"[PickController] IK failed for {label}!")
                self.state = PickState.FAILED
                return self.state, True
            self.traj_index = 0
            print(f"[PickController] Planned {label} trajectory: {len(self.trajectory)} waypoints")

        # Execute trajectory
        if self.traj_index < len(self.trajectory):
            self.traj_index = self.arm.execute_trajectory_step(self.trajectory, self.traj_index)
            return self.state, False

        # Trajectory complete
        print(f"[PickController] Reached {label} position")
        self.trajectory = []
        self.traj_index = 0
        self.wait_counter = 0
        self.state = next_state
        return self.state, False

    @property
    def is_active(self):
        return self.state not in (PickState.IDLE, PickState.DONE, PickState.FAILED)
