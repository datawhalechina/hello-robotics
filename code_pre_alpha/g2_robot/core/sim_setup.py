"""
SimSetup - Robot-specific initialization after bootstrap.

Handles body posture, IK solvers, gripper, and Ruckig trajectory controller.
Expects an already-initialized articulation and world from SimBootstrap.

Key fix: Body/head joints must be set to their expected posture BEFORE
initializing IK solvers, and the Lula YAML must be regenerated with actual
joint values. Otherwise, the Lula kinematic model doesn't match the real
robot state, causing IK failures.
"""

import os
import tempfile

import numpy as np
import yaml as yaml_lib

from g2_robot.core.constants import (
    FIXED_JOINT_NAMES,
    FIXED_JOINT_TARGETS,
    RIGHT_ARM_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    find_joint_indices,
)


class SimSetup:
    """Sets up IK solvers, gripper, and trajectory controller for the G2 robot."""

    def __init__(self, world, articulation, config: dict):
        """
        Args:
            world: Isaac Sim World instance (from SimBootstrap)
            articulation: SingleArticulation instance (from SimBootstrap)
            config: full config dict
        """
        self.world = world
        self.articulation = articulation
        self.config = config

        self.ik_solvers = {}
        self.gripper = None
        self.ruckig = None
        self._temp_yaml_paths = []

    def setup_all(self):
        """Run the full robot-specific setup sequence."""
        self._set_body_posture()
        self._init_ik_solvers()
        self._init_gripper()
        self._init_ruckig()
        return self

    def cleanup(self):
        """Remove temp Lula YAML files created during setup."""
        for path in self._temp_yaml_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _set_body_posture(self):
        """Set body/head joints to expected posture BEFORE IK solver init.

        Without this, body joints may read as 0 when the Lula YAML expects
        non-zero values (e.g. body_joint3=0.261), causing the Lula kinematic
        model to diverge from the real robot state and IK to fail.
        """
        dof_names = self.articulation.dof_names
        print("[SimSetup] Setting body/head joints to standard posture...")
        for joint_name, target_val in FIXED_JOINT_TARGETS.items():
            idx = next(i for i, d in enumerate(dof_names) if d == joint_name)
            self.articulation.set_joint_positions(
                positions=np.array([target_val]),
                joint_indices=np.array([idx]),
            )
        for _ in range(60):
            self.world.step(render=True)

        # Log actual values for verification
        current_joints = self.articulation.get_joint_positions()
        for joint_name in FIXED_JOINT_NAMES:
            idx = next(i for i, d in enumerate(dof_names) if d == joint_name)
            print(f"  {joint_name} = {current_joints[idx]:.4f} (target: {FIXED_JOINT_TARGETS[joint_name]:.4f})")

    def _create_temp_lula_yaml(self, orig_yaml_path):
        """Create a temp Lula YAML with actual body joint values from simulation.

        Returns:
            Path to the temporary YAML file.
        """
        dof_names = self.articulation.dof_names
        current_joints = self.articulation.get_joint_positions()

        with open(orig_yaml_path, "r") as f:
            lula_desc = yaml_lib.safe_load(f)

        for rule in lula_desc["cspace_to_urdf_rules"]:
            if rule["rule"] == "fixed" and rule["name"] in FIXED_JOINT_TARGETS:
                idx = next(i for i, d in enumerate(dof_names) if d == rule["name"])
                rule["value"] = float(current_joints[idx])

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml_lib.dump(lula_desc, tmp, default_flow_style=False)
        tmp.close()
        self._temp_yaml_paths.append(tmp.name)
        print(f"[SimSetup] Created temp Lula YAML: {tmp.name}")
        return tmp.name

    def _init_ik_solvers(self):
        """Initialize IK solvers for left and right arms using 7-DOF configs."""
        from isaacsim.robot_motion.motion_generation import (
            LulaKinematicsSolver,
            ArticulationKinematicsSolver,
        )

        repo_root = os.getenv("SIM_REPO_ROOT")

        right_desc_orig = os.path.join(
            repo_root,
            "source/data_collection/config/robot_cfg/G2/G2_omnipicker_fixed_right.yaml",
        )
        left_desc_orig = os.path.join(
            repo_root,
            "source/data_collection/config/robot_cfg/G2/G2_omnipicker_fixed_left.yaml",
        )
        urdf_path = os.path.join(
            repo_root,
            "source/robot_cfg/G2_omnipicker/G2_omnipicker.urdf",
        )

        right_desc = self._create_temp_lula_yaml(right_desc_orig)
        left_desc = self._create_temp_lula_yaml(left_desc_orig)

        right_lula = LulaKinematicsSolver(
            robot_description_path=right_desc,
            urdf_path=urdf_path,
        )
        right_solver = ArticulationKinematicsSolver(
            self.articulation, right_lula, "gripper_r_center_link"
        )

        left_lula = LulaKinematicsSolver(
            robot_description_path=left_desc,
            urdf_path=urdf_path,
        )
        left_solver = ArticulationKinematicsSolver(
            self.articulation, left_lula, "gripper_l_center_link"
        )

        self.ik_solvers = {
            "right": {"lula": right_lula, "art": right_solver},
            "left": {"lula": left_lula, "art": left_solver},
        }
        print("[SimSetup] IK solvers initialized for left and right arms")

    def _init_gripper(self):
        """Initialize the right-hand parallel gripper."""
        from g2_robot.controllers.gripper import ParallelGripper

        self.gripper = ParallelGripper(
            end_effector_prim_path="/genie/gripper_r_center_link",
            joint_prim_names=[
                "idx81_gripper_r_outer_joint1",
                "idx71_gripper_r_inner_joint1",
            ],
            joint_opened_positions=np.array([2.0, 2.0]),
            joint_closed_positions=np.array([0.0, 0.0]),
            joint_closed_velocities=np.array([-50.0, -50.0]),
            joint_control_prim="/genie/joints/idx81_gripper_r_outer_joint1",
            gripper_type="angular",
            gripper_max_force=5,
        )
        self.gripper.initialize(
            articulation_apply_action_func=self.articulation.apply_action,
            get_joint_positions_func=self.articulation.get_joint_positions,
            set_joint_positions_func=self.articulation.set_joint_positions,
            dof_names=self.articulation.dof_names,
        )
        print("[SimSetup] Right gripper initialized")

    def _init_ruckig(self):
        """Initialize Ruckig trajectory controller for 7-DOF arm."""
        from g2_robot.controllers.trajectory import RuckigTrajectory

        self.ruckig = RuckigTrajectory(dof_num=7, delta_time=0.01)
        print("[SimSetup] Ruckig trajectory controller initialized")

    def get_arm_joint_indices(self, arm="right"):
        """Get the DOF indices for left or right arm joints."""
        names = RIGHT_ARM_JOINT_NAMES if arm == "right" else LEFT_ARM_JOINT_NAMES
        return find_joint_indices(self.articulation.dof_names, names)
