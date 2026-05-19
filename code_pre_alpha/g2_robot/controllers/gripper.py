# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.

"""Parallel gripper controller for the G2 robot."""

import asyncio
from typing import Callable, List

import numpy as np
import omni.kit.app
import omni

from g2_robot.utils.logger import Logger

logger = Logger()

from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper
from pxr import UsdPhysics


class ParallelGripper(Gripper):
    """Parallel gripper controller (two-finger gripper).

    Supports position-mode open and force-mode close with configurable
    friction materials for reliable grasping.

    Args:
        end_effector_prim_path: prim path of the gripper root / end effector
        joint_prim_names: left and right finger joint prim names
        joint_opened_positions: joint positions when opened
        joint_closed_positions: joint positions when closed
        joint_opened_velocities: joint velocities when opening (optional)
        joint_closed_velocities: joint velocities when closing (optional)
        action_deltas: deltas for finger positions (optional)
        joint_control_prim: prim path of the drive joint for force control
        gripper_type: "angular" or "linear"
        gripper_max_force: maximum grasp force
    """

    def __init__(
        self,
        end_effector_prim_path: str,
        joint_prim_names: List[str],
        joint_opened_velocities: np.ndarray = None,
        joint_closed_velocities: np.ndarray = None,
        joint_opened_positions: np.ndarray = None,
        joint_closed_positions: np.ndarray = None,
        action_deltas: np.ndarray = None,
        joint_control_prim=None,
        gripper_type: str = "angular",
        gripper_max_force=5,
    ) -> None:
        Gripper.__init__(self, end_effector_prim_path=end_effector_prim_path)

        self._joint_prim_names = joint_prim_names
        self._joint_dof_indicies = np.array([None, None, None, None])
        self._joint_opened_velocities = joint_opened_velocities
        self._joint_closed_velocities = joint_closed_velocities
        self._joint_opened_positions = joint_opened_positions
        self._joint_closed_positions = joint_closed_positions
        self._joint_control_prim = joint_control_prim
        self._get_joint_positions_func = None
        self._set_joint_positions_func = None
        self._action_deltas = np.array([-0.0628, 0.0628])
        self._articulation_num_dofs = None
        self.physics_material = PhysicsMaterial(
            prim_path="/World/gripper_physics",
            static_friction=1,
            dynamic_friction=1,
            restitution=0.1,
        )
        self.object_material = PhysicsMaterial(
            prim_path="/World/object_physics",
            static_friction=1,
            dynamic_friction=1,
            restitution=0.1,
        )
        self.modify_friction_mode("/World/gripper_physics")
        self.modify_friction_mode("/World/object_physics")
        self.is_reached = False
        self.gripper_type = gripper_type
        self.gripper_max_force = gripper_max_force

    def modify_friction_mode(self, prim_path):
        from pxr import PhysxSchema

        stage = omni.usd.get_context().get_stage()
        obj_physics_prim = stage.GetPrimAtPath(prim_path)
        physx_material_api = PhysxSchema.PhysxMaterialAPI(obj_physics_prim)
        if physx_material_api is not None:
            fric_combine_mode = physx_material_api.GetFrictionCombineModeAttr().Get()
            if fric_combine_mode is None:
                physx_material_api.CreateFrictionCombineModeAttr().Set("max")
            elif fric_combine_mode != "max":
                physx_material_api.GetFrictionCombineModeAttr().Set("max")

    @property
    def joint_opened_positions(self) -> np.ndarray:
        return self._joint_opened_positions

    @property
    def joint_closed_positions(self) -> np.ndarray:
        return self._joint_closed_positions

    @property
    def joint_dof_indicies(self) -> np.ndarray:
        return self._joint_dof_indicies

    @property
    def joint_prim_names(self) -> List[str]:
        return self._joint_prim_names

    def initialize(
        self,
        articulation_apply_action_func: Callable,
        get_joint_positions_func: Callable,
        set_joint_positions_func: Callable,
        dof_names: List,
        physics_sim_view=None,
    ) -> None:
        Gripper.initialize(self)
        self._get_joint_positions_func = get_joint_positions_func
        self._articulation_num_dofs = len(dof_names)
        for index in range(len(dof_names)):
            if self._joint_prim_names[0] == dof_names[index]:
                self._joint_dof_indicies[0] = index
            elif self._joint_prim_names[1] == dof_names[index]:
                self._joint_dof_indicies[1] = index
            if len(self._joint_prim_names) > 2:
                if self._joint_prim_names[0] == dof_names[index]:
                    self._joint_dof_indicies[0] = index
                elif self._joint_prim_names[1] == dof_names[index]:
                    self._joint_dof_indicies[1] = index
                elif self._joint_prim_names[2] == dof_names[index]:
                    self._joint_dof_indicies[2] = index
                elif self._joint_prim_names[3] == dof_names[index]:
                    self._joint_dof_indicies[3] = index

        if self._joint_dof_indicies[0] is None or self._joint_dof_indicies[1] is None:
            raise Exception("Not all gripper dof names were resolved to dof handles and dof indices.")
        self._articulation_apply_action_func = articulation_apply_action_func
        current_joint_positions = get_joint_positions_func()
        if self._default_state is None:
            self._default_state = np.array(
                [
                    current_joint_positions[self._joint_dof_indicies[0]],
                    current_joint_positions[self._joint_dof_indicies[1]],
                ]
            )
            if len(self._joint_prim_names) > 2:
                self._default_state = np.array(
                    [
                        current_joint_positions[self._joint_dof_indicies[0]],
                        current_joint_positions[self._joint_dof_indicies[1]],
                        current_joint_positions[self._joint_dof_indicies[2]],
                        current_joint_positions[self._joint_dof_indicies[3]],
                    ]
                )
        self._set_joint_positions_func = set_joint_positions_func

    def open(self) -> None:
        """Open the gripper (position mode)."""
        self._articulation_apply_action_func(self.forward(action="open"))

    def close(self) -> None:
        """Close the gripper (force mode)."""
        self._articulation_apply_action_func(self.forward(action="close"))

    def set_action_deltas(self, value: np.ndarray) -> None:
        self._action_deltas = value

    def get_action_deltas(self) -> np.ndarray:
        return self._action_deltas

    def set_default_state(self, joint_positions: np.ndarray) -> None:
        self._default_state = joint_positions

    def get_default_state(self) -> np.ndarray:
        return self._default_state

    def post_reset(self):
        Gripper.post_reset(self)
        self._set_joint_positions_func(
            positions=self._default_state,
            joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]],
        )

    def set_joint_positions(self, positions: np.ndarray) -> None:
        self._set_joint_positions_func(
            positions=positions,
            joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]],
        )

    def get_joint_positions(self) -> np.ndarray:
        return self._get_joint_positions_func(
            joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]]
        )

    def forward(self, action: str) -> ArticulationAction:
        """Calculate ArticulationAction for "open" or "close".

        Args:
            action: "open" or "close"

        Returns:
            ArticulationAction for the full articulation.
        """
        target_action = None
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._joint_control_prim)
        if prim:
            drive = UsdPhysics.DriveAPI.Get(prim, self.gripper_type)

        if action == "open":
            self.is_reached = False
            target_joint_positions = [None] * self._articulation_num_dofs
            target_joint_positions[self._joint_dof_indicies[0]] = self._joint_opened_positions[0]
            target_joint_positions[self._joint_dof_indicies[1]] = self._joint_opened_positions[1]
            target_action = ArticulationAction(joint_positions=target_joint_positions)
        elif action == "close":
            self.is_reached = False
            current_joint_positions = self._get_joint_positions_func()
            current_drive_finger_position = current_joint_positions[self._joint_dof_indicies[0]]
            target_force = self.gripper_max_force + 2 * np.abs(current_drive_finger_position)
            if prim:
                drive.GetMaxForceAttr().Set(target_force)
            target_joint_velocities = [None] * self._articulation_num_dofs
            target_joint_velocities[self._joint_dof_indicies[0]] = self._joint_closed_velocities[0]
            target_joint_velocities[self._joint_dof_indicies[1]] = self._joint_closed_velocities[1]
            target_action = ArticulationAction(joint_velocities=target_joint_velocities)
        else:
            raise Exception("action {} is not defined for ParallelGripper".format(action))

        self.is_reached = True
        return target_action

    def apply_action(self, control_actions: ArticulationAction) -> None:
        joint_actions = ArticulationAction()
        if control_actions.joint_positions is not None:
            joint_actions.joint_positions = [None] * self._articulation_num_dofs
            joint_actions.joint_positions[self._joint_dof_indicies[0]] = control_actions.joint_positions[0]
            joint_actions.joint_positions[self._joint_dof_indicies[1]] = control_actions.joint_positions[1]
            if len(self._joint_prim_names) > 2:
                joint_actions.joint_positions[self._joint_dof_indicies[2]] = control_actions.joint_positions[2]
                joint_actions.joint_positions[self._joint_dof_indicies[3]] = control_actions.joint_positions[3]
        if control_actions.joint_velocities is not None:
            joint_actions.joint_velocities = [None] * self._articulation_num_dofs
            joint_actions.joint_velocities[self._joint_dof_indicies[0]] = control_actions.joint_velocities[0]
            joint_actions.joint_velocities[self._joint_dof_indicies[1]] = control_actions.joint_velocities[1]
        if control_actions.joint_efforts is not None:
            joint_actions.joint_efforts = [None] * self._articulation_num_dofs
            joint_actions.joint_efforts[self._joint_dof_indicies[0]] = control_actions.joint_efforts[0]
            joint_actions.joint_efforts[self._joint_dof_indicies[1]] = control_actions.joint_efforts[1]
        self._articulation_apply_action_func(control_actions=joint_actions)
