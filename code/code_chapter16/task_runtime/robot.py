"""Minimal G2 joint and camera interface."""

from __future__ import annotations
import numpy as np
from config import (
    ACTION_DIM,
    ARM_LOWER_14,
    ARM_UPPER_14,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_ARMS_14,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    WAIST_JOINTS,
)


def rgb8(frame):
    x = np.asarray(frame)[..., :3]
    if np.issubdtype(x.dtype, np.floating):
        x = np.nan_to_num(x * 255 if x.size and np.nanmax(x) <= 1 else x)
    return np.ascontiguousarray(np.clip(x, 0, 255).astype(np.uint8))


def smoothstep(t):
    t = float(np.clip(t, 0, 1))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def model_to_joint_gripper(x):
    return float(np.clip(-x, GRIPPER_CLOSED, GRIPPER_OPEN))


def joint_to_model_gripper(x):
    return -float(np.clip(x, GRIPPER_CLOSED, GRIPPER_OPEN))


def closed_fraction(x):
    return (GRIPPER_OPEN - model_to_joint_gripper(x)) / GRIPPER_OPEN


class CameraRig:
    def __init__(self, prims, resolution, frequency):
        from isaacsim.sensors.camera import Camera

        self.cameras = {
            name: Camera(
                path, f"chapter15_{name}", frequency=frequency, resolution=resolution
            )
            for name, path in prims.items()
        }
        for camera in self.cameras.values():
            camera.initialize()
            camera.add_rgb_to_frame()

    def capture(self):
        frames = [self.cameras[k].get_rgba() for k in ("head", "left", "right")]
        if any(x is None for x in frames):
            raise RuntimeError("camera has no RGB frame")
        return tuple(rgb8(x) for x in frames)


class G2Robot:
    def __init__(self, articulation):
        self.articulation = articulation
        names = list(articulation.dof_names)
        required = (
            *LEFT_ARM_JOINTS,
            *RIGHT_ARM_JOINTS,
            *LEFT_GRIPPER_JOINTS,
            *RIGHT_GRIPPER_JOINTS,
            *WAIST_JOINTS,
        )
        missing = [x for x in required if x not in names]
        if missing:
            raise RuntimeError(f"G2 USD missing joints: {missing}")
        self.arm_ids = np.array(
            [names.index(x) for x in (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)]
        )
        self.left_gripper_ids = np.array([names.index(x) for x in LEFT_GRIPPER_JOINTS])
        self.right_gripper_ids = np.array(
            [names.index(x) for x in RIGHT_GRIPPER_JOINTS]
        )
        self.waist_ids = np.array([names.index(x) for x in WAIST_JOINTS])
        self.initial_waist = self._q()[self.waist_ids].copy()
        self.initial_root = tuple(np.asarray(x) for x in articulation.get_world_pose())

    def _q(self):
        return np.asarray(self.articulation.get_joint_positions(), float)

    def state(self):
        q = self._q()
        return np.concatenate(
            [
                q[self.arm_ids],
                [
                    joint_to_model_gripper(q[self.left_gripper_ids[0]]),
                    joint_to_model_gripper(q[self.right_gripper_ids[0]]),
                ],
            ]
        ).astype(np.float32)

    @property
    def home(self):
        return np.concatenate([HOME_ARMS_14, [-GRIPPER_OPEN, -GRIPPER_OPEN]]).astype(
            np.float32
        )

    def apply(self, action):
        from isaacsim.core.utils.types import ArticulationAction

        action = np.asarray(action, float).reshape(ACTION_DIM)
        arms = np.clip(action[:14], ARM_LOWER_14, ARM_UPPER_14)
        grips = np.clip(action[14:], -GRIPPER_OPEN, 0)
        ids = np.concatenate(
            [self.arm_ids, [self.left_gripper_ids[0], self.right_gripper_ids[0]]]
        )
        targets = np.concatenate(
            [arms, [model_to_joint_gripper(grips[0]), model_to_joint_gripper(grips[1])]]
        )
        self.articulation.apply_action(
            ArticulationAction(joint_positions=targets, joint_indices=ids)
        )
        return np.concatenate([arms, grips]).astype(np.float32)

    def reset(self):
        ids = np.concatenate(
            [
                self.arm_ids,
                [self.left_gripper_ids[0], self.right_gripper_ids[0]],
                self.waist_ids,
            ]
        )
        positions = np.concatenate(
            [HOME_ARMS_14, [GRIPPER_OPEN, GRIPPER_OPEN], self.initial_waist]
        )
        self.articulation.set_world_pose(*self.initial_root)
        self.articulation.set_linear_velocity(np.zeros(3))
        self.articulation.set_angular_velocity(np.zeros(3))
        self.articulation.set_joint_velocities(
            np.zeros(len(self.articulation.dof_names))
        )
        self.articulation.set_joint_positions(positions, joint_indices=ids)
        self.apply(self.home)
