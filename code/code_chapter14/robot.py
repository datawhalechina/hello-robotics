"""G2双臂、双夹爪、腰部和三路机载相机接口。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from .config import (
        ARM_LOWER_14, ARM_UPPER_14, G2_ARM_GRIPPER_DIM, G2_WITH_WAIST_DIM,
        GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD, HOME_ARMS_14, LEFT_ARM_JOINTS,
        LEFT_GRIPPER_JOINTS, RIGHT_ARM_JOINTS, RIGHT_GRIPPER_JOINTS, WAIST_JOINTS,
    )
except ImportError:
    from config import (
        ARM_LOWER_14, ARM_UPPER_14, G2_ARM_GRIPPER_DIM, G2_WITH_WAIST_DIM,
        GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD, HOME_ARMS_14, LEFT_ARM_JOINTS,
        LEFT_GRIPPER_JOINTS, RIGHT_ARM_JOINTS, RIGHT_GRIPPER_JOINTS, WAIST_JOINTS,
    )


def rgba_to_rgb(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"相机图像形状错误：{image.shape}")
    rgb = image[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if rgb.size and float(np.nanmax(rgb)) <= 1.0 else 1.0
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0) * scale
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))


def quintic_blend(progress: float) -> float:
    t = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def model_gripper_to_joint(value: float) -> float:
    return float(np.clip(-value, GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD))


def joint_gripper_to_model(value: float) -> float:
    return -float(np.clip(value, GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD))


def gripper_closed_amount(model_value: float) -> float:
    joint = model_gripper_to_joint(model_value)
    return float((GRIPPER_OPEN_RAD - joint) / GRIPPER_OPEN_RAD)


class G2Cameras:
    def __init__(self, head_path, left_path, right_path, resolution, frequency) -> None:
        from isaacsim.sensors.camera import Camera

        self.head = Camera(head_path, "chapter14_head", frequency=frequency, resolution=resolution)
        self.left = Camera(left_path, "chapter14_left_wrist", frequency=frequency, resolution=resolution)
        self.right = Camera(right_path, "chapter14_right_wrist", frequency=frequency, resolution=resolution)
        for camera in (self.head, self.left, self.right):
            camera.initialize()
            camera.add_rgb_to_frame()

    def capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = (self.head.get_rgba(), self.left.get_rgba(), self.right.get_rgba())
        if any(frame is None for frame in frames):
            raise RuntimeError("G2机载相机尚未产生图像")
        return tuple(rgba_to_rgb(frame) for frame in frames)


class G2Robot:
    """模型顺序：左臂7 + 右臂7 + 左夹爪 + 右夹爪 [+ 腰部5]。"""

    def __init__(self, articulation) -> None:
        self.articulation = articulation
        names = list(articulation.dof_names)
        required = (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS, *LEFT_GRIPPER_JOINTS,
                    *RIGHT_GRIPPER_JOINTS, *WAIST_JOINTS)
        missing = [name for name in required if name not in names]
        if missing:
            raise RuntimeError(f"G2 USD缺少关节：{missing}")
        self.arm_indices = np.array([names.index(n) for n in (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)])
        self.left_gripper_indices = np.array([names.index(n) for n in LEFT_GRIPPER_JOINTS])
        self.right_gripper_indices = np.array([names.index(n) for n in RIGHT_GRIPPER_JOINTS])
        self.waist_indices = np.array([names.index(n) for n in WAIST_JOINTS])

        # Isaac Sim 5.1 的 SingleArticulation 通过 dof_properties 暴露关节限位。
        properties = articulation.dof_properties
        self.waist_lower = np.asarray(properties["lower"], dtype=np.float64)[self.waist_indices]
        self.waist_upper = np.asarray(properties["upper"], dtype=np.float64)[self.waist_indices]

    def _positions(self) -> np.ndarray:
        return np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)

    def state16(self) -> np.ndarray:
        q = self._positions()
        return np.concatenate([
            q[self.arm_indices],
            [joint_gripper_to_model(q[self.left_gripper_indices[0]]),
             joint_gripper_to_model(q[self.right_gripper_indices[0]])],
        ]).astype(np.float32)

    def state21(self) -> np.ndarray:
        return np.concatenate([self.state16(), self._positions()[self.waist_indices]]).astype(np.float32)

    def state(self, include_waist: bool = False) -> np.ndarray:
        return self.state21() if include_waist else self.state16()

    def apply_absolute(self, action: Sequence[float], control_waist: bool = False) -> np.ndarray:
        from isaacsim.core.utils.types import ArticulationAction

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        required = G2_WITH_WAIST_DIM if control_waist else G2_ARM_GRIPPER_DIM
        if action.size < required or not np.all(np.isfinite(action[:required])):
            raise ValueError(f"G2动作必须至少包含{required}个有限数")

        arms = np.clip(action[:14], ARM_LOWER_14, ARM_UPPER_14)
        grippers = np.clip(action[14:16], -GRIPPER_OPEN_RAD, -GRIPPER_CLOSED_RAD)
        left_outer = model_gripper_to_joint(grippers[0])
        right_outer = model_gripper_to_joint(grippers[1])
        # inner关节在G2 USD中是outer的PhysX mimic关节，没有独立drive。
        # 与Genie Sim的G2控制接口一致，只给两个outer关节下发目标。
        indices = [*self.arm_indices, self.left_gripper_indices[0], self.right_gripper_indices[0]]
        targets = [*arms, left_outer, right_outer]
        applied = [*arms, *grippers]
        if control_waist:
            waist = np.clip(action[16:21], self.waist_lower, self.waist_upper)
            indices.extend(self.waist_indices)
            targets.extend(waist)
            applied.extend(waist)
        self.articulation.apply_action(ArticulationAction(
            joint_positions=np.asarray(targets), joint_indices=np.asarray(indices)
        ))
        return np.asarray(applied, dtype=np.float32)

    @property
    def home_action(self) -> np.ndarray:
        return np.concatenate([HOME_ARMS_14, [-GRIPPER_OPEN_RAD, -GRIPPER_OPEN_RAD]]).astype(np.float32)
