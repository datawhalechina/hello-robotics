"""G2双臂、双夹爪、腰部和三路机载相机接口。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from .config import (
        ARM_LOWER_14,
        ARM_UPPER_14,
        G2_ARM_GRIPPER_DIM,
        G2_WITH_WAIST_DIM,
        GRIPPER_CLOSED_RAD,
        GRIPPER_OPEN_RAD,
        HOME_ARMS_14,
        LEFT_ARM_JOINTS,
        LEFT_GRIPPER_JOINTS,
        RIGHT_ARM_JOINTS,
        RIGHT_GRIPPER_JOINTS,
        WAIST_JOINTS,
    )
except ImportError:
    from config import (
        ARM_LOWER_14,
        ARM_UPPER_14,
        G2_ARM_GRIPPER_DIM,
        G2_WITH_WAIST_DIM,
        GRIPPER_CLOSED_RAD,
        GRIPPER_OPEN_RAD,
        HOME_ARMS_14,
        LEFT_ARM_JOINTS,
        LEFT_GRIPPER_JOINTS,
        RIGHT_ARM_JOINTS,
        RIGHT_GRIPPER_JOINTS,
        WAIST_JOINTS,
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
    def __init__(self, head_path, left_path, right_path, resolution) -> None:
        from isaacsim.sensors.camera import Camera

        self.head = Camera(
            head_path, "acotvla_head", frequency=-1, resolution=resolution
        )
        self.left = Camera(
            left_path,
            "acotvla_left_wrist",
            frequency=-1,
            resolution=resolution,
        )
        self.right = Camera(
            right_path,
            "acotvla_right_wrist",
            frequency=-1,
            resolution=resolution,
        )
        for camera in (self.head, self.left, self.right):
            camera.initialize()
            camera.add_rgb_to_frame()

    def capture(
        self, expected_time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 每次渲染更新缓存；数据保存频率仍由采集器控制。
        # 图像和时间戳必须来自同一缓存，不能混用 get_rgba() 的最新缓冲区。
        frames = [c.get_current_frame() for c in (self.head, self.left, self.right)]
        times = np.asarray([f.get("rendering_time", np.nan) for f in frames])
        if not np.all(np.isfinite(times)) or not np.allclose(
            times, expected_time, rtol=0, atol=1e-6
        ):
            raise RuntimeError(
                f"相机与状态时间不一致：images={times}, state={expected_time}"
            )
        if any(f.get("rgb") is None or np.asarray(f["rgb"]).size == 0 for f in frames):
            raise RuntimeError("G2 机载相机尚未产生图像")
        return tuple(rgba_to_rgb(f["rgb"]).copy() for f in frames)


class G2Robot:
    """模型顺序：左臂7 + 右臂7 + 左夹爪 + 右夹爪 [+ 腰部5]。"""

    def __init__(self, articulation) -> None:
        self.articulation = articulation
        names = list(articulation.dof_names)
        required = (
            *LEFT_ARM_JOINTS,
            *RIGHT_ARM_JOINTS,
            *LEFT_GRIPPER_JOINTS,
            *RIGHT_GRIPPER_JOINTS,
            *WAIST_JOINTS,
        )
        missing = [name for name in required if name not in names]
        if missing:
            raise RuntimeError(f"G2 USD缺少关节：{missing}")
        self.arm_indices = np.array(
            [names.index(n) for n in (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)]
        )
        self.left_gripper_indices = np.array(
            [names.index(n) for n in LEFT_GRIPPER_JOINTS]
        )
        self.right_gripper_indices = np.array(
            [names.index(n) for n in RIGHT_GRIPPER_JOINTS]
        )
        self.waist_indices = np.array([names.index(n) for n in WAIST_JOINTS])

        # Isaac Sim 5.1 的 SingleArticulation 通过 dof_properties 暴露关节限位。
        properties = articulation.dof_properties
        self.waist_lower = np.asarray(properties["lower"], dtype=np.float64)[
            self.waist_indices
        ]
        self.waist_upper = np.asarray(properties["upper"], dtype=np.float64)[
            self.waist_indices
        ]
        self._initial_waist = self._positions()[self.waist_indices].copy()
        self._initial_root_position, self._initial_root_orientation = (
            articulation.get_world_pose()
        )
        self._initial_root_position = np.asarray(
            self._initial_root_position, dtype=np.float64
        )
        self._initial_root_orientation = np.asarray(
            self._initial_root_orientation, dtype=np.float64
        )

    def _positions(self) -> np.ndarray:
        return np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)

    def state16(self) -> np.ndarray:
        q = self._positions()
        return np.concatenate(
            [
                q[self.arm_indices],
                [
                    joint_gripper_to_model(q[self.left_gripper_indices[0]]),
                    joint_gripper_to_model(q[self.right_gripper_indices[0]]),
                ],
            ]
        ).astype(np.float32)

    def state21(self) -> np.ndarray:
        return np.concatenate(
            [self.state16(), self._positions()[self.waist_indices]]
        ).astype(np.float32)

    def state(self, include_waist: bool = False) -> np.ndarray:
        return self.state21() if include_waist else self.state16()

    def apply_absolute(
        self, action: Sequence[float], control_waist: bool = False
    ) -> np.ndarray:
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
        indices = [
            *self.arm_indices,
            self.left_gripper_indices[0],
            self.right_gripper_indices[0],
        ]
        targets = [*arms, left_outer, right_outer]
        applied = [*arms, *grippers]
        if control_waist:
            waist = np.clip(action[16:21], self.waist_lower, self.waist_upper)
            indices.extend(self.waist_indices)
            targets.extend(waist)
            applied.extend(waist)
        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(targets), joint_indices=np.asarray(indices)
            )
        )
        return np.asarray(applied, dtype=np.float32)

    def reset_episode(self) -> None:
        """硬重置机器人状态，避免跨 episode 的速度、底座和关节状态残留。"""
        indices = np.concatenate(
            [
                self.arm_indices,
                [self.left_gripper_indices[0], self.right_gripper_indices[0]],
                self.waist_indices,
            ]
        ).astype(np.int64)
        positions = np.concatenate(
            [
                HOME_ARMS_14,
                [GRIPPER_OPEN_RAD, GRIPPER_OPEN_RAD],
                self._initial_waist,
            ]
        )
        self.articulation.set_world_pose(
            self._initial_root_position, self._initial_root_orientation
        )
        self.articulation.set_linear_velocity(np.zeros(3))
        self.articulation.set_angular_velocity(np.zeros(3))
        self.articulation.set_joint_velocities(
            np.zeros(len(self.articulation.dof_names), dtype=np.float64)
        )
        self.articulation.set_joint_positions(positions, joint_indices=indices)
        home_with_waist = np.concatenate([self.home_action, self._initial_waist])
        self.apply_absolute(home_with_waist, control_waist=True)

    @property
    def home_action(self) -> np.ndarray:
        return np.concatenate(
            [HOME_ARMS_14, [-GRIPPER_OPEN_RAD, -GRIPPER_OPEN_RAD]]
        ).astype(np.float32)
