"""用于生成脚本示教的 G2 右臂 FK 和阻尼最小二乘 IK。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    from .config import ARM_LOWER_7, ARM_UPPER_7
except ImportError:
    from config import ARM_LOWER_7, ARM_UPPER_7


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class IKResult:
    joints: np.ndarray
    success: bool
    position_error: float


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)) -> np.ndarray:
    roll, pitch, yaw = rpy
    result = np.eye(4)
    result[:3, :3] = _rz(yaw) @ _ry(pitch) @ _rx(roll)
    result[:3, 3] = xyz
    return result


def axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c, s, v = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    result = np.eye(4)
    result[:3, :3] = np.array(
        [
            [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
        ]
    )
    return result


def orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """小角度下稳定的旋转误差，足够用于固定向下抓取姿态。"""
    relative = target @ current.T
    return 0.5 * np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )


class RightArmKinematics:
    def __init__(self) -> None:
        pi = math.pi
        self.origins = (
            transform((0, 0, -0.069), (pi, 0, 0)),
            transform((0, 0, 0.1745), (pi / 2, 0, 0)),
            transform((0, 0, 0), (-pi / 2, 0, 0)),
            transform((0.018, 0, 0.287), (pi / 2, 0, 0)),
            transform((-0.018, 0, 0), (-pi / 2, 0, 0)),
            transform((0, 0, 0.314), (pi / 2, 0, 0)),
            transform((0, 0, 0), (pi / 2, 0, pi / 2)),
        )
        self.axes = tuple(np.array([0.0, 0.0, 1.0]) for _ in range(7))
        self.tool = transform((0.23645, 0, 0), (pi, -pi / 2, 0))

    def _chain(self, joints):
        current = np.eye(4)
        points, axes = [], []
        for origin, axis, angle in zip(
            self.origins, self.axes, np.asarray(joints), strict=True
        ):
            current = current @ origin
            points.append(current[:3, 3].copy())
            axes.append(current[:3, :3] @ axis)
            current = current @ axis_rotation(axis, float(angle))
        return current @ self.tool, points, axes

    def forward(self, joints) -> Pose:
        transform_, _, _ = self._chain(joints)
        return Pose(transform_[:3, 3].copy(), transform_[:3, :3].copy())

    def jacobian(self, joints) -> np.ndarray:
        transform_, points, axes = self._chain(joints)
        end = transform_[:3, 3]
        result = np.zeros((6, 7))
        for i, (point, axis) in enumerate(zip(points, axes, strict=True)):
            result[:3, i] = np.cross(axis, end - point)
            result[3:, i] = axis
        return result

    def inverse(
        self, position, rotation=None, initial=None, max_iterations=300
    ) -> IKResult:
        """位置优先的DLS IK。

        桌面抓取时位置是否到达比固定末端朝向更重要，因此默认只求位置；
        ``rotation`` 参数为可选约束；脚本专家默认不固定末端姿态。
        """
        del rotation
        if initial is None:
            initial = (ARM_LOWER_7 + ARM_UPPER_7) / 2
        q = np.clip(np.asarray(initial, dtype=float).copy(), ARM_LOWER_7, ARM_UPPER_7)
        center = (ARM_LOWER_7 + ARM_UPPER_7) / 2
        half = np.maximum((ARM_UPPER_7 - ARM_LOWER_7) / 2, 1e-6)
        position_error = float("inf")
        for _ in range(max_iterations):
            pose = self.forward(q)
            error = np.asarray(position) - pose.position
            position_error = float(np.linalg.norm(error))
            if position_error < 0.003:
                return IKResult(q.copy(), True, position_error)
            jacobian = self.jacobian(q)[:3]
            damping = 0.04
            pinv = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(3), np.eye(3)
            )
            dq = pinv @ error
            dq += 0.02 * (np.eye(7) - pinv @ jacobian) @ ((center - q) / half**2)
            largest = float(np.max(np.abs(dq)))
            if largest > 0.12:
                dq *= 0.12 / largest
            q = np.clip(q + dq, ARM_LOWER_7, ARM_UPPER_7)
        return IKResult(q.copy(), False, position_error)
