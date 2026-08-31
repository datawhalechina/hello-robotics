"""Numerical position IK used only by the scripted demonstrator/corrector."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from config import ARM_LOWER_7, ARM_UPPER_7


@dataclass(frozen=True)
class IKResult:
    joints: np.ndarray
    success: bool
    error: float


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    out = np.eye(4)
    out[:3, :3] = rz @ ry @ rx
    out[:3, 3] = xyz
    return out


def axis_angle(axis, angle):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s, v = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    out = np.eye(4)
    out[:3, :3] = [
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ]
    return out


class RightArmIK:
    def __init__(self):
        p = math.pi
        self.origins = (
            transform((0, 0, -0.069), (p, 0, 0)),
            transform((0, 0, 0.1745), (p / 2, 0, 0)),
            transform(rpy=(-p / 2, 0, 0)),
            transform((0.018, 0, 0.287), (p / 2, 0, 0)),
            transform((-0.018, 0, 0), (-p / 2, 0, 0)),
            transform((0, 0, 0.314), (p / 2, 0, 0)),
            transform(rpy=(p / 2, 0, p / 2)),
        )
        self.tool = transform((0.23645, 0, 0), (p, -p / 2, 0))
        self.axis = np.array([0.0, 0.0, 1.0])

    def chain(self, q):
        tf = np.eye(4)
        points = []
        axes = []
        for origin, angle in zip(self.origins, np.asarray(q), strict=True):
            tf = tf @ origin
            points.append(tf[:3, 3].copy())
            axes.append(tf[:3, :3] @ self.axis)
            tf = tf @ axis_angle(self.axis, float(angle))
        return tf @ self.tool, points, axes

    def solve(self, position, initial, iterations=300):
        q = np.clip(np.asarray(initial, float), ARM_LOWER_7, ARM_UPPER_7).copy()
        center = (ARM_LOWER_7 + ARM_UPPER_7) / 2
        scale = np.maximum((ARM_UPPER_7 - ARM_LOWER_7) / 2, 1e-6)
        for _ in range(iterations):
            end, points, axes = self.chain(q)
            error = np.asarray(position, float) - end[:3, 3]
            norm = float(np.linalg.norm(error))
            if norm < 0.003:
                return IKResult(q.copy(), True, norm)
            jac = np.column_stack(
                [np.cross(a, end[:3, 3] - p) for p, a in zip(points, axes, strict=True)]
            )
            inv = jac.T @ np.linalg.inv(jac @ jac.T + 0.04**2 * np.eye(3))
            delta = inv @ error + 0.02 * (np.eye(7) - inv @ jac) @ (
                (center - q) / scale**2
            )
            peak = np.max(np.abs(delta))
            delta *= min(1, 0.12 / max(peak, 1e-9))
            q = np.clip(q + delta, ARM_LOWER_7, ARM_UPPER_7)
        final_error = float(np.linalg.norm(np.asarray(position, float) - self.chain(q)[0][:3, 3]))
        return IKResult(q.copy(), False, final_error)
