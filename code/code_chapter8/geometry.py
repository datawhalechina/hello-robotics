"""二维位姿与点云坐标变换。"""

from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(angle: float) -> float:
    """把角度归一化到 [-pi, pi)。"""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array((self.x, self.y, self.yaw), dtype=np.float64)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """将机器人坐标系的 Nx2/Nx3 点转换到地图坐标系。"""
        points = np.asarray(points)
        if points.size == 0:
            return points.copy()
        result = np.asarray(points, dtype=np.float64).copy()
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x = points[:, 0]
        y = points[:, 1]
        result[:, 0] = c * x - s * y + self.x
        result[:, 1] = s * x + c * y + self.y
        return result


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """位姿复合：T_world_result = T_world_a * T_a_b。"""
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(
        x=a.x + c * b.x - s * b.y,
        y=a.y + s * b.x + c * b.y,
        yaw=wrap_angle(a.yaw + b.yaw),
    )


def inverse(pose: Pose2D) -> Pose2D:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return Pose2D(
        x=-c * pose.x - s * pose.y,
        y=+s * pose.x - c * pose.y,
        yaw=wrap_angle(-pose.yaw),
    )


def between(a: Pose2D, b: Pose2D) -> Pose2D:
    """返回 b 在 a 坐标系中的相对位姿。"""
    return compose(inverse(a), b)


def pose_distance(a: Pose2D, b: Pose2D) -> tuple[float, float]:
    return math.hypot(b.x - a.x, b.y - a.y), abs(wrap_angle(b.yaw - a.yaw))


def pose_from_vector(vector: np.ndarray) -> Pose2D:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    return Pose2D(float(vector[0]), float(vector[1]), wrap_angle(float(vector[2])))


def quaternion_wxyz_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    """单位四元数 wxyz 转 3x3 旋转矩阵。"""
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("四元数长度不能为 0")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
