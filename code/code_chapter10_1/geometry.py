"""导航算法共用的二维几何类型和小工具。

本文件不依赖 Isaac Sim 或 ROS 2，便于单独学习和测试。
坐标约定：x 向前、y 向左、yaw 逆时针为正。
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass
class Velocity2D:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass(frozen=True)
class CircleObstacle:
    """局部规划器使用的圆形动态障碍物。"""

    x: float
    y: float
    radius: float


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def world_to_body(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def cumulative_lengths(path: Sequence[Pose2D]) -> list[float]:
    if not path:
        return []
    lengths = [0.0]
    for previous, current in zip(path, path[1:]):
        lengths.append(lengths[-1] + distance(previous, current))
    return lengths


def nearest_path_index(path: Sequence[Pose2D], pose: Pose2D, start: int = 0) -> int:
    if not path:
        raise ValueError("path 不能为空")
    start = max(0, min(start, len(path) - 1))
    return min(
        range(start, len(path)),
        key=lambda index: (path[index].x - pose.x) ** 2 + (path[index].y - pose.y) ** 2,
    )


def path_length(path: Iterable[Pose2D]) -> float:
    points = list(path)
    return sum(distance(a, b) for a, b in zip(points, points[1:]))
