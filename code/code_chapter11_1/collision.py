"""机械臂碰撞检测：连杆胶囊近似、场景 AABB 和简化自碰撞。"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:
    from .arm_model import G2PlanningModel
    from .config import BoxObstacle, PlannerConfig
except ImportError:
    from arm_model import G2PlanningModel
    from config import BoxObstacle, PlannerConfig


@dataclass(frozen=True)
class CollisionReport:
    collision: bool
    reason: str = ""


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator < 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(((point - start) @ delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def segment_segment_distance(a0, a1, b0, b1) -> float:
    """两线段距离；使用小型凸二次问题的稳定闭式实现。"""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    denominator = a * c - b * b
    s_n, s_d = denominator, denominator
    t_n, t_d = denominator, denominator
    eps = 1e-12
    if denominator < eps:
        s_n, s_d = 0.0, 1.0
        t_n, t_d = e, c
    else:
        s_n = b * e - c * d
        t_n = a * e - b * d
        if s_n < 0.0:
            s_n, t_n, t_d = 0.0, e, c
        elif s_n > s_d:
            s_n, t_n, t_d = s_d, e + b, c
    if t_n < 0.0:
        t_n = 0.0
        if -d < 0.0:
            s_n = 0.0
        elif -d > a:
            s_n = s_d
        else:
            s_n, s_d = -d, a
    elif t_n > t_d:
        t_n = t_d
        if -d + b < 0.0:
            s_n = 0.0
        elif -d + b > a:
            s_n = s_d
        else:
            s_n, s_d = -d + b, a
    sc = 0.0 if abs(s_n) < eps else s_n / s_d
    tc = 0.0 if abs(t_n) < eps else t_n / t_d
    return float(np.linalg.norm(w + sc * u - tc * v))


def segment_intersects_box(
    start: np.ndarray,
    end: np.ndarray,
    obstacle: BoxObstacle,
    inflation: float,
) -> bool:
    """Slab 法检测线段是否与膨胀后的 AABB 相交。"""
    half = obstacle.size_array * 0.5 + inflation
    lower, upper = obstacle.center_array - half, obstacle.center_array + half
    direction = end - start
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        inv = 1.0 / direction[axis]
        t1 = (lower[axis] - start[axis]) * inv
        t2 = (upper[axis] - start[axis]) * inv
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
        if t_min > t_max:
            return False
    return True


class ArmCollisionChecker:
    """对关节状态和两状态之间的边进行碰撞检查。"""

    def __init__(
        self,
        model: G2PlanningModel,
        obstacles: Iterable[BoxObstacle],
        config: PlannerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or PlannerConfig()
        self.obstacles = [obstacle for obstacle in obstacles if obstacle.collision]

    def set_obstacles(self, obstacles: Iterable[BoxObstacle]) -> None:
        self.obstacles = [obstacle for obstacle in obstacles if obstacle.collision]

    def state_report(self, joint_positions: Sequence[float]) -> CollisionReport:
        q = np.asarray(joint_positions, dtype=np.float64)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            return CollisionReport(True, "非法关节状态")
        if np.any(q < self.model.lower_limits) or np.any(q > self.model.upper_limits):
            return CollisionReport(True, "超过关节限位")

        points = self.model.link_points(q)
        inflation = self.config.link_radius + self.config.safety_margin
        # 第一个零长度段和肩部附近由 body_box 单独约束，从第 1 段开始检测。
        for link_index, (start, end) in enumerate(zip(points[1:-1], points[2:]), start=1):
            for obstacle in self.obstacles:
                # 肩部两段本来就从躯干内部伸出，不能按普通外部障碍处理。
                if obstacle.name == "robot_body" and link_index <= 3:
                    continue
                if segment_intersects_box(start, end, obstacle, inflation):
                    return CollisionReport(
                        True, f"link_{link_index} 与 {obstacle.name} 碰撞"
                    )

        # 非相邻连杆之间使用胶囊中心线距离近似自碰撞。
        segments = list(zip(points[1:-1], points[2:]))
        for i, first in enumerate(segments):
            for j in range(i + 3, len(segments)):
                # 肩部与末端在正常折叠姿态下可能很近，使用稍小阈值。
                threshold = self.config.self_collision_distance
                if segment_segment_distance(*first, *segments[j]) < threshold:
                    return CollisionReport(True, f"link_{i + 1} 与 link_{j + 1} 自碰撞")
        return CollisionReport(False)

    def state_is_valid(self, joint_positions: Sequence[float]) -> bool:
        return not self.state_report(joint_positions).collision

    def edge_is_valid(self, start: Sequence[float], goal: Sequence[float]) -> bool:
        q0, q1 = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
        distance = float(np.linalg.norm(q1 - q0))
        samples = max(1, int(np.ceil(distance / self.config.edge_resolution)))
        for ratio in np.linspace(0.0, 1.0, samples + 1):
            if not self.state_is_valid(q0 + ratio * (q1 - q0)):
                return False
        return True

    def path_is_valid(self, path: Sequence[Sequence[float]]) -> bool:
        return bool(path) and all(
            self.edge_is_valid(start, end) for start, end in zip(path[:-1], path[1:])
        )
