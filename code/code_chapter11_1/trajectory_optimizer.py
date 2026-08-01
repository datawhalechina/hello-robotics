"""碰撞安全的路径捷径优化、重采样和五次时间参数化。"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from .collision import ArmCollisionChecker
    from .config import (
        JOINT_ACCELERATION_LIMITS,
        JOINT_VELOCITY_LIMITS,
        TrajectoryConfig,
    )
except ImportError:
    from collision import ArmCollisionChecker
    from config import JOINT_ACCELERATION_LIMITS, JOINT_VELOCITY_LIMITS, TrajectoryConfig


@dataclass(frozen=True)
class TrajectoryPoint:
    time_from_start: float
    positions: np.ndarray
    velocities: np.ndarray


class TrajectoryOptimizer:
    def __init__(
        self, checker: ArmCollisionChecker, config: TrajectoryConfig | None = None
    ) -> None:
        self.checker = checker
        self.config = config or TrajectoryConfig()
        self.rng = np.random.default_rng(1101)

    @staticmethod
    def length(path: Sequence[Sequence[float]]) -> float:
        return float(
            sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(path[:-1], path[1:]))
        )

    def shortcut(self, path: Sequence[Sequence[float]]) -> list[np.ndarray]:
        result = [np.asarray(q, dtype=np.float64).copy() for q in path]
        for _ in range(self.config.shortcut_attempts):
            if len(result) <= 2:
                break
            i, j = sorted(self.rng.integers(0, len(result), size=2).tolist())
            if j <= i + 1:
                continue
            if self.checker.edge_is_valid(result[i], result[j]):
                result = result[: i + 1] + result[j:]
        return result

    def resample(self, path: Sequence[Sequence[float]]) -> list[np.ndarray]:
        output = [np.asarray(path[0], dtype=np.float64).copy()]
        for start, end in zip(path[:-1], path[1:]):
            q0, q1 = np.asarray(start), np.asarray(end)
            steps = max(1, int(np.ceil(np.linalg.norm(q1 - q0) / self.config.path_resolution)))
            output.extend(q0 + ratio * (q1 - q0) for ratio in np.linspace(0.0, 1.0, steps + 1)[1:])
        return [np.asarray(q, dtype=np.float64) for q in output]

    def optimize(self, path: Sequence[Sequence[float]]) -> list[np.ndarray]:
        if len(path) < 2:
            raise ValueError("path 至少需要两个点")
        optimized = self.resample(self.shortcut(path))
        if not self.checker.path_is_valid(optimized):
            raise RuntimeError("优化后路径发生碰撞")
        return optimized

    def time_parameterize(self, path: Sequence[Sequence[float]]) -> list[TrajectoryPoint]:
        """按速度/加速度上限给路径加时间，并用五次 smoothstep 插值。"""
        if len(path) < 2:
            raise ValueError("path 至少需要两个点")
        velocity_limits = JOINT_VELOCITY_LIMITS * self.config.speed_scale
        acceleration_limits = JOINT_ACCELERATION_LIMITS * self.config.speed_scale
        trajectory = [TrajectoryPoint(0.0, np.asarray(path[0]).copy(), np.zeros(7))]
        current_time = 0.0
        for start, end in zip(path[:-1], path[1:]):
            q0, q1 = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
            delta = np.abs(q1 - q0)
            # 五次曲线峰值速度约为 1.875*dq/T；保守加入加速度约束。
            duration_v = float(np.max(1.875 * delta / np.maximum(velocity_limits, 1e-6)))
            duration_a = float(np.max(np.sqrt(5.8 * delta / np.maximum(acceleration_limits, 1e-6))))
            duration = max(0.12, duration_v, duration_a)
            steps = max(2, int(np.ceil(duration / self.config.sample_dt)))
            for step in range(1, steps + 1):
                tau = step / steps
                scale = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                scale_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
                t = current_time + tau * duration
                trajectory.append(TrajectoryPoint(t, q0 + scale * (q1 - q0), scale_dot * (q1 - q0)))
            current_time += duration
        return trajectory
