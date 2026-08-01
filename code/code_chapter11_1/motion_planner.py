"""双向 RRT-Connect 关节空间规划器。"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from .collision import ArmCollisionChecker
    from .config import PlannerConfig
except ImportError:
    from collision import ArmCollisionChecker
    from config import PlannerConfig


@dataclass
class _Node:
    q: np.ndarray
    parent: int | None


@dataclass(frozen=True)
class PlanResult:
    path: list[np.ndarray]
    success: bool
    iterations: int
    message: str


class RRTConnectPlanner:
    """适合教学的最小双向 RRT-Connect。"""

    def __init__(
        self, checker: ArmCollisionChecker, config: PlannerConfig | None = None
    ) -> None:
        self.checker = checker
        self.config = config or PlannerConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    @staticmethod
    def _nearest(tree: list[_Node], sample: np.ndarray) -> int:
        distances = [np.linalg.norm(node.q - sample) for node in tree]
        return int(np.argmin(distances))

    def _steer(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - start
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.step_size:
            return target.copy()
        return start + delta * (self.config.step_size / distance)

    def _extend(self, tree: list[_Node], sample: np.ndarray) -> tuple[int | None, bool]:
        nearest_index = self._nearest(tree, sample)
        candidate = self._steer(tree[nearest_index].q, sample)
        if not self.checker.edge_is_valid(tree[nearest_index].q, candidate):
            return None, False
        tree.append(_Node(candidate, nearest_index))
        reached = float(np.linalg.norm(candidate - sample)) < 1e-8
        return len(tree) - 1, reached

    def _connect(self, tree: list[_Node], target: np.ndarray) -> tuple[int | None, bool]:
        last_index = None
        while True:
            last_index, reached = self._extend(tree, target)
            if last_index is None:
                return None, False
            if reached:
                return last_index, True

    @staticmethod
    def _trace(tree: list[_Node], index: int) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        while index is not None:
            path.append(tree[index].q.copy())
            index = tree[index].parent
        return list(reversed(path))

    def plan(self, start: Sequence[float], goal: Sequence[float]) -> PlanResult:
        start_q, goal_q = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
        for label, state in (("起点", start_q), ("终点", goal_q)):
            report = self.checker.state_report(state)
            if report.collision:
                return PlanResult([], False, 0, f"{label}无效：{report.reason}")
        if self.checker.edge_is_valid(start_q, goal_q):
            return PlanResult([start_q.copy(), goal_q.copy()], True, 0, "直线连接成功")

        start_tree, goal_tree = [_Node(start_q.copy(), None)], [_Node(goal_q.copy(), None)]
        tree_a, tree_b = start_tree, goal_tree
        a_is_start = True
        for iteration in range(1, self.config.max_iterations + 1):
            if self.rng.random() < self.config.goal_bias:
                sample = tree_b[0].q.copy()
            else:
                sample = self.rng.uniform(
                    self.checker.model.lower_limits, self.checker.model.upper_limits
                )
            new_index, _ = self._extend(tree_a, sample)
            if new_index is not None:
                connect_index, connected = self._connect(tree_b, tree_a[new_index].q)
                if connected and connect_index is not None:
                    path_a = self._trace(tree_a, new_index)
                    path_b = self._trace(tree_b, connect_index)
                    if a_is_start:
                        path = path_a + list(reversed(path_b))[1:]
                    else:
                        path = path_b + list(reversed(path_a))[1:]
                    return PlanResult(path, True, iteration, "RRT-Connect 找到路径")
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start
        return PlanResult([], False, self.config.max_iterations, "超过最大采样次数")
