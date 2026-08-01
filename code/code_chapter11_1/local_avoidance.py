"""局部避障：只重规划发生碰撞附近的一小段剩余路径。"""

from dataclasses import replace
from typing import Sequence

import numpy as np

try:
    from .collision import ArmCollisionChecker
    from .config import PlannerConfig
    from .motion_planner import RRTConnectPlanner
except ImportError:
    from collision import ArmCollisionChecker
    from config import PlannerConfig
    from motion_planner import RRTConnectPlanner


class LocalPathRepair:
    """碰撞监测 + 局部 RRT 修复。

    与每次从头做全局规划相比，它保留仍然安全的路径前后段，只替换碰撞窗口。
    静态案例中通常不会触发；移动/新增障碍后可直接复用。
    """

    def __init__(self, checker: ArmCollisionChecker, window: int = 4) -> None:
        self.checker = checker
        self.window = max(2, int(window))

    def first_invalid_edge(self, path: Sequence[Sequence[float]]) -> int | None:
        for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
            if not self.checker.edge_is_valid(start, end):
                return index
        return None

    def repair(self, path: Sequence[Sequence[float]]) -> list[np.ndarray]:
        copied = [np.asarray(q, dtype=np.float64).copy() for q in path]
        invalid = self.first_invalid_edge(copied)
        if invalid is None:
            return copied
        left = max(0, invalid - self.window + 1)
        right = min(len(copied) - 1, invalid + self.window)
        local_config = replace(
            self.checker.config,
            max_iterations=min(2200, self.checker.config.max_iterations),
            goal_bias=0.28,
            random_seed=self.checker.config.random_seed + invalid + 101,
        )
        result = RRTConnectPlanner(self.checker, local_config).plan(
            copied[left], copied[right]
        )
        if not result.success:
            raise RuntimeError(f"局部重规划失败：{result.message}")
        repaired = copied[:left] + result.path + copied[right + 1 :]
        if not self.checker.path_is_valid(repaired):
            raise RuntimeError("局部重规划生成了无效路径")
        return repaired
