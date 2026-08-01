"""滚动窗口局部重规划。

局部规划器不直接控制车轮，而是输出一小段无碰撞参考轨迹：
1. 找到机器人在全局路径上的最近点；
2. 沿全局路径选择局部目标；
3. 把动态障碍写入临时代价地图；
4. 若原路径安全则复用，否则在局部窗口内运行 A*；
5. 对仍然安全的绕行路径做短暂保持，避免障碍两侧反复切换。
"""

from dataclasses import dataclass
from typing import Sequence

try:
    from .costmap import OccupancyGrid2D
    from .geometry import (
        CircleObstacle,
        Pose2D,
        distance,
        nearest_path_index,
        path_length,
    )
    from .global_planner import AStarPlanner, PlanningError
    from .trajectory import TrajectoryOptimizer
except ImportError:
    from costmap import OccupancyGrid2D
    from geometry import CircleObstacle, Pose2D, distance, nearest_path_index, path_length
    from global_planner import AStarPlanner, PlanningError
    from trajectory import TrajectoryOptimizer


@dataclass(frozen=True)
class LocalPlanResult:
    path: list[Pose2D]
    used_detour: bool
    blocked: bool
    target_index: int


class LocalPlanner:
    def __init__(
        self,
        static_costmap: OccupancyGrid2D,
        lookahead_distance: float = 2.4,
        window_radius: float = 3.0,
        path_spacing: float = 0.12,
        dynamic_padding: float = 0.44,
        detour_hold_cycles: int = 3,
        reuse_min_distance: float = 0.55,
    ) -> None:
        self.static_costmap = static_costmap
        self.lookahead_distance = lookahead_distance
        self.window_radius = window_radius
        self.path_spacing = path_spacing
        self.dynamic_padding = dynamic_padding
        self.detour_hold_cycles = max(0, int(detour_hold_cycles))
        self.reuse_min_distance = max(0.0, reuse_min_distance)
        self._last_global_index = 0
        self._previous_path: list[Pose2D] = []
        self._previous_used_detour = False
        self._clear_cycles = 0

    def reset(self) -> None:
        self._last_global_index = 0
        self._previous_path = []
        self._previous_used_detour = False
        self._clear_cycles = 0

    def plan(
        self,
        current: Pose2D,
        global_path: Sequence[Pose2D],
        dynamic_obstacles: Sequence[CircleObstacle] = (),
    ) -> LocalPlanResult:
        if len(global_path) < 2:
            self.reset()
            return LocalPlanResult(list(global_path), False, not bool(global_path), 0)

        nearest = nearest_path_index(global_path, current, max(0, self._last_global_index - 5))
        self._last_global_index = nearest
        target_index = self._target_index(global_path, nearest)
        target = global_path[target_index]

        local_map = self.static_costmap.copy()
        local_map.add_circles(dynamic_obstacles, padding=self.dynamic_padding)
        reference = [Pose2D(current.x, current.y, current.yaw), *global_path[nearest : target_index + 1]]
        reference_is_free = local_map.path_is_free(reference)

        # 一条绕行路径只要仍然安全，就不要每次重规划时重新选择障碍物左右侧。
        # 当全局参考线恢复畅通后，再连续确认若干周期才切回，形成简单的滞回。
        previous = self._remaining_previous_path(current)
        if self._previous_used_detour and previous and local_map.path_is_free(previous):
            if reference_is_free:
                self._clear_cycles += 1
                if self._clear_cycles <= self.detour_hold_cycles:
                    self._remember(previous, used_detour=True)
                    return LocalPlanResult(previous, True, False, target_index)
            else:
                self._clear_cycles = 0
                self._remember(previous, used_detour=True)
                return LocalPlanResult(previous, True, False, target_index)

        self._clear_cycles = 0
        if reference_is_free:
            self._remember(reference, used_detour=False)
            return LocalPlanResult(reference, False, False, target_index)

        bounds = (
            current.x - self.window_radius,
            current.y - self.window_radius,
            current.x + self.window_radius,
            current.y + self.window_radius,
        )
        try:
            raw = AStarPlanner(local_map).plan(current, target, bounds=bounds).path
            optimized = TrajectoryOptimizer(local_map, self.path_spacing).optimize(raw)
            self._remember(optimized, used_detour=True)
            return LocalPlanResult(optimized, True, False, target_index)
        except PlanningError:
            self._previous_path = []
            self._previous_used_detour = False
            return LocalPlanResult([Pose2D(current.x, current.y, current.yaw)], True, True, target_index)

    def _remaining_previous_path(self, current: Pose2D) -> list[Pose2D]:
        """裁掉机器人已经走过的旧路径，只保留当前位置之后的部分。"""
        if len(self._previous_path) < 2:
            return []
        nearest = nearest_path_index(self._previous_path, current)
        remaining = [Pose2D(current.x, current.y, current.yaw), *self._previous_path[nearest + 1 :]]
        if len(remaining) < 2 or path_length(remaining) < self.reuse_min_distance:
            return []
        return remaining

    def _remember(self, path: Sequence[Pose2D], used_detour: bool) -> None:
        self._previous_path = list(path)
        self._previous_used_detour = used_detour

    def _target_index(self, path: Sequence[Pose2D], start: int) -> int:
        travelled = 0.0
        for index in range(start + 1, len(path)):
            travelled += distance(path[index - 1], path[index])
            if travelled >= self.lookahead_distance:
                return index
        return len(path) - 1
