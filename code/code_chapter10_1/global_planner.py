"""A* 全局路径规划。

教学重点：开放列表、累计代价、启发函数、父节点回溯和防止斜向穿墙。
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

try:
    from .costmap import OccupancyGrid2D
    from .geometry import Pose2D
except ImportError:
    from costmap import OccupancyGrid2D
    from geometry import Pose2D


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanResult:
    path: list[Pose2D]
    expanded_nodes: int
    path_cost: float


class AStarPlanner:
    def __init__(
        self,
        costmap: OccupancyGrid2D,
        allow_diagonal: bool = True,
        heuristic_weight: float = 1.0,
    ) -> None:
        if heuristic_weight <= 0.0:
            raise ValueError("heuristic_weight 必须大于 0")
        self.costmap = costmap
        self.allow_diagonal = allow_diagonal
        self.heuristic_weight = heuristic_weight

    def plan(
        self,
        start: Pose2D,
        goal: Pose2D,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> PlanResult:
        start_cell = self._nearest_free(self.costmap.world_to_grid(start.x, start.y), bounds)
        goal_cell = self._nearest_free(self.costmap.world_to_grid(goal.x, goal.y), bounds)
        if start_cell is None:
            raise PlanningError("起点附近没有可通行栅格")
        if goal_cell is None:
            raise PlanningError("终点附近没有可通行栅格")

        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        counter = 0
        heapq.heappush(open_heap, (0.0, counter, start_cell))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_cost = {start_cell: 0.0}
        closed: set[tuple[int, int]] = set()
        expanded = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_cell:
                cells = self._reconstruct(came_from, current)
                path = [Pose2D(*self.costmap.grid_to_world(row, column)) for row, column in cells]
                path[0] = Pose2D(start.x, start.y, start.yaw)
                path[-1] = Pose2D(goal.x, goal.y, goal.yaw)
                return PlanResult(path, expanded, g_cost[current] * self.costmap.resolution)

            closed.add(current)
            expanded += 1
            for neighbor, step_cost in self._neighbors(current, bounds):
                if neighbor in closed:
                    continue
                tentative = g_cost[current] + step_cost
                if tentative >= g_cost.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_cost[neighbor] = tentative
                counter += 1
                priority = tentative + self.heuristic_weight * self._heuristic(neighbor, goal_cell)
                heapq.heappush(open_heap, (priority, counter, neighbor))

        raise PlanningError("A* 未找到从起点到终点的可行路径")

    def _neighbors(
        self,
        cell: tuple[int, int],
        bounds: tuple[float, float, float, float] | None,
    ) -> Iterable[tuple[tuple[int, int], float]]:
        row, column = cell
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
        if self.allow_diagonal:
            diagonal = math.sqrt(2.0)
            moves += [(1, 1, diagonal), (1, -1, diagonal), (-1, 1, diagonal), (-1, -1, diagonal)]
        for dr, dc, cost in moves:
            next_cell = row + dr, column + dc
            if not self._cell_allowed(next_cell, bounds):
                continue
            # 对角运动时两侧正交栅格也必须空闲，避免从障碍角点“穿墙”。
            if dr and dc:
                if self.costmap.is_occupied_cell(row + dr, column):
                    continue
                if self.costmap.is_occupied_cell(row, column + dc):
                    continue
            yield next_cell, cost

    def _cell_allowed(
        self,
        cell: tuple[int, int],
        bounds: tuple[float, float, float, float] | None,
    ) -> bool:
        row, column = cell
        if self.costmap.is_occupied_cell(row, column):
            return False
        if bounds is None:
            return True
        x, y = self.costmap.grid_to_world(row, column)
        min_x, min_y, max_x, max_y = bounds
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _nearest_free(
        self,
        requested: tuple[int, int],
        bounds: tuple[float, float, float, float] | None,
        max_radius_cells: int = 20,
    ) -> tuple[int, int] | None:
        if self._cell_allowed(requested, bounds):
            return requested
        row, column = requested
        for radius in range(1, max_radius_cells + 1):
            candidates = []
            for dr in range(-radius, radius + 1):
                candidates.append((row + dr, column - radius))
                candidates.append((row + dr, column + radius))
            for dc in range(-radius + 1, radius):
                candidates.append((row - radius, column + dc))
                candidates.append((row + radius, column + dc))
            valid = [cell for cell in candidates if self._cell_allowed(cell, bounds)]
            if valid:
                return min(valid, key=lambda cell: (cell[0] - row) ** 2 + (cell[1] - column) ** 2)
        return None

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        dx, dy = abs(a[1] - b[1]), abs(a[0] - b[0])
        # Octile distance：适合 8 邻域移动。
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
