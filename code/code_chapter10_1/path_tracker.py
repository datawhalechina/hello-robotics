"""适用于 G2 全向底盘的前视轨迹跟踪器。"""

from dataclasses import dataclass
import math
from typing import Sequence

try:
    from .geometry import Pose2D, Velocity2D, distance, nearest_path_index, normalize_angle, world_to_body
except ImportError:
    from geometry import Pose2D, Velocity2D, distance, nearest_path_index, normalize_angle, world_to_body


@dataclass(frozen=True)
class TrackingResult:
    command: Velocity2D
    target: Pose2D
    position_error: float
    yaw_error: float
    reached: bool


class HolonomicPathTracker:
    """前视点 + 比例反馈。

    与差速底盘不同，G2 可同时输出 vx、vy，因此无需先原地转向再前进。
    """

    def __init__(
        self,
        lookahead_distance: float = 0.55,
        position_gain: float = 1.25,
        yaw_gain: float = 1.8,
        max_linear_speed: float = 0.55,
        max_angular_speed: float = 1.0,
        detour_speed_limit: float | None = None,
        slow_down_distance: float = 0.90,
        position_tolerance: float = 0.12,
        yaw_tolerance: float = 0.12,
        costmap=None,
    ) -> None:
        self.lookahead_distance = lookahead_distance
        self.position_gain = position_gain
        self.yaw_gain = yaw_gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.detour_speed_limit = (
            max_linear_speed if detour_speed_limit is None else min(max_linear_speed, detour_speed_limit)
        )
        self.slow_down_distance = slow_down_distance
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.costmap = costmap

    def compute(self, current: Pose2D, path: Sequence[Pose2D], goal: Pose2D) -> TrackingResult:
        if not path:
            return TrackingResult(Velocity2D(), current, math.inf, 0.0, False)
        goal_distance = distance(current, goal)
        goal_yaw_error = normalize_angle(goal.yaw - current.yaw)
        reached = goal_distance <= self.position_tolerance and abs(goal_yaw_error) <= self.yaw_tolerance
        if reached:
            return TrackingResult(Velocity2D(), goal, goal_distance, goal_yaw_error, True)

        nearest = nearest_path_index(path, current)
        target = path[min(nearest + 1, len(path) - 1)]
        travelled = 0.0
        for index in range(nearest + 1, len(path)):
            travelled += distance(path[index - 1], path[index])
            candidate = path[index]
            # 弯道附近不能让前视点越过障碍，否则控制器会“切弯”。
            if self.costmap is not None and not self.costmap.line_is_free(current, candidate):
                break
            target = candidate
            if travelled >= self.lookahead_distance:
                break

        dx_body, dy_body = world_to_body(target.x - current.x, target.y - current.y, current.yaw)
        vx = self.position_gain * dx_body
        vy = self.position_gain * dy_body
        speed = math.hypot(vx, vy)
        speed_limit = self.max_linear_speed * min(1.0, max(0.18, goal_distance / self.slow_down_distance))
        if speed > speed_limit and speed > 1e-12:
            scale = speed_limit / speed
            vx *= scale
            vy *= scale

        # 行进时跟随轨迹切线；进入目标位置容差后再精确对齐目标朝向。
        desired_yaw = goal.yaw if goal_distance <= self.position_tolerance else target.yaw
        yaw_error = normalize_angle(desired_yaw - current.yaw)
        wz = max(-self.max_angular_speed, min(self.max_angular_speed, self.yaw_gain * yaw_error))
        if goal_distance <= self.position_tolerance:
            vx = vy = 0.0
        return TrackingResult(Velocity2D(vx, vy, wz), target, goal_distance, yaw_error, False)
