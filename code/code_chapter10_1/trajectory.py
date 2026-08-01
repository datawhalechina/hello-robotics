"""全局路径裁剪、平滑、等距采样和航向生成。"""

from __future__ import annotations

import math
from typing import Sequence

try:
    from .costmap import OccupancyGrid2D
    from .geometry import Pose2D, distance, normalize_angle
except ImportError:
    from costmap import OccupancyGrid2D
    from geometry import Pose2D, distance, normalize_angle


class TrajectoryOptimizer:
    def __init__(self, costmap: OccupancyGrid2D, spacing: float = 0.12) -> None:
        if spacing <= 0.0:
            raise ValueError("spacing 必须大于 0")
        self.costmap = costmap
        self.spacing = spacing

    def optimize(self, path: Sequence[Pose2D], shortcut_passes: int = 2) -> list[Pose2D]:
        """依次进行视线裁剪、保碰撞约束的 Chaikin 平滑和等距重采样。"""
        if len(path) < 2:
            return list(path)
        result = list(path)
        for _ in range(max(1, shortcut_passes)):
            result = self.shortcut(result)
        smoothed = self.chaikin(result, iterations=2)
        if self.costmap.path_is_free(smoothed):
            result = smoothed
        result = self.resample(result, self.spacing)
        return self.assign_yaws(result, final_yaw=path[-1].yaw)

    def shortcut(self, path: Sequence[Pose2D]) -> list[Pose2D]:
        if len(path) <= 2:
            return list(path)
        output = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            candidate = len(path) - 1
            while candidate > anchor + 1:
                if self.costmap.line_is_free(path[anchor], path[candidate]):
                    break
                candidate -= 1
            output.append(path[candidate])
            anchor = candidate
        return output

    def chaikin(self, path: Sequence[Pose2D], iterations: int = 2) -> list[Pose2D]:
        points = list(path)
        for _ in range(max(0, iterations)):
            if len(points) < 3:
                break
            refined = [points[0]]
            for first, second in zip(points, points[1:]):
                q = Pose2D(0.75 * first.x + 0.25 * second.x, 0.75 * first.y + 0.25 * second.y)
                r = Pose2D(0.25 * first.x + 0.75 * second.x, 0.25 * first.y + 0.75 * second.y)
                # 新点必须位于自由空间；否则保留原线段端点。
                if not self.costmap.is_occupied_world(q.x, q.y):
                    refined.append(q)
                if not self.costmap.is_occupied_world(r.x, r.y):
                    refined.append(r)
            refined.append(points[-1])
            points = refined
        return points

    @staticmethod
    def resample(path: Sequence[Pose2D], spacing: float) -> list[Pose2D]:
        if len(path) < 2:
            return list(path)
        output = [Pose2D(path[0].x, path[0].y, path[0].yaw)]
        carry = 0.0
        previous = path[0]
        for target in path[1:]:
            segment = distance(previous, target)
            while carry + segment >= spacing and segment > 1e-12:
                ratio = (spacing - carry) / segment
                previous = Pose2D(
                    previous.x + ratio * (target.x - previous.x),
                    previous.y + ratio * (target.y - previous.y),
                )
                output.append(previous)
                segment = distance(previous, target)
                carry = 0.0
            carry += segment
            previous = target
        if distance(output[-1], path[-1]) > 1e-6:
            output.append(Pose2D(path[-1].x, path[-1].y, path[-1].yaw))
        return output

    @staticmethod
    def assign_yaws(path: Sequence[Pose2D], final_yaw: float | None = None) -> list[Pose2D]:
        if not path:
            return []
        output: list[Pose2D] = []
        for index, pose in enumerate(path):
            if index < len(path) - 1:
                following = path[index + 1]
                yaw = math.atan2(following.y - pose.y, following.x - pose.x)
            elif final_yaw is not None:
                yaw = normalize_angle(final_yaw)
            elif output:
                yaw = output[-1].yaw
            else:
                yaw = pose.yaw
            output.append(Pose2D(pose.x, pose.y, yaw))
        return output
