"""Nav2 bridge 所需的最小二维地图工具。

本文件只生成静态地图和模拟 LaserScan，不包含任何规划或轨迹跟踪算法。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from .config import CircleObstacle, Pose2D
except ImportError:
    from config import CircleObstacle, Pose2D


class OccupancyGrid2D:
    """简化占据栅格：0 表示空闲，100 表示占用。"""

    def __init__(self, data: np.ndarray, resolution: float, origin_x: float, origin_y: float):
        self.data = np.asarray(data, dtype=np.int8)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def copy(self) -> "OccupancyGrid2D":
        return OccupancyGrid2D(self.data.copy(), self.resolution, self.origin_x, self.origin_y)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        column = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return row, column

    def is_occupied_world(self, x: float, y: float) -> bool:
        row, column = self.world_to_grid(x, y)
        if not (0 <= row < self.height and 0 <= column < self.width):
            return True
        return bool(self.data[row, column] >= 50)

    def set_rectangle(self, center_x: float, center_y: float, size_x: float, size_y: float) -> None:
        row0, col0 = self.world_to_grid(center_x - size_x / 2, center_y - size_y / 2)
        row1, col1 = self.world_to_grid(center_x + size_x / 2, center_y + size_y / 2)
        row0, row1 = sorted((max(0, row0), min(self.height - 1, row1)))
        col0, col1 = sorted((max(0, col0), min(self.width - 1, col1)))
        self.data[row0 : row1 + 1, col0 : col1 + 1] = 100

    def add_circles(self, obstacles: Iterable[CircleObstacle]) -> None:
        """把移动障碍临时加入传感器地图，但不写入 Nav2 静态地图。"""
        for obstacle in obstacles:
            cells = int(math.ceil(obstacle.radius / self.resolution))
            center_row, center_col = self.world_to_grid(obstacle.x, obstacle.y)
            for row in range(center_row - cells, center_row + cells + 1):
                for col in range(center_col - cells, center_col + cells + 1):
                    if not (0 <= row < self.height and 0 <= col < self.width):
                        continue
                    x = self.origin_x + (col + 0.5) * self.resolution
                    y = self.origin_y + (row + 0.5) * self.resolution
                    if math.hypot(x - obstacle.x, y - obstacle.y) <= obstacle.radius:
                        self.data[row, col] = 100

    def ray_cast(
        self,
        pose: Pose2D,
        angles: Sequence[float],
        range_min: float,
        range_max: float,
    ) -> list[float]:
        """二维射线步进，生成 sensor_msgs/LaserScan 的 ranges。"""
        output: list[float] = []
        step = max(self.resolution * 0.5, 0.02)
        for relative_angle in angles:
            angle = pose.yaw + relative_angle
            cosine, sine = math.cos(angle), math.sin(angle)
            measured = range_max
            distance = range_min
            while distance <= range_max:
                if self.is_occupied_world(
                    pose.x + distance * cosine, pose.y + distance * sine
                ):
                    measured = distance
                    break
                distance += step
            output.append(measured)
        return output

    def save(self, yaml_path: str | Path) -> tuple[Path, Path]:
        """保存为 nav2_map_server 使用的 PGM + YAML。"""
        yaml_path = Path(yaml_path)
        pgm_path = yaml_path.with_suffix(".pgm")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.where(self.data >= 50, 0, 254).astype(np.uint8)
        with pgm_path.open("wb") as stream:
            stream.write(
                f"P5\n# Chapter 10-2 Nav2 map\n{self.width} {self.height}\n255\n".encode("ascii")
            )
            stream.write(np.flipud(image).tobytes())
        yaml_path.write_text(
            "\n".join(
                [
                    f"image: {pgm_path.name}",
                    "mode: trinary",
                    f"resolution: {self.resolution:.6f}",
                    f"origin: [{self.origin_x:.6f}, {self.origin_y:.6f}, 0.000000]",
                    "negate: 0",
                    "occupied_thresh: 0.650000",
                    "free_thresh: 0.250000",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return yaml_path, pgm_path
