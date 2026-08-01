"""二维占据栅格、障碍膨胀、碰撞检测和激光射线模拟。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np

try:
    from .geometry import CircleObstacle, Pose2D
except ImportError:
    from geometry import CircleObstacle, Pose2D


@dataclass
class OccupancyGrid2D:
    """ROS 地图约定的二维栅格：0 空闲，100 占用，-1 未知。"""

    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.int8)
        if self.data.ndim != 2:
            raise ValueError("data 必须是二维数组")
        if self.resolution <= 0.0:
            raise ValueError("resolution 必须大于 0")

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def max_x(self) -> float:
        return self.origin_x + self.width * self.resolution

    @property
    def max_y(self) -> float:
        return self.origin_y + self.height * self.resolution

    def copy(self) -> "OccupancyGrid2D":
        return OccupancyGrid2D(self.data.copy(), self.resolution, self.origin_x, self.origin_y)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        column = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return row, column

    def grid_to_world(self, row: int, column: int) -> tuple[float, float]:
        return (
            self.origin_x + (column + 0.5) * self.resolution,
            self.origin_y + (row + 0.5) * self.resolution,
        )

    def contains_cell(self, row: int, column: int) -> bool:
        return 0 <= row < self.height and 0 <= column < self.width

    def is_occupied_cell(self, row: int, column: int, unknown_is_occupied: bool = True) -> bool:
        if not self.contains_cell(row, column):
            return True
        value = int(self.data[row, column])
        return value >= 50 or (unknown_is_occupied and value < 0)

    def is_occupied_world(self, x: float, y: float, unknown_is_occupied: bool = True) -> bool:
        return self.is_occupied_cell(*self.world_to_grid(x, y), unknown_is_occupied)

    def set_rectangle(self, center_x: float, center_y: float, size_x: float, size_y: float) -> None:
        row0, col0 = self.world_to_grid(center_x - size_x / 2.0, center_y - size_y / 2.0)
        row1, col1 = self.world_to_grid(center_x + size_x / 2.0, center_y + size_y / 2.0)
        row0, row1 = sorted((max(0, row0), min(self.height - 1, row1)))
        col0, col1 = sorted((max(0, col0), min(self.width - 1, col1)))
        self.data[row0 : row1 + 1, col0 : col1 + 1] = 100

    def add_circles(self, obstacles: Iterable[CircleObstacle], padding: float = 0.0) -> None:
        for obstacle in obstacles:
            radius = obstacle.radius + padding
            center_row, center_column = self.world_to_grid(obstacle.x, obstacle.y)
            cells = int(math.ceil(radius / self.resolution))
            for row in range(center_row - cells, center_row + cells + 1):
                for column in range(center_column - cells, center_column + cells + 1):
                    if not self.contains_cell(row, column):
                        continue
                    x, y = self.grid_to_world(row, column)
                    if math.hypot(x - obstacle.x, y - obstacle.y) <= radius:
                        self.data[row, column] = 100

    def inflated(self, radius: float) -> "OccupancyGrid2D":
        """按机器人外接圆膨胀障碍物，配置空间中机器人可视为一个点。"""
        if radius <= 0.0:
            return self.copy()
        occupied = self.data >= 50
        result = self.copy()
        cells = int(math.ceil(radius / self.resolution))
        offsets = [
            (dr, dc)
            for dr in range(-cells, cells + 1)
            for dc in range(-cells, cells + 1)
            if math.hypot(dr, dc) * self.resolution <= radius
        ]
        rows, columns = np.nonzero(occupied)
        for dr, dc in offsets:
            shifted_rows = rows + dr
            shifted_columns = columns + dc
            valid = (
                (shifted_rows >= 0)
                & (shifted_rows < self.height)
                & (shifted_columns >= 0)
                & (shifted_columns < self.width)
            )
            result.data[shifted_rows[valid], shifted_columns[valid]] = 100
        return result

    def line_is_free(self, start: Pose2D, end: Pose2D, step: float | None = None) -> bool:
        length = math.hypot(end.x - start.x, end.y - start.y)
        sample_step = step or self.resolution * 0.5
        samples = max(1, int(math.ceil(length / sample_step)))
        for index in range(samples + 1):
            ratio = index / samples
            x = start.x + ratio * (end.x - start.x)
            y = start.y + ratio * (end.y - start.y)
            if self.is_occupied_world(x, y):
                return False
        return True

    def path_is_free(self, path: Sequence[Pose2D]) -> bool:
        return all(self.line_is_free(a, b) for a, b in zip(path, path[1:]))

    def ray_cast(
        self,
        pose: Pose2D,
        angles: Sequence[float],
        range_min: float,
        range_max: float,
    ) -> list[float]:
        """在地图中做二维射线步进，生成教学用 LaserScan 距离。"""
        ranges: list[float] = []
        step = max(self.resolution * 0.5, 0.02)
        for relative_angle in angles:
            angle = pose.yaw + relative_angle
            cosine, sine = math.cos(angle), math.sin(angle)
            measured = range_max
            distance = range_min
            while distance <= range_max:
                x = pose.x + distance * cosine
                y = pose.y + distance * sine
                if self.is_occupied_world(x, y):
                    measured = distance
                    break
                distance += step
            ranges.append(measured)
        return ranges

    def save(self, yaml_path: str | Path) -> tuple[Path, Path]:
        yaml_path = Path(yaml_path)
        pgm_path = yaml_path.with_suffix(".pgm")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        # PGM 原点在左上，ROS OccupancyGrid 原点在左下，因此垂直翻转。
        image = np.where(self.data < 0, 205, np.where(self.data >= 50, 0, 254)).astype(np.uint8)
        with pgm_path.open("wb") as stream:
            stream.write(f"P5\n# Chapter 10 navigation map\n{self.width} {self.height}\n255\n".encode("ascii"))
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

    @classmethod
    def load(cls, yaml_path: str | Path) -> "OccupancyGrid2D":
        yaml_path = Path(yaml_path)
        text = yaml_path.read_text(encoding="utf-8")
        image_name = _yaml_value(text, "image")
        resolution = float(_yaml_value(text, "resolution"))
        origin_values = re.findall(r"[-+]?\d*\.?\d+", _yaml_value(text, "origin"))
        if len(origin_values) < 2:
            raise ValueError(f"地图 origin 格式错误：{yaml_path}")
        image = _read_pgm(yaml_path.parent / image_name)
        image = np.flipud(image)
        data = np.full(image.shape, -1, dtype=np.int8)
        data[image >= 250] = 0
        data[image <= 65] = 100
        return cls(data, resolution, float(origin_values[0]), float(origin_values[1]))


def _yaml_value(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"地图 YAML 缺少 {key}")
    return match.group(1).strip().strip("'\"")


def _read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        magic = stream.readline().strip()
        if magic not in (b"P5", b"P2"):
            raise ValueError(f"仅支持 P5/P2 PGM：{path}")
        tokens: list[bytes] = []
        while len(tokens) < 3:
            line = stream.readline()
            if not line:
                break
            line = line.split(b"#", 1)[0]
            tokens.extend(line.split())
        width, height, maximum = map(int, tokens[:3])
        if magic == b"P5":
            dtype = np.uint8 if maximum < 256 else ">u2"
            image = np.frombuffer(stream.read(), dtype=dtype, count=width * height)
        else:
            image = np.fromstring(stream.read().decode("ascii"), sep=" ", dtype=np.uint16)
        return image.reshape(height, width).astype(np.uint8)
