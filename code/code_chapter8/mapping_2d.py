"""三维点云投影、逆传感器模型和二维占据栅格地图。"""

from pathlib import Path
import math

import numpy as np

try:
    from .config import Map2DConfig
    from .geometry import Pose2D
    from .pointcloud import PointCloud
except ImportError:
    from config import Map2DConfig
    from geometry import Pose2D
    from pointcloud import PointCloud


def logit(probability: float) -> float:
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return math.log(probability / (1.0 - probability))


class OccupancyGrid2D:
    """动态边界的 log-odds 占据栅格。"""

    def __init__(self, config: Map2DConfig = Map2DConfig()) -> None:
        self.config = config
        self._log_odds: dict[tuple[int, int], float] = {}
        self._free_delta = logit(config.free_probability)
        self._occupied_delta = logit(config.occupied_probability)
        self.frames_integrated = 0
        self.rays_integrated = 0

    def clear(self) -> None:
        self._log_odds.clear()
        self.frames_integrated = 0
        self.rays_integrated = 0

    def integrate(self, cloud: PointCloud, pose: Pose2D) -> int:
        if len(cloud) == 0:
            return 0
        world_points = pose.transform_points(cloud.xyz)
        world_origins = pose.transform_points(cloud.origin)
        height = (world_points[:, 2] >= self.config.min_hit_z) & (
            world_points[:, 2] <= self.config.max_hit_z
        )
        world_points, world_origins = world_points[height], world_origins[height]
        if world_points.size == 0:
            return 0

        selected = self._select_nearest_angular_rays(world_points, world_origins)
        changed = 0
        for index in selected:
            start = self.world_to_cell(world_origins[index, :2])
            end = self.world_to_cell(world_points[index, :2])
            ray = list(self.bresenham(start, end))
            if not ray:
                continue
            for cell in ray[:-1]:
                changed += self._update(cell, self._free_delta)
            changed += self._update(ray[-1], self._occupied_delta)
            self.rays_integrated += 1
        self.frames_integrated += 1
        return changed

    def _select_nearest_angular_rays(self, points: np.ndarray, origins: np.ndarray) -> np.ndarray:
        """每个雷达、每个方位角只保留最近障碍，减少射线数量和穿墙伪影。"""
        origin_keys = np.round(origins[:, :2] / 0.01).astype(np.int64)
        selected: list[int] = []
        bin_width = math.radians(max(0.1, self.config.angular_resolution_deg))
        for origin_key in np.unique(origin_keys, axis=0):
            group = np.flatnonzero(np.all(origin_keys == origin_key, axis=1))
            delta = points[group, :2] - origins[group, :2]
            ranges = np.linalg.norm(delta, axis=1)
            angles = np.arctan2(delta[:, 1], delta[:, 0])
            bins = np.floor((angles + math.pi) / bin_width).astype(np.int64)
            order = np.lexsort((ranges, bins))
            sorted_bins = bins[order]
            first = np.ones(len(order), dtype=bool)
            first[1:] = sorted_bins[1:] != sorted_bins[:-1]
            selected.extend(group[order[first]].tolist())
        return np.asarray(selected, dtype=np.int64)

    def _update(self, cell: tuple[int, int], delta: float) -> int:
        old = self._log_odds.get(cell, 0.0)
        new = float(np.clip(old + delta, -self.config.max_log_odds, self.config.max_log_odds))
        self._log_odds[cell] = new
        return int(new != old)

    def world_to_cell(self, xy: np.ndarray) -> tuple[int, int]:
        return tuple(np.floor(np.asarray(xy) / self.config.resolution).astype(np.int64))

    @staticmethod
    def bresenham(start: tuple[int, int], end: tuple[int, int]):
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx - dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled > -dy:
                error -= dy
                x0 += sx
            if doubled < dx:
                error += dx
                y0 += sy

    def to_arrays(self, padding_cells: int = 4):
        if not self._log_odds:
            return (
                np.full((1, 1), 205, dtype=np.uint8),
                np.array([-1], dtype=np.int8),
                1,
                1,
                (0.0, 0.0),
            )
        keys = np.asarray(list(self._log_odds), dtype=np.int64)
        min_x, min_y = keys.min(axis=0) - padding_cells
        max_x, max_y = keys.max(axis=0) + padding_cells
        width, height = int(max_x - min_x + 1), int(max_y - min_y + 1)
        image = np.full((height, width), 205, dtype=np.uint8)
        ros_grid = np.full((height, width), -1, dtype=np.int16)
        for (x, y), odds in self._log_odds.items():
            col, row_bottom = x - min_x, y - min_y
            row_image = height - 1 - row_bottom
            probability = 1.0 / (1.0 + math.exp(-odds))
            if probability >= self.config.occupied_threshold:
                image[row_image, col] = 0
                ros_grid[row_bottom, col] = 100
            elif probability <= self.config.free_threshold:
                image[row_image, col] = 254
                ros_grid[row_bottom, col] = 0
            else:
                image[row_image, col] = np.uint8(np.clip(round(254 * (1 - probability)), 1, 253))
                ros_grid[row_bottom, col] = round(100 * probability)
        origin = (float(min_x * self.config.resolution), float(min_y * self.config.resolution))
        return image, ros_grid.astype(np.int8).reshape(-1), width, height, origin

    def save(self, output_prefix: str | Path) -> tuple[Path, Path, Path]:
        image, _, width, height, origin = self.to_arrays()
        prefix = Path(output_prefix)
        pgm_path = prefix.with_suffix(".pgm")
        yaml_path = prefix.with_suffix(".yaml")
        png_path = prefix.with_suffix(".png")
        pgm_path.parent.mkdir(parents=True, exist_ok=True)
        with pgm_path.open("wb") as stream:
            stream.write(f"P5\n# Chapter 8 occupancy grid\n{width} {height}\n255\n".encode("ascii"))
            stream.write(image.tobytes())
        with yaml_path.open("w", encoding="utf-8") as stream:
            stream.write(f"image: {pgm_path.name}\nmode: trinary\n")
            stream.write(f"resolution: {self.config.resolution:.6f}\n")
            stream.write(f"origin: [{origin[0]:.6f}, {origin[1]:.6f}, 0.000000]\n")
            stream.write("negate: 0\n")
            stream.write(f"occupied_thresh: {self.config.occupied_threshold:.6f}\n")
            stream.write(f"free_thresh: {self.config.free_threshold:.6f}\n")
        self._save_png(png_path, image)
        return pgm_path, yaml_path, png_path

    @staticmethod
    def _save_png(path: Path, image: np.ndarray) -> None:
        try:
            from PIL import Image

            Image.fromarray(image).save(path)
        except Exception:
            try:
                import cv2

                cv2.imwrite(str(path), image)
            except Exception:
                pass
