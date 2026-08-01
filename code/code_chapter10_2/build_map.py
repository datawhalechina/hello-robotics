"""根据第十章场景参数生成 ROS/Nav2 通用 PGM + YAML 地图。"""

import math

try:
    from .config import MAP_YAML, MapConfig, STATIC_OBSTACLES
    from .map_utils import OccupancyGrid2D
except ImportError:
    from config import MAP_YAML, MapConfig, STATIC_OBSTACLES
    from map_utils import OccupancyGrid2D

import numpy as np


def build_navigation_map(config: MapConfig | None = None) -> OccupancyGrid2D:
    config = config or MapConfig()
    width = int(math.ceil((config.max_x - config.min_x) / config.resolution))
    height = int(math.ceil((config.max_y - config.min_y) / config.resolution))
    grid = OccupancyGrid2D(
        np.zeros((height, width), dtype=np.int8),
        config.resolution,
        config.min_x,
        config.min_y,
    )
    thickness = max(1, int(math.ceil(config.wall_thickness / config.resolution)))
    grid.data[:thickness, :] = 100
    grid.data[-thickness:, :] = 100
    grid.data[:, :thickness] = 100
    grid.data[:, -thickness:] = 100
    for obstacle in STATIC_OBSTACLES:
        grid.set_rectangle(
            obstacle.center_x,
            obstacle.center_y,
            obstacle.size_x,
            obstacle.size_y,
        )
    return grid


def main() -> None:
    grid = build_navigation_map()
    yaml_path, pgm_path = grid.save(MAP_YAML)
    print(f"地图已生成：{yaml_path}")
    print(f"地图图像：{pgm_path}")
    print(f"尺寸：{grid.width} x {grid.height}，分辨率：{grid.resolution:.3f} m")


if __name__ == "__main__":
    main()
