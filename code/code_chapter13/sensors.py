"""G2 头部 RGB-D 相机与双 OS1 RTX LiDAR。"""

from __future__ import annotations

import numpy as np

try:
    from .config import HEAD_CAMERA_PRIM_PATH, SensorConfig
    from .lidar import DualRtxLidar, LidarConfig
except ImportError:
    from config import HEAD_CAMERA_PRIM_PATH, SensorConfig
    from lidar import DualRtxLidar, LidarConfig


def rgba_to_bgr(image: np.ndarray) -> np.ndarray:
    """Isaac RGB/RGBA 图像转 OpenCV BGR。"""
    import cv2

    array = np.asarray(image)[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(np.nan_to_num(array) * scale, 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_RGB2BGR)


def pointcloud_to_laserscan(
    xyz: np.ndarray,
    bins: int = 360,
    min_range: float = 0.38,
    max_range: float = 8.0,
    min_z: float = 0.08,
    max_z: float = 1.80,
) -> np.ndarray:
    """把 base_link 三维点云压成 Nav2 使用的二维 LaserScan。"""
    ranges = np.full(int(bins), float(max_range), dtype=np.float32)
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        return ranges

    distance = np.hypot(points[:, 0], points[:, 1])
    valid = np.isfinite(points[:, :3]).all(axis=1)
    valid &= (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    valid &= (distance >= min_range) & (distance <= max_range)
    if not np.any(valid):
        return ranges

    angle = np.arctan2(points[valid, 1], points[valid, 0])
    distance = distance[valid]
    indices = np.floor((angle + np.pi) / (2.0 * np.pi) * bins).astype(np.int32)
    indices = np.clip(indices, 0, bins - 1)
    np.minimum.at(ranges, indices, distance)
    return ranges


class G2Sensors:
    """只暴露本章真正需要的 capture_camera() 和 capture_scan()。"""

    def __init__(self, simulation, config: SensorConfig = SensorConfig()) -> None:
        from isaacsim.sensors.camera import Camera

        self.simulation = simulation
        self.config = config
        self.camera = Camera(
            prim_path=HEAD_CAMERA_PRIM_PATH,
            name="chapter13_head_rgbd",
            frequency=config.camera_hz,
            resolution=config.camera_resolution,
        )
        self.camera.initialize()
        self.camera.add_rgb_to_frame()
        self.camera.add_distance_to_image_plane_to_frame()

        lidar_config = LidarConfig(
            min_range=config.lidar_min_range,
            max_range=config.lidar_max_range,
        )
        self.lidar = DualRtxLidar(lidar_config)
        for _ in range(30):
            simulation.step()
        print("[传感器] G2 头部 RGB-D 相机与双 OS1 LiDAR 已启动", flush=True)

    def capture_camera(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        rgba = self.camera.get_rgba()
        depth = self.camera.get_depth()
        if rgba is None or depth is None:
            return None, None
        image = np.asarray(rgba)
        depth = np.asarray(depth, dtype=np.float32)
        if image.size == 0 or depth.ndim != 2:
            return None, None
        return rgba_to_bgr(image), depth

    def capture_scan(self) -> np.ndarray:
        return pointcloud_to_laserscan(
            self.lidar.capture_xyz(),
            bins=self.config.lidar_bins,
            min_range=self.config.lidar_min_range,
            max_range=self.config.lidar_max_range,
            min_z=self.config.lidar_min_z,
            max_z=self.config.lidar_max_z,
        )
