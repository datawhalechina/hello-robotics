"""第十三章自包含的 G2 双 RTX LiDAR 读取与点云预处理。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .config import LIDAR_PARENT_PATH
except ImportError:
    from config import LIDAR_PARENT_PATH


SCAN_BUFFER_ANNOTATOR = "IsaacCreateRTXLidarScanBuffer"


@dataclass(frozen=True)
class LidarConfig:
    model: str = "OS1"
    profile: str = "OS1_REV6_128ch10hz1024res"
    min_range: float = 0.38
    max_range: float = 8.0
    point_stride: int = 2
    scan_voxel_size: float = 0.06


@dataclass(frozen=True)
class LidarMount:
    name: str
    prim_path: str
    translation: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


LIDAR_MOUNTS = (
    LidarMount(
        "chapter13_lidar_left",
        f"{LIDAR_PARENT_PATH}/chapter13_lidar_left",
        (0.17, 0.21, 0.34),
    ),
    LidarMount(
        "chapter13_lidar_right",
        f"{LIDAR_PARENT_PATH}/chapter13_lidar_right",
        (-0.23, -0.20, 0.34),
        (0.0, 0.0, 0.0, 1.0),
    ),
)


def quaternion_matrix(quaternion) -> np.ndarray:
    """标量在前四元数 ``(w, x, y, z)`` 转旋转矩阵。"""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _xyz_array(data) -> np.ndarray:
    if data is None:
        return np.empty((0, 3), dtype=np.float32)
    points = np.squeeze(np.asarray(data))
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.ndim == 1:
        usable = points.size - points.size % 3
        points = points[:usable].reshape((-1, 3))
    elif points.ndim > 2:
        points = points.reshape((-1, points.shape[-1]))
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return points[:, :3].astype(np.float32, copy=False)


def _voxel_sample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0.0:
        return points
    voxel_ids = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(voxel_ids, axis=0, return_index=True)
    return points[np.sort(indices)]


class DualRtxLidar:
    """创建 G2 左右两个 OS1，并返回融合后的 base_link 点云。"""

    def __init__(self, config: LidarConfig = LidarConfig()) -> None:
        from isaacsim.sensors.rtx import LidarRtx

        self.config = config
        self.handles = []
        for mount in LIDAR_MOUNTS:
            sensor = LidarRtx(
                prim_path=mount.prim_path,
                name=mount.name,
                config_file_name=config.model,
                variant=config.profile,
                translation=np.asarray(mount.translation, dtype=np.float64),
                orientation=np.asarray(mount.orientation_wxyz, dtype=np.float64),
            )
            sensor.initialize()
            sensor.attach_annotator(
                SCAN_BUFFER_ANNOTATOR,
                outputIntensity=True,
                enablePerFrameOutput=True,
            )
            self.handles.append((mount, sensor))

    def capture_xyz(self) -> np.ndarray:
        clouds = []
        for mount, sensor in self.handles:
            scan = sensor.get_current_frame().get(SCAN_BUFFER_ANNOTATOR, {})
            data = next(
                (scan[key] for key in ("data", "pointCloudData", "points", "pointcloud") if key in scan),
                None,
            )
            points = _xyz_array(data)
            if len(points) == 0:
                continue
            valid = np.isfinite(points).all(axis=1)
            distance = np.linalg.norm(points, axis=1)
            valid &= (distance >= self.config.min_range) & (distance <= self.config.max_range)
            points = points[valid][:: max(1, self.config.point_stride)]
            points = _voxel_sample(points, self.config.scan_voxel_size)
            rotation = quaternion_matrix(mount.orientation_wxyz)
            translation = np.asarray(mount.translation, dtype=np.float64)
            clouds.append((points.astype(np.float64) @ rotation.T + translation).astype(np.float32))
        return np.concatenate(clouds, axis=0) if clouds else np.empty((0, 3), dtype=np.float32)
