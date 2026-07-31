"""RTX LiDAR 数据解析、双雷达外参变换、滤波与融合。"""

from dataclasses import dataclass

import numpy as np

try:
    from .config import LidarConfig, LidarMount
    from .geometry import quaternion_wxyz_to_matrix
except ImportError:
    from config import LidarConfig, LidarMount
    from geometry import quaternion_wxyz_to_matrix


@dataclass
class PointCloud:
    """统一的 XYZI 点云；xyz 和 origin 都位于 base_link 坐标系。"""

    xyz: np.ndarray
    intensity: np.ndarray
    origin: np.ndarray

    @classmethod
    def empty(cls) -> "PointCloud":
        return cls(
            xyz=np.empty((0, 3), dtype=np.float32),
            intensity=np.empty((0,), dtype=np.float32),
            origin=np.empty((0, 3), dtype=np.float32),
        )

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def xyzi(self) -> np.ndarray:
        if len(self) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.column_stack((self.xyz, self.intensity)).astype(np.float32, copy=False)

    @staticmethod
    def concatenate(clouds: list["PointCloud"]) -> "PointCloud":
        valid = [cloud for cloud in clouds if len(cloud) > 0]
        if not valid:
            return PointCloud.empty()
        return PointCloud(
            xyz=np.concatenate([cloud.xyz for cloud in valid], axis=0),
            intensity=np.concatenate([cloud.intensity for cloud in valid], axis=0),
            origin=np.concatenate([cloud.origin for cloud in valid], axis=0),
        )


class PointCloudProcessor:
    """把不同版本 annotator 的输出整理为干净的 base_link 点云。"""

    POINT_KEYS = ("data", "pointCloudData", "points", "pointcloud")
    INTENSITY_KEYS = ("intensity", "intensities", "intensitiesData")

    def __init__(self, config: LidarConfig = LidarConfig()) -> None:
        self.config = config

    def process(self, scan_data, mount: LidarMount) -> PointCloud:
        points, intensity = self.parse_scan(scan_data)
        if points.size == 0:
            return PointCloud.empty()

        finite = np.isfinite(points).all(axis=1) & np.isfinite(intensity)
        points, intensity = points[finite], intensity[finite]
        distances = np.linalg.norm(points, axis=1)
        effective_max = max(
            self.config.min_range,
            self.config.max_range - self.config.max_range_margin,
        )
        valid = (distances >= self.config.min_range) & (distances <= effective_max)
        points, intensity = points[valid], intensity[valid]
        if points.size == 0:
            return PointCloud.empty()

        stride = max(1, int(self.config.point_stride))
        points, intensity = points[::stride], intensity[::stride]
        selected = voxel_sample_indices(points, self.config.scan_voxel_size)
        points, intensity = points[selected], intensity[selected]

        rotation = quaternion_wxyz_to_matrix(mount.orientation_wxyz)
        translation = np.asarray(mount.translation, dtype=np.float64)
        points_base = points.astype(np.float64) @ rotation.T + translation
        origins = np.repeat(translation.reshape(1, 3), points_base.shape[0], axis=0)
        return PointCloud(
            xyz=points_base.astype(np.float32),
            intensity=normalize_intensity(intensity),
            origin=origins.astype(np.float32),
        )

    @classmethod
    def parse_scan(cls, scan_data) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(scan_data, dict):
            points_data = next((scan_data[key] for key in cls.POINT_KEYS if key in scan_data), None)
            intensity_data = next(
                (scan_data[key] for key in cls.INTENSITY_KEYS if key in scan_data),
                None,
            )
        else:
            points_data = scan_data
            intensity_data = None

        points = as_xyz_array(points_data)
        intensity = as_scalar_array(intensity_data, len(points))
        count = min(len(points), len(intensity))
        return points[:count], intensity[:count]


def as_xyz_array(data) -> np.ndarray:
    if data is None:
        return np.empty((0, 3), dtype=np.float32)
    if isinstance(data, dict):
        data = next((data[key] for key in PointCloudProcessor.POINT_KEYS if key in data), None)
        if data is None:
            return np.empty((0, 3), dtype=np.float32)
    points = np.asarray(data)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    points = np.squeeze(points)
    if points.ndim == 1:
        if points.shape[0] < 3:
            return np.empty((0, 3), dtype=np.float32)
        usable = points.shape[0] - points.shape[0] % 3
        points = points[:usable].reshape((-1, 3))
    elif points.ndim > 2:
        points = points.reshape((-1, points.shape[-1]))
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return points[:, :3].astype(np.float32, copy=False)


def as_scalar_array(data, count: int) -> np.ndarray:
    if data is None:
        return np.zeros((count,), dtype=np.float32)
    if isinstance(data, dict):
        nested_keys = ("data",) + PointCloudProcessor.INTENSITY_KEYS
        data = next((data[key] for key in nested_keys if key in data), None)
        if data is None:
            return np.zeros((count,), dtype=np.float32)
    values = np.asarray(data, dtype=np.float32)
    if values.size == 0:
        return np.zeros((count,), dtype=np.float32)
    values = np.squeeze(values).reshape(-1)
    if len(values) >= count:
        return values[:count]
    padded = np.zeros((count,), dtype=np.float32)
    padded[: len(values)] = values
    return padded


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    """保留真实 intensity；仅处理 NaN/Inf，不按帧拉伸，避免地图闪烁。"""
    return np.nan_to_num(np.asarray(intensity, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def voxel_sample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """每个体素保留最靠近体素中心的点，返回原数组下标。"""
    points = np.asarray(points)
    if points.size == 0:
        return np.empty((0,), dtype=np.int64)
    if voxel_size <= 0.0:
        return np.arange(len(points), dtype=np.int64)
    voxel_ids = np.floor(points / float(voxel_size)).astype(np.int64)
    centers = (voxel_ids.astype(np.float64) + 0.5) * float(voxel_size)
    distances = np.sum((points - centers) ** 2, axis=1)
    order = np.lexsort((distances, voxel_ids[:, 2], voxel_ids[:, 1], voxel_ids[:, 0]))
    sorted_ids = voxel_ids[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_ids[1:] != sorted_ids[:-1], axis=1)
    return order[first]
