"""增量式三维体素点云地图，保存标准 XYZI PLY。"""

from pathlib import Path

import numpy as np

try:
    from .config import Map3DConfig
    from .geometry import Pose2D
    from .pointcloud import PointCloud
except ImportError:
    from config import Map3DConfig
    from geometry import Pose2D
    from pointcloud import PointCloud


class VoxelMap3D:
    """使用体素哈希表保存均值点。
    三个思想：先滤波、增量插入、只维护有信息的空间单元。
    """

    def __init__(self, config: Map3DConfig = Map3DConfig()) -> None:
        self.config = config
        self._sum_by_voxel: dict[tuple[int, int, int], np.ndarray] = {}
        self._count_by_voxel: dict[tuple[int, int, int], int] = {}
        self.frames_integrated = 0
        self.points_integrated = 0

    def clear(self) -> None:
        self._sum_by_voxel.clear()
        self._count_by_voxel.clear()
        self.frames_integrated = 0
        self.points_integrated = 0

    def integrate(self, cloud: PointCloud, pose: Pose2D) -> int:
        if len(cloud) == 0:
            return 0
        world_xyz = pose.transform_points(cloud.xyz)
        intensity = cloud.intensity.astype(np.float64, copy=False)
        valid = np.isfinite(world_xyz).all(axis=1) & np.isfinite(intensity)
        valid &= (world_xyz[:, 2] >= self.config.min_z) & (world_xyz[:, 2] <= self.config.max_z)
        if self.config.ground_margin > 0.0:
            valid &= np.abs(world_xyz[:, 2] - self.config.ground_z) > self.config.ground_margin
        world_xyz, intensity = world_xyz[valid], intensity[valid]
        if world_xyz.size == 0:
            return 0

        voxel_ids = np.floor(world_xyz / self.config.voxel_size).astype(np.int64)
        unique_ids, inverse = np.unique(voxel_ids, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.column_stack(
            [np.bincount(inverse, weights=world_xyz[:, axis]) for axis in range(3)]
            + [np.bincount(inverse, weights=intensity)]
        )

        changed = 0
        for voxel, value_sum, count in zip(unique_ids, sums, counts):
            key = tuple(int(v) for v in voxel)
            if key not in self._sum_by_voxel and len(self._sum_by_voxel) >= self.config.max_voxels:
                continue
            if key in self._sum_by_voxel:
                self._sum_by_voxel[key] += value_sum
                self._count_by_voxel[key] += int(count)
            else:
                self._sum_by_voxel[key] = value_sum.astype(np.float64)
                self._count_by_voxel[key] = int(count)
            changed += 1

        self.frames_integrated += 1
        self.points_integrated += int(len(world_xyz))
        return changed

    def points(self, min_observations: int | None = None) -> np.ndarray:
        threshold = self.config.min_observations if min_observations is None else int(min_observations)
        rows = []
        for key, value_sum in self._sum_by_voxel.items():
            count = self._count_by_voxel[key]
            if count >= threshold:
                rows.append(value_sum / float(count))
        if not rows:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def save_ply(self, path: str | Path, min_observations: int | None = None) -> int:
        points = self.points(min_observations)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii") as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write(f"element vertex {len(points)}\n")
            stream.write("property float x\nproperty float y\nproperty float z\n")
            stream.write("property float intensity\nend_header\n")
            for x, y, z, intensity in points:
                stream.write(f"{x:.5f} {y:.5f} {z:.5f} {intensity:.5f}\n")
        return int(len(points))
