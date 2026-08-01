"""G2 头部深度建图。

红色物块的位姿由场景配置直接给出，不使用右夹爪相机定位，也不做末端视觉伺服。
头部深度点云只负责生成环境障碍体素，障碍尺寸不做人为膨胀。
"""
from __future__ import annotations

import numpy as np

try:
    from .config import (
        HEAD_DEPTH_CAMERA_PRIM_PATH,
        RED_OBJECT_POSITION,
        RED_OBJECT_SIZE,
        BoxObstacle,
        PerceptionConfig,
    )
    from .rgbd_geometry import depth_pixels_to_world
except ImportError:
    from config import (
        HEAD_DEPTH_CAMERA_PRIM_PATH,
        RED_OBJECT_POSITION,
        RED_OBJECT_SIZE,
        BoxObstacle,
        PerceptionConfig,
    )
    from rgbd_geometry import depth_pixels_to_world


def remove_known_target_points(
    points: np.ndarray, center=RED_OBJECT_POSITION, size=RED_OBJECT_SIZE
) -> np.ndarray:
    """按已知物块真实 AABB 从障碍点云中删除目标点，不增加额外边界。"""
    points = np.asarray(points)
    center = np.asarray(center, dtype=np.float64)
    half_size = 0.5 * np.asarray(size, dtype=np.float64)
    inside = np.all((points >= center - half_size) & (points <= center + half_size), axis=1)
    return points[~inside]


class G2HeadDepthPerception:
    """只启动 G2 头部深度相机，用于全局障碍地图。"""

    def __init__(self, simulation, config: PerceptionConfig | None = None) -> None:
        from isaacsim.sensors.camera import Camera

        self.simulation = simulation
        self.config = config or PerceptionConfig()
        self.head = Camera(
            prim_path=HEAD_DEPTH_CAMERA_PRIM_PATH,
            name="chapter11_head_depth",
            frequency=self.config.camera_hz,
            resolution=self.config.head_resolution,
        )
        self.head.initialize()
        self.head.add_distance_to_image_plane_to_frame()
        for _ in range(20):
            simulation.step(render=True)
        print("[感知] G2 头部深度相机已启动；红色物块使用已知位姿", flush=True)

    def _world_to_arm(self, points: np.ndarray) -> np.ndarray:
        base_position, base_quaternion = self.simulation.arm_base_pose()
        rotation = self.simulation._quaternion_to_matrix(base_quaternion)
        return (np.asarray(points) - np.asarray(base_position)) @ rotation

    def _selected_points_arm(self, depth: np.ndarray, pixel_mask: np.ndarray) -> np.ndarray:
        points_world = depth_pixels_to_world(self.head, depth, pixel_mask)
        if not len(points_world):
            return np.empty((0, 3), dtype=np.float64)
        return self._world_to_arm(points_world)

    def capture_head_obstacles(self) -> tuple[np.ndarray, list[BoxObstacle]]:
        """返回头部相机体素中心和规划器可直接使用的未膨胀 AABB。"""
        depth = self.head.get_depth()
        if depth is None:
            raise RuntimeError("头部深度相机没有返回深度帧")
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim != 2:
            raise RuntimeError("头部深度图尺寸不正确")

        pixel_mask = np.isfinite(depth) & (depth > 0.0)
        points = self._selected_points_arm(depth, pixel_mask)
        if not len(points):
            raise RuntimeError("头部深度相机没有返回有效点云")

        lo, hi = np.asarray(self.config.workspace_min), np.asarray(self.config.workspace_max)
        valid = np.all(np.isfinite(points), axis=1)
        valid &= np.all(points >= lo, axis=1) & np.all(points <= hi, axis=1)
        # 去除机器人胸腔附近点；机器人自身碰撞由 ROBOT_BODY_BOX/连杆模型处理。
        valid &= ~(
            (points[:, 0] < 0.22)
            & (points[:, 1] < 0.52)
            & (np.abs(points[:, 2]) < 0.34)
        )
        points = remove_known_target_points(points[valid])
        if not len(points):
            raise RuntimeError("头部点云经过工作区裁剪后为空")

        voxel = self.config.map_voxel_size
        keys = np.unique(np.floor(points / voxel).astype(np.int32), axis=0)
        if len(keys) > self.config.map_max_points:
            stride = int(np.ceil(len(keys) / self.config.map_max_points))
            keys = keys[::stride]
        centers = (keys.astype(np.float64) + 0.5) * voxel
        boxes = [
            BoxObstacle(
                f"depth_voxel_{i}",
                tuple(center),
                (voxel, voxel, voxel),
                (0.35, 0.70, 0.95),
            )
            for i, center in enumerate(centers)
        ]
        print(
            f"[头部深度] 工作区点云生成 {len(boxes)} 个 {voxel:.2f} m 未膨胀体素",
            flush=True,
        )
        return centers, boxes


# 保留旧类名，避免已有教学调用立即失效；当前实现不会启动右夹爪相机。
G2VisualPerception = G2HeadDepthPerception
