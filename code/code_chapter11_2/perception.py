"""Isaac 端 G2 头部深度感知。

红色物块位姿由配置直接给出；右夹爪相机不启动，不发布目标，也不做视觉伺服。
头部深度点云负责全局 OctoMap，并过滤机器人自身和已知抓取目标。
"""
from __future__ import annotations

import numpy as np

try:
    from .config import (
        HEAD_DEPTH_CAMERA_PRIM_PATH,
        RED_OBJECT_POSITION,
        RED_OBJECT_SIZE,
        RIGHT_ARM_JOINT_NAMES,
        PerceptionConfig,
    )
    from .rgbd_geometry import depth_pixels_to_world
    from .robot_self_filter import remove_points_inside_obbs, remove_right_arm_points
except ImportError:
    from config import (
        HEAD_DEPTH_CAMERA_PRIM_PATH,
        RED_OBJECT_POSITION,
        RED_OBJECT_SIZE,
        RIGHT_ARM_JOINT_NAMES,
        PerceptionConfig,
    )
    from rgbd_geometry import depth_pixels_to_world
    from robot_self_filter import remove_points_inside_obbs, remove_right_arm_points


def remove_known_target_points(
    points: np.ndarray, center=RED_OBJECT_POSITION, size=RED_OBJECT_SIZE
) -> np.ndarray:
    """按已知物块真实 AABB 从 OctoMap 输入中删除目标点，不增加额外边界。"""
    points = np.asarray(points)
    center = np.asarray(center, dtype=np.float64)
    half_size = 0.5 * np.asarray(size, dtype=np.float64)
    inside = np.all((points >= center - half_size) & (points <= center + half_size), axis=1)
    return points[~inside]


class G2HeadDepthPerception:
    """只启动头部深度相机，生成 MoveIt 2 的全局障碍点云。"""

    def __init__(self, simulation, config: PerceptionConfig | None = None):
        from isaacsim.sensors.camera import Camera

        self.simulation = simulation
        self.config = config or PerceptionConfig()
        names = list(simulation.robot.dof_names)
        self._arm_indices = np.asarray(
            [names.index(name) for name in RIGHT_ARM_JOINT_NAMES], dtype=np.int64
        )
        self.head = Camera(
            HEAD_DEPTH_CAMERA_PRIM_PATH,
            "chapter11_moveit_head_depth",
            frequency=self.config.camera_hz,
            resolution=self.config.head_resolution,
        )
        self.head.initialize()
        self.head.add_distance_to_image_plane_to_frame()
        for _ in range(20):
            simulation.step(render=True)

        # 读取 G2 USD 中右臂和右夹爪的真实可视边界，用于机器人自身过滤。
        from isaacsim.core.utils import bounds as bounds_utils

        self._bounds_utils = bounds_utils
        self._bbox_cache = bounds_utils.create_bbox_cache()
        self._right_robot_prim_paths = tuple(
            f"/genie/{name}"
            for name in (
                *(f"arm_r_link{i}" for i in range(1, 8)),
                "arm_r_end_link",
                "gripper_r_base_link",
                "gripper_r_center_link",
                *(f"gripper_r_inner_link{i}" for i in range(1, 5)),
                *(f"gripper_r_outer_link{i}" for i in range(1, 5)),
            )
        )
        print("[感知] 头部深度 OctoMap 数据源已启动；红色物块使用已知位姿", flush=True)

    def _world_to_arm(self, points):
        position, quaternion = self.simulation.arm_base_pose()
        rotation = self.simulation._quat_matrix(quaternion)
        return (np.asarray(points) - np.asarray(position)) @ rotation

    def _selected_points_arm(self, depth, pixel_mask):
        points_world = depth_pixels_to_world(self.head, depth, pixel_mask)
        if not len(points_world):
            return np.empty((0, 3), dtype=np.float64)
        return self._world_to_arm(points_world)

    def _right_robot_obbs_arm(self):
        """读取当前帧实际机器人几何的 OBB，并转换到 arm_base_link。"""
        self._bbox_cache.Clear()
        _, base_quaternion = self.simulation.arm_base_pose()
        world_to_arm_rotation = self.simulation._quat_matrix(base_quaternion)
        obbs = []
        for prim_path in self._right_robot_prim_paths:
            center_world, axes_world, half_extent = self._bounds_utils.compute_obb(
                self._bbox_cache, prim_path
            )
            axes_arm = np.asarray(axes_world, dtype=np.float64) @ world_to_arm_rotation
            scales = np.linalg.norm(axes_arm, axis=1)
            if np.any(scales < 1e-9) or np.any(np.asarray(half_extent) <= 0.0):
                continue
            unit_axes = axes_arm / scales[:, None]
            center_arm = self._world_to_arm(np.asarray([center_world]))[0]
            obbs.append((center_arm, unit_axes, np.asarray(half_extent) * scales))
        return obbs

    def head_obstacle_points(self):
        # 运动时相机帧与关节反馈可能相差一帧。暂停该帧更新，避免夹爪残影。
        velocities = np.asarray(self.simulation.robot.get_joint_velocities(), dtype=np.float64)
        if np.max(np.abs(velocities[self._arm_indices])) > self.config.map_arm_stationary_velocity:
            return np.empty((0, 3), dtype=np.float32)

        depth = self.head.get_depth()
        if depth is None:
            return np.empty((0, 3), dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim != 2:
            return np.empty((0, 3), dtype=np.float32)

        pixel_mask = np.isfinite(depth) & (depth > 0.0)
        points = self._selected_points_arm(depth, pixel_mask)
        if not len(points):
            return np.empty((0, 3), dtype=np.float32)

        lo, hi = np.asarray(self.config.workspace_min), np.asarray(self.config.workspace_max)
        valid = np.all(np.isfinite(points), axis=1)
        valid &= np.all(points >= lo, axis=1) & np.all(points <= hi, axis=1)
        points = remove_known_target_points(points[valid])

        # 固定躯干、头部和左臂靠近本体；桌面障碍均在 x > 0.40 m。
        static_robot = (
            (points[:, 0] < 0.24)
            & (points[:, 1] < 0.55)
            & (np.abs(points[:, 2]) < 0.38)
        )
        points = points[~static_robot]

        # 过滤头部相机看到的右臂、夹爪和手指；这不是环境障碍物膨胀。
        points = remove_points_inside_obbs(
            points,
            self._right_robot_obbs_arm(),
            self.config.robot_self_filter_tolerance,
        )
        all_positions = np.asarray(
            self.simulation.robot.get_joint_positions(), dtype=np.float64
        )
        arm_positions = all_positions[self._arm_indices]
        points = remove_right_arm_points(
            points, arm_positions, self.config.robot_self_filter_tolerance
        )
        if not len(points):
            return np.empty((0, 3), dtype=np.float32)

        # 只做 OctoMap 分辨率对应的体素采样，不额外膨胀障碍物。
        voxel = self.config.map_voxel_size
        keys = np.unique(np.floor(points / voxel).astype(np.int32), axis=0)
        if len(keys) > self.config.map_max_points:
            keys = keys[:: int(np.ceil(len(keys) / self.config.map_max_points))]
        return ((keys.astype(np.float32) + 0.5) * voxel).astype(np.float32)


# 保留旧类名，已有脚本仍可导入；当前实现不会启动右夹爪相机。
G2MoveItPerception = G2HeadDepthPerception
