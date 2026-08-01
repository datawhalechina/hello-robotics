"""从头部深度点云中剔除 G2 自身右臂。

这一步是点云的机器人自滤波，不是给环境障碍物做安全膨胀：只删除落在当前
机器人可见几何内部的点，桌子和黄色障挡物的尺寸保持不变。
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _transform(xyz, rpy):
    """生成与 G2 右臂 URDF 一致的 4x4 固定变换。"""
    x, y, z = xyz
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = (x, y, z)
    return result


def _axis_rotation_z(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    return result


_PI = math.pi
_JOINT_ORIGINS = (
    _transform((0.0, 0.0, -0.069), (_PI, 0.0, 0.0)),
    _transform((0.0, 0.0, 0.1745), (_PI / 2.0, 0.0, 0.0)),
    _transform((0.0, 0.0, 0.0), (-_PI / 2.0, 0.0, 0.0)),
    _transform((0.018, 0.0, 0.287), (_PI / 2.0, 0.0, 0.0)),
    _transform((-0.018, 0.0, 0.0), (-_PI / 2.0, 0.0, 0.0)),
    _transform((0.0, 0.0, 0.314), (_PI / 2.0, 0.0, 0.0)),
    _transform((0.0, 0.0, 0.0), (_PI / 2.0, 0.0, _PI / 2.0)),
)
_TOOL_TRANSFORM = _transform((0.23645, 0.0, 0.0), (_PI, -_PI / 2.0, 0.0))


def right_arm_link_transforms(joint_positions: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """返回七个关节变换和夹爪中心变换（``arm_base_link``）。"""
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError("joint_positions 必须包含 7 个关节角")
    current = np.eye(4, dtype=np.float64)
    joints = []
    for origin, angle in zip(_JOINT_ORIGINS, q):
        current = current @ origin @ _axis_rotation_z(float(angle))
        joints.append(current.copy())
    return np.asarray(joints), current @ _TOOL_TRANSFORM


def right_arm_link_points(joint_positions: Sequence[float]) -> np.ndarray:
    """返回七个关节中心和夹爪中心，坐标系为 ``arm_base_link``。"""
    joints, gripper = right_arm_link_transforms(joint_positions)
    return np.vstack((joints[:, :3, 3], gripper[:3, 3]))


def _inside_capsule(points, start, end, radius):
    direction = end - start
    length_squared = float(direction @ direction)
    if length_squared < 1e-12:
        closest = np.broadcast_to(start, points.shape)
    else:
        ratio = np.clip(((points - start) @ direction) / length_squared, 0.0, 1.0)
        closest = start + ratio[:, None] * direction
    return np.sum((points - closest) ** 2, axis=1) <= radius * radius


def _inside_oriented_box(points, transform, size, tolerance):
    """检测点是否在机器人自身的有向包围盒内。"""
    center = transform[:3, 3]
    rotation = transform[:3, :3]
    local = (points - center) @ rotation
    half_size = 0.5 * np.asarray(size, dtype=np.float64) + tolerance
    return np.all(np.abs(local) <= half_size, axis=1)


def remove_points_inside_obbs(points, obbs, tolerance=0.008):
    """按机器人 USD 的实际有向包围盒删除自身点云。

    ``obbs`` 每项为 ``(中心, 三个单位轴, 半边长)``。这里只扩大机器人自身
    掩码来容纳毫米级深度噪声，不会扩大环境障碍物或 OctoMap 体素。
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud):
        return cloud.reshape((-1, 3))
    robot = np.zeros(len(cloud), dtype=bool)
    for center, axes, half_extent in obbs:
        center = np.asarray(center, dtype=np.float64)
        axes = np.asarray(axes, dtype=np.float64)
        half_extent = np.asarray(half_extent, dtype=np.float64) + tolerance
        local = (cloud - center) @ axes.T
        robot |= np.all(np.abs(local) <= half_extent, axis=1)
    return cloud[~robot]


def remove_right_arm_points(points, joint_positions, tolerance=0.008):
    """删除右臂可见几何内的点，返回剩余环境点。

    ``tolerance`` 只补偿深度噪声和视觉/碰撞模型的毫米级差异，不改变任何环境
    障碍物的体素尺寸，也不作为规划安全边界使用。
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud):
        return cloud.reshape((-1, 3))

    joint_transforms, gripper_transform = right_arm_link_transforms(joint_positions)
    links = np.vstack((joint_transforms[:, :3, 3], gripper_transform[:3, 3]))
    robot = np.zeros(len(cloud), dtype=bool)

    # G2 URDF 中肩/腕关节球的可见半径为 0.065 m。
    for center in links[:-1]:
        robot |= np.sum((cloud - center) ** 2, axis=1) <= (0.065 + tolerance) ** 2

    # 三段主要臂杆和末端组件。半径来自本章 URDF/可见几何，而非障碍膨胀值。
    capsules = (
        (links[0], links[1], 0.055),
        (links[2], links[3], 0.052),
        (links[4], links[5], 0.046),
        (links[6], links[7], 0.075),
    )
    for start, end, radius in capsules:
        robot |= _inside_capsule(cloud, start, end, radius + tolerance)

    # 深度相机看到的是 0.11 x 0.10 x 0.12 m 的夹爪可视外形。这里按夹爪
    # 当前姿态删除这个有向盒，避免盒角残留为 OctoMap 障碍。它是自滤波，
    # 不是对桌子或阻挡物做膨胀。
    robot |= _inside_oriented_box(
        cloud, gripper_transform, (0.11, 0.10, 0.12), tolerance
    )

    # MoveIt 教学 URDF 中的躯干碰撞盒：中心 (-0.12, 0, 0)，尺寸 0.34x0.52x0.45。
    torso_center = np.array([-0.12, 0.0, 0.0])
    torso_half = np.array([0.17, 0.26, 0.225]) + tolerance
    robot |= np.all(np.abs(cloud - torso_center) <= torso_half, axis=1)
    return cloud[~robot]
