"""RGB-D 反投影、桌面平面与物体几何姿态；不依赖 Isaac Sim/深度学习。"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


def transform_points(T, points):
    return np.asarray(points) @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def validate_transform(T):
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("位姿应为有限 4x4 矩阵")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5) or not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-4) or np.linalg.det(T[:3,:3]) < .999:
        raise ValueError("位姿旋转必须属于 SO(3)，最后一行必须为 [0,0,0,1]")
    return T


def unproject(depth, K, T_world_camera, mask=None, stride=1):
    """depth 为光轴深度 Z（米），不是 distance_to_camera 的欧氏距离。"""
    depth, K = np.asarray(depth), np.asarray(K, float)
    validate_transform(T_world_camera)
    if depth.ndim != 2 or K.shape != (3, 3) or not np.isfinite(K).all() or K[0,0] <= 0 or K[1,1] <= 0:
        raise ValueError("非法深度图或相机内参")
    if stride < 1 or (mask is not None and np.shape(mask) != depth.shape):
        raise ValueError("非法采样步长或掩码尺寸")
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[v, u]
    valid = np.isfinite(z) & (z > .05) & (z < 4.)
    if mask is not None:
        valid &= np.asarray(mask, bool)[v, u]
    pixels = np.column_stack([u[valid], v[valid], np.ones(valid.sum())])
    rays = pixels @ np.linalg.inv(K).T
    return transform_points(T_world_camera, rays * z[valid, None])


def voxel_downsample(points, size=.003):
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if not len(p):
        return p
    if size <= 0:
        raise ValueError("体素尺寸必须为正")
    _, inverse = np.unique(np.floor(p / size).astype(np.int64), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    return np.column_stack([np.bincount(inverse, weights=p[:, i]) / count for i in range(3)])


def clean_cloud(points, voxel=.003):
    p = voxel_downsample(points, voxel)
    if len(p) < 12:
        return p
    distance, _ = cKDTree(p).query(p, k=min(12, len(p)))
    mean = distance[:, 1:].mean(axis=1)
    # MAD 避免少数飞点把均值/方差拉大。
    median = np.median(mean)
    return p[mean <= median + max(3 * 1.4826 * np.median(abs(mean - median)), voxel * 1.5)]


def fit_table(points, expected_z, seed=9):
    """RANSAC + SVD，法向朝上；已知桌高仅作 ROI 先验，最终高度来自深度。"""
    p = np.asarray(points)
    p = p[abs(p[:,2] - expected_z) < .035]
    if len(p) < 80:
        raise ValueError("桌面有效深度不足，拒绝猜测桌高")
    rng = np.random.default_rng(seed)
    sample = p[rng.choice(len(p), min(5000, len(p)), replace=False)]
    best = np.zeros(len(sample), bool)
    for _ in range(120):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b-a, c-a)
        if np.linalg.norm(n) < 1e-8:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < .98:
            continue
        keep = abs((sample-a) @ n) < .004
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < 80 or best.mean() < .35:
        raise ValueError("未找到可靠桌面平面")
    q = sample[best]
    _, _, v = np.linalg.svd(q-q.mean(0), full_matrices=False)
    n = v[-1] * (1 if v[-1,2] > 0 else -1)
    if n[2] < .98:
        raise ValueError("本入门示例要求近水平桌面")
    return np.r_[n, -n @ q.mean(0)]


def table_height(plane, xy):
    return -(np.asarray(xy) @ plane[:2] + plane[3]) / plane[2]

@dataclass
class ObjectPose:
    center: np.ndarray
    rotation: np.ndarray
    extents: np.ndarray
    yaw_observable: bool
    height: float


def estimate_pose(points, plane):
    """2.5D OBB：仅估计桌上直立物体的中心/主轴；不是 CAD 6D 配准。

    单视角不可见背面不凭空补全；高度用桌面支撑先验。
    球/圆柱及接近正方形的物体偏航不可观测，输出标记而不是虚假精度。
    """
    p = np.asarray(points, float)
    if len(p) < 12:
        raise ValueError("物体点云不足")
    xy = p[:,:2]
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    major = vecs[:, -1]
    if major[0] < 0:
        major = -major
    R = np.eye(3)
    R[:2,0], R[:2,1] = major, [-major[1], major[0]]
    local = p @ R
    lo, hi = np.percentile(local, [1,99], axis=0)
    center = R @ ((lo+hi)/2)
    bottom = table_height(plane, center[:2])
    top = float(np.percentile(p[:,2], 97))
    height = top-bottom
    if not .012 < height < .16:
        raise ValueError(f"物体高度异常：{height:.3f}m")
    center[2] = bottom + height/2
    extents = hi-lo
    extents[2] = height
    return ObjectPose(center, R, extents, bool(vals[-1] / max(vals[0], 1e-10) > 1.35), height)


def write_ply(path, points):
    p = np.asarray(points).reshape(-1, 3)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(p)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, p, fmt='%.6f')
