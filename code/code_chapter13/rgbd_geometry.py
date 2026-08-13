"""RGB-D 几何小工具：把深度图按同像素顺序反投影到三维。

为什么不直接调用 ``Camera.get_pointcloud()``？
G2 相机使用 OpenCV Pinhole 畸变模型，而 Isaac Sim 5.1 的深度回退接口只接受
纯 ``pinhole`` 字符串，会反复报警甚至抛异常。这里直接读取相机标定参数，既保留
RGB 与深度的一一对应关系，也适合讲解 RGB-D 相机的反投影过程。
"""
from __future__ import annotations

import numpy as np


def _attribute(camera, name: str, default=None):
    """读取 USD Camera 属性；属性不存在时返回默认值。"""
    attribute = camera.prim.GetAttribute(name)
    if not attribute or not attribute.IsValid():
        return default
    value = attribute.Get()
    return default if value is None else value


def _quaternion_matrix(quaternion) -> np.ndarray:
    """标量在前的四元数 ``(w, x, y, z)`` 转旋转矩阵。"""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _scaled_calibration(camera, width: int, height: int):
    """读取 OpenCV 标定参数，并缩放到当前渲染分辨率。"""
    prefix = "omni:lensdistortion:opencvPinhole:"
    source_size = _attribute(camera, prefix + "imageSize")

    if source_size is not None:
        source_width, source_height = map(float, source_size)
        scale_x = width / source_width
        scale_y = height / source_height
        fx = float(_attribute(camera, prefix + "fx")) * scale_x
        fy = float(_attribute(camera, prefix + "fy")) * scale_y
        cx = float(_attribute(camera, prefix + "cx")) * scale_x
        cy = float(_attribute(camera, prefix + "cy")) * scale_y
        distortion = np.array(
            [float(_attribute(camera, prefix + key, 0.0)) for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")],
            dtype=np.float64,
        )
        return fx, fy, cx, cy, distortion

    # 普通针孔相机的教学级回退：f / aperture 得到像素焦距。
    focal = float(camera.get_focal_length())
    fx = focal / float(camera.get_horizontal_aperture()) * width
    fy = focal / float(camera.get_vertical_aperture()) * height
    return fx, fy, width / 2.0, height / 2.0, np.zeros(8, dtype=np.float64)


def _undistort_normalized(x_distorted, y_distorted, coefficients, iterations: int = 5):
    """用 OpenCV Brown-Conrady 模型迭代求无畸变归一化坐标。"""
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    x = np.asarray(x_distorted, dtype=np.float64).copy()
    y = np.asarray(y_distorted, dtype=np.float64).copy()

    for _ in range(iterations):
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        radial_inverse = np.divide(
            denominator,
            numerator,
            out=np.ones_like(numerator),
            where=np.abs(numerator) > 1e-12,
        )
        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (x_distorted - delta_x) * radial_inverse
        y = (y_distorted - delta_y) * radial_inverse
    return x, y


def depth_pixels_to_world(camera, depth, pixel_mask=None) -> np.ndarray:
    """将选中的深度像素转换到 Isaac 世界坐标。

    ``depth`` 是到成像平面的距离，因此直接作为相机光轴 ``Z``；相机坐标采用 ROS
    光学约定：``+X`` 向右、``+Y`` 向下、``+Z`` 向前。
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        return np.empty((0, 3), dtype=np.float64)

    valid = np.isfinite(depth) & (depth > 0.0)
    if pixel_mask is not None:
        mask = np.asarray(pixel_mask, dtype=bool)
        if mask.shape != depth.shape:
            return np.empty((0, 3), dtype=np.float64)
        valid &= mask

    rows, columns = np.nonzero(valid)
    if not len(rows):
        return np.empty((0, 3), dtype=np.float64)

    z = depth[rows, columns]
    height, width = depth.shape
    fx, fy, cx, cy, distortion = _scaled_calibration(camera, width, height)
    x_distorted = (columns.astype(np.float64) + 0.5 - cx) / fx
    y_distorted = (rows.astype(np.float64) + 0.5 - cy) / fy
    x, y = _undistort_normalized(x_distorted, y_distorted, distortion)
    camera_points = np.column_stack((x * z, y * z, z))

    camera_position, camera_quaternion = camera.get_world_pose(camera_axes="ros")
    camera_rotation = _quaternion_matrix(camera_quaternion)
    return np.asarray(camera_position, dtype=np.float64) + camera_points @ camera_rotation.T
