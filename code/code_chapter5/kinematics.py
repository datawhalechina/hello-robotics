"""G2 七自由度机械臂的正运动学、Jacobian 与数值逆运动学。

本文件只依赖 NumPy，不调用 Isaac Sim、Lula、MoveIt 或其他现成 IK 求解器。
所有目标位姿都以 ``arm_base_link`` 坐标系表示。
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

try:
    from .config import IKConfig, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS
except ImportError:
    from config import IKConfig, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS


@dataclass(frozen=True)
class Pose:
    """三维位姿：位置单位为 m，旋转矩阵形状为 (3, 3)。"""

    position: np.ndarray
    rotation: np.ndarray

    @property
    def quaternion(self) -> np.ndarray:
        """返回 Isaac Sim 常用的 [w, x, y, z] 四元数。"""
        return matrix_to_quaternion(self.rotation)


@dataclass(frozen=True)
class IKResult:
    """逆运动学求解结果。"""

    joint_positions: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float


def _as_vector(values: Sequence[float], length: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (length,):
        raise ValueError(f"{name} 必须包含 {length} 个数，实际形状为 {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 中不能包含 NaN 或无穷大")
    return vector


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """按照 URDF 约定生成齐次变换：R = Rz(yaw) Ry(pitch) Rx(roll)。"""
    roll, pitch, yaw = rpy
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues 公式：绕任意单位轴旋转。"""
    axis = _as_vector(axis, 3, "axis")
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("旋转轴长度不能为 0")
    x, y, z = axis / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    return result


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """[w, x, y, z] 四元数转旋转矩阵。"""
    q = _as_vector(quaternion, 4, "quaternion").copy()
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("四元数长度不能为 0")
    q /= norm
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵转 [w, x, y, z] 四元数。"""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError("rotation 必须是 3x3 矩阵")

    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                          (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                          0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                          (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """计算从 current 转到 target 的世界坐标系旋转误差向量。"""
    relative = np.asarray(target, dtype=np.float64) @ np.asarray(
        current, dtype=np.float64
    ).T
    quaternion = matrix_to_quaternion(relative)
    vector = quaternion[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-10:
        return 2.0 * vector
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[0]))
    return vector * (angle / vector_norm)



class G2ArmKinematics:
    """由 G2 机器人模型参数直接构建的七自由度单臂模型。"""

    def __init__(self, arm: str = "right") -> None:
        if arm not in ("left", "right"):
            raise ValueError("arm 只能是 'left' 或 'right'")
        self.arm = arm
        pi = math.pi

        first_xyz = (0.0, 0.0, 0.069) if arm == "left" else (0.0, 0.0, -0.069)
        first_rpy = (0.0, 0.0, 0.0) if arm == "left" else (pi, 0.0, 0.0)
        self.joint_origins = (
            transform(first_xyz, first_rpy),
            transform((0.0, 0.0, 0.1745), (pi / 2.0, 0.0, 0.0)),
            transform((0.0, 0.0, 0.0), (-pi / 2.0, 0.0, 0.0)),
            transform((0.018, 0.0, 0.287), (pi / 2.0, 0.0, 0.0)),
            transform((-0.018, 0.0, 0.0), (-pi / 2.0, 0.0, 0.0)),
            transform((0.0, 0.0, 0.314), (pi / 2.0, 0.0, 0.0)),
            transform((0.0, 0.0, 0.0), (pi / 2.0, 0.0, pi / 2.0)),
        )
        self.joint_axes = tuple(np.array([0.0, 0.0, 1.0]) for _ in range(7))

        # arm_link7 -> gripper_center 的组合固定变换。当前 G2 USD 与仓库中
        # 辅助 URDF 的末端长度略有差异，因此以实际仿真模型的 0.23645 m 为准。
        self.tool_transform = transform(
            (0.23645, 0.0, 0.0), (pi, -pi / 2.0, 0.0)
        )
        self.lower_limits = JOINT_LOWER_LIMITS.copy()
        self.upper_limits = JOINT_UPPER_LIMITS.copy()

    def _chain(self, joint_positions: Sequence[float]):
        q = _as_vector(joint_positions, 7, "joint_positions")
        current = np.eye(4, dtype=np.float64)
        joint_points, joint_axes = [], []

        for origin, local_axis, angle in zip(self.joint_origins, self.joint_axes, q):
            current = current @ origin
            joint_points.append(current[:3, 3].copy())
            joint_axes.append(current[:3, :3] @ local_axis)
            current = current @ axis_rotation(local_axis, float(angle))

        end_transform = current @ self.tool_transform
        return end_transform, joint_points, joint_axes

    def forward(self, joint_positions: Sequence[float]) -> Pose:
        """正运动学：7 个关节角 -> 末端在 arm_base_link 下的位姿。"""
        end_transform, _, _ = self._chain(joint_positions)
        return Pose(end_transform[:3, 3].copy(), end_transform[:3, :3].copy())

    def jacobian(self, joint_positions: Sequence[float]) -> np.ndarray:
        """计算 6x7 几何 Jacobian，上三行为线速度，下三行为角速度。"""
        end_transform, joint_points, joint_axes = self._chain(joint_positions)
        end_position = end_transform[:3, 3]
        jacobian = np.zeros((6, 7), dtype=np.float64)
        for i, (point, axis) in enumerate(zip(joint_points, joint_axes)):
            jacobian[:3, i] = np.cross(axis, end_position - point)
            jacobian[3:, i] = axis
        return jacobian

    def inverse(
        self,
        target_position: Sequence[float],
        target_rotation: np.ndarray | None = None,
        initial_positions: Sequence[float] | None = None,
        config: IKConfig | None = None,
    ) -> IKResult:
        """阻尼最小二乘 IK，可求位置或完整位姿。

        7 自由度机械臂存在冗余解。代码在主任务之外加入很小的零空间关节居中项，
        使结果尽量远离限位，但不会覆盖末端位姿任务。
        """
        cfg = config or IKConfig()
        target_p = _as_vector(target_position, 3, "target_position")
        if target_rotation is not None:
            target_r = np.asarray(target_rotation, dtype=np.float64)
            if target_r.shape != (3, 3):
                raise ValueError("target_rotation 必须是 3x3 矩阵")
        else:
            target_r = None

        if initial_positions is None:
            q = (self.lower_limits + self.upper_limits) * 0.5
        else:
            q = _as_vector(initial_positions, 7, "initial_positions").copy()
        q = np.clip(q, self.lower_limits, self.upper_limits)

        center = (self.lower_limits + self.upper_limits) * 0.5
        half_range = np.maximum((self.upper_limits - self.lower_limits) * 0.5, 1e-6)
        position_norm = math.inf
        orientation_norm = 0.0

        for iteration in range(1, cfg.max_iterations + 1):
            pose = self.forward(q)
            position_delta = target_p - pose.position
            position_norm = float(np.linalg.norm(position_delta))

            full_jacobian = self.jacobian(q)
            if target_r is None:
                error = position_delta
                jacobian = full_jacobian[:3]
                orientation_norm = 0.0
            else:
                rotation_delta = orientation_error(target_r, pose.rotation)
                orientation_norm = float(np.linalg.norm(rotation_delta))
                error = np.concatenate(
                    (position_delta, cfg.orientation_weight * rotation_delta)
                )
                jacobian = np.vstack(
                    (full_jacobian[:3], cfg.orientation_weight * full_jacobian[3:])
                )

            if position_norm <= cfg.position_tolerance and (
                target_r is None or orientation_norm <= cfg.orientation_tolerance
            ):
                return IKResult(q.copy(), True, iteration - 1,
                                position_norm, orientation_norm)

            # J# = J^T (J J^T + lambda^2 I)^-1，比直接求普通逆矩阵更稳健。
            task_size = jacobian.shape[0]
            damped = jacobian @ jacobian.T + (cfg.damping ** 2) * np.eye(task_size)
            pseudo_inverse = jacobian.T @ np.linalg.solve(damped, np.eye(task_size))
            delta_q = pseudo_inverse @ error

            # 冗余自由度用于远离关节上下限。
            normalized_center_error = (center - q) / (half_range * half_range)
            nullspace = np.eye(7) - pseudo_inverse @ jacobian
            delta_q += cfg.joint_center_gain * (nullspace @ normalized_center_error)

            largest_step = float(np.max(np.abs(delta_q)))
            if largest_step > cfg.max_joint_step:
                delta_q *= cfg.max_joint_step / largest_step
            q = np.clip(q + delta_q, self.lower_limits, self.upper_limits)

        return IKResult(q.copy(), False, cfg.max_iterations,
                        position_norm, orientation_norm)
