"""G2 底盘控制器。

这里没有调用 Isaac Sim 自带的 DifferentialController、HolonomicController
或 WheelBasePoseController。底盘运动学和位姿反馈控制均由本代码实现。
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

try:
    from .config import (
        ControlLimits,
        DRIVE_JOINT_NAMES,
        STEERING_JOINT_NAMES,
    )
    from .kinematics import (
        ChassisVelocity,
        Pose2D,
        SwerveKinematics,
        WheelState,
        normalize_angle,
    )
except ImportError:  # 支持直接执行本目录中的示例
    from config import ControlLimits, DRIVE_JOINT_NAMES, STEERING_JOINT_NAMES
    from kinematics import (
        ChassisVelocity,
        Pose2D,
        SwerveKinematics,
        WheelState,
        normalize_angle,
    )


@dataclass(frozen=True)
class PoseControlResult:
    command: ChassisVelocity
    distance_error: float
    yaw_error: float
    reached: bool


def _find_joint_indices(all_names: Sequence[str], required_names: Sequence[str]) -> list[int]:
    """严格查找关节，缺少任意一个关节都立即报错。"""
    name_to_index = {name: index for index, name in enumerate(all_names)}
    missing = [name for name in required_names if name not in name_to_index]
    if missing:
        raise RuntimeError(
            "G2 USD 中缺少底盘关节："
            + ", ".join(missing)
            + "\n实际可用关节："
            + ", ".join(all_names)
        )
    return [name_to_index[name] for name in required_names]


def _limit_linear_speed(command: ChassisVelocity, max_speed: float) -> ChassisVelocity:
    length = math.hypot(command.vx, command.vy)
    if length <= max_speed or length < 1e-12:
        return command
    scale = max_speed / length
    return ChassisVelocity(command.vx * scale, command.vy * scale, command.wz)


def _move_toward(current: float, target: float, max_change: float) -> float:
    difference = target - current
    if abs(difference) <= max_change:
        return target
    return current + math.copysign(max_change, difference)


class G2BaseController:
    """把底盘速度命令转换为 G2 的 8 个车轮关节命令。"""

    def __init__(self, articulation, kinematics: SwerveKinematics, limits: ControlLimits) -> None:
        self.articulation = articulation
        self.kinematics = kinematics
        self.limits = limits

        dof_names = articulation.dof_names
        self.steering_indices = _find_joint_indices(
            dof_names, STEERING_JOINT_NAMES
        )
        self.drive_indices = _find_joint_indices(
            dof_names, DRIVE_JOINT_NAMES
        )

        self._requested = ChassisVelocity()
        self._applied = ChassisVelocity()
        self._time_since_command = math.inf

        print("[G2BaseController] 底盘关节检查通过", flush=True)
        print(f"  转向关节索引: {self.steering_indices}")
        print(f"  驱动关节索引: {self.drive_indices}")

    @property
    def applied_velocity(self) -> ChassisVelocity:
        """经过限速与加速度限制后，当前实际下发的底盘命令。"""
        return self._applied

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """设置底盘速度；调用 update() 才会真正写入关节。"""
        command = ChassisVelocity(float(vx), float(vy), float(wz))
        command = _limit_linear_speed(command, self.limits.max_linear_speed)
        command = ChassisVelocity(
            command.vx,
            command.vy,
            max(-self.limits.max_angular_speed, min(self.limits.max_angular_speed, command.wz)),
        )
        self._requested = command
        self._time_since_command = 0.0

    def update(self, dt: float) -> list[WheelState]:
        """执行一次控制计算。应在每个物理步调用一次。"""
        if dt <= 0.0:
            raise ValueError("dt 必须大于 0")

        self._time_since_command += dt
        desired = self._requested
        if self._time_since_command > self.limits.command_timeout:
            desired = ChassisVelocity()  # 看门狗：通信中断时自动减速停车

        self._applied = self._apply_acceleration_limit(desired, dt)
        current_angles = self._read_steering_angles()
        wheel_states = self.kinematics.inverse(
            self._applied,
            current_angles=current_angles,
            max_wheel_speed=self.limits.max_wheel_speed,
        )
        self._write_wheel_states(wheel_states)
        return wheel_states

    def stop(self, center_steering: bool = False) -> None:
        """立即停止车轮；默认保持当前转向角，避免不必要的原地摆动。"""
        from isaacsim.core.utils.types import ArticulationAction

        self._requested = ChassisVelocity()
        self._applied = ChassisVelocity()
        self._time_since_command = math.inf

        self.articulation.apply_action(
            ArticulationAction(
                joint_velocities=np.zeros(len(self.drive_indices), dtype=np.float64),
                joint_indices=self.drive_indices,
            )
        )
        if center_steering:
            self.articulation.apply_action(
                ArticulationAction(
                    joint_positions=np.zeros(len(self.steering_indices), dtype=np.float64),
                    joint_indices=self.steering_indices,
                )
            )

    def measured_wheel_states(self) -> list[WheelState]:
        """读取仿真中的转向角和车轮实际角速度。"""
        all_positions = np.asarray(self.articulation.get_joint_positions())
        all_velocities = np.asarray(self.articulation.get_joint_velocities())
        return [
            WheelState(float(all_positions[steer]), float(all_velocities[drive]))
            for steer, drive in zip(self.steering_indices, self.drive_indices)
        ]

    def measured_chassis_velocity(self) -> ChassisVelocity:
        """由实际关节状态通过正运动学估计底盘速度。"""
        return self.kinematics.forward(self.measured_wheel_states())

    def _apply_acceleration_limit(
        self, desired: ChassisVelocity, dt: float
    ) -> ChassisVelocity:
        # 对二维线速度增量整体限幅，避免斜向运动时合加速度超限。
        delta_vx = desired.vx - self._applied.vx
        delta_vy = desired.vy - self._applied.vy
        delta_length = math.hypot(delta_vx, delta_vy)
        max_linear_change = self.limits.max_linear_acceleration * dt
        if delta_length > max_linear_change and delta_length > 1e-12:
            scale = max_linear_change / delta_length
            delta_vx *= scale
            delta_vy *= scale

        new_wz = _move_toward(
            self._applied.wz,
            desired.wz,
            self.limits.max_angular_acceleration * dt,
        )
        return ChassisVelocity(
            self._applied.vx + delta_vx,
            self._applied.vy + delta_vy,
            new_wz,
        )

    def _read_steering_angles(self) -> list[float]:
        all_positions = np.asarray(self.articulation.get_joint_positions())
        return [float(all_positions[index]) for index in self.steering_indices]

    def _write_wheel_states(self, wheel_states: Sequence[WheelState]) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        steering_targets = np.array(
            [state.steering_angle for state in wheel_states], dtype=np.float64
        )
        drive_targets = np.array(
            [state.wheel_speed for state in wheel_states], dtype=np.float64
        )

        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=steering_targets,
                joint_indices=self.steering_indices,
            )
        )
        self.articulation.apply_action(
            ArticulationAction(
                joint_velocities=drive_targets,
                joint_indices=self.drive_indices,
            )
        )


class PoseController:
    """简单、直观的二维位姿比例控制器。

    位置误差先从世界坐标系转换到底盘坐标系，再输出 vx、vy；因此 G2 可以
    不先转头，直接向任意方向平移。该控制器只负责局部闭环，不负责避障。
    """

    def __init__(
        self,
        position_gain: float = 1.2,
        yaw_gain: float = 2.0,
        max_linear_speed: float = 0.55,
        max_angular_speed: float = 1.0,
        position_tolerance: float = 0.04,
        yaw_tolerance: float = 0.05,
    ) -> None:
        self.position_gain = position_gain
        self.yaw_gain = yaw_gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance

    def compute(self, current: Pose2D, target: Pose2D) -> PoseControlResult:
        error_world_x = target.x - current.x
        error_world_y = target.y - current.y
        distance_error = math.hypot(error_world_x, error_world_y)
        yaw_error = normalize_angle(target.yaw - current.yaw)

        reached = (
            distance_error <= self.position_tolerance
            and abs(yaw_error) <= self.yaw_tolerance
        )
        if reached:
            return PoseControlResult(ChassisVelocity(), distance_error, yaw_error, True)

        cos_yaw = math.cos(current.yaw)
        sin_yaw = math.sin(current.yaw)
        error_body_x = cos_yaw * error_world_x + sin_yaw * error_world_y
        error_body_y = -sin_yaw * error_world_x + cos_yaw * error_world_y

        if distance_error <= self.position_tolerance:
            vx = 0.0
            vy = 0.0
        else:
            vx = self.position_gain * error_body_x
            vy = self.position_gain * error_body_y

        command = _limit_linear_speed(
            ChassisVelocity(vx, vy, self.yaw_gain * yaw_error),
            self.max_linear_speed,
        )
        command = ChassisVelocity(
            command.vx,
            command.vy,
            max(-self.max_angular_speed, min(self.max_angular_speed, command.wz)),
        )
        return PoseControlResult(command, distance_error, yaw_error, False)
