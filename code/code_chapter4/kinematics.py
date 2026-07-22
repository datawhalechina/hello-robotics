"""四轮独立转向（swerve）底盘的二维运动学。

坐标约定：底盘 x 向前、y 向左、z 向上，逆时针角速度为正。
"""

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class ChassisVelocity:
    """底盘坐标系中的速度指令。"""

    vx: float = 0.0       # 前向速度，m/s
    vy: float = 0.0       # 左向速度，m/s
    wz: float = 0.0       # 逆时针角速度，rad/s


@dataclass(frozen=True)
class WheelState:
    """一个转向轮的目标状态。"""

    steering_angle: float  # 转向角，rad
    wheel_speed: float     # 车轮自转角速度，rad/s


@dataclass
class Pose2D:
    """平面位姿。"""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


def normalize_angle(angle: float) -> float:
    """把角度归一化到 [-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SwerveKinematics:
    """G2 四轮独立转向底盘的正、逆运动学。"""

    def __init__(
        self,
        wheel_positions: Sequence[tuple[float, float]],
        wheel_radius: float,
    ) -> None:
        if len(wheel_positions) < 3:
            raise ValueError("至少需要 3 个轮子才能计算平面底盘运动学")
        if wheel_radius <= 0.0:
            raise ValueError("wheel_radius 必须大于 0")

        self.wheel_positions = tuple(wheel_positions)
        self.wheel_radius = float(wheel_radius)

    def inverse(
        self,
        velocity: ChassisVelocity,
        current_angles: Sequence[float] | None = None,
        max_wheel_speed: float | None = None,
    ) -> list[WheelState]:
        """逆运动学：底盘速度 -> 每个轮子的转向角和自转速度。

        对位于 (x_i, y_i) 的轮子，轮心速度为：
            v_ix = vx - wz * y_i
            v_iy = vy + wz * x_i

        若提供 current_angles，会自动选择“转向少于 90°并反转车轮”的等价解，
        避免转向机构绕远路。底盘静止时保持当前转向角，不强制回正。
        """
        if current_angles is not None and len(current_angles) != len(self.wheel_positions):
            raise ValueError("current_angles 数量必须与轮子数量一致")
        if max_wheel_speed is not None and max_wheel_speed <= 0.0:
            raise ValueError("max_wheel_speed 必须大于 0")

        states: list[WheelState] = []
        for index, (wheel_x, wheel_y) in enumerate(self.wheel_positions):
            wheel_vx = velocity.vx - velocity.wz * wheel_y
            wheel_vy = velocity.vy + velocity.wz * wheel_x
            linear_speed = math.hypot(wheel_vx, wheel_vy)

            if linear_speed < 1e-8:
                hold_angle = 0.0 if current_angles is None else float(current_angles[index])
                states.append(WheelState(hold_angle, 0.0))
                continue

            desired_angle = math.atan2(wheel_vy, wheel_vx)
            wheel_speed = linear_speed / self.wheel_radius

            if current_angles is not None:
                desired_angle, wheel_speed = self._nearest_equivalent_state(
                    desired_angle,
                    wheel_speed,
                    float(current_angles[index]),
                )

            states.append(WheelState(desired_angle, wheel_speed))

        return self._desaturate(states, max_wheel_speed)

    def forward(self, wheel_states: Sequence[WheelState]) -> ChassisVelocity:
        """正运动学：轮子状态 -> 底盘速度。

        先把各轮自转速度投影为轮心二维速度，再利用对称轮组的最小二乘
        闭式结果估计 vx、vy、wz。它也可用于简单的轮速里程计。
        """
        if len(wheel_states) != len(self.wheel_positions):
            raise ValueError("wheel_states 数量必须与轮子数量一致")

        wheel_vectors: list[tuple[float, float]] = []
        rotation_numerator = 0.0
        rotation_denominator = 0.0

        for state, (wheel_x, wheel_y) in zip(wheel_states, self.wheel_positions):
            linear_speed = state.wheel_speed * self.wheel_radius
            wheel_vx = linear_speed * math.cos(state.steering_angle)
            wheel_vy = linear_speed * math.sin(state.steering_angle)
            wheel_vectors.append((wheel_vx, wheel_vy))

            rotation_numerator += -wheel_y * wheel_vx + wheel_x * wheel_vy
            rotation_denominator += wheel_x * wheel_x + wheel_y * wheel_y

        wz = rotation_numerator / rotation_denominator
        count = len(wheel_vectors)
        vx = sum(
            wheel_vx + wz * wheel_y
            for (wheel_vx, _), (_, wheel_y) in zip(wheel_vectors, self.wheel_positions)
        ) / count
        vy = sum(
            wheel_vy - wz * wheel_x
            for (_, wheel_vy), (wheel_x, _) in zip(wheel_vectors, self.wheel_positions)
        ) / count
        return ChassisVelocity(vx, vy, wz)

    @staticmethod
    def _nearest_equivalent_state(
        desired_angle: float,
        wheel_speed: float,
        current_angle: float,
    ) -> tuple[float, float]:
        """选择离当前角度最近的等价“转向角 + 轮速”组合。"""
        delta = normalize_angle(desired_angle - current_angle)
        if abs(delta) > math.pi / 2.0:
            delta = normalize_angle(delta + math.pi)
            wheel_speed = -wheel_speed
        return current_angle + delta, wheel_speed

    @staticmethod
    def _desaturate(
        states: list[WheelState],
        max_wheel_speed: float | None,
    ) -> list[WheelState]:
        """等比例缩放所有轮速，保持期望运动方向不变。"""
        if max_wheel_speed is None:
            return states

        largest = max(abs(state.wheel_speed) for state in states)
        if largest <= max_wheel_speed:
            return states

        scale = max_wheel_speed / largest
        return [
            WheelState(state.steering_angle, state.wheel_speed * scale)
            for state in states
        ]


class SwerveOdometry:
    """使用车轮正运动学积分得到的简易二维里程计。"""

    def __init__(self, kinematics: SwerveKinematics, initial_pose: Pose2D | None = None) -> None:
        self.kinematics = kinematics
        self.pose = initial_pose if initial_pose is not None else Pose2D()

    def update(self, wheel_states: Sequence[WheelState], dt: float) -> Pose2D:
        if dt <= 0.0:
            raise ValueError("dt 必须大于 0")

        velocity = self.kinematics.forward(wheel_states)
        cos_yaw = math.cos(self.pose.yaw)
        sin_yaw = math.sin(self.pose.yaw)

        world_vx = cos_yaw * velocity.vx - sin_yaw * velocity.vy
        world_vy = sin_yaw * velocity.vx + cos_yaw * velocity.vy
        self.pose.x += world_vx * dt
        self.pose.y += world_vy * dt
        self.pose.yaw = normalize_angle(self.pose.yaw + velocity.wz * dt)
        return Pose2D(self.pose.x, self.pose.y, self.pose.yaw)
