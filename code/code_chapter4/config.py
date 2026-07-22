from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = PROJECT_ROOT / "assets/robot/G2_omnipicker/robot.usda"
ROOM_USD = PROJECT_ROOT / "assets/background/room/room_1/background.usda"
ROBOT_PRIM_PATH = "/genie"
ROOM_PRIM_PATH = "/World"

# 顺序必须始终保持：左前、左后、右前、右后。
WHEEL_NAMES = ("left_front", "left_rear", "right_front", "right_rear")
STEERING_JOINT_NAMES = (
    "idx111_chassis_lwheel_front_joint1",
    "idx121_chassis_lwheel_rear_joint1",
    "idx131_chassis_rwheel_front_joint1",
    "idx141_chassis_rwheel_rear_joint1",
)
DRIVE_JOINT_NAMES = (
    "idx112_chassis_lwheel_front_joint2",
    "idx122_chassis_lwheel_rear_joint2",
    "idx132_chassis_rwheel_front_joint2",
    "idx142_chassis_rwheel_rear_joint2",
)


@dataclass(frozen=True)
class RobotGeometry:
    """G2 四轮独立转向底盘的几何尺寸，单位为米。"""

    wheel_radius: float = 0.070
    wheel_base: float = 0.460
    track_width: float = 0.436

    @property
    def wheel_positions(self) -> tuple[tuple[float, float], ...]:
        """每个轮子在底盘坐标系中的 (x, y)。

        坐标约定：x 向前，y 向左，z 向上。
        """
        half_length = self.wheel_base / 2.0
        half_width = self.track_width / 2.0
        return (
            (+half_length, +half_width),  # 左前
            (-half_length, +half_width),  # 左后
            (+half_length, -half_width),  # 右前
            (-half_length, -half_width),  # 右后
        )


@dataclass(frozen=True)
class ControlLimits:
    """速度、加速度与安全超时限制。"""

    max_linear_speed: float = 0.70       # m/s，限制 sqrt(vx^2 + vy^2)
    max_angular_speed: float = 1.20      # rad/s
    max_wheel_speed: float = 18.0        # rad/s
    max_linear_acceleration: float = 0.80   # m/s^2
    max_angular_acceleration: float = 1.80  # rad/s^2
    command_timeout: float = 0.50        # s，超时后自动刹车


@dataclass(frozen=True)
class SimulationConfig:
    """Isaac Sim 的最小运行配置。"""

    physics_hz: int = 120
    rendering_hz: int = 60
    headless: bool = False
    renderer: str = "RaytracedLighting"
    warmup_steps: int = 120
    robot_position: tuple[float, float, float] = (0.0, 0.0, -0.01)
    robot_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz
