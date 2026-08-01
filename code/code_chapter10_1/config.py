"""不使用 Nav2 的教学导航统一配置。

优先修改本文件，观察机器人半径、地图膨胀、规划分辨率和控制参数的影响。
"""

from dataclasses import dataclass, field
from pathlib import Path

try:
    from .geometry import CircleObstacle, Pose2D
except ImportError:
    from geometry import CircleObstacle, Pose2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAPTER_DIR = Path(__file__).resolve().parent
ROBOT_USD = PROJECT_ROOT / "assets/robot/G2_omnipicker/robot.usda"
ROOM_USD = PROJECT_ROOT / "assets/background/room/room_1/background.usda"
MAP_YAML = CHAPTER_DIR / "maps/chapter10_1_map.yaml"
RVIZ_CONFIG = CHAPTER_DIR / "config/chapter10_1_navigation.rviz"
ROBOT_PRIM_PATH = "/genie"
ROOM_PRIM_PATH = "/World"


@dataclass(frozen=True)
class RectangleObstacle:
    name: str
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    height: float = 0.70


# 栅格地图和 Isaac Sim 障碍物使用同一份定义。
STATIC_OBSTACLES = (
    RectangleObstacle("center_block", 0.0, 0.0, 1.0, 2.4),
    RectangleObstacle("northwest_block", -2.2, 1.9, 0.9, 1.4),
    RectangleObstacle("southeast_block", 2.1, -1.8, 1.3, 0.8),
)


@dataclass(frozen=True)
class MapConfig:
    resolution: float = 0.05
    min_x: float = -4.75
    min_y: float = -4.75
    max_x: float = 4.75
    max_y: float = 4.75
    wall_thickness: float = 0.12
    robot_radius: float = 0.34
    safety_margin: float = 0.10
    unknown_is_occupied: bool = True

    @property
    def inflation_radius(self) -> float:
        return self.robot_radius + self.safety_margin


@dataclass(frozen=True)
class PlannerConfig:
    allow_diagonal: bool = True
    heuristic_weight: float = 1.0
    shortcut_passes: int = 2
    path_spacing: float = 0.12
    local_lookahead_distance: float = 2.4
    local_window_radius: float = 3.0
    local_replan_period: float = 0.40
    detour_hold_cycles: int = 3


@dataclass(frozen=True)
class TrackerConfig:
    # 稍长的前视距离与较低增益可减少局部路径更新时的左右摆动。
    lookahead_distance: float = 0.70
    position_gain: float = 1.0
    yaw_gain: float = 1.5
    max_linear_speed: float = 0.48
    max_angular_speed: float = 0.80
    detour_speed_limit: float = 0.34
    slow_down_distance: float = 0.90
    position_tolerance: float = 0.12
    yaw_tolerance: float = 0.12


@dataclass(frozen=True)
class SafetyConfig:
    max_tilt_angle: float = 0.35  # rad，约 20°，超过后立即停车
    collision_horizon: float = 0.12
    collision_step: float = 0.05
    progress_timeout: float = 7.0
    minimum_progress: float = 0.12
    max_recovery_attempts: int = 3


@dataclass(frozen=True)
class SimulationConfig:
    headless: bool = False
    physics_hz: int = 120
    rendering_hz: int = 60
    warmup_steps: int = 120
    renderer: str = "RaytracedLighting"
    robot_start: Pose2D = field(default_factory=lambda: Pose2D(-3.6, -3.3, 0.0))
    default_goal: Pose2D = field(default_factory=lambda: Pose2D(3.6, 3.2, 1.57))
    dynamic_obstacle: bool = True

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz


@dataclass(frozen=True)
class DynamicObstacleConfig:
    x: float = 1.20
    center_y: float = 1.05
    travel: float = 1.15
    period: float = 8.0
    radius: float = 0.34

    def at_time(self, seconds: float) -> CircleObstacle:
        import math

        y = self.center_y + self.travel * math.sin(2.0 * math.pi * seconds / self.period)
        return CircleObstacle(self.x, y, self.radius)
