"""Nav2 专用示例的场景、地图与仿真配置。"""

from dataclasses import dataclass, field
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAPTER_DIR = Path(__file__).resolve().parent
ROBOT_USD = PROJECT_ROOT / "assets/robot/G2_omnipicker/robot.usda"
ROOM_USD = PROJECT_ROOT / "assets/background/room/room_1/background.usda"
MAP_YAML = CHAPTER_DIR / "maps/chapter10_2_map.yaml"
ROBOT_PRIM_PATH = "/genie"
ROOM_PRIM_PATH = "/World"

@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0

@dataclass(frozen=True)
class Velocity2D:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

@dataclass(frozen=True)
class CircleObstacle:
    x: float
    y: float
    radius: float

@dataclass(frozen=True)
class RectangleObstacle:
    name: str
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    height: float = 0.70

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
        y = self.center_y + self.travel * math.sin(2.0 * math.pi * seconds / self.period)
        return CircleObstacle(self.x, y, self.radius)
