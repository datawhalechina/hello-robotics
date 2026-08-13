"""第十三章统一配置：场景、传感器、模型和语义目标。"""

from dataclasses import dataclass, field
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parent
CODE_DIR = CHAPTER_DIR.parent
SHARED_ASSET_DIR = CODE_DIR / "assets"  # 全教程共用，只有仿真资源允许放在本章外

ROBOT_USD = SHARED_ASSET_DIR / "robot/G2_omnipicker/robot.usda"
ROOM_USD = SHARED_ASSET_DIR / "background/room/room_1/background.usda"
YOLO_WORLD_MODEL = CHAPTER_DIR / "yolov8l-world.pt"
QWEN3_VL_MODEL = CHAPTER_DIR / "Qwen3-VL-4B-Instruct"
DEFAULT_OUTPUT_DIR = CHAPTER_DIR / "outputs"

ROBOT_PRIM_PATH = "/genie"
ROOM_PRIM_PATH = "/World"
HEAD_CAMERA_PRIM_PATH = f"{ROBOT_PRIM_PATH}/head_link3/head_front_Camera"
LIDAR_PARENT_PATH = f"{ROBOT_PRIM_PATH}/base_link"

# G2 四轮独立转向底盘关节，顺序为左前、左后、右前、右后。
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
class RobotGeometry:
    """G2 四轮独立转向底盘几何参数，单位为米。"""

    wheel_radius: float = 0.070
    wheel_base: float = 0.460
    track_width: float = 0.436

    @property
    def wheel_positions(self) -> tuple[tuple[float, float], ...]:
        half_length = self.wheel_base / 2.0
        half_width = self.track_width / 2.0
        return (
            (+half_length, +half_width),
            (-half_length, +half_width),
            (+half_length, -half_width),
            (-half_length, -half_width),
        )


@dataclass(frozen=True)
class ControlLimits:
    """底盘速度、加速度和看门狗限制。"""

    max_linear_speed: float = 0.70
    max_angular_speed: float = 1.20
    max_wheel_speed: float = 18.0
    max_linear_acceleration: float = 0.80
    max_angular_acceleration: float = 1.80
    command_timeout: float = 0.50


@dataclass(frozen=True)
class BoxObject:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color_rgb: tuple[float, float, float]


# 与第十章静态地图一致；目标物体不写进静态地图，由 G2 雷达实时加入代价地图。
STATIC_OBSTACLES = (
    BoxObject("center_block", (0.0, 0.0, 0.35), (1.0, 2.4, 0.70), (0.45, 0.45, 0.45)),
    BoxObject("northwest_block", (-2.2, 1.9, 0.35), (0.9, 1.4, 0.70), (0.45, 0.45, 0.45)),
    BoxObject("southeast_block", (2.1, -1.8, 0.35), (1.3, 0.8, 0.70), (0.45, 0.45, 0.45)),
)

# 三个目标分散在房间不同方向，G2 需要原地扫描后再导航。
# 相对起点的距离约为 3.8 m、4.2 m 和 3.8 m，同时避开静态障碍物。
TARGET_OBJECTS = (
    BoxObject("red", (-3.00, 0.50, 0.42), (0.48, 0.48, 0.84), (0.95, 0.06, 0.04)),
    BoxObject("blue", (0.60, -3.50, 0.42), (0.48, 0.48, 0.84), (0.04, 0.20, 0.95)),
    BoxObject("yellow", (0.00, -2.00, 0.42), (0.48, 0.48, 0.84), (0.98, 0.82, 0.04)),
)
TARGET_COLORS = tuple(item.name for item in TARGET_OBJECTS)
YOLO_WORLD_CLASSES = ["box"]


@dataclass(frozen=True)
class SimulationConfig:
    headless: bool = False
    physics_hz: int = 60
    rendering_hz: int = 30
    warmup_steps: int = 90
    renderer: str = "RaytracedLighting"
    robot_start: Pose2D = field(default_factory=lambda: Pose2D(-3.60, -3.30, 0.0))

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz


@dataclass(frozen=True)
class SensorConfig:
    camera_resolution: tuple[int, int] = (640, 400)
    camera_hz: int = 15
    lidar_min_range: float = 0.38  # 去掉机器人自身附近点
    lidar_max_range: float = 8.0
    lidar_min_z: float = 0.08
    lidar_max_z: float = 1.80
    lidar_bins: int = 360


@dataclass(frozen=True)
class PerceptionConfig:
    confidence: float = 0.08
    iou: float = 0.45
    image_size: int = 640
    depth_tolerance: float = 0.18
    settle_seconds: float = 2.0
    scan_speed: float = 0.42
    scan_timeout: float = 17.0


@dataclass(frozen=True)
class NavigationConfig:
    stand_off_distance: float = 0.85
    nav2_wait_timeout: float = 60.0
    nav2_activation_grace: float = 2.0  # action 建立后等待 lifecycle 完成激活
    navigation_timeout: float = 120.0
    publish_hz: int = 20
    lidar_hz: int = 10


@dataclass(frozen=True)
class VLMConfig:
    # Python 解释器由 VLM_PYTHON 或 --vlm-python 指定，不在配置中绑定项目外路径。
    python: str = ""
    model_path: Path = QWEN3_VL_MODEL
    max_new_tokens: int = 128
