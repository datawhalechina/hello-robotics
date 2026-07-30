"""第七章统一配置：场景、机器人、相机和视觉模型路径。"""

from dataclasses import dataclass
from pathlib import Path


# code_chapter7 -> new -> g2_robot
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = PROJECT_ROOT / "assets/robot/G2_omnipicker/robot.usda"
ROOM_USD = PROJECT_ROOT / "assets/background/room/room_1/background.usda"
ROBOT_PRIM_PATH = "/genie"
ROOM_PRIM_PATH = "/World"
WAREHOUSE_PRIM_PATH = "/background"
WAREHOUSE_ASSET = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

DEFAULT_DETECTION_MODEL = PROJECT_ROOT / "yolo26s.pt"
DEFAULT_SEGMENTATION_MODEL = PROJECT_ROOT / "yolo26s-seg.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/chapter7"


@dataclass(frozen=True)
class CameraConfig:
    """机器人相机配置；resolution 顺序为 (宽, 高)。"""

    name: str
    prim_path: str
    resolution: tuple[int, int]


CAMERAS = {
    # 头部 3 个普通针孔相机
    "head_front": CameraConfig(
        name="head_front",
        prim_path="/genie/head_link3/head_front_Camera",
        resolution=(640, 400),
    ),
    "head_left": CameraConfig(
        name="head_left",
        prim_path="/genie/head_link3/head_left_Camera",
        resolution=(640, 400),
    ),
    "head_right": CameraConfig(
        name="head_right",
        prim_path="/genie/head_link3/head_right_Camera",
        resolution=(640, 400),
    ),
    # 头部 3 个鱼眼相机
    "head_back_fisheye": CameraConfig(
        name="head_back_fisheye",
        prim_path="/genie/head_link3/head_back_fisheye",
        resolution=(640, 400),
    ),
    "head_left_fisheye": CameraConfig(
        name="head_left_fisheye",
        prim_path="/genie/head_link3/head_left_fisheye",
        resolution=(640, 400),
    ),
    "head_right_fisheye": CameraConfig(
        name="head_right_fisheye",
        prim_path="/genie/head_link3/head_right_fisheye",
        resolution=(640, 400),
    ),
    # 左右夹爪各 1 个针孔相机
    "gripper_left": CameraConfig(
        name="gripper_left",
        prim_path="/genie/gripper_l_base_link/Left_Camera",
        resolution=(1280, 1056),
    ),
    "gripper_right": CameraConfig(
        name="gripper_right",
        prim_path="/genie/gripper_r_base_link/Right_Camera",
        resolution=(1280, 1056),
    ),
}


@dataclass(frozen=True)
class SimulationConfig:
    """Isaac Sim 视觉示例的最小运行参数。"""

    physics_hz: int = 120
    rendering_hz: int = 30
    headless: bool = False
    renderer: str = "RaytracedLighting"
    warmup_steps: int = 120
    scene: str = "warehouse"  # warehouse 更适合目标检测；room 与第四章一致
    room_robot_position: tuple[float, float, float] = (0.0, 0.0, -0.01)
    warehouse_robot_position: tuple[float, float, float] = (-8.0, 13.0, -0.01)
    robot_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def robot_position(self) -> tuple[float, float, float]:
        return self.warehouse_robot_position if self.scene == "warehouse" else self.room_robot_position

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz


@dataclass(frozen=True)
class InferenceConfig:
    """YOLO 通用推理参数。"""

    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 640
    mask_threshold: float = 0.50
