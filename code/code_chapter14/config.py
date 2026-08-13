"""第十四章统一配置：场景、G2关节、数据和checkpoint路径。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CHAPTER_ROOT = HERE
CODE_ROOT = CHAPTER_ROOT.parent
# OpenPI、模型权重、数据和训练产物全部放在本章目录中，避免引用外部副本。
OPENPI_ROOT = CHAPTER_ROOT / "openpi"
CHECKPOINT_ROOT = CHAPTER_ROOT / "checkpoints"

# 机器人和背景USD是各章节共用资源，统一从code/assets读取，不在本章重复保存。
SHARED_ASSET_ROOT = CODE_ROOT / "assets"
ROBOT_USD = SHARED_ASSET_ROOT / "robot/G2_omnipicker/robot.usda"
ROOM_USD = SHARED_ASSET_ROOT / "background/room/room_1/background.usda"

ROBOT_PRIM_PATH = "/genie"
ROOM_PRIM_PATH = "/World"
ARM_BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/arm_base_link"

LEFT_ARM_JOINTS = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
WAIST_JOINTS = tuple(f"idx0{i}_body_joint{i}" for i in range(1, 6))
LEFT_GRIPPER_JOINTS = (
    "idx41_gripper_l_outer_joint1",
    "idx31_gripper_l_inner_joint1",
)
RIGHT_GRIPPER_JOINTS = (
    "idx81_gripper_r_outer_joint1",
    "idx71_gripper_r_inner_joint1",
)

HEAD_CAMERA_PATH = f"{ROBOT_PRIM_PATH}/head_link3/head_front_Camera"
LEFT_WRIST_CAMERA_PATH = f"{ROBOT_PRIM_PATH}/gripper_l_base_link/Left_Camera"
RIGHT_WRIST_CAMERA_PATH = f"{ROBOT_PRIM_PATH}/gripper_r_base_link/Right_Camera"

ARM_LOWER_7 = np.array(
    [-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708]
)
ARM_UPPER_7 = np.array(
    [3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708]
)
ARM_LOWER_14 = np.tile(ARM_LOWER_7, 2)
ARM_UPPER_14 = np.tile(ARM_UPPER_7, 2)

# 与公开G2数据使用的双臂初始姿态一致。
HOME_ARMS_14 = np.array(
    [
        0.739033, -0.717023, -1.524419, -1.537612, 0.278110, -0.925845, -0.839257,
        -0.739033, -0.717023, 1.524419, -1.537612, -0.278110, -0.925845, 0.839257,
    ],
    dtype=np.float64,
)
GRIPPER_OPEN_RAD = 0.785
GRIPPER_CLOSED_RAD = 0.0
G2_ARM_GRIPPER_DIM = 16
G2_WITH_WAIST_DIM = 21

CHECKPOINTS = {
    "instruction": CHECKPOINT_ROOT / "instruction_and_robust_pi05",
    "manipulation": CHECKPOINT_ROOT / "manipulation_pi05",
    "base": CHECKPOINT_ROOT / "pi05_base",
    "droid": CHECKPOINT_ROOT / "pi05_droid",
}

# 权重不存在时可手动取消下面命令的注释；程序不会自动下载。
# 在code_chapter14目录执行；脚本固定下载到本章checkpoints/：
# bash download_checkpoint.sh instruction_and_robust_pi05
# bash download_checkpoint.sh manipulation_pi05

RAW_DATA_DIR = HERE / "data/raw"
LEROBOT_HOME = HERE / "data/lerobot"
REPO_ID = "chapter14/g2_color_blocks"
TRAIN_ASSETS_DIR = HERE / "assets"
TRAIN_CHECKPOINT_DIR = CHECKPOINT_ROOT
TRAIN_CONFIG_NAME = "pi05_g2_color_blocks"
DATASET_FPS = 10

BLOCK_COLORS = {
    "red": (0.90, 0.05, 0.05),
    "green": (0.05, 0.75, 0.12),
    "blue": (0.05, 0.30, 0.92),
}


def checkpoint_path(model: str, custom: str | None = None) -> Path:
    """解析本章内部权重，不允许意外加载章节目录外的checkpoint。"""
    checkpoint_root = CHECKPOINT_ROOT.resolve()
    if custom:
        raw = Path(custom).expanduser()
        if raw.is_absolute():
            path = raw.resolve()
        else:
            # 推荐传 checkpoints/...；无论从项目根目录还是本章目录启动都能解析。
            cwd_path = raw.resolve()
            chapter_path = (CHAPTER_ROOT / raw).resolve()
            path = cwd_path if cwd_path.is_relative_to(checkpoint_root) else chapter_path
    else:
        path = CHECKPOINTS.get(model)
        if path is None:
            raise ValueError(f"未知模型类型：{model}")
        path = path.resolve()
    if not path.is_relative_to(checkpoint_root):
        raise ValueError(
            f"checkpoint必须位于本章目录：{checkpoint_root}，实际为：{path}"
        )
    if not (path / "params").is_dir():
        raise FileNotFoundError(
            f"找不到模型参数：{path / 'params'}。下载命令已注释保留在config.py。"
        )
    return path


@dataclass(frozen=True)
class SimulationConfig:
    physics_hz: int = 120
    rendering_hz: int = 30
    headless: bool = False
    renderer: str = "RaytracedLighting"
    warmup_steps: int = 90
    image_size: tuple[int, int] = (320, 240)

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz


@dataclass(frozen=True)
class TaskConfig:
    """第十一章桌面坐标；所有位置都位于arm_base_link坐标系。"""

    table_arm_center: tuple[float, float, float] = (0.72, 0.610, -0.28)
    table_size: tuple[float, float, float] = (0.95, 0.08, 1.15)
    block_size: float = 0.075
    block_arm_positions: tuple[tuple[float, float, float], ...] = (
        (0.56, 0.535, -0.43),
        (0.56, 0.535, -0.18),
        (0.56, 0.535, 0.04),
    )
    box_arm_position: tuple[float, float, float] = (0.59, 0.535, -0.63)
    box_inner_size: tuple[float, float] = (0.24, 0.20)
    box_wall_height: float = 0.11
    box_wall_thickness: float = 0.018
    box_floor_thickness: float = 0.024
    max_replans: int = 180
    # 夹爪中心位于物块中心上方的距离。G2夹指会继续向下延伸。
    pregrasp_clearance: float = 0.19
    # 三个位置的IK姿态略有不同，因此分别给出红、绿、蓝的物理夹取深度。
    grasp_clearances: tuple[float, float, float] = (0.015, 0.005, -0.005)
    place_clearance: float = 0.13
    block_mass: float = 0.05
    contact_offset: float = 0.003
    static_friction: float = 1.8
    dynamic_friction: float = 1.5

    @property
    def table_top_arm_y(self) -> float:
        return self.table_arm_center[1] - self.table_size[1] / 2


@dataclass(frozen=True)
class ControlConfig:
    # 0表示执行模型返回的完整动作块，与Genie Sim官方G2 baseline一致。
    execute_chunk: int = 0
    physics_steps_per_action: int = 4
    home_duration_s: float = 3.0
