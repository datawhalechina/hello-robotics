"""G2 三色物块采集与 ACoT-VLA 闭环评测配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHAPTER_ROOT = Path(__file__).resolve().parent
CODE_ROOT = CHAPTER_ROOT.parent
OPENPI_ROOT = CHAPTER_ROOT / "acotvla"
ASSETS_ROOT = CODE_ROOT / "assets"
DATA_ROOT = CHAPTER_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
RESULTS_ROOT = CHAPTER_ROOT / "results"

ROBOT_USD = ASSETS_ROOT / "robot/G2_omnipicker/robot.usda"
ROBOT_PRIM_PATH = "/genie"
ARM_BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/arm_base_link"
HEAD_CAMERA_PATH = f"{ROBOT_PRIM_PATH}/head_link3/head_front_Camera"
LEFT_WRIST_CAMERA_PATH = f"{ROBOT_PRIM_PATH}/gripper_l_base_link/Left_Camera"
RIGHT_WRIST_CAMERA_PATH = f"{ROBOT_PRIM_PATH}/gripper_r_base_link/Right_Camera"

LEFT_ARM_JOINTS = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
WAIST_JOINTS = tuple(f"idx0{i}_body_joint{i}" for i in range(1, 6))
LEFT_GRIPPER_JOINTS = ("idx41_gripper_l_outer_joint1", "idx31_gripper_l_inner_joint1")
RIGHT_GRIPPER_JOINTS = ("idx81_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1")

ARM_LOWER_7 = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708])
ARM_UPPER_7 = np.array([3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708])
ARM_LOWER_14 = np.tile(ARM_LOWER_7, 2)
ARM_UPPER_14 = np.tile(ARM_UPPER_7, 2)
HOME_ARMS_14 = np.array(
    [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.278110,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.278110,
        -0.925845,
        0.839257,
    ],
    dtype=np.float64,
)
GRIPPER_OPEN_RAD = 0.785
GRIPPER_CLOSED_RAD = 0.0
G2_ARM_GRIPPER_DIM = 16
G2_WITH_WAIST_DIM = 21

COLORS = ("red", "green", "blue")
BLOCK_COLORS = {
    "red": (0.90, 0.05, 0.05),
    "green": (0.05, 0.75, 0.12),
    "blue": (0.05, 0.30, 0.92),
}
DATASET_FPS = 30


def demonstration_dir() -> Path:
    return RAW_DATA_DIR


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
    table_arm_center: tuple[float, float, float] = (0.72, 0.610, -0.28)
    table_size: tuple[float, float, float] = (0.95, 0.08, 1.15)
    block_size: float = 0.050
    block_arm_positions: tuple[tuple[float, float, float], ...] = (
        (0.56, 0.545, -0.43),
        (0.56, 0.545, -0.18),
        (0.56, 0.545, 0.04),
    )
    box_arm_position: tuple[float, float, float] = (0.59, 0.535, -0.63)
    box_inner_size: tuple[float, float] = (0.24, 0.20)
    box_wall_height: float = 0.11
    box_wall_thickness: float = 0.018
    box_floor_thickness: float = 0.024
    max_replans: int = 180
    pregrasp_clearance: float = 0.18
    grasp_clearances: tuple[float, float, float] = (0.010, 0.000, 0.000)
    place_clearance: float = 0.13
    block_mass: float = 0.025
    contact_offset: float = 0.002
    static_friction: float = 1.8
    dynamic_friction: float = 1.5

    @property
    def table_top_arm_y(self) -> float:
        return self.table_arm_center[1] - self.table_size[1] / 2


@dataclass(frozen=True)
class ControlConfig:
    execute_chunk: int = 8
    physics_steps_per_action: int = 4
    home_duration_s: float = 3.0
