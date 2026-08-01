"""Isaac Sim/MoveIt 2 桥接端配置。"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = PROJECT_ROOT / "assets/robot/G2_omnipicker/robot.usda"
ROOM_USD = PROJECT_ROOT / "assets/background/room/room_1/background.usda"
ROBOT_PRIM_PATH = "/genie"
ARM_BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/arm_base_link"
END_EFFECTOR_PRIM_PATH = f"{ROBOT_PRIM_PATH}/gripper_r_center_link"
HEAD_DEPTH_CAMERA_PRIM_PATH = f"{ROBOT_PRIM_PATH}/head_link3/head_front_Camera"
RIGHT_ARM_JOINT_NAMES = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
GRIPPER_JOINT_NAMES = ("idx81_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1")

# 红色物块位姿已知，单位为 arm_base_link 下的米。
RED_OBJECT_POSITION = (0.56, 0.535, -0.43)
RED_OBJECT_SIZE = (0.075, 0.075, 0.075)

# 与示例 1 完全相同的 arm_base_link 场景。
SCENE_BOXES = (
    ("table_top", (0.72, 0.610, -0.28), (0.95, 0.08, 1.15), (0.48, 0.28, 0.12)),
    ("blocker", (0.59, 0.385, -0.630), (0.17, 0.37, 0.16), (0.95, 0.72, 0.08)),
    ("red_object", RED_OBJECT_POSITION, RED_OBJECT_SIZE, (0.90, 0.05, 0.05)),
    ("green_object", (0.56, 0.535, -0.18), (0.075, 0.075, 0.075), (0.05, 0.75, 0.12)),
    ("blue_object", (0.62, 0.535, 0.04), (0.075, 0.075, 0.075), (0.05, 0.30, 0.92)),
)


@dataclass(frozen=True)
class SimulationConfig:
    physics_hz: int = 120
    rendering_hz: int = 60
    warmup_steps: int = 120
    headless: bool = False
    renderer: str = "RaytracedLighting"

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz


@dataclass(frozen=True)
class PerceptionConfig:
    head_resolution: tuple[int, int] = (960, 600)
    camera_hz: int = 15
    map_voxel_size: float = 0.04
    map_publish_hz: float = 1.0
    # 运动期间暂停 OctoMap 帧，避免 RGB-D/关节状态一帧延迟形成机器人残影。
    map_arm_stationary_velocity: float = 0.08
    map_max_points: int = 5000
    # 仅用于删除机器人自身深度点，不会膨胀环境障碍物。
    robot_self_filter_tolerance: float = 0.008
    workspace_min: tuple[float, float, float] = (0.12, 0.12, -0.90)
    workspace_max: tuple[float, float, float] = (1.15, 0.66, 0.25)
