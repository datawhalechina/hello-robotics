"""第十一章示例 1 的统一参数。

规划坐标系统一使用 ``arm_base_link``。G2 的该坐标系中：

- x：机器人前方；
- y：向下；
- z：机器人左方（右臂工作区通常 z < 0）。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = PROJECT_ROOT / "assets/robot/G2_omnipicker/robot.usda"
ROOM_USD = PROJECT_ROOT / "assets/background/room/room_1/background.usda"
ROBOT_PRIM_PATH = "/genie"
ARM_BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/arm_base_link"
END_EFFECTOR_PRIM_PATH = f"{ROBOT_PRIM_PATH}/gripper_r_center_link"
HEAD_DEPTH_CAMERA_PRIM_PATH = f"{ROBOT_PRIM_PATH}/head_link3/head_front_Camera"

RIGHT_ARM_JOINT_NAMES = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
GRIPPER_JOINT_NAMES = (
    "idx81_gripper_r_outer_joint1",
    "idx71_gripper_r_inner_joint1",
)

HOME_JOINT_POSITIONS = np.array(
    [0.0, -0.35, 0.0, -1.10, 0.0, 0.35, 0.0], dtype=np.float64
)
JOINT_LOWER_LIMITS = np.array(
    [-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708],
    dtype=np.float64,
)
JOINT_UPPER_LIMITS = np.array(
    [3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708],
    dtype=np.float64,
)
JOINT_VELOCITY_LIMITS = np.full(7, 1.25, dtype=np.float64)
JOINT_ACCELERATION_LIMITS = np.full(7, 2.2, dtype=np.float64)

# 红色物体的中心和抓取点，均在 arm_base_link 下。
RED_OBJECT_POSITION = np.array([0.56, 0.535, -0.43], dtype=np.float64)
RED_OBJECT_SIZE = np.array([0.075, 0.075, 0.075], dtype=np.float64)
PRE_GRASP_POSITION = RED_OBJECT_POSITION + np.array([0.0, -0.19, 0.0])
GRASP_POSITION = RED_OBJECT_POSITION + np.array([0.0, -0.065, 0.0])
LIFT_POSITION = RED_OBJECT_POSITION + np.array([-0.02, -0.27, 0.0])


@dataclass(frozen=True)
class BoxObstacle:
    """轴对齐包围盒；center/size 使用 arm_base_link 坐标。"""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]
    collision: bool = True

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)

    @property
    def size_array(self) -> np.ndarray:
        return np.asarray(self.size, dtype=np.float64)


# 桌面、阻挡物和不同颜色物体。红色物体是抓取目标，规划时不把它当障碍，
# 否则末端执行器无法进入抓取位姿。
SCENE_BOXES = (
    BoxObstacle("table_top", (0.72, 0.610, -0.28), (0.95, 0.08, 1.15), (0.48, 0.28, 0.12)),
    BoxObstacle("blocker", (0.59, 0.385, -0.630), (0.17, 0.37, 0.16), (0.95, 0.72, 0.08)),
    BoxObstacle("red_object", tuple(RED_OBJECT_POSITION), tuple(RED_OBJECT_SIZE), (0.90, 0.05, 0.05), False),
    BoxObstacle("green_object", (0.56, 0.535, -0.18), (0.075, 0.075, 0.075), (0.05, 0.75, 0.12)),
    BoxObstacle("blue_object", (0.62, 0.535, 0.04), (0.075, 0.075, 0.075), (0.05, 0.30, 0.92)),
)

# 机器人躯干的简化碰撞盒。右臂规划时只需防止手臂穿过躯干。
ROBOT_BODY_BOX = BoxObstacle(
    "robot_body", (-0.04, 0.05, 0.0), (0.48, 0.86, 0.50), (0.35, 0.35, 0.38)
)


@dataclass(frozen=True)
class PlannerConfig:
    step_size: float = 0.22
    edge_resolution: float = 0.045
    goal_bias: float = 0.18
    max_iterations: int = 5000
    random_seed: int = 11
    link_radius: float = 0.055
    # 不对障碍物做额外预膨胀；link_radius 只是机械臂自身物理半径。
    safety_margin: float = 0.0
    self_collision_distance: float = 0.085


@dataclass(frozen=True)
class TrajectoryConfig:
    sample_dt: float = 1.0 / 120.0
    shortcut_attempts: int = 180
    path_resolution: float = 0.08
    speed_scale: float = 0.55
    tracking_tolerance: float = 0.10
    abort_tolerance: float = 0.38


@dataclass(frozen=True)
class PerceptionConfig:
    """G2 机载相机的三维感知参数。"""

    head_resolution: tuple[int, int] = (960, 600)
    camera_hz: int = 15
    map_voxel_size: float = 0.04
    map_max_points: int = 600
    # arm_base_link 下的机械臂桌面工作区，点云只保留这里。
    workspace_min: tuple[float, float, float] = (0.12, 0.12, -0.90)
    workspace_max: tuple[float, float, float] = (1.15, 0.66, 0.25)


@dataclass(frozen=True)
class SimulationConfig:
    physics_hz: int = 120
    rendering_hz: int = 60
    warmup_steps: int = 120
    headless: bool = False
    enable_rviz: bool = True
    renderer: str = "RaytracedLighting"

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz
