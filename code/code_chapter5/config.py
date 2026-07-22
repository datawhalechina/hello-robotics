from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = PROJECT_ROOT / "assets/robot/G2_omnipicker/robot.usda"
ROOM_USD = PROJECT_ROOT / "assets/background/room/room_1/background.usda"

ROBOT_PRIM_PATH = "/genie"
ROOM_PRIM_PATH = "/World"
ARM_BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/arm_base_link"
END_EFFECTOR_PRIM_PATHS = {
    "left": f"{ROBOT_PRIM_PATH}/gripper_l_center_link",
    "right": f"{ROBOT_PRIM_PATH}/gripper_r_center_link",
}

ARM_JOINT_NAMES = {
    "left": tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)),
    "right": tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)),
}

# 关节限位来自 G2 URDF，单位为 rad。
JOINT_LOWER_LIMITS = np.array(
    [-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708],
    dtype=np.float64,
)
JOINT_UPPER_LIMITS = np.array(
    [3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708],
    dtype=np.float64,
)
JOINT_VELOCITY_LIMITS = np.full(7, 3.1416, dtype=np.float64)

# 左右臂是镜像安装。若希望两侧关节在画面中表现为“同向动作”，
# J1/J3/J5/J7 的增量需要反号，J2/J4/J6 保持同号。
MIRRORED_JOINT_DELTA_SIGNS = np.array(
    [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64
)

# 在 arm_base_link 坐标系中，视觉同向的左右末端位置满足：
# right_position = [left_x, left_y, -left_z]。
# 注意 y 不反号；旧示例把 y、z 都反号，会让两侧奇数关节同号并表现为反向。
MIRRORED_END_EFFECTOR_POSITION_SIGNS = np.array(
    [1.0, 1.0, -1.0], dtype=np.float64
)

# 一个避开关节极限、适合开始教学实验的姿态。
HOME_JOINT_POSITIONS = np.array(
    [0.0, -0.35, 0.0, -1.10, 0.0, 0.35, 0.0], dtype=np.float64
)

# FK/IK 演示使用的左臂可达目标。单臂演示直接使用该姿态；双臂演示会
# 以 HOME 为中心生成右臂镜像姿态，再分别做 FK 和自编数值 IK。
DEMO_TARGET_JOINT_POSITIONS = np.array(
    [0.45, -0.55, 0.30, -1.35, 0.20, 0.50, -0.25], dtype=np.float64
)


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


@dataclass(frozen=True)
class IKConfig:
    """阻尼最小二乘 IK 参数。"""

    max_iterations: int = 300
    position_tolerance: float = 0.002       # m
    orientation_tolerance: float = 0.015    # rad
    damping: float = 0.04
    max_joint_step: float = 0.12            # 每次迭代的最大关节增量，rad
    orientation_weight: float = 0.35
    joint_center_gain: float = 0.03
