"""第九章参数。米、弧度；世界 Z 向上；RGB-D 使用 OpenCV 光学坐标。"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ROBOT_USD = ROOT / "assets/robot/G2_omnipicker/robot.usda"
ROBOT_PRIM_PATH = "/genie"
ARM_BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/arm_base_link"
END_EFFECTOR_PRIM_PATHS = {a: f"{ROBOT_PRIM_PATH}/gripper_{a[0]}_center_link" for a in ("left", "right")}
ARM_JOINT_NAMES = {a: tuple(f"idx{n}{i}_arm_{a[0]}_joint{i}" for i in range(1, 8)) for a, n in (("left", 2), ("right", 6))}
JOINT_LOWER_LIMITS = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708])
JOINT_UPPER_LIMITS = np.array([3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708])
JOINT_VELOCITY_LIMITS = np.full(7, 3.1416)
MIRRORED_JOINT_DELTA_SIGNS = np.array([-1, 1, -1, 1, -1, 1, -1])
HOME_LEFT = np.array([.739033, -.717023, -1.524419, -1.537612, .278110, -.925845, -.839257])
HOME_RIGHT = np.array([-.739033, -.717023, 1.524419, -1.537612, -.278110, -.925845, .839257])
GRIPPER_JOINT = "idx81_gripper_r_outer_joint1"  # 只驱动主关节，其他指节由 USD mimic/闭环约束带动
GRIPPER_OPEN, GRIPPER_CLOSED = .785, 0.0
# 任务区域先验：生成场景和视觉过滤共享边界，不能让待抓物体出生在放置区。
TRAY_EXCLUSION_HALF_SIZE = np.array([.12, .11])
PICK_SLOTS = tuple((x, z) for x in (.34, .46) for z in (-.32, -.16, 0.))

@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 350
    position_tolerance: float = .003
    orientation_tolerance: float = .025
    damping: float = .025
    max_joint_step: float = .12
    orientation_weight: float = .35
    joint_center_gain: float = .005

@dataclass(frozen=True)
class SimulationConfig:
    headless: bool = False
    physics_hz: int = 120
    rendering_hz: int = 30
    resolution: tuple = (640, 480)
    @property
    def physics_dt(self):
        return 1 / self.physics_hz

@dataclass(frozen=True)
class GripperGeometry:
    """夹爪简化碰撞盒；实测标定由 simulation.calibrate_gripper 更新。

    抓取坐标 G 原点为 center_link，X 闭合方向，+Z 从掌部朝指尖。
    指尖/掌部范围相对 G；不是其他品牌夹爪的参数。
    """
    max_width: float = .085
    finger_thickness: float = .014
    finger_depth: float = .028
    finger_z_min: float = -.055
    finger_z_max: float = .016
    palm_z_min: float = -.14
    palm_z_max: float = -.055
    margin: float = .004

@dataclass(frozen=True)
class PlannerConfig:
    voxel: float = .003
    min_points: int = 40
    table_clearance: float = .008
    pregrasp: float = .12
    lift: float = .16
    max_tilt_deg: float = 25
    min_width: float = .014
    max_candidates: int = 24

# 小物体尺寸刻意限制在 G2 小夹爪适用范围。程序生成教学外形，不冒称扫描资产。
@dataclass(frozen=True)
class ObjectSpec:
    name: str
    prompt: str
    shape: str
    size: tuple  # X,Y,Z 包围盒（m）
    color: tuple
    mass: float = .035

OBJECTS = (
    ObjectSpec("apple", "a small red apple", "apple", (.050, .048, .050), (.78, .05, .025)),
    ObjectSpec("orange", "an orange fruit", "sphere", (.047, .047, .047), (.95, .32, .015)),
    ObjectSpec("lemon", "a yellow lemon", "ellipsoid", (.066, .039, .039), (.92, .80, .025)),
    ObjectSpec("tomato", "a red tomato", "sphere", (.046, .046, .038), (.90, .07, .035)),
    ObjectSpec("potato", "a small potato", "ellipsoid", (.068, .043, .040), (.49, .31, .12)),
    ObjectSpec("soap", "a bar of soap", "box", (.070, .038, .027), (.82, .58, .71)),
    ObjectSpec("sponge", "a kitchen sponge", "box", (.070, .040, .030), (.93, .78, .09)),
    ObjectSpec("tea_box", "a small cardboard tea box", "box", (.060, .040, .055), (.17, .46, .22)),
    ObjectSpec("can", "a small soda can", "can", (.040, .040, .073), (.70, .10, .04)),
    ObjectSpec("bottle", "a small opaque plastic bottle", "bottle", (.038, .038, .085), (.10, .35, .70)),
    ObjectSpec("cup", "a small cup", "cup", (.050, .050, .060), (.78, .82, .90)),
    ObjectSpec("spool", "a spool of thread", "spool", (.045, .045, .050), (.20, .62, .60)),
    ObjectSpec("wood_block", "a wooden toy block", "box", (.042, .042, .042), (.62, .38, .16)),
    ObjectSpec("eraser", "a rectangular eraser", "box", (.060, .026, .026), (.32, .53, .80)),
    ObjectSpec("battery", "a cylindrical battery", "can", (.024, .024, .050), (.12, .12, .13)),
)
