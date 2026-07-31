"""第八章统一配置。

先改这个文件，再观察雷达安装位置、滤波参数和建图结果的变化。
"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = PROJECT_ROOT / "assets/robot/G2_omnipicker/robot_fix.usda"
ROBOT_PRIM_PATH = "/genie"
WAREHOUSE_PRIM_PATH = "/background"
WAREHOUSE_ASSET = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
LIDAR_PARENT_PATH = "/genie/base_link"
OUTPUT_DIR = PROJECT_ROOT / "outputs/chapter8"


@dataclass(frozen=True)
class LidarMount:
    """一个 RTX LiDAR 在 base_link 下的固定外参。"""

    name: str
    prim_path: str
    frame_id: str
    topic: str
    translation: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


LEFT_LIDAR = LidarMount(
    name="chapter8_lidar_left",
    prim_path=f"{LIDAR_PARENT_PATH}/chapter8_lidar_left",
    frame_id="lidar_left",
    topic="/lidar/left/points",
    translation=(0.17, 0.21, 0.34),
)
RIGHT_LIDAR = LidarMount(
    name="chapter8_lidar_right",
    prim_path=f"{LIDAR_PARENT_PATH}/chapter8_lidar_right",
    frame_id="lidar_right",
    topic="/lidar/right/points",
    translation=(-0.23, -0.20, 0.34),
    # 绕 base_link 的 Z 轴旋转 180°，让逐帧扫描扇区与左雷达互补。
    orientation_wxyz=(0.0, 0.0, 0.0, 1.0),
)
LIDAR_MOUNTS = (LEFT_LIDAR, RIGHT_LIDAR)


@dataclass(frozen=True)
class SimulationConfig:
    headless: bool = False
    physics_hz: int = 60
    rendering_hz: int = 60
    warmup_steps: int = 30
    renderer: str = "RaytracedLighting"
    robot_z: float = -0.01

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz


@dataclass(frozen=True)
class LidarConfig:
    """Ouster OS1 RTX LiDAR 和单帧点云预处理参数。"""

    model: str = "OS1"
    profile: str = "OS1_REV6_128ch10hz1024res"
    min_range: float = 0.25
    max_range: float = 50.0
    max_range_margin: float = 0.0
    point_stride: int = 2
    scan_voxel_size: float = 0.10


@dataclass(frozen=True)
class Map3DConfig:
    voxel_size: float = 0.08
    min_z: float = 0.08
    max_z: float = 6.0
    ground_z: float = 0.0
    ground_margin: float = 0.06
    min_observations: int = 1
    max_voxels: int = 800_000


@dataclass(frozen=True)
class Map2DConfig:
    resolution: float = 0.05
    min_hit_z: float = 0.10
    max_hit_z: float = 3.0
    angular_resolution_deg: float = 1.0
    free_probability: float = 0.35
    occupied_probability: float = 0.70
    occupied_threshold: float = 0.65
    free_threshold: float = 0.25
    max_log_odds: float = 5.0


@dataclass(frozen=True)
class SlamConfig:
    """教学版前端/后端参数。

    odom_* 参数故意给里程计加入轻微系统误差，用于观察漂移和回环优化。
    设为 1、1、0 即可关闭人为漂移。
    """

    keyframe_distance: float = 0.22
    keyframe_yaw: float = 0.18
    odom_linear_scale: float = 1.012
    odom_lateral_scale: float = 0.994
    odom_yaw_scale: float = 1.008
    odom_yaw_bias_per_meter: float = 0.002
    odom_noise_std: float = 0.0005
    random_seed: int = 8
    loop_min_keyframe_gap: int = 24
    loop_search_radius: float = 2.0
    loop_descriptor_threshold: float = 0.34
    loop_icp_max_correspondence: float = 0.45
    loop_icp_max_rmse: float = 0.22
    loop_icp_min_inlier_ratio: float = 0.30
    optimize_iterations: int = 8
