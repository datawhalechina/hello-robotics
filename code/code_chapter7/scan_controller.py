"""视觉扫描控制：复用第四章真实车轮控制器让机器人缓慢转动。"""

import sys

try:
    from .config import PROJECT_ROOT
except ImportError:
    from config import PROJECT_ROOT

# 直接执行 demo_xxx.py 时，确保能够导入兄弟目录 code_chapter4。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.code_chapter4.base_controller import G2BaseController
from code.code_chapter4.config import ControlLimits, RobotGeometry
from code.code_chapter4.kinematics import SwerveKinematics


class VisionScanController:
    """只暴露扫描所需的角速度，底层运动学仍由第四章实现。"""

    def __init__(self, articulation) -> None:
        geometry = RobotGeometry()
        kinematics = SwerveKinematics(
            geometry.wheel_positions,
            geometry.wheel_radius,
        )
        self.base = G2BaseController(articulation, kinematics, ControlLimits())

    def update(self, dt: float, angular_speed: float = 0.0) -> None:
        self.base.set_velocity(vx=0.0, vy=0.0, wz=angular_speed)
        self.base.update(dt)

    def stop(self) -> None:
        self.base.stop(center_steering=False)
