"""
启动应用、加载 room_1、创建地面、加载 G2、推进物理仿真。
控制算法全部位于 kinematics.py 和 base_controller.py。
"""

import math
import time

import numpy as np

try:
    from .config import (
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        SimulationConfig,
    )
    from .kinematics import Pose2D
except ImportError:
    from config import (
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        SimulationConfig,
    )
    from kinematics import Pose2D


class G2Simulation:
    """G2 底盘教学示例的 Isaac Sim 运行环境。"""

    def __init__(self, config: SimulationConfig) -> None:
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(f"找不到 G2 USD：{ROBOT_USD}")
        if not ROOM_USD.is_file():
            raise FileNotFoundError(f"找不到 room_1 USD：{ROOM_USD}")

        self.config = config
        self.app = None
        self.world = None
        self.robot = None

        # 必须先创建 SimulationApp，再导入大部分 Isaac Sim 模块。
        from isaacsim import SimulationApp

        self.app = SimulationApp(
            {
                "headless": config.headless,
                "disable_viewport_updates": config.headless,
                "renderer": config.renderer,
                "limit_cpu_threads": 16,
            }
        )
        print("[G2Simulation] Isaac Sim 应用已启动", flush=True)
        self._build_world()

    def _build_world(self) -> None:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        print("[G2Simulation] 正在创建物理世界", flush=True)
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=1.0 / self.config.rendering_hz,
        )
        # room_1 的 defaultPrim 是 /World。引用到 /World 后会加载完整仓库环境。
        add_reference_to_stage(str(ROOM_USD), ROOM_PRIM_PATH)
        print(f"[G2Simulation] room_1 场景已加载：{ROOM_USD}", flush=True)

        # 保留一个可靠的物理地面，做法与仓库原有 G2 示例一致。
        self.world.scene.add_default_ground_plane()
        print("[G2Simulation] 物理地面已创建", flush=True)

        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        print("[G2Simulation] G2 USD 已加入场景", flush=True)
        SingleXFormPrim(
            prim_path=ROBOT_PRIM_PATH,
            position=np.array(self.config.robot_position, dtype=np.float64),
            orientation=np.array(self.config.robot_orientation, dtype=np.float64),
        )

        # 先让 USD 和物理场景完成加载，再创建 Python articulation 句柄。
        # 这个顺序与本仓库现有 G2 示例一致，也避免 reset() 重置浮动底座。
        self.world.play()
        self._set_camera()
        # 大型 G2 USD 的材质和物理数据需要一点时间完成同步加载。
        time.sleep(1.0)
        for _ in range(self.config.warmup_steps):
            self.world.step(render=not self.config.headless)
        time.sleep(0.5)

        self.robot = SingleArticulation(
            prim_path=ROBOT_PRIM_PATH,
            name="G2_chapter4",
        )
        self.world.scene.add(self.robot)
        self.robot.initialize()
        print("[G2Simulation] G2 articulation 已注册", flush=True)

        print(f"[G2Simulation] G2 已加载：{self.robot.num_dof} 个自由度", flush=True)
        print(f"[G2Simulation] 物理频率：{self.config.physics_hz} Hz", flush=True)

    def step(self, render: bool | None = None) -> None:
        if render is None:
            render = not self.config.headless
        self.world.step(render=render)

    def is_running(self) -> bool:
        return bool(self.app and self.app.is_running())

    def get_pose2d(self) -> Pose2D:
        position, quaternion = self.robot.get_world_pose()
        return Pose2D(
            x=float(position[0]),
            y=float(position[1]),
            yaw=self._quaternion_to_yaw(quaternion),
        )

    def close(self) -> None:
        if self.app is not None:
            self.app.close()
            self.app = None

    def _set_camera(self) -> None:
        if self.config.headless:
            return
        try:
            from pxr import Gf
            from omni.kit.viewport.utility.camera_state import ViewportCameraState

            camera = ViewportCameraState("/OmniverseKit_Persp")
            camera.set_position_world(Gf.Vec3d(3.2, 3.2, 2.4), True)
            camera.set_target_world(Gf.Vec3d(0.0, 0.0, 0.5), True)
        except Exception as exc:  # 相机失败不影响物理与控制
            print(f"[G2Simulation] 视角设置失败，可忽略：{exc}")

    @staticmethod
    def _quaternion_to_yaw(quaternion) -> float:
        w, x, y, z = (float(value) for value in quaternion)
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(sin_yaw, cos_yaw)
