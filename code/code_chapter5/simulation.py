import time

import numpy as np

try:
    from .config import (
        ARM_BASE_PRIM_PATH,
        END_EFFECTOR_PRIM_PATHS,
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        SimulationConfig,
    )
except ImportError:
    from config import (
        ARM_BASE_PRIM_PATH,
        END_EFFECTOR_PRIM_PATHS,
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        SimulationConfig,
    )


class G2Simulation:
    """G2 机械臂教学示例的仿真环境。"""

    def __init__(self, config: SimulationConfig) -> None:
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(f"找不到 G2 USD：{ROBOT_USD}")
        if not ROOM_USD.is_file():
            raise FileNotFoundError(f"找不到 room_1 USD：{ROOM_USD}")

        self.config = config
        self.app = None
        self.world = None
        self.robot = None

        # Isaac Sim 要求 SimulationApp 在其他 isaacsim 模块之前创建。
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

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=1.0 / self.config.rendering_hz,
        )
        add_reference_to_stage(str(ROOM_USD), ROOM_PRIM_PATH)
        self.world.scene.add_default_ground_plane()
        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        SingleXFormPrim(
            prim_path=ROBOT_PRIM_PATH,
            position=np.asarray(self.config.robot_position, dtype=np.float64),
            orientation=np.asarray(self.config.robot_orientation, dtype=np.float64),
        )

        self.world.play()
        self._set_camera()
        time.sleep(1.0)
        for _ in range(self.config.warmup_steps):
            self.world.step(render=not self.config.headless)
        time.sleep(0.5)

        self.robot = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="G2_chapter5")
        self.world.scene.add(self.robot)
        self.robot.initialize()
        print(f"[G2Simulation] room_1 与 G2 已加载：{self.robot.num_dof} 个自由度", flush=True)

    def step(self, render: bool | None = None) -> None:
        self.world.step(render=not self.config.headless if render is None else render)

    def is_running(self) -> bool:
        return bool(self.app and self.app.is_running())

    def get_relative_link_pose(self, child_prim_path: str):
        """读取 child 相对 arm_base_link 的位姿，便于核对自编 FK。"""
        from pxr import Usd, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        base_prim = stage.GetPrimAtPath(ARM_BASE_PRIM_PATH)
        child_prim = stage.GetPrimAtPath(child_prim_path)
        if not base_prim.IsValid() or not child_prim.IsValid():
            raise RuntimeError(f"无法读取 Prim：{ARM_BASE_PRIM_PATH} 或 {child_prim_path}")

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        base_world = np.asarray(cache.GetLocalToWorldTransform(base_prim), dtype=np.float64).T
        child_world = np.asarray(cache.GetLocalToWorldTransform(child_prim), dtype=np.float64).T
        relative = np.linalg.inv(base_world) @ child_world
        return relative[:3, 3].copy(), relative[:3, :3].copy()

    def get_end_effector_pose(self, arm: str = "right"):
        return self.get_relative_link_pose(END_EFFECTOR_PRIM_PATHS[arm])

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
            camera.set_position_world(Gf.Vec3d(2.4, 2.4, 1.8), True)
            camera.set_target_world(Gf.Vec3d(0.0, 0.0, 0.9), True)
        except Exception as exc:
            print(f"[G2Simulation] 视角设置失败，继续运行：{exc}", flush=True)
