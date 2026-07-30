"""第七章使用的 Isaac Sim 环境封装。

本文件只负责启动应用、加载教学场景、加载 G2 和推进仿真。
相机采集、图像处理和模型推理分别放在其他文件中。
"""

import time

import numpy as np

try:
    from .config import (
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        WAREHOUSE_ASSET,
        WAREHOUSE_PRIM_PATH,
        SimulationConfig,
    )
except ImportError:  # 支持直接执行本目录中的示例
    from config import (
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        WAREHOUSE_ASSET,
        WAREHOUSE_PRIM_PATH,
        SimulationConfig,
    )


class G2VisionSimulation:
    """G2 视觉教学示例的最小仿真环境。"""

    def __init__(self, config: SimulationConfig) -> None:
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(f"找不到 G2 USD：{ROBOT_USD}")
        if config.scene not in {"room", "warehouse"}:
            raise ValueError("scene 必须是 room 或 warehouse")
        if config.scene == "room" and not ROOM_USD.is_file():
            raise FileNotFoundError(f"找不到 room_1 USD：{ROOM_USD}")
        if config.physics_hz <= 0 or config.rendering_hz <= 0:
            raise ValueError("physics_hz 和 rendering_hz 必须大于 0")

        self.config = config
        self.app = None
        self.world = None
        self.robot = None

        # Isaac Sim 要求先创建 SimulationApp，再导入大部分 isaacsim 模块。
        from isaacsim import SimulationApp

        self.app = SimulationApp(
            {
                "headless": config.headless,
                "renderer": config.renderer,
                "limit_cpu_threads": 16,
            }
        )
        print("[G2VisionSimulation] Isaac Sim 应用已启动", flush=True)
        try:
            self._build_world()
        except Exception:
            self.close()
            raise

    def _build_world(self) -> None:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=1.0 / self.config.rendering_hz,
        )
        scene_description = self._load_scene(add_reference_to_stage)
        self.world.scene.add_default_ground_plane()

        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        SingleXFormPrim(
            prim_path=ROBOT_PRIM_PATH,
            position=np.asarray(self.config.robot_position, dtype=np.float64),
            orientation=np.asarray(self.config.robot_orientation, dtype=np.float64),
        )

        self.world.play()
        self._set_viewport_camera()
        time.sleep(1.0)
        for _ in range(self.config.warmup_steps):
            # RTX 相机在无窗口模式下也需要渲染步。
            self.world.step(render=True)
        time.sleep(0.5)

        self.robot = SingleArticulation(
            prim_path=ROBOT_PRIM_PATH,
            name="G2_chapter7",
        )
        self.world.scene.add(self.robot)
        self.robot.initialize()

        print(f"[G2VisionSimulation] 场景：{scene_description}", flush=True)
        print(f"[G2VisionSimulation] G2：{self.robot.num_dof} 个自由度", flush=True)

    def _load_scene(self, add_reference_to_stage) -> str:
        if self.config.scene == "room":
            add_reference_to_stage(str(ROOM_USD), ROOM_PRIM_PATH)
            return str(ROOM_USD)

        from isaacsim.storage.native import get_assets_root_path

        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("找不到 Isaac Sim Assets 根路径，无法加载 warehouse")
        warehouse = assets_root + WAREHOUSE_ASSET
        add_reference_to_stage(warehouse, WAREHOUSE_PRIM_PATH)
        return warehouse

    def step(self, render: bool = True) -> None:
        """推进一步。视觉任务应保持 render=True。"""
        self.world.step(render=render)

    def is_running(self) -> bool:
        return bool(self.app and self.app.is_running())

    def close(self) -> None:
        if self.app is not None:
            self.app.close()
            self.app = None

    def _set_viewport_camera(self) -> None:
        if self.config.headless:
            return
        try:
            from pxr import Gf
            from omni.kit.viewport.utility.camera_state import ViewportCameraState

            camera = ViewportCameraState("/OmniverseKit_Persp")
            camera.set_position_world(Gf.Vec3d(3.2, 3.2, 2.4), True)
            camera.set_target_world(Gf.Vec3d(0.0, 0.0, 0.7), True)
        except Exception as exc:
            print(f"[G2VisionSimulation] 视角设置失败，可忽略：{exc}")
