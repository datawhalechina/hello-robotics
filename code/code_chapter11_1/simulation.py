"""第十一章统一桌面抓取场景的 Isaac Sim 加载器。"""

from __future__ import annotations

import time

import numpy as np

try:
    from .config import (
        ARM_BASE_PRIM_PATH,
        END_EFFECTOR_PRIM_PATH,
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_USD,
        SCENE_BOXES,
        SimulationConfig,
    )
except ImportError:
    from config import ARM_BASE_PRIM_PATH, END_EFFECTOR_PRIM_PATH, ROBOT_PRIM_PATH, ROBOT_USD, ROOM_USD, SCENE_BOXES, SimulationConfig


class G2ManipulationSimulation:
    def __init__(self, config: SimulationConfig) -> None:
        if not ROBOT_USD.is_file() or not ROOM_USD.is_file():
            raise FileNotFoundError("G2 或 room_1 USD 不存在")
        self.config = config
        self.app = self.world = self.robot = None
        self.red_object = None
        self.red_attached = False
        self._build()

    def _build(self) -> None:
        from isaacsim import SimulationApp

        self.app = SimulationApp(
            {
                "headless": self.config.headless,
                "disable_viewport_updates": self.config.headless,
                "renderer": self.config.renderer,
                "limit_cpu_threads": 16,
            }
        )
        if self.config.enable_rviz:
            try:
                from isaacsim.core.utils.extensions import enable_extension
                enable_extension("isaacsim.ros2.bridge")
                self.app.update()
            except Exception as exc:
                print(f"[Simulation] ROS 2 Bridge 启用失败：{exc}", flush=True)

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=1.0 / self.config.rendering_hz,
        )
        add_reference_to_stage(str(ROOM_USD), "/World")
        self.world.scene.add_default_ground_plane()
        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        SingleXFormPrim(
            prim_path=ROBOT_PRIM_PATH,
            position=np.array([0.0, 0.0, -0.01]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.world.play()
        for _ in range(self.config.warmup_steps):
            self.world.step(render=not self.config.headless)
        self.robot = SingleArticulation(ROBOT_PRIM_PATH, name="G2_chapter11_1")
        self.world.scene.add(self.robot)
        self.robot.initialize()
        self._create_table_scene()
        self._set_camera()
        for _ in range(30):
            self.world.step(render=not self.config.headless)
        print("[Simulation] 已加载桌子、三色物体、黄色阻挡物和 G2", flush=True)

    @staticmethod
    def _quaternion_to_matrix(quaternion) -> np.ndarray:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])

    def arm_base_pose(self):
        from isaacsim.core.prims import SingleXFormPrim
        return SingleXFormPrim(ARM_BASE_PRIM_PATH).get_world_pose()

    def local_to_world(self, position) -> np.ndarray:
        base_position, base_quaternion = self.arm_base_pose()
        return np.asarray(base_position) + self._quaternion_to_matrix(base_quaternion) @ np.asarray(position)

    def _create_table_scene(self) -> None:
        from isaacsim.core.api.objects import FixedCuboid, VisualCuboid

        _, orientation = self.arm_base_pose()
        for box in SCENE_BOXES:
            cube_class = VisualCuboid if box.name == "red_object" else FixedCuboid
            cube = cube_class(
                prim_path=f"/World/Chapter11/{box.name}",
                name=f"chapter11_{box.name}",
                position=self.local_to_world(box.center_array),
                orientation=np.asarray(orientation),
                scale=box.size_array,
                size=1.0,
                color=np.asarray(box.color),
            )
            self.world.scene.add(cube)
            if box.name == "red_object":
                self.red_object = cube

    def attach_red_object(self) -> None:
        self.red_attached = True
        self._update_attached_object()

    def _update_attached_object(self) -> None:
        if not self.red_attached or self.red_object is None:
            return
        from isaacsim.core.prims import SingleXFormPrim
        position, orientation = SingleXFormPrim(END_EFFECTOR_PRIM_PATH).get_world_pose()
        self.red_object.set_world_pose(position=np.asarray(position), orientation=np.asarray(orientation))

    def _set_camera(self) -> None:
        if self.config.headless:
            return
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(
                eye=np.array([2.1, -2.5, 2.25]),
                target=np.array([0.55, -0.35, 1.0]),
                camera_prim_path="/OmniverseKit_Persp",
            )
        except Exception as exc:
            print(f"[Simulation] 相机设置失败：{exc}", flush=True)

    def step(self, render: bool | None = None) -> None:
        self._update_attached_object()
        # headless 模式也必须渲染传感器帧；否则深度/RGB annotator 不更新。
        self.world.step(render=True if render is None else render)

    def is_running(self) -> bool:
        return bool(self.app and self.app.is_running())

    def close(self) -> None:
        if self.app is not None:
            self.app.close()
            self.app = None
