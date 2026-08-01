"""MoveIt 2 案例的 Isaac Sim 场景加载器。"""

from __future__ import annotations

import numpy as np

try:
    from .config import ARM_BASE_PRIM_PATH, END_EFFECTOR_PRIM_PATH, ROBOT_PRIM_PATH, ROBOT_USD, ROOM_USD, SCENE_BOXES, SimulationConfig
except ImportError:
    from config import ARM_BASE_PRIM_PATH, END_EFFECTOR_PRIM_PATH, ROBOT_PRIM_PATH, ROBOT_USD, ROOM_USD, SCENE_BOXES, SimulationConfig


class G2MoveItSimulation:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.app = self.world = self.robot = None
        self.red_object = None
        self.red_attached = False
        self._build()

    def _build(self) -> None:
        from isaacsim import SimulationApp
        self.app = SimulationApp({
            "headless": self.config.headless,
            "disable_viewport_updates": self.config.headless,
            "renderer": self.config.renderer,
            "limit_cpu_threads": 16,
        })
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.ros2.bridge")
        self.app.update()

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        self.world = World(stage_units_in_meters=1.0, physics_dt=self.config.physics_dt,
                           rendering_dt=1.0 / self.config.rendering_hz)
        add_reference_to_stage(str(ROOM_USD), "/World")
        self.world.scene.add_default_ground_plane()
        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        SingleXFormPrim(ROBOT_PRIM_PATH, position=np.array([0.0, 0.0, -0.01]),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        self.world.play()
        for _ in range(self.config.warmup_steps):
            self.world.step(render=not self.config.headless)
        self.robot = SingleArticulation(ROBOT_PRIM_PATH, name="G2_chapter11_moveit")
        self.world.scene.add(self.robot)
        self.robot.initialize()
        self._create_scene()
        self._set_camera()
        for _ in range(30):
            self.world.step(render=not self.config.headless)
        print("[Isaac] MoveIt 2 教学桌面场景已加载", flush=True)

    @staticmethod
    def _quat_matrix(quaternion):
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
        ])

    def arm_base_pose(self):
        from isaacsim.core.prims import SingleXFormPrim
        return SingleXFormPrim(ARM_BASE_PRIM_PATH).get_world_pose()

    def local_to_world(self, point):
        position, quaternion = self.arm_base_pose()
        return np.asarray(position) + self._quat_matrix(quaternion) @ np.asarray(point)

    def _create_scene(self):
        from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
        _, orientation = self.arm_base_pose()
        for name, center, size, color in SCENE_BOXES:
            cube_class = VisualCuboid if name == "red_object" else FixedCuboid
            cube = cube_class(
                prim_path=f"/World/Chapter11/{name}", name=f"chapter11_{name}",
                position=self.local_to_world(center), orientation=np.asarray(orientation),
                scale=np.asarray(size), size=1.0, color=np.asarray(color),
            )
            self.world.scene.add(cube)
            if name == "red_object":
                self.red_object = cube

    def set_red_attached(self, attached=True):
        self.red_attached = attached

    def _update_red(self):
        if not self.red_attached or self.red_object is None:
            return
        from isaacsim.core.prims import SingleXFormPrim
        position, orientation = SingleXFormPrim(END_EFFECTOR_PRIM_PATH).get_world_pose()
        self.red_object.set_world_pose(position=np.asarray(position), orientation=np.asarray(orientation))

    def _set_camera(self):
        if self.config.headless:
            return
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([2.1, -2.5, 2.25]), target=np.array([0.55, -0.35, 1.0]),
                            camera_prim_path="/OmniverseKit_Persp")
        except Exception as exc:
            print(f"[Isaac] 相机设置失败：{exc}", flush=True)

    def step(self, render=None):
        self._update_red()
        # headless 模式也必须渲染传感器帧。
        self.world.step(render=True if render is None else render)

    def is_running(self):
        return bool(self.app and self.app.is_running())

    def close(self):
        if self.app is not None:
            self.app.close()
            self.app = None
