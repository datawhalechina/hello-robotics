"""加载 room_1、G2、静态障碍和红蓝黄目标物体。"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    from .config import (
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        STATIC_OBSTACLES,
        TARGET_OBJECTS,
        Pose2D,
        SimulationConfig,
    )
except ImportError:
    from config import (
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        STATIC_OBSTACLES,
        TARGET_OBJECTS,
        Pose2D,
        SimulationConfig,
    )


class G2Chapter13Simulation:
    """第十三章最小 Isaac Sim 场景封装。"""

    def __init__(self, config: SimulationConfig) -> None:
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(f"找不到 G2 USD：{ROBOT_USD}")
        if not ROOM_USD.is_file():
            raise FileNotFoundError(f"找不到 room_1 USD：{ROOM_USD}")

        from isaacsim import SimulationApp

        self.config = config
        self.app = SimulationApp(
            {
                "headless": config.headless,
                "disable_viewport_updates": config.headless,
                "renderer": config.renderer,
                "limit_cpu_threads": 16,
            }
        )
        self.world = None
        self.robot = None
        self._enable_ros2_bridge()
        self._build_world()

    def _enable_ros2_bridge(self) -> None:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")
        self.app.update()
        print("[仿真] Isaac ROS 2 Bridge 已启用", flush=True)

    def _build_world(self) -> None:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=1.0 / self.config.rendering_hz,
        )
        add_reference_to_stage(str(ROOM_USD), ROOM_PRIM_PATH)
        self.world.scene.add_default_ground_plane()

        for group, objects in (("obstacle", STATIC_OBSTACLES), ("target", TARGET_OBJECTS)):
            for item in objects:
                self.world.scene.add(
                    FixedCuboid(
                        prim_path=f"/World/Chapter13/{group}_{item.name}",
                        name=f"chapter13_{group}_{item.name}",
                        position=np.asarray(item.center, dtype=np.float64),
                        scale=np.asarray(item.size, dtype=np.float64),
                        size=1.0,
                        color=np.asarray(item.color_rgb, dtype=np.float64),
                    )
                )

        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        start = self.config.robot_start
        SingleXFormPrim(
            prim_path=ROBOT_PRIM_PATH,
            position=np.array([start.x, start.y, -0.01], dtype=np.float64),
            orientation=np.array(
                [math.cos(start.yaw / 2.0), 0.0, 0.0, math.sin(start.yaw / 2.0)],
                dtype=np.float64,
            ),
        )

        self.world.play()
        self._set_viewport()
        time.sleep(0.5)
        for _ in range(self.config.warmup_steps):
            self.world.step(render=True)

        self.robot = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="G2_chapter13")
        self.world.scene.add(self.robot)
        self.robot.initialize()
        print(
            f"[仿真] 场景加载完成：G2 自由度 {self.robot.num_dof}，"
            f"语义目标 {len(TARGET_OBJECTS)} 个",
            flush=True,
        )

    def step(self) -> None:
        # 相机和 RTX LiDAR 都依赖渲染更新，headless 时也必须 render=True。
        self.world.step(render=True)

    def is_running(self) -> bool:
        return bool(self.app and self.app.is_running())

    @property
    def sim_time(self) -> float:
        return float(self.world.current_time) if self.world is not None else 0.0

    def get_pose2d(self) -> Pose2D:
        position, quaternion = self.robot.get_world_pose()
        w, x, y, z = (float(value) for value in quaternion)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return Pose2D(float(position[0]), float(position[1]), yaw)

    def close(self) -> None:
        if self.world is not None:
            self.world.stop()
            self.world = None
        if self.app is not None:
            self.app.close()
            self.app = None

    def _set_viewport(self) -> None:
        if self.config.headless:
            return
        try:
            from pxr import Gf
            from omni.kit.viewport.utility.camera_state import ViewportCameraState

            camera = ViewportCameraState("/OmniverseKit_Persp")
            camera.set_position_world(Gf.Vec3d(-4.8, -5.1, 3.4), True)
            camera.set_target_world(Gf.Vec3d(-1.2, -3.0, 0.4), True)
        except Exception as exc:
            print(f"[仿真] 视角设置失败（不影响任务）：{exc}", flush=True)
