"""第十章 Isaac Sim 场景加载器。

场景中的静态障碍与 maps/chapter10_2_map.yaml 一一对应；移动障碍仅用于演示
局部规划和动态避障，不写入静态地图。
"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    from .config import (
        DynamicObstacleConfig,
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        STATIC_OBSTACLES,
        SimulationConfig,
        CircleObstacle,
        Pose2D,
    )
except ImportError:
    from config import (
        DynamicObstacleConfig,
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        ROOM_PRIM_PATH,
        ROOM_USD,
        STATIC_OBSTACLES,
        SimulationConfig,
        CircleObstacle,
        Pose2D,
    )


class G2NavigationSimulation:
    def __init__(self, config: SimulationConfig, enable_ros2: bool = True) -> None:
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(f"找不到 G2 USD：{ROBOT_USD}")
        if not ROOM_USD.is_file():
            raise FileNotFoundError(f"找不到 room_1 USD：{ROOM_USD}")
        self.config = config
        self.dynamic_config = DynamicObstacleConfig()
        self.sim_time = 0.0
        self.app = None
        self.world = None
        self.robot = None
        self.dynamic_visual = None

        from isaacsim import SimulationApp

        self.app = SimulationApp(
            {
                "headless": config.headless,
                "disable_viewport_updates": config.headless,
                "renderer": config.renderer,
                "limit_cpu_threads": 16,
            }
        )
        if enable_ros2:
            self._enable_ros2_bridge()
        self._build_world()

    def _enable_ros2_bridge(self) -> None:
        try:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.ros2.bridge")
            # 扩展启用是异步的；先推进一帧，再让后续代码导入 Isaac 自带 rclpy。
            self.app.update()
            print("[Simulation] ROS 2 Bridge 已启用", flush=True)
        except Exception as exc:
            print(f"[Simulation] ROS 2 Bridge 启用失败，RViz 功能将不可用：{exc}", flush=True)

    def _build_world(self) -> None:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=1.0 / self.config.rendering_hz,
        )
        add_reference_to_stage(str(ROOM_USD), ROOM_PRIM_PATH)
        self.world.scene.add_default_ground_plane()

        for obstacle in STATIC_OBSTACLES:
            cube = FixedCuboid(
                prim_path=f"/World/Chapter10_2Obstacles/{obstacle.name}",
                name=f"chapter10_2_{obstacle.name}",
                position=np.array(
                    [obstacle.center_x, obstacle.center_y, obstacle.height / 2.0],
                    dtype=np.float64,
                ),
                scale=np.array([obstacle.size_x, obstacle.size_y, obstacle.height]),
                size=1.0,
                color=np.array([0.82, 0.38, 0.12]),
            )
            self.world.scene.add(cube)

        if self.config.dynamic_obstacle:
            obstacle = self.dynamic_config.at_time(0.0)
            self.dynamic_visual = VisualCuboid(
                prim_path="/World/Chapter10_2Obstacles/dynamic_obstacle",
                name="chapter10_2_dynamic_obstacle",
                position=np.array([obstacle.x, obstacle.y, 0.42]),
                scale=np.array([0.55, 0.55, 0.84]),
                size=1.0,
                color=np.array([0.10, 0.42, 0.95]),
            )
            self.world.scene.add(self.dynamic_visual)

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
        self._set_camera()
        time.sleep(1.0)
        for _ in range(self.config.warmup_steps):
            self.world.step(render=not self.config.headless)
        time.sleep(0.5)

        self.robot = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="G2_chapter10_2")
        self.world.scene.add(self.robot)
        self.robot.initialize()
        print(
            f"[Simulation] 第十章 10-2 场景加载完成，G2 自由度：{self.robot.num_dof}，"
            f"静态障碍：{len(STATIC_OBSTACLES)}",
            flush=True,
        )

    def step(self, render: bool | None = None) -> None:
        self.sim_time += self.config.physics_dt
        if self.dynamic_visual is not None:
            obstacle = self.dynamic_config.at_time(self.sim_time)
            self.dynamic_visual.set_world_pose(position=np.array([obstacle.x, obstacle.y, 0.42]))
        self.world.step(render=not self.config.headless if render is None else render)

    def dynamic_obstacles(self) -> list[CircleObstacle]:
        if not self.config.dynamic_obstacle:
            return []
        return [self.dynamic_config.at_time(self.sim_time)]

    def get_pose2d(self) -> Pose2D:
        position, quaternion = self.robot.get_world_pose()
        return Pose2D(float(position[0]), float(position[1]), self._quaternion_to_yaw(quaternion))

    def is_running(self) -> bool:
        return bool(self.app and self.app.is_running())

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
            camera.set_position_world(Gf.Vec3d(7.5, 7.5, 8.0), True)
            camera.set_target_world(Gf.Vec3d(0.0, 0.0, 0.0), True)
        except Exception as exc:
            print(f"[Simulation] 视角设置失败，可忽略：{exc}")

    @staticmethod
    def _quaternion_to_yaw(quaternion) -> float:
        w, x, y, z = (float(value) for value in quaternion)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
