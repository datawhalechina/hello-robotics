"""Isaac Sim 中独立运行的 G2 小物块入盒场景。"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    from .config import (
        ARM_BASE_PRIM_PATH,
        BLOCK_COLORS,
        HEAD_CAMERA_PATH,
        LEFT_WRIST_CAMERA_PATH,
        RIGHT_WRIST_CAMERA_PATH,
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        SimulationConfig,
        TaskConfig,
    )
    from .robot import G2Cameras
except ImportError:
    from config import (
        ARM_BASE_PRIM_PATH,
        BLOCK_COLORS,
        HEAD_CAMERA_PATH,
        LEFT_WRIST_CAMERA_PATH,
        RIGHT_WRIST_CAMERA_PATH,
        ROBOT_PRIM_PATH,
        ROBOT_USD,
        SimulationConfig,
        TaskConfig,
    )
    from robot import G2Cameras


def require_nvidia_render_gpu() -> None:
    """机载 RGB 相机需要 NVIDIA GPU；在启动 Kit 前给出明确错误。"""
    command = shutil.which("nvidia-smi")
    if command is not None:
        result = subprocess.run(
            [command, "-L"], capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            return
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "未检测到可用的 NVIDIA GPU，G2 头部/腕部 RGB 相机无法渲染。"
            f" nvidia-smi: {detail or '未返回 GPU'}。"
            "请检查 NVIDIA 驱动；如果在 Docker 中运行，请确认已传入 GPU。"
        )

    if not Path("/dev/nvidia0").exists() and not Path("/dev/dxg").exists():
        raise RuntimeError(
            "未找到 nvidia-smi 或 NVIDIA GPU 设备。G2 机载 RGB 相机需要可用 GPU；"
            "请先配置 NVIDIA 驱动或容器 GPU 透传。"
        )


class G2BlockTask:
    """桌子、红绿蓝动态积木、空盒和纯物理抓取任务。"""

    def __init__(self, simulation: G2Simulation, config: TaskConfig) -> None:
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from pxr import PhysxSchema

        self.sim = simulation
        self.config = config
        self.blocks = {}
        self.block_arm_positions = {}
        self.completed: set[str] = set()

        orientation = simulation.arm_base_orientation
        self.table_top_arm_y = config.table_top_arm_y
        self.block_material = PhysicsMaterial(
            prim_path="/World/ACoTVLA/Materials/block_grip",
            name="acotvla_block_grip",
            static_friction=config.static_friction,
            dynamic_friction=config.dynamic_friction,
            restitution=0.0,
        )
        material_api = PhysxSchema.PhysxMaterialAPI.Apply(self.block_material.prim)
        material_api.CreateFrictionCombineModeAttr().Set("max")
        material_api.CreateRestitutionCombineModeAttr().Set("min")

        # 用 Isaac Sim 基础几何体创建桌面，不依赖其他章节场景资产。
        self.table = simulation.world.scene.add(
            FixedCuboid(
                prim_path="/World/ACoTVLA/table_top",
                name="acotvla_table_top",
                position=self.arm_to_world(config.table_arm_center),
                orientation=orientation,
                scale=np.asarray(config.table_size),
                size=1.0,
                color=np.array([0.48, 0.28, 0.12]),
            )
        )
        self._configure_contact(self.table)

        # 黄色 blocker 不再创建；在它原来的水平区域放置一个灰色空盒。
        box_arm = np.asarray(config.box_arm_position, dtype=np.float64)
        inner_x, inner_z = config.box_inner_size
        wall_t = config.box_wall_thickness
        wall_h = config.box_wall_height
        floor_t = config.box_floor_thickness
        floor_y = self.table_top_arm_y - floor_t / 2
        wall_y = self.table_top_arm_y - wall_h / 2

        self.box_wall_top_arm_y = self.table_top_arm_y - wall_h

        def add_box_part(name: str, arm_position, scale):
            part = simulation.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/ACoTVLA/{name}",
                    name=f"acotvla_{name}",
                    position=self.arm_to_world(arm_position),
                    orientation=orientation,
                    scale=np.asarray(scale, dtype=np.float64),
                    size=1.0,
                    color=np.array([0.68, 0.68, 0.68]),
                )
            )
            self._configure_contact(part)
            return part

        floor_position = box_arm.copy()
        floor_position[1] = floor_y
        self.box_parts = [
            add_box_part(
                "box_floor",
                floor_position,
                (inner_x + 2 * wall_t, floor_t, inner_z + 2 * wall_t),
            )
        ]
        wall_specs = (
            ((inner_x / 2 + wall_t / 2, 0.0), (wall_t, wall_h, inner_z + 2 * wall_t)),
            ((-inner_x / 2 - wall_t / 2, 0.0), (wall_t, wall_h, inner_z + 2 * wall_t)),
            ((0.0, inner_z / 2 + wall_t / 2), (inner_x, wall_h, wall_t)),
            ((0.0, -inner_z / 2 - wall_t / 2), (inner_x, wall_h, wall_t)),
        )
        for index, ((offset_x, offset_z), scale) in enumerate(wall_specs):
            wall_position = box_arm.copy()
            wall_position += np.array([offset_x, 0.0, offset_z])
            wall_position[1] = wall_y
            self.box_parts.append(
                add_box_part(f"box_wall_{index}", wall_position, scale)
            )

        # 使用较小、便于夹取的动态刚体物块。
        for name, arm_position in zip(
            BLOCK_COLORS, config.block_arm_positions, strict=True
        ):
            arm_position = np.asarray(arm_position, dtype=np.float64)
            self.block_arm_positions[name] = arm_position.copy()
            block = simulation.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/ACoTVLA/{name}_object",
                    name=f"acotvla_{name}_object",
                    position=self.arm_to_world(arm_position),
                    orientation=orientation,
                    scale=np.full(3, config.block_size),
                    size=1.0,
                    color=np.asarray(BLOCK_COLORS[name]),
                    mass=config.block_mass,
                    physics_material=self.block_material,
                )
            )
            self._configure_contact(block)
            self.blocks[name] = block

    def _configure_contact(self, geometry) -> None:
        """使用毫米级接触距离，避免小物块在碰撞体外提前悬停。"""
        geometry.set_rest_offset(0.0)
        geometry.set_contact_offset(self.config.contact_offset)

    def arm_to_world(self, point) -> np.ndarray:
        homogeneous = np.concatenate([np.asarray(point, dtype=np.float64), [1.0]])
        return (self.sim.arm_base_world @ homogeneous)[:3]

    def world_to_arm(self, point) -> np.ndarray:
        homogeneous = np.concatenate([np.asarray(point, dtype=np.float64), [1.0]])
        return (self.sim.world_to_arm_base @ homogeneous)[:3]

    def randomize(
        self, rng: np.random.Generator, position_noise: float = 0.025
    ) -> None:
        """在任务初始位置附近随机化桌面水平位置。"""
        self.completed.clear()
        for name, nominal in zip(
            BLOCK_COLORS, self.config.block_arm_positions, strict=True
        ):
            arm = np.asarray(nominal, dtype=np.float64).copy()
            arm[[0, 2]] += rng.uniform(-position_noise, position_noise, size=2)
            self.block_arm_positions[name] = arm
            block = self.blocks[name]
            block.set_world_pose(
                position=self.arm_to_world(arm),
                orientation=self.sim.arm_base_orientation,
            )
            block.set_linear_velocity(np.zeros(3))
            block.set_angular_velocity(np.zeros(3))

    def update(self, gripper_closed_amount: float) -> None:
        """只读取真实物块位置做成功判定，不锁定、吸附或移动物块。"""
        if gripper_closed_amount >= 0.45:
            return
        for name, block in self.blocks.items():
            if name in self.completed:
                continue
            position, _ = block.get_world_pose()
            if self._inside_box(np.asarray(position)):
                self.completed.add(name)
                print(
                    f"[Task] {name} block 已通过物理抓取放入盒子"
                    f"（{len(self.completed)}/3）",
                    flush=True,
                )

    def _inside_box(self, world_position: np.ndarray) -> bool:
        arm = self.world_to_arm(world_position)
        half_x, half_z = np.asarray(self.config.box_inner_size) / 2
        horizontal_ok = (
            abs(arm[0] - self.config.box_arm_position[0]) <= half_x + 0.02
            and abs(arm[2] - self.config.box_arm_position[2]) <= half_z + 0.02
        )
        vertical_ok = (
            self.box_wall_top_arm_y - self.config.block_size / 2
            <= arm[1]
            <= self.table_top_arm_y + self.config.block_size / 2
        )
        return bool(horizontal_ok and vertical_ok)

    @property
    def success(self) -> bool:
        return len(self.completed) == len(BLOCK_COLORS)


class G2Simulation:
    def __init__(
        self, config: SimulationConfig, task_config: TaskConfig | None = None
    ) -> None:
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(f"找不到 G2 USD：{ROBOT_USD}")
        self.config = config
        self.task_config = task_config or TaskConfig()
        self.app = None
        self.world = None
        self.robot = None
        self.cameras = None
        self.task = None

        require_nvidia_render_gpu()
        from isaacsim import SimulationApp

        self.app = SimulationApp(
            {
                "headless": config.headless,
                "disable_viewport_updates": False,  # headless 时仍要渲染机载相机
                "renderer": config.renderer,
                "limit_cpu_threads": 16,
            }
        )
        self._build_world()

    def _build_world(self) -> None:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Sdf, UsdPhysics

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=1.0 / self.config.rendering_hz,
        )
        # 背景和桌面均由本脚本创建，避免依赖外部房间USD。
        self.world.scene.add_default_ground_plane()
        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        SingleXFormPrim(
            prim_path=ROBOT_PRIM_PATH,
            position=np.array([0.0, 0.0, -0.01]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        # 该 G2 USD 默认是可移动底座。连续挥动机械臂会让整机因反作用力缓慢漂移，
        # IK 和桌面坐标都以初始 arm_base_link 为基准，因此必须把底座固定到世界。
        # 否则通常运行十几个 episode 后，相机视角和抓取点会一起偏移并连续失败。
        stage = omni.usd.get_context().get_stage()
        fixed_joint = UsdPhysics.FixedJoint.Define(
            stage, "/World/ACoTVLA/G2WorldFixedJoint"
        )
        fixed_joint.CreateBody1Rel().SetTargets(
            [Sdf.Path(f"{ROBOT_PRIM_PATH}/base_link")]
        )

        self.world.play()
        time.sleep(0.5)
        for _ in range(self.config.warmup_steps):
            self.world.step(render=True)

        self.robot = SingleArticulation(ROBOT_PRIM_PATH, "G2_acotvla")
        self.world.scene.add(self.robot)
        self.robot.initialize()
        # 增加接触求解迭代次数，让夹指与小物块的摩擦接触更稳定。
        self.robot.set_solver_position_iteration_count(32)
        self.robot.set_solver_velocity_iteration_count(4)
        self.arm_base_world = self.prim_world_matrix(ARM_BASE_PRIM_PATH)
        self.world_to_arm_base = np.linalg.inv(self.arm_base_world)
        _, orientation = SingleXFormPrim(ARM_BASE_PRIM_PATH).get_world_pose()
        self.arm_base_orientation = np.asarray(orientation, dtype=np.float64)

        self.task = G2BlockTask(self, self.task_config)
        self._set_viewport_camera()
        for _ in range(20):
            self.world.step(render=True)

        self.cameras = G2Cameras(
            HEAD_CAMERA_PATH,
            LEFT_WRIST_CAMERA_PATH,
            RIGHT_WRIST_CAMERA_PATH,
            self.config.image_size,
        )
        for _ in range(15):
            self.world.step(render=True)
        print(
            "[G2Simulation] 红绿蓝物块、空盒和 G2 已加载",
            flush=True,
        )

    def _set_viewport_camera(self) -> None:
        if self.config.headless:
            return
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(
                eye=np.array([2.1, -2.5, 2.25]),
                target=np.array([0.55, -0.35, 1.0]),
                camera_prim_path="/OmniverseKit_Persp",
            )
        # Kit/viewport 扩展可能抛出不同异常；视口设置失败不影响 headless 采集。
        except Exception as error:  # noqa: BLE001
            print(f"[G2Simulation] 视口相机设置失败：{error}", flush=True)

    def prim_world_matrix(self, prim_path: str) -> np.ndarray:
        import omni.usd
        from pxr import Usd, UsdGeom

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"找不到 Prim：{prim_path}")
        matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            prim
        )
        return np.asarray(matrix, dtype=np.float64).T

    def step(self, render: bool = True) -> None:
        # 带渲染的 World.step 会按 rendering_dt 推进，可能包含多个物理步。
        # 显式推进一个 physics_dt，再独立渲染，保证采集与执行的计数准确。
        self.world.step(render=False)
        if render:
            self.world.render()

    def observe(self, robot) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """固定物理时刻刷新相机，再读取该时刻的图像和关节状态。"""
        timestamp = self.world.current_time
        step_index = self.world.current_time_step_index
        # 排空上一物理步的渲染结果，不推进物理，也不挪动时间戳。
        for attempt in range(4):
            self.world.render()
            if (
                self.world.current_time_step_index != step_index
                or self.world.current_time != timestamp
            ):
                raise RuntimeError("读取观测时仿真意外推进")
            try:
                images = self.cameras.capture(expected_time=timestamp)
            except RuntimeError:
                # 渲染流水线偶尔需要多个更新才能交付当前时刻的图像。
                if attempt == 3:
                    raise
            else:
                return robot.state16().copy(), images

    def close(self) -> None:
        if self.app is not None:
            self.app.close()
            self.app = None
