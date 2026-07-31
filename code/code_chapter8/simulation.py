"""第八章 Isaac Sim 环境：加载 Simple Warehouse、G2，并生成闭环巡航轨迹。"""

import math
import time


try:
    from .config import ROBOT_PRIM_PATH, ROBOT_USD, WAREHOUSE_ASSET, WAREHOUSE_PRIM_PATH, SimulationConfig
    from .geometry import Pose2D, wrap_angle
except ImportError:
    from config import ROBOT_PRIM_PATH, ROBOT_USD, WAREHOUSE_ASSET, WAREHOUSE_PRIM_PATH, SimulationConfig
    from geometry import Pose2D, wrap_angle


class SquarePatrol:
    """带原地转弯的矩形闭环；便于稳定触发回环检测。"""

    def __init__(self, move_frames: int = 300, turn_frames: int = 75) -> None:
        self.move_frames = int(move_frames)
        self.turn_frames = int(turn_frames)
        self.corners = (
            (-1.5, -1.2),
            (+1.5, -1.2),
            (+1.5, +1.2),
            (-1.5, +1.2),
        )
        self.yaws = (0.0, math.pi / 2, math.pi, -math.pi / 2)
        self.segment_frames = self.move_frames + self.turn_frames
        self.frames_per_lap = 4 * self.segment_frames

    def pose(self, frame: int) -> Pose2D:
        local = int(frame) % self.frames_per_lap
        side = local // self.segment_frames
        phase = local % self.segment_frames
        start = self.corners[side]
        end = self.corners[(side + 1) % 4]
        yaw_start = self.yaws[side]
        yaw_end = self.yaws[(side + 1) % 4]
        if phase < self.move_frames:
            ratio = phase / float(self.move_frames)
            return Pose2D(
                x=start[0] + ratio * (end[0] - start[0]),
                y=start[1] + ratio * (end[1] - start[1]),
                yaw=yaw_start,
            )
        ratio = (phase - self.move_frames) / float(self.turn_frames)
        yaw_delta = wrap_angle(yaw_end - yaw_start)
        return Pose2D(end[0], end[1], wrap_angle(yaw_start + ratio * yaw_delta))


class G2MappingSimulation:

    def __init__(self, config: SimulationConfig, enable_ros2: bool = True) -> None:
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(f"找不到机器人：{ROBOT_USD}")
        self.config = config
        self.app = None
        self.context = None
        self.stage = None
        self._translate_op = None
        self._yaw_op = None
        self.true_pose = Pose2D()

        from isaacsim import SimulationApp

        self.app = SimulationApp(
            {
                "headless": config.headless,
                "renderer": config.renderer,
                "limit_cpu_threads": 16,
            }
        )
        self._build(enable_ros2)

    def _build(self, enable_ros2: bool) -> None:
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.prims import get_prim_at_path
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.storage.native import get_assets_root_path
        from pxr import UsdGeom

        if enable_ros2:
            enable_extension("isaacsim.ros2.bridge")
            self.app.update()
            self._create_ros_clock_graph()

        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("找不到 Isaac Sim Assets 根路径，无法加载 Simple Warehouse")
        warehouse_usd = assets_root + WAREHOUSE_ASSET
        add_reference_to_stage(warehouse_usd, WAREHOUSE_PRIM_PATH)
        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        self.app.update()
        self.stage = get_current_stage()

        robot_prim = get_prim_at_path(ROBOT_PRIM_PATH)
        if not robot_prim.IsValid():
            raise RuntimeError(f"G2 prim 加载失败：{ROBOT_PRIM_PATH}")
        xform = UsdGeom.Xformable(robot_prim)
        xform.ClearXformOpOrder()
        self._translate_op = xform.AddTranslateOp()
        self._yaw_op = xform.AddRotateZOp()

        self.context = SimulationContext(
            physics_dt=self.config.physics_dt,
            rendering_dt=1.0 / self.config.rendering_hz,
            stage_units_in_meters=1.0,
        )
        self.context.play()
        self._set_camera()
        time.sleep(0.5)
        for _ in range(self.config.warmup_steps):
            self.app.update()
        print(
            f"[G2MappingSimulation] 已加载 Simple Warehouse 与 G2，"
            f"频率 {self.config.physics_hz} Hz",
            flush=True,
        )

    @staticmethod
    def _create_ros_clock_graph() -> None:
        """发布 Isaac Sim 仿真时间，保证 RViz、点云和 TF 使用同一时钟。"""
        try:
            import omni.graph.core as og

            og.Controller.edit(
                {
                    "graph_path": "/Chapter8RosClockGraph",
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
                        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                        ("RosContext.outputs:context", "PublishClock.inputs:context"),
                    ],
                },
            )
            print("[ROS2] 已发布 /clock（Isaac Sim 仿真时间）", flush=True)
        except Exception as exc:
            print(f"[ROS2] /clock 发布未启用：{exc}", flush=True)

    def set_pose(self, pose: Pose2D) -> None:
        from pxr import Gf

        self._translate_op.Set(Gf.Vec3d(float(pose.x), float(pose.y), self.config.robot_z))
        self._yaw_op.Set(math.degrees(float(pose.yaw)))
        self.true_pose = pose

    def step(self, pose: Pose2D | None = None) -> None:
        if pose is not None:
            self.set_pose(pose)
        self.app.update()

    def is_running(self) -> bool:
        return bool(self.app and self.app.is_running())

    @property
    def sim_time(self) -> float:
        return float(self.context.current_time) if self.context is not None else 0.0

    def close(self) -> None:
        if self.context is not None:
            self.context.stop()
            self.context = None
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
            camera.set_position_world(Gf.Vec3d(4.5, 4.5, 3.4), True)
            camera.set_target_world(Gf.Vec3d(0.0, 0.0, 0.5), True)
        except Exception as exc:
            print(f"[G2MappingSimulation] 视角设置失败（不影响建图）：{exc}", flush=True)
