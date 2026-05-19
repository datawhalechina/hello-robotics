"""
SimBootstrap - Shared Isaac Sim + ROS2 initialization sequence.

Eliminates the massive boilerplate duplicated across main.py, sensor_publisher.py,
and all test_*.py files. Provides a single entry point for:
  1. Environment setup
  2. Isaac Sim app launch
  3. ROS2 bridge initialization (optional)
  4. World creation with physics
  5. Robot/scene loading
  6. Articulation initialization
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from g2_robot.utils import system_utils


@dataclass
class _AppCfg:
    headless: bool = False
    render_mode: str = "RaytracedLighting"


class SimBootstrap:
    """One-call setup for Isaac Sim simulation with optional ROS2 support."""

    def __init__(self, config: dict):
        """
        Initialize Isaac Sim application and create the simulation world.

        Args:
            config: dict with at least 'sim' key containing:
                - headless (bool)
                - physics_step (int, Hz)
                - rendering_step (int, Hz)
                - render_mode (str, optional)
        """
        self.config = config
        self._rclpy = None
        self._app = None
        self._world = None
        self._stage = None
        self._articulation = None

        # Step 1: Fix environment
        system_utils.check_and_fix_env()

        # Step 2: Launch Isaac Sim
        sim_cfg = config.get("sim", {})
        app_cfg = _AppCfg(
            headless=sim_cfg.get("headless", False),
            render_mode=sim_cfg.get("render_mode", "RaytracedLighting"),
        )

        from g2_robot.core.app_launcher import AppLauncher
        launcher = AppLauncher(app_cfg)
        self._app = launcher.app

        # Step 3: Create World (imports only available after app launch)
        from isaacsim.core.api import World

        physics_dt = 1.0 / sim_cfg.get("physics_step", 120)
        rendering_dt = 1.0 / sim_cfg.get("rendering_step", 60)

        self._world = World(
            stage_units_in_meters=1,
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
        )

        # Step 4: Get stage reference
        import omni.usd
        self._stage = omni.usd.get_context().get_stage()

    @property
    def app(self):
        """The SimulationApp instance."""
        return self._app

    @property
    def world(self):
        """The Isaac Sim World instance."""
        return self._world

    @property
    def stage(self):
        """The USD stage."""
        return self._stage

    @property
    def articulation(self):
        """The robot articulation (None until init_articulation is called)."""
        return self._articulation

    @property
    def rclpy(self):
        """The rclpy module (None until init_ros2 is called)."""
        return self._rclpy

    def setup_physics_scene(self, add_ground_plane=True):
        """Configure physics scene with gravity and GPU parameters.

        Args:
            add_ground_plane: if True, add an invisible collision ground plane at z=0.
                Required for floating-base robots (robot.usda). Harmless for fixed-base.
        """
        from pxr import Gf, Sdf, UsdPhysics, PhysxSchema

        scene = UsdPhysics.Scene.Define(self._stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

        physics_scene = PhysxSchema.PhysxSceneAPI.Get(self._stage, "/physicsScene")
        physics_scene.CreateGpuMaxRigidContactCountAttr(8388608)
        physics_scene.CreateGpuMaxRigidPatchCountAttr(163840)
        physics_scene.CreateGpuFoundLostPairsCapacityAttr(2097152)
        physics_scene.CreateGpuFoundLostAggregatePairsCapacityAttr(33554432)
        physics_scene.CreateGpuTotalAggregatePairsCapacityAttr(2097152)

        if add_ground_plane:
            self._world.scene.add_default_ground_plane()
            print("[Bootstrap] Ground plane added at z=0")

    def init_ros2(self):
        """Enable ROS2 bridge extension, fix rclpy path, and initialize rclpy.

        Returns:
            The rclpy module.
        """
        os.environ.setdefault("ROS_DISTRO", "humble")
        os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

        from isaacsim.core.utils import extensions
        extensions.enable_extension("isaacsim.ros2.bridge")
        extensions.enable_extension("isaacsim.robot.wheeled_robots")

        # Fix rclpy path for Python 3.11 compatibility
        import isaacsim
        _bridge_ext_path = os.path.join(
            os.path.dirname(isaacsim.__file__),
            "exts", "isaacsim.ros2.bridge", "humble",
        )
        _bridge_python_path = os.path.join(_bridge_ext_path, "rclpy")
        if os.path.isdir(_bridge_python_path):
            sys.path = [
                p for p in sys.path
                if not (p.startswith("/opt/ros") and "python3.10" in p)
            ]
            for _key in [k for k in sys.modules if k == "rclpy" or k.startswith("rclpy.")]:
                del sys.modules[_key]
            if _bridge_python_path not in sys.path:
                sys.path.insert(0, _bridge_python_path)
            _bridge_lib_path = os.path.join(_bridge_ext_path, "lib")
            _ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if _bridge_lib_path not in _ld_path:
                os.environ["LD_LIBRARY_PATH"] = _bridge_lib_path + ":" + _ld_path

        # Wait for rclpy to become available
        self._rclpy = self._wait_rclpy(_bridge_ext_path)
        self._rclpy.init()
        return self._rclpy

    def load_robot(self, robot_usd_rel, prim_path, position, orientation):
        """Load robot USD into the stage.

        Args:
            robot_usd_rel: relative path under assets dir (e.g. "robot/G2_omnipicker/robot_fix.usda")
            prim_path: USD prim path (e.g. "/genie")
            position: [x, y, z] initial position
            orientation: [w, x, y, z] initial orientation quaternion

        Returns:
            SingleXFormPrim for the robot root.
        """
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        assets_dir = system_utils.assets_path()
        robot_usd_path = os.path.join(assets_dir, robot_usd_rel)
        add_reference_to_stage(robot_usd_path, prim_path)

        robot_xform = SingleXFormPrim(
            prim_path=prim_path,
            position=np.array(position, dtype=np.float64),
            orientation=np.array(orientation, dtype=np.float64),
        )
        print(f"[Bootstrap] Robot loaded at {prim_path}")
        return robot_xform

    def load_scene(self, scene_usd_rel, prim_path="/World"):
        """Load scene USD into the stage.

        Args:
            scene_usd_rel: relative path under assets dir
            prim_path: USD prim path for the scene root
        """
        from isaacsim.core.utils.stage import add_reference_to_stage

        assets_dir = system_utils.assets_path()
        scene_usd_path = os.path.join(assets_dir, scene_usd_rel)
        add_reference_to_stage(scene_usd_path, prim_path)
        print(f"[Bootstrap] Scene loaded at {prim_path}")

    def set_viewport_camera(self, position, target):
        """Set the free viewport camera position and look-at target.

        Args:
            position: [x, y, z] camera world position
            target: [x, y, z] look-at target
        """
        try:
            from pxr import Gf
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
            cam_state = ViewportCameraState("/OmniverseKit_Persp")
            cam_state.set_position_world(Gf.Vec3d(*position), True)
            cam_state.set_target_world(Gf.Vec3d(*target), True)
            print("[Bootstrap] Viewport camera positioned")
        except Exception as e:
            print(f"[Bootstrap] Could not set viewport camera: {e}")

    def play_and_warmup(self, warmup_steps=120, sleep_before=1.0, sleep_after=0.5):
        """Start physics simulation and warm up.

        Args:
            warmup_steps: number of physics steps to warm up
            sleep_before: seconds to sleep before stepping
            sleep_after: seconds to sleep after stepping
        """
        self._world.play()
        time.sleep(sleep_before)
        for _ in range(warmup_steps):
            self._world.step(render=True)
        time.sleep(sleep_after)
        print(f"[Bootstrap] Simulation warmed up ({warmup_steps} steps)")

    def init_articulation(self, prim_path, name="G2_omnipicker"):
        """Initialize and register robot articulation.

        Args:
            prim_path: USD prim path of the robot
            name: articulation name

        Returns:
            SingleArticulation instance.
        """
        from isaacsim.core.prims import SingleArticulation

        self._articulation = SingleArticulation(prim_path=prim_path, name=name)
        self._world.scene.add(self._articulation)
        self._articulation.initialize()
        print(f"[Bootstrap] Articulation initialized: {self._articulation.num_dof} DOFs")
        return self._articulation

    def cleanup(self):
        """Shutdown rclpy (if initialized) and close the simulation app."""
        if self._rclpy is not None:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass
        if self._app is not None:
            self._app.close()

    @staticmethod
    def _wait_rclpy(bridge_ext_path, timeout=10, tick=0.1):
        """Wait for rclpy to become importable."""
        start = time.time()
        while True:
            try:
                import rclpy
                return rclpy
            except (ModuleNotFoundError, ImportError) as e:
                if time.time() - start > timeout:
                    raise RuntimeError(
                        f"rclpy not available after timeout: {e}\n"
                        f"Ensure ROS_DISTRO=humble is set and LD_LIBRARY_PATH "
                        f"includes {bridge_ext_path}/lib"
                    )
                time.sleep(tick)
