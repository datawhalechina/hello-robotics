"""Standalone Isaac Sim scene for one G2 and a three-color block-to-box task."""

from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from config import (
    ARM_BASE_PRIM,
    CAMERA_PRIMS,
    COLORS,
    COLOR_RGB,
    ROBOT_PRIM,
    ROBOT_USD,
    SimulationConfig,
    TaskConfig,
)
from robot import CameraRig

SCENE_ROOT = "/World/Chapter15"


def require_gpu():
    command = shutil.which("nvidia-smi")
    if (
        command
        and subprocess.run(
            [command, "-L"], capture_output=True, text=True, check=False, timeout=10
        ).returncode
        == 0
    ):
        return
    if not Path("/dev/nvidia0").exists() and not Path("/dev/dxg").exists():
        raise RuntimeError("Isaac Sim RGB cameras require an NVIDIA GPU")


class BlockTask:
    def __init__(self, sim, cfg):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from pxr import PhysxSchema

        self.sim, self.cfg, self.blocks, self.completed = sim, cfg, {}, set()
        orientation = sim.arm_base_orientation
        material = PhysicsMaterial(
            f"{SCENE_ROOT}/Materials/block",
            "chapter15_block_material",
            static_friction=cfg.static_friction,
            dynamic_friction=cfg.dynamic_friction,
            restitution=0.0,
        )
        PhysxSchema.PhysxMaterialAPI.Apply(
            material.prim
        ).CreateFrictionCombineModeAttr().Set("max")
        self.table = sim.world.scene.add(
            FixedCuboid(
                f"{SCENE_ROOT}/table",
                "chapter15_table",
                position=self.arm_to_world(cfg.table_center),
                orientation=orientation,
                scale=np.asarray(cfg.table_size),
                size=1,
                color=np.array([0.48, 0.28, 0.12]),
            )
        )
        self._contact(self.table)
        box = np.asarray(cfg.box_position, float)
        ix, iz = cfg.box_inner_size
        wt, wh, ft = (
            cfg.box_wall_thickness,
            cfg.box_wall_height,
            cfg.box_floor_thickness,
        )
        table_y = cfg.table_top_y

        def fixed(name, pos, scale):
            body = sim.world.scene.add(
                FixedCuboid(
                    f"{SCENE_ROOT}/{name}",
                    f"chapter15_{name}",
                    position=self.arm_to_world(pos),
                    orientation=orientation,
                    scale=np.asarray(scale),
                    size=1,
                    color=np.array([0.68, 0.68, 0.68]),
                )
            )
            self._contact(body)
            return body

        floor = box.copy()
        floor[1] = table_y - ft / 2
        self.box_parts = [fixed("box_floor", floor, (ix + 2 * wt, ft, iz + 2 * wt))]
        walls = (
            ((ix / 2 + wt / 2, 0), (wt, wh, iz + 2 * wt)),
            ((-ix / 2 - wt / 2, 0), (wt, wh, iz + 2 * wt)),
            ((0, iz / 2 + wt / 2), (ix, wh, wt)),
            ((0, -iz / 2 - wt / 2), (ix, wh, wt)),
        )
        for i, ((dx, dz), scale) in enumerate(walls):
            pos = box + np.array([dx, 0, dz])
            pos[1] = table_y - wh / 2
            self.box_parts.append(fixed(f"box_wall_{i}", pos, scale))
        self.box_top_y = table_y - wh
        for color, pos in zip(COLORS, cfg.block_positions, strict=True):
            block = sim.world.scene.add(
                DynamicCuboid(
                    f"{SCENE_ROOT}/{color}_block",
                    f"chapter15_{color}_block",
                    position=self.arm_to_world(pos),
                    orientation=orientation,
                    scale=np.full(3, cfg.block_size),
                    size=1,
                    color=np.asarray(COLOR_RGB[color]),
                    mass=cfg.block_mass,
                    physics_material=material,
                )
            )
            self._contact(block)
            self.blocks[color] = block

    def _contact(self, body):
        body.set_rest_offset(0.0)
        body.set_contact_offset(self.cfg.contact_offset)

    def arm_to_world(self, point):
        return (self.sim.arm_base_world @ np.append(np.asarray(point, float), 1))[:3]

    def world_to_arm(self, point):
        return (self.sim.world_to_arm_base @ np.append(np.asarray(point, float), 1))[:3]

    def block_position(self, color):
        return self.world_to_arm(self.blocks[color].get_world_pose()[0])

    def randomize(self, rng, noise):
        self.completed.clear()
        for color, base in zip(COLORS, self.cfg.block_positions, strict=True):
            pos = np.asarray(base, float).copy()
            pos[[0, 2]] += rng.uniform(-noise, noise, 2)
            block = self.blocks[color]
            block.set_world_pose(
                position=self.arm_to_world(pos),
                orientation=self.sim.arm_base_orientation,
            )
            block.set_linear_velocity(np.zeros(3))
            block.set_angular_velocity(np.zeros(3))

    def inside_box(self, color):
        pos = self.block_position(color)
        box = np.asarray(self.cfg.box_position)
        hx, hz = np.asarray(self.cfg.box_inner_size) / 2
        return bool(
            abs(pos[0] - box[0]) <= hx + 0.02
            and abs(pos[2] - box[2]) <= hz + 0.02
            and self.box_top_y - self.cfg.block_size / 2
            <= pos[1]
            <= self.cfg.table_top_y + self.cfg.block_size / 2
        )

    def update(self, right_gripper_closed):
        if right_gripper_closed >= 0.45:
            return
        for color in COLORS:
            if color not in self.completed and self.inside_box(color):
                self.completed.add(color)

    def success(self, color):
        return color in self.completed and self.inside_box(color)

    def goal_distance(self, color):
        return float(
            np.linalg.norm(
                self.block_position(color)[[0, 2]]
                - np.asarray(self.cfg.box_position)[[0, 2]]
            )
        )


class G2Simulation:
    def __init__(self, cfg: SimulationConfig, task_cfg: TaskConfig | None = None):
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(ROBOT_USD)
        require_gpu()
        from isaacsim import SimulationApp

        self.cfg, self.task_cfg = cfg, task_cfg or TaskConfig()
        self.app = SimulationApp(
            {
                "headless": cfg.headless,
                "disable_viewport_updates": False,
                "renderer": cfg.renderer,
                "limit_cpu_threads": 16,
            }
        )
        self._build()

    def _build(self):
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        import omni.usd

        self.world = World(
            stage_units_in_meters=1,
            physics_dt=self.cfg.physics_dt,
            rendering_dt=1 / self.cfg.render_hz,
        )
        self.world.scene.add_default_ground_plane()
        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM)
        SingleXFormPrim(
            ROBOT_PRIM,
            position=np.array([0, 0, -0.01]),
            orientation=np.array([1, 0, 0, 0]),
        )
        stage = omni.usd.get_context().get_stage()
        joint = UsdPhysics.FixedJoint.Define(stage, f"{SCENE_ROOT}/G2FixedJoint")
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{ROBOT_PRIM}/base_link")])
        self.world.play()
        time.sleep(0.5)
        for _ in range(self.cfg.warmup_steps):
            self.world.step(render=True)
        self.articulation = SingleArticulation(ROBOT_PRIM, "G2_chapter15")
        self.world.scene.add(self.articulation)
        self.articulation.initialize()
        self.articulation.set_solver_position_iteration_count(32)
        self.articulation.set_solver_velocity_iteration_count(4)
        matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            stage.GetPrimAtPath(ARM_BASE_PRIM)
        )
        self.arm_base_world = np.asarray(matrix, float).T
        self.world_to_arm_base = np.linalg.inv(self.arm_base_world)
        self.arm_base_orientation = np.asarray(
            SingleXFormPrim(ARM_BASE_PRIM).get_world_pose()[1], float
        )
        self.task = BlockTask(self, self.task_cfg)
        for _ in range(20):
            self.world.step(render=True)
        self.cameras = CameraRig(CAMERA_PRIMS, self.cfg.image_size, self.cfg.render_hz)
        for _ in range(15):
            self.world.step(render=True)

    def step(self, render=True):
        self.world.step(render=render)

    def close(self):
        if getattr(self, "app", None):
            self.app.close()
            self.app = None
