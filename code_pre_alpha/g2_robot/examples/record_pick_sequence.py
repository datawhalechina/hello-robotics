"""Record a G2 gripper pick sequence video for Chapter 5.

This script avoids the full room scene so it can run with the minimal G2 assets
used in the tutorial. It follows the high-level approach used by Genie Sim
benchmark demos: close the gripper on an object, then attach the object to the
gripper frame for the lift phase.

Usage:
    python -m g2_robot.examples.record_pick_sequence --trial-index 0
"""

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from g2_robot.core.bootstrap import SimBootstrap


LIFT_HEIGHT = 0.16
CUBE_SIZE = 0.06
RECORDER_WARMUP_FRAMES = 260
APPROACH_STEPS = 120


def _cosine_path(start, target, steps):
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    for i in range(steps):
        alpha = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / steps))
        yield start + alpha * (target - start)


def _set_cube_pose(translate_op, position):
    from pxr import Gf

    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))


def _quat_wxyz_to_matrix(quat):
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _world_to_local(parent_pos, parent_quat, child_pos):
    rot = _quat_wxyz_to_matrix(parent_quat)
    return rot.T @ (np.asarray(child_pos, dtype=np.float64) - np.asarray(parent_pos, dtype=np.float64))


def _local_to_world(parent_pos, parent_quat, local_pos):
    rot = _quat_wxyz_to_matrix(parent_quat)
    return np.asarray(parent_pos, dtype=np.float64) + rot @ np.asarray(local_pos, dtype=np.float64)


def _create_demo_cube(position):
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    stage = sys.modules["omni.usd"].get_context().get_stage()
    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/chapter5_dome_light"))
    light.CreateIntensityAttr(900.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    cube = UsdGeom.Cube.Define(stage, Sdf.Path("/World/snow_block"))
    cube.CreateSizeAttr(CUBE_SIZE)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.86, 1.0)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    translate_op = xform.AddTranslateOp()
    _set_cube_pose(translate_op, position)
    return translate_op


def _latest_rgb_pngs(frames_dir):
    pngs = sorted(glob.glob(os.path.join(frames_dir, "**", "*rgb*.png"), recursive=True))
    if not pngs:
        pngs = sorted(glob.glob(os.path.join(frames_dir, "**", "*.png"), recursive=True))
    return pngs


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def _sample_reachable_joint_pose(base_joints, rng):
    return np.asarray(base_joints, dtype=np.float64) + np.array(
        [
            rng.uniform(-0.08, 0.08),
            rng.uniform(-0.22, 0.18),
            rng.uniform(-0.10, 0.10),
            rng.uniform(-0.20, 0.22),
            rng.uniform(-0.10, 0.10),
            rng.uniform(-0.16, 0.14),
            rng.uniform(-0.10, 0.10),
        ],
        dtype=np.float64,
    )


def _encode_video(frames_dir, output):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-start_number",
            str(RECORDER_WARMUP_FRAMES),
            "-framerate",
            "30",
            "-i",
            str(Path(frames_dir) / "rgb_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(output),
        ],
        check=True,
    )


def main():
    args = _parse_args()
    sys.argv = [sys.argv[0]]
    output_name = args.output_name or f"g2_pick_random_{args.trial_index + 1}.mp4"
    output = Path(__file__).resolve().parents[3] / "docs/chapter5/assets" / output_name
    frames_dir = Path(f"/tmp/g2_pick_sequence_frames_{args.trial_index}")
    rng = np.random.default_rng(args.seed + args.trial_index)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "sim": {
            "headless": True,
            "physics_step": 120,
            "rendering_step": 60,
            "render_mode": "RaytracedLighting",
        }
    }

    boot = SimBootstrap(config)
    sim_setup = None
    try:
        boot.setup_physics_scene()
        boot.load_robot(
            "robot/G2_omnipicker/robot_fix.usda",
            "/genie",
            position=[-3.55, -0.032, -0.01],
            orientation=[0.0, 0.0, 0.0, 1.0],
        )

        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.prims import SingleXFormPrim

        from g2_robot.controllers.arm import ArmController
        from g2_robot.core.sim_setup import SimSetup

        camera = rep.create.camera(position=(-3.00, 2.95, 1.62), look_at=(-3.72, 0.73, 1.24))
        render_product = rep.create.render_product(camera, (1280, 720))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(frames_dir), rgb=True)
        writer.attach([render_product])

        boot.world.reset()
        art = boot.init_articulation("/genie")
        sim_setup = SimSetup(boot.world, art, config)
        sim_setup._set_body_posture()
        sim_setup._init_ik_solvers()
        sim_setup._init_gripper()

        arm = ArmController(
            articulation=art,
            arm="right",
            ik_solver=sim_setup.ik_solvers["right"],
            gripper=sim_setup.gripper,
        )

        right_inner_tip = SingleXFormPrim("/genie/gripper_r_inner_link4")
        right_outer_tip = SingleXFormPrim("/genie/gripper_r_outer_link4")
        gripper_base = SingleXFormPrim("/genie/gripper_r_base_link")
        gripper_center = SingleXFormPrim("/genie/gripper_r_center_link")

        def step_and_record(count=1):
            for _ in range(count):
                boot.world.step(render=True)
                rep.orchestrator.step()

        sim_setup.gripper.open()
        for _ in range(80):
            boot.world.step(render=True)

        base_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        grasp_joints = _sample_reachable_joint_pose(base_joints, rng)
        arm.set_joint_positions(grasp_joints)
        for _ in range(80):
            boot.world.step(render=True)

        target_pos, _ = gripper_center.get_world_pose()
        inner_pos, _ = right_inner_tip.get_world_pose()
        outer_pos, _ = right_outer_tip.get_world_pose()
        object_pos = (np.asarray(inner_pos, dtype=np.float64) + np.asarray(outer_pos, dtype=np.float64)) * 0.5
        cube_translate = _create_demo_cube(object_pos)
        print(f"[RecordPick] sampled reachable target {np.round(target_pos, 4).tolist()}", flush=True)
        print(f"[RecordPick] object initialized at final fingertip midpoint {np.round(object_pos, 4).tolist()}", flush=True)
        approach_joints = grasp_joints + np.array([0.0, -0.45, 0.0, 0.55, 0.0, -0.35, 0.0], dtype=np.float64)
        arm.set_joint_positions(approach_joints)
        for _ in range(80):
            boot.world.step(render=True)

        step_and_record(RECORDER_WARMUP_FRAMES)

        for q in _cosine_path(approach_joints, grasp_joints, APPROACH_STEPS):
            arm.set_joint_positions(q)
            step_and_record(1)
        step_and_record(30)

        sim_setup.gripper.close()
        step_and_record(90)

        base_pos, base_quat = gripper_base.get_world_pose()
        attached_local_pos = _world_to_local(base_pos, base_quat, object_pos)
        print(f"[RecordPick] attached object local offset {np.round(attached_local_pos, 4).tolist()}", flush=True)

        current = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        lift_joints = current + np.array([0.0, 0.28, 0.0, -0.22, 0.0, 0.12, 0.0], dtype=np.float64)
        print(f"[RecordPick] joint-space lift target {np.round(lift_joints, 3).tolist()}", flush=True)
        for q in _cosine_path(current, lift_joints, 120):
            arm.set_joint_positions(q)
            base_pos, base_quat = gripper_base.get_world_pose()
            _set_cube_pose(cube_translate, _local_to_world(base_pos, base_quat, attached_local_pos))
            step_and_record(1)

        step_and_record(70)
        rep.orchestrator.wait_until_complete()

        pngs = _latest_rgb_pngs(str(frames_dir))
        if not pngs:
            raise RuntimeError(f"No RGB frames were written under {frames_dir}")
        if len(pngs) <= RECORDER_WARMUP_FRAMES:
            raise RuntimeError("No RGB frames remained after recorder warmup trimming")

        output.parent.mkdir(parents=True, exist_ok=True)
        _encode_video(frames_dir, output)
        print(f"[RecordPick] wrote {len(pngs) - RECORDER_WARMUP_FRAMES} frames to {output}", flush=True)
    finally:
        if sim_setup is not None:
            sim_setup.cleanup()
        boot.cleanup()


if __name__ == "__main__":
    main()
