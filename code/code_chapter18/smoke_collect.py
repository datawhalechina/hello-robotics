"""正式采集前：RGB 各试采一次，再从文件回放动作并检查真实计时。"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import simulation
from config import (
    COLORS,
    DATASET_FPS,
    RESULTS_ROOT,
    ControlConfig,
    SimulationConfig,
    TaskConfig,
)
from dataset import EpisodeRecorder
from expert import ScriptedExpert
from robot import G2Robot
from training.convert_dataset import validate_timing
from vla_client import ChunkRunner


def check_arrays(data):
    frames = len(data["state"])
    if frames < 2:
        raise ValueError("轨迹至少需要两帧才能检查频率")
    for key in ("state", "actions"):
        if data[key].shape != (frames, 16) or not np.isfinite(data[key]).all():
            raise ValueError(f"{key} 维度或数值异常")
    for key in ("head_image", "left_image", "right_image"):
        if data[key].shape != (frames, 240, 320, 3) or data[key].dtype != np.uint8:
            raise ValueError(f"{key} 图像格式异常")
        if not np.any(data[key][1:] != data[key][:-1]):
            raise ValueError(f"{key} 整条轨迹没有图像更新")
    validate_timing(data, DATASET_FPS)
    return frames


def reset(sim, robot, rng_state, noise):
    robot.reset_episode()
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    sim.task.randomize(rng, noise)
    for _ in range(40):
        sim.step()


def succeeded(sim, color):
    position, _ = sim.task.blocks[color].get_world_pose()
    return color in sim.task.completed and sim.task._inside_box(position)


def run_checks(args):
    report = {
        "passed": False,
        "seed": args.seed,
        "position_noise": args.position_noise,
        "trials": [],
    }
    sim = None
    try:
        if args.robot_usd is not None:
            simulation.ROBOT_USD = args.robot_usd.resolve()
        sim = simulation.G2Simulation(
            SimulationConfig(headless=args.headless), TaskConfig()
        )
        robot = G2Robot(sim.robot)
        rng = np.random.default_rng(args.seed)
        for index, color in enumerate(COLORS):
            rng_state = rng.bit_generator.state
            # 与正式采集相同：每回合为三个物块各生成两个水平偏移。
            rng.uniform(-args.position_noise, args.position_noise, size=(3, 2))
            reset(sim, robot, rng_state, args.position_noise)
            recorder = EpisodeRecorder(
                args.run_dir,
                index,
                f"Pick up the {color} block and place it into the empty box.",
                color,
            )
            start_time, start_step = (
                sim.world.current_time,
                sim.world.current_time_step_index,
            )
            ScriptedExpert(sim, robot, recorder).run(color)
            collected = succeeded(sim, color)
            elapsed = sim.world.current_time - start_time
            steps = sim.world.current_time_step_index - start_step
            # 失败样本也只写到独立的验收目录，用于定位，不进入训练目录。
            path = recorder.save(collected)
            with np.load(path, allow_pickle=False) as data:
                frames = check_arrays(data)
                actions = data["actions"].copy()
                sync_error = float(
                    np.max(
                        np.abs(data["image_time"] - data["observation_time"][:, None])
                    )
                )
            expected_steps = frames * sim.config.physics_hz // DATASET_FPS
            if steps != expected_steps or not np.isclose(
                elapsed, frames / DATASET_FPS, rtol=0, atol=1e-5
            ):
                raise ValueError("采集执行时间与保存帧数不一致")
            reset(sim, robot, rng_state, args.position_noise)
            runner = ChunkRunner(robot, sim, ControlConfig())
            start_time, start_step = (
                sim.world.current_time,
                sim.world.current_time_step_index,
            )
            for offset in range(0, frames, runner.config.execute_chunk):
                sim.observe(robot)
                runner.execute(actions[offset : offset + runner.config.execute_chunk])
            sim.observe(robot)
            replay_steps = sim.world.current_time_step_index - start_step
            replay_time = sim.world.current_time - start_time
            if replay_steps != expected_steps or not np.isclose(
                replay_time, frames / DATASET_FPS, rtol=0, atol=1e-5
            ):
                raise ValueError("回放执行时间与动作数量不一致")
            row = {
                "color": color,
                "capture_success": collected,
                "replay_success": succeeded(sim, color),
                "frames": frames,
                "physics_steps": steps,
                "sim_seconds": elapsed,
                "sync_error": sync_error,
                "replay_physics_steps": replay_steps,
                "replay_sim_seconds": replay_time,
            }
            report["trials"].append(row)
            (args.run_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[验收] {color}: capture={collected}, replay={row['replay_success']}, frames={frames}",
                flush=True,
            )
        report["passed"] = all(
            row["capture_success"] and row["replay_success"] for row in report["trials"]
        )
    # 验收入口需要把任意运行错误写入 report.json，交由外层进程统一判定失败。
    except Exception as error:  # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        print(f"[验收失败] {report['error']}", flush=True)
    finally:
        (args.run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if sim is not None:
            sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--position-noise", type=float, default=0.01)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=RESULTS_ROOT / "collection_smoke"
    )
    parser.add_argument(
        "--robot-usd", type=Path, help="可选：使用已下载的 G2 robot.usda"
    )
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not np.isfinite(args.position_noise) or args.position_noise < 0:
        parser.error("position-noise 必须为非负有限数")
    if args.run_dir is not None:
        run_checks(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=args.output.resolve()))
    print(f"[验收目录] {run_dir}", flush=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(run_dir),
        "--seed",
        str(args.seed),
        "--position-noise",
        str(args.position_noise),
    ]
    if args.headless:
        command.append("--headless")
    if args.robot_usd:
        command.extend(["--robot-usd", str(args.robot_usd.resolve())])
    # Kit 关闭可能直接退出进程；外层根据报告判断结果，避免误报退出码 0。
    process = subprocess.run(command, check=False)
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    passed = process.returncode == 0 and report.get("passed") is True
    print(f"[验收{'通过' if passed else '未通过'}] {report_path}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
