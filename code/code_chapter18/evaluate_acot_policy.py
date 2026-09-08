"""评测 G2 三色物块场景中的 ACoT-VLA 服务。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from config import COLORS, ControlConfig, SimulationConfig, TaskConfig
from robot import G2Robot
from simulation import G2Simulation
from vla_client import ChunkRunner, VLAClient

COLOR_IDS = {color: index for index, color in enumerate(COLORS)}


def prompt(color: str) -> str:
    return f"Pick up the {color} block and place it into the empty box."


def atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def episode_rng(seed: int, color: str, index: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([seed, COLOR_IDS[color], index])
    )


def reset_episode(
    sim: G2Simulation,
    robot: G2Robot,
    rng: np.random.Generator,
    position_noise: float,
) -> None:
    robot.reset_episode()
    for _ in range(10):
        sim.step(render=True)
    sim.task.randomize(rng, position_noise)
    for _ in range(30):
        sim.step(render=True)


def inside_box(sim: G2Simulation, color: str) -> bool:
    position, _ = sim.task.blocks[color].get_world_pose()
    return color in sim.task.completed and sim.task._inside_box(np.asarray(position))


def summarize(results: dict) -> None:
    colors = tuple(results["per_color"])
    for color in colors:
        trials = results["per_color"][color]["trials"]
        successes = sum(bool(trial["success"]) for trial in trials)
        results["per_color"][color]["summary"] = {
            "episodes": len(trials),
            "successes": successes,
            "success_rate": successes / len(trials) if trials else 0.0,
        }
    total_episodes = sum(
        results["per_color"][color]["summary"]["episodes"] for color in colors
    )
    total_successes = sum(
        results["per_color"][color]["summary"]["successes"] for color in colors
    )
    results["overall"] = {
        "episodes": total_episodes,
        "successes": total_successes,
        "micro_success_rate": total_successes / total_episodes
        if total_episodes
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--colors",
        nargs="+",
        choices=COLORS,
        default=list(COLORS),
        help="需要评测的颜色；默认依次评测 red green blue",
    )
    parser.add_argument("--seed", type=int, default=1506)
    parser.add_argument("--position-noise", type=float, default=0.01)
    parser.add_argument("--max-replans", type=int, default=60)
    parser.add_argument("--execute-chunk", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.max_replans, args.execute_chunk) < 1:
        raise ValueError("episodes、max-replans 和 execute-chunk 必须大于 0")
    if args.position_noise < 0:
        raise ValueError("position-noise 不能小于 0")
    if args.resume and args.overwrite:
        raise ValueError("--resume 和 --overwrite 不能同时使用")

    client = VLAClient(args.host, args.port)
    if client.state_dim != 16 or client.action_dim != 16:
        raise ValueError(
            f"G2 需要 16 维 state/action，服务返回 "
            f"{client.state_dim}/{client.action_dim}"
        )
    expected = {
        "colors": args.colors,
        "episodes_per_color": args.episodes,
        "seed": args.seed,
        "position_noise": args.position_noise,
        "max_replans": args.max_replans,
        "execute_chunk": args.execute_chunk,
        "server_metadata": client.metadata,
        "headless": True,
    }
    if args.resume and args.output.is_file():
        results = json.loads(args.output.read_text(encoding="utf-8"))
        if results.get("config") != expected:
            raise ValueError("续测配置与已有结果不一致")
    else:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"结果已存在：{args.output}")
        results = {
            "config": expected,
            "per_color": {color: {"trials": []} for color in args.colors},
        }
        summarize(results)
        atomic_save(args.output, results)

    sim = G2Simulation(
        SimulationConfig(headless=True), TaskConfig(max_replans=args.max_replans)
    )
    try:
        robot = G2Robot(sim.robot)
        runner = ChunkRunner(
            robot,
            sim,
            replace(ControlConfig(), execute_chunk=args.execute_chunk),
        )
        runner.move_home()
        for color in args.colors:
            trials = results["per_color"][color]["trials"]
            for index in range(len(trials), args.episodes):
                reset_episode(
                    sim,
                    robot,
                    episode_rng(args.seed, color, index),
                    args.position_noise,
                )
                runner.move_home()
                replans = 0
                for _ in range(args.max_replans):
                    if inside_box(sim, color):
                        break
                    state, images = sim.observe(robot)
                    actions = client.infer(state, images, prompt(color))
                    runner.execute(actions)
                    replans += 1
                success = inside_box(sim, color)
                trials.append(
                    {"episode": index, "success": success, "replans": replans}
                )
                summarize(results)
                atomic_save(args.output, results)
                print(
                    f"[评测] {color} {index + 1}/{args.episodes}: "
                    f"success={success}, replans={replans}",
                    flush=True,
                )
    finally:
        sim.close()

    summarize(results)
    atomic_save(args.output, results)
    print(
        json.dumps(
            results["per_color"] | {"overall": results["overall"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[保存] {args.output}")


if __name__ == "__main__":
    main()
