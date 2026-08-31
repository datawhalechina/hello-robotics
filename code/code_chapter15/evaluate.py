"""Pure-policy evaluation: no automatic expert and no intervention state machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from config import (
    COLORS,
    POSITION_NOISE,
    REPORT_ROOT,
    SEED,
    TASK_TEMPLATE,
    SimulationConfig,
)
from policy_client import RemotePolicy, unsafe_action
from robot import G2Robot, closed_fraction
from simulation import G2Simulation


def build_report(rows: list[dict], *, complete: bool) -> dict:
    singles = [row for row in rows if row["mode"] == "single"]
    sequential = [row for row in rows if row["mode"] == "sequential"]
    per_color = {}
    for color in COLORS:
        values = [row["success"] for row in singles if row["color"] == color]
        per_color[color] = float(np.mean(values)) if values else None
    return {
        "complete": complete,
        "no_intervention": True,
        "completed_single_episodes": len(singles),
        "completed_sequential_episodes": len(sequential),
        "single_color_success": per_color,
        "sequential_three_color_success": (
            float(np.mean([row["success"] for row in sequential]))
            if sequential
            else None
        ),
        "episodes": rows,
    }


def save_report(path: Path, rows: list[dict], *, complete: bool) -> dict:
    """Atomically checkpoint results so a later simulator shutdown cannot lose them."""
    report = build_report(rows, complete=complete)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)
    return report


def one_task(
    sim,
    robot,
    runtime,
    color: str,
    max_frames: int,
    *,
    progress_every: int,
    label: str,
) -> tuple[bool, str, int]:
    task = TASK_TEMPLATE.format(color=color)
    runtime.reset()
    started = time.monotonic()
    for frame in range(max_frames):
        if progress_every and frame and frame % progress_every == 0:
            print(
                f"{label}: frame={frame}/{max_frames} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
        if sim.task.success(color):
            return True, "success", frame
        images, state = sim.cameras.capture(), robot.state()
        try:
            action = runtime.next_action(images, state, task)
        except Exception as exc:
            print(
                f"{label}: policy failed at frame={frame}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False, f"policy_error:{type(exc).__name__}", frame
        if unsafe_action(action, state):
            return False, "unsafe_action", frame
        applied = robot.apply(action)
        for _ in range(sim.cfg.record_every):
            sim.task.update(closed_fraction(applied[15]))
            sim.step(True)
    return sim.task.success(color), "timeout", max_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-color", type=int, default=20)
    parser.add_argument("--sequential-episodes", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--position-noise", type=float, default=POSITION_NOISE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--connect-timeout", type=float, default=60.0)
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED + 10_000)
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "evaluation.json")
    args = parser.parse_args()

    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")
    rng = np.random.default_rng(args.seed)
    runtime = RemotePolicy(
        args.host,
        args.port,
        connect_timeout=args.connect_timeout,
        inference_timeout=args.inference_timeout,
        verbose=True,
    )
    print("initializing Isaac Sim scene...", flush=True)
    sim = G2Simulation(SimulationConfig(headless=args.headless))
    print("Isaac Sim scene ready; evaluation starts", flush=True)
    rows: list[dict] = []
    try:
        robot = G2Robot(sim.articulation)
        for color in COLORS:
            for episode in range(args.episodes_per_color):
                robot.reset()
                sim.task.randomize(rng, args.position_noise)
                for _ in range(30):
                    sim.step(True)
                label = (
                    f"single color={color} "
                    f"episode={episode + 1}/{args.episodes_per_color}"
                )
                print(f"START {label}", flush=True)
                success, reason, frames = one_task(
                    sim,
                    robot,
                    runtime,
                    color,
                    args.max_frames,
                    progress_every=args.progress_every,
                    label=label,
                )
                rows.append(
                    {
                        "mode": "single",
                        "episode": episode,
                        "color": color,
                        "success": success,
                        "reason": reason,
                        "frames": frames,
                    }
                )
                print(
                    f"DONE {label}: success={success} reason={reason} frames={frames}",
                    flush=True,
                )
                save_report(args.output, rows, complete=False)

        for episode in range(args.sequential_episodes):
            robot.reset()
            sim.task.randomize(rng, args.position_noise)
            for _ in range(30):
                sim.step(True)
            colors = list(COLORS)
            rng.shuffle(colors)
            completed = 0
            details = []
            for color in colors:
                label = (
                    f"sequential episode={episode + 1}/{args.sequential_episodes} "
                    f"task={completed + 1}/3 color={color}"
                )
                print(f"START {label}", flush=True)
                success, reason, frames = one_task(
                    sim,
                    robot,
                    runtime,
                    color,
                    args.max_frames,
                    progress_every=args.progress_every,
                    label=label,
                )
                print(
                    f"DONE {label}: success={success} reason={reason} frames={frames}",
                    flush=True,
                )
                details.append(
                    {
                        "color": color,
                        "success": success,
                        "reason": reason,
                        "frames": frames,
                    }
                )
                if not success:
                    break
                completed += 1
            rows.append(
                {
                    "mode": "sequential",
                    "episode": episode,
                    "order": colors,
                    "success": completed == 3,
                    "completed": completed,
                    "details": details,
                }
            )
            save_report(args.output, rows, complete=False)
        complete = True
    finally:
        report = save_report(args.output, rows, complete=locals().get("complete", False))
        print(f"evaluation report saved: {args.output.resolve()}", flush=True)
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "episodes"},
                indent=2,
            ),
            flush=True,
        )
        runtime.close()
        sim.close()


if __name__ == "__main__":
    main()
