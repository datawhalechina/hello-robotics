"""Execute a fine-tuned Motus Stage-3 policy in the Chapter-16 G2 task.

``official`` executes the complete 16-action absolute-qpos chunk, matching Motus'
RoboTwin evaluator. ``receding`` executes a shorter prefix with an optional rate
limiter and then observes the scene again.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from control import GraspReleaseGuard, clip_joint_limits, rate_limit_target
from environment import COLORS, G2Robot, G2Simulation, MotusSimulationConfig, closed_fraction
from policy_rpc import RemotePolicy
from settings import OUTPUT_ROOT, TASK_TEXT


def keep_simulation_alive(sim, robot, hold_target: np.ndarray) -> None:
    """Advance/render Isaac Sim while holding the request-time robot posture."""
    hold_target = np.asarray(hold_target, np.float32)
    applied = robot.apply(hold_target)
    sim.task.update(closed_fraction(applied[15]))
    sim.step(True)


def wait_for_prediction(
    sim, robot, future: Future, hold_target: np.ndarray,
    timeout: float | None = None,
):
    """Wait for Motus in the RPC thread while Isaac Sim keeps rendering."""
    started = time.monotonic()
    # Pace display updates approximately at the configured render rate instead
    # of busy-stepping the simulator as fast as the CPU allows.
    frame_period = 1.0 / max(float(sim.cfg.render_hz), 1.0)
    while not future.done():
        tick = time.monotonic()
        keep_simulation_alive(sim, robot, hold_target)
        if timeout is not None and time.monotonic() - started > timeout:
            raise TimeoutError(f"Motus inference exceeded {timeout:.1f}s")
        remaining = frame_period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)
    return future.result()


def execute_target(sim, robot, target: np.ndarray, seconds: float = 0.1) -> None:
    start = robot.state().astype(np.float32)
    steps = max(1, round(seconds * sim.cfg.physics_hz))
    for index in range(1, steps + 1):
        alpha = index / steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        applied = robot.apply(start + alpha * (target - start))
        sim.task.update(closed_fraction(applied[15]))
        sim.step(True)


def main() -> None:
    parser = argparse.ArgumentParser(description="closed-loop Motus evaluation in Isaac Sim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8616)
    parser.add_argument("--colors", nargs="*", choices=COLORS, default=list(COLORS))
    parser.add_argument("--episodes-per-color", type=int, default=10)
    parser.add_argument("--execution-mode", choices=("official", "receding"), default="official")
    parser.add_argument(
        "--execute-steps", type=int, default=None,
        help="actions per prediction; defaults to 16 in official mode and 4 in receding mode",
    )
    parser.add_argument("--max-replans", type=int, default=20)
    parser.add_argument("--position-noise", type=float, default=0.01)
    parser.add_argument("--max-arm-step", type=float, default=0.40)
    parser.add_argument("--max-gripper-step", type=float, default=0.80)
    parser.add_argument("--action-seconds", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=16160)
    parser.add_argument(
        "--enable-grasp-guard", action="store_true",
        help="optional non-official right-gripper state machine (off by default)",
    )
    parser.add_argument(
        "--inference-timeout", type=float, default=900.0,
        help="maximum wait for one Motus RPC request while Isaac Sim keeps stepping",
    )
    parser.add_argument(
        "--pause-during-inference", action="store_true",
        help="block Isaac Sim while waiting; default keeps the display running",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "stage3_evaluation.json")
    args = parser.parse_args()

    execute_steps = args.execute_steps
    if execute_steps is None:
        execute_steps = 16 if args.execution_mode == "official" else 4
    if not 1 <= execute_steps <= 16:
        parser.error("--execute-steps must be in [1,16]")
    if args.execution_mode == "official" and execute_steps != 16:
        parser.error("Motus official mode always executes the complete 16-action chunk")
    if args.execution_mode == "official" and args.enable_grasp_guard:
        parser.error("--enable-grasp-guard is not part of the Motus official evaluation flow")

    policy = RemotePolicy(args.host, args.port)
    rng = np.random.default_rng(args.seed)
    # Keep Isaac Sim responsive during the remote GPU inference by default.
    # In official mode only the display/wait implementation is asynchronous;
    # the returned full 16-action chunk is still executed with the official
    # action protocol and no guard/limiter.
    use_async_rpc = not args.pause_during_inference
    inference_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="motus-rpc")
        if use_async_rpc else None
    )
    print(
        "evaluation protocol: "
        + (f"Motus official action protocol (async_display={use_async_rpc}, "
             "full 16-action qpos chunk, no guard/limiter)"
           if args.execution_mode == "official"
           else f"Chapter-16 receding (execute_steps={execute_steps}, async_rpc={use_async_rpc}, "
                f"grasp_guard={args.enable_grasp_guard})"),
        flush=True,
    )
    rows = []
    sim = G2Simulation(MotusSimulationConfig(headless=args.headless))
    try:
        robot = G2Robot(sim.articulation)
        for color in args.colors:
            for episode in range(args.episodes_per_color):
                robot.reset()
                sim.task.randomize(rng, args.position_noise)
                for _ in range(30):
                    sim.step(True)
                started = time.monotonic()
                limited = 0
                reason = "max_replans"
                replans = 0
                executed = 0
                max_raw_arm_delta = 0.0
                max_raw_gripper_delta = 0.0
                right_gripper_min = float("inf")
                right_gripper_max = float("-inf")
                grasp_guard = GraspReleaseGuard(color) if args.enable_grasp_guard else None
                if grasp_guard is not None:
                    grasp_guard.reset(color)
                try:
                    for replan in range(args.max_replans):
                        replans = replan + 1
                        if sim.task.success(color):
                            reason = "success"
                            break
                        images = sim.cameras.capture()
                        state_snapshot = robot.state().copy()
                        instruction = TASK_TEXT[color]
                        if not use_async_rpc:
                            chunk = policy.predict(images, state_snapshot, instruction)
                        else:
                            # Optional Chapter-16 convenience mode only.
                            future = inference_pool.submit(
                                policy.predict, images, state_snapshot, instruction
                            )
                            chunk = wait_for_prediction(
                                sim, robot, future, state_snapshot,
                                args.inference_timeout,
                            )
                        if chunk.shape != (16, 16):
                            raise ValueError(f"expected action chunk (16,16), got {chunk.shape}")
                        for raw in chunk[:execute_steps]:
                            current = robot.state()
                            if args.execution_mode == "official":
                                target, report = clip_joint_limits(raw, current)
                            else:
                                target, report = rate_limit_target(
                                    raw, current, args.max_arm_step, args.max_gripper_step
                                )
                            guard_report = {}
                            if grasp_guard is not None:
                                target, guard_report = grasp_guard.filter_target(sim, robot, target)
                            limited += int(report["limited"])
                            max_raw_arm_delta = max(max_raw_arm_delta, report["raw_max_arm_delta"])
                            max_raw_gripper_delta = max(max_raw_gripper_delta, report["raw_max_gripper_delta"])
                            right_gripper_min = min(right_gripper_min, float(target[15]))
                            right_gripper_max = max(right_gripper_max, float(target[15]))
                            execute_target(sim, robot, target, args.action_seconds)
                            executed += 1
                            # Official Motus executes the complete predicted
                            # action chunk before the benchmark checks success.
                            # Receding mode may stop early as a Chapter-16 safety
                            # convenience.
                            if args.execution_mode != "official" and sim.task.success(color):
                                reason = "success"
                                break
                        if args.execution_mode != "official" and reason == "success":
                            break
                except Exception as exc:
                    reason = f"error:{type(exc).__name__}:{exc}"
                success = sim.task.success(color)
                final_state = robot.state()
                final_block = np.asarray(sim.task.block_position(color), dtype=float)
                row = {
                    "color": color,
                    "episode": episode,
                    "success": bool(success),
                    "reason": "success" if success else reason,
                    "execution_mode": args.execution_mode,
                    "protocol": (
                        "motus_official_full_chunk"
                        if args.execution_mode == "official"
                        else "chapter16_receding"
                    ),
                    "execute_steps": execute_steps,
                    "replans": replans,
                    "executed_actions": executed,
                    "limited_actions": limited,
                    "limited_fraction": limited / max(executed, 1),
                    "max_raw_arm_delta": max_raw_arm_delta,
                    "max_raw_gripper_delta": max_raw_gripper_delta,
                    "right_gripper_target_range": [
                        None if not np.isfinite(right_gripper_min) else right_gripper_min,
                        None if not np.isfinite(right_gripper_max) else right_gripper_max,
                    ],
                    "final_right_gripper_state": float(final_state[15]),
                    "grasp_guard": None if grasp_guard is None else {
                        "phase": grasp_guard.phase,
                        "locked_actions": grasp_guard.locked_actions,
                        "release_actions": grasp_guard.release_actions,
                    },
                    "final_block_position": final_block.tolist(),
                    "inside_box": bool(sim.task.inside_box(color)),
                    "seconds": time.monotonic() - started,
                }
                rows.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                print(row, flush=True)
    finally:
        if inference_pool is not None:
            inference_pool.shutdown(wait=True, cancel_futures=True)
        sim.close()


if __name__ == "__main__":
    main()
