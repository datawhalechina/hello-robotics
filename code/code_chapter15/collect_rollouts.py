"""Collect one round with automatic, rather than manual, correction."""

from __future__ import annotations

import argparse
import signal
import traceback

import numpy as np

from auto_expert import AutoExpert, IKError
from config import (
    COLORS,
    POSITION_NOISE,
    SEED,
    TASK_TEMPLATE,
    SimulationConfig,
    rollout_dir,
)
from dataset import EpisodeRecorder, prepare_dir
from hil import HILController, HILState, ProgressDetector
from policy_client import RemotePolicy, unsafe_action
from robot import G2Robot
from rollout_core import execute_recorded_action, execute_release_frame
from simulation import G2Simulation


class CollectionSignal(BaseException):
    """Keep termination signals out of broad ``except Exception`` recovery paths."""

    def __init__(self, signum: int):
        self.signum = signum
        self.signal_name = signal.Signals(signum).name
        super().__init__(f"received {self.signal_name} ({signum})")


def run_automatic_correction(
    sim, robot, recorder, runtime, task, color, controller
):
    """Run one Evo-RL ACTIVE segment and its one-frame RELEASE transition.

    ``AutoExpert.correct`` plans every required IK waypoint before it records any
    ACTIVE frame. Therefore an IK failure can safely abort ACTIVE without
    inventing a RELEASE frame.
    """
    controller.start()
    expert = AutoExpert(sim, robot, recorder=recorder, intervention=True)
    try:
        segment = expert.correct(color)
    except IKError as exc:
        controller.abort()
        runtime.reset()
        return None, exc

    controller.finish()
    execute_release_frame(sim, robot, recorder, runtime, str(task), controller)
    if controller.state is not HILState.POLICY:
        raise AssertionError("RELEASE must last exactly one frame")
    return segment, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--episodes-per-color", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--position-noise", type=float, default=POSITION_NOISE)
    parser.add_argument(
        "--stagnation-frames",
        type=int,
        default=None,
        help="compatibility alias: sets both progress and motion patience",
    )
    parser.add_argument("--progress-patience", type=int, default=80)
    parser.add_argument("--motion-patience", type=int, default=35)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep completed episode files and collect only missing episodes",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume cannot be used together")

    output = rollout_dir(args.round)
    prepare_dir(output, args.overwrite, resume=args.resume)
    rng = np.random.default_rng(args.seed + args.round)
    runtime = RemotePolicy(args.host, args.port)
    sim = G2Simulation(SimulationConfig(headless=args.headless))
    current_episode = "initialization"
    current_phase = "initialize robot"
    shutdown_reason = "normal completion"

    def stop_on_signal(signum, _frame):
        raise CollectionSignal(signum)

    # SimulationApp installs its own SIGINT handler. Replace it after startup so an
    # external timeout/Ctrl-C is reported before the application shuts down.
    signal.signal(signal.SIGINT, stop_on_signal)
    signal.signal(signal.SIGTERM, stop_on_signal)

    try:
        robot = G2Robot(sim.articulation)
        total = len(COLORS) * args.episodes_per_color
        print(
            f"collection plan: colors={COLORS}, episodes_per_color="
            f"{args.episodes_per_color}, total={total}",
            flush=True,
        )
        for color_index, color in enumerate(COLORS):
            for episode_in_color in range(args.episodes_per_color):
                global_episode = (
                    args.round * 1_000_000
                    + color_index * args.episodes_per_color
                    + episode_in_color
                )
                episode_path = output / f"episode_{global_episode:06d}.npz"
                current_episode = (
                    f"{episode_path.name} color={color} "
                    f"index={episode_in_color + 1}/{args.episodes_per_color}"
                )
                current_phase = "resume validation"
                if args.resume and episode_path.exists():
                    with np.load(episode_path, allow_pickle=False) as existing:
                        saved_color = str(existing["target_color"].item())
                        saved_task_index = int(existing["task_index"][0])
                    if saved_color != color or saved_task_index != color_index:
                        raise RuntimeError(
                            f"cannot resume: {episode_path} belongs to "
                            f"color={saved_color}, task_index={saved_task_index}; "
                            f"expected color={color}, task_index={color_index}. "
                            "Use the same --episodes-per-color as the original run."
                        )
                    # Keep RNG aligned with an uninterrupted run. randomize() draws
                    # two values for each task color on every episode.
                    rng.uniform(
                        -args.position_noise,
                        args.position_noise,
                        (len(COLORS), 2),
                    )
                    print(f"resume: keeping {episode_path.name}")
                    continue

                current_phase = "episode reset and randomization"
                robot.reset()
                sim.task.randomize(rng, args.position_noise)
                runtime.reset()
                for _ in range(30):
                    sim.step(True)
                task = TASK_TEMPLATE.format(color=color)
                recorder = EpisodeRecorder(
                    output,
                    global_episode,
                    task,
                    color,
                    color_index,
                    collector_policy_id=f"acp_round_{args.round:03d}",
                )
                controller = HILController()
                progress_patience = args.progress_patience
                motion_patience = args.motion_patience
                if args.stagnation_frames is not None:
                    progress_patience = motion_patience = args.stagnation_frames
                detector = ProgressDetector(
                    progress_patience=progress_patience,
                    motion_patience=motion_patience,
                )
                detector.reset(sim.task.goal_distance(color), robot.state())
                correction_segments: list[str] = []
                trigger = False
                stop_reason = "timeout"

                current_phase = "policy rollout"
                while len(recorder) < args.max_frames:
                    if sim.task.success(color):
                        stop_reason = "success"
                        break

                    if trigger:
                        current_phase = "automatic correction"
                        segment, correction_error = run_automatic_correction(
                            sim,
                            robot,
                            recorder,
                            runtime,
                            task,
                            color,
                            controller,
                        )
                        if correction_error is not None:
                            stop_reason = "automatic_correction_ik_failed"
                            print(
                                f"{current_episode}: {stop_reason}: "
                                f"{correction_error}",
                                flush=True,
                            )
                            break
                        correction_segments.append(segment)
                        detector.reset(sim.task.goal_distance(color), robot.state())
                        trigger = False
                        current_phase = "policy rollout"
                        continue

                    images = sim.cameras.capture()
                    state = robot.state()
                    try:
                        action = runtime.next_action(images, state, task)
                    except Exception as exc:
                        print(
                            f"policy inference failed, automatic correction starts: {exc}"
                        )
                        trigger = True
                        continue
                    if unsafe_action(action, state):
                        runtime.reset()
                        trigger = True
                        continue
                    execute_recorded_action(
                        sim,
                        robot,
                        recorder,
                        action,
                        state=HILState.POLICY,
                        source="policy",
                        policy_action=action,
                    )
                    trigger = detector.observe(
                        sim.task.goal_distance(color), robot.state()
                    )

                current_phase = "episode save"
                success = sim.task.success(color)
                path = recorder.save(
                    success=success,
                    episode_kind="corrected_rollout",
                    stop_reason="success" if success else stop_reason,
                    correction_segments=correction_segments,
                    use_for_sft=False,
                    use_for_value=True,
                )
                print(
                    f"{path.name}: color={color} success={success} frames={len(recorder)} "
                    f"corrections={correction_segments} stop_reason="
                    f"{'success' if success else stop_reason}",
                    flush=True,
                )
        current_episode = "all requested episodes"
        current_phase = "completed"
        print(f"collection completed: {total}/{total} episodes", flush=True)
    except BaseException as exc:
        shutdown_reason = f"{type(exc).__name__}: {exc}"
        print(
            "COLLECTION STOPPED: "
            f"reason={shutdown_reason}; episode={current_episode}; phase={current_phase}",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        print(
            f"closing Isaac Sim: reason={shutdown_reason}; "
            f"episode={current_episode}; phase={current_phase}",
            flush=True,
        )
        runtime.close()
        sim.close()


if __name__ == "__main__":
    main()
