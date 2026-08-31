"""Dependency-light helpers that make the Evo-RL RELEASE semantics testable."""

from __future__ import annotations

import numpy as np

from hil import HILController, HILState


def execute_recorded_action(
    sim, robot, recorder, action, *, state, source: str, policy_action=None
) -> np.ndarray:
    """Record one 10 Hz frame, then hold its absolute target for one control period."""
    images = sim.cameras.capture()
    observation_state = robot.state()
    applied = robot.apply(action)
    recorder.add(
        images,
        observation_state,
        applied,
        intervention_state=int(state),
        source=source,
        policy_action=policy_action,
    )
    for _ in range(sim.cfg.record_every):
        sim.task.update((0.785 + float(applied[15])) / 0.785)
        sim.step(True)
    return applied


def execute_release_frame(
    sim, robot, recorder, runtime, task: str, controller: HILController
) -> np.ndarray:
    """Clear the queue, infer from the corrected observation, and record exactly one S2 frame."""
    if controller.state is not HILState.RELEASE:
        raise RuntimeError("release frame requires RELEASE state")
    runtime.reset()
    images = sim.cameras.capture()
    state = robot.state()
    action = runtime.next_action(images, state, task, force_fresh=False)
    applied = robot.apply(action)
    recorder.add(
        images,
        state,
        applied,
        intervention_state=int(HILState.RELEASE),
        source="policy_release",
        policy_action=np.asarray(action, dtype=np.float32),
    )
    for _ in range(sim.cfg.record_every):
        sim.task.update((0.785 + float(applied[15])) / 0.785)
        sim.step(True)
    controller.after_frame()
    return applied
