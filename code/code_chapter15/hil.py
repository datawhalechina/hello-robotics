"""Automatic replacement for human HIL with 0/1/2 state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class HILState(IntEnum):
    """Intervention state stored on every recorded frame."""

    POLICY = 0
    ACTIVE = 1
    RELEASE = 2


@dataclass
class ProgressDetector:
    """Trigger correction when the task stalls or the robot stops moving.

    Arm motion and task progress use separate clocks. A bad policy may keep the
    joints moving forever without bringing the selected block closer to its
    goal, so joint motion must not reset the task-progress clock.
    """

    progress_patience: int = 80
    motion_patience: int = 35
    goal_epsilon: float = 0.008
    joint_epsilon: float = 0.025

    def reset(self, goal_distance: float, state: np.ndarray) -> None:
        self.best_goal_distance = float(goal_distance)
        self.previous_state = np.asarray(state, dtype=np.float32).copy()
        self.frames_without_goal_progress = 0
        self.frames_without_joint_motion = 0

    def observe(self, goal_distance: float, state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=np.float32)
        goal_distance = float(goal_distance)
        progressed = goal_distance < self.best_goal_distance - self.goal_epsilon
        moved = (
            np.linalg.norm(state[7:14] - self.previous_state[7:14]) > self.joint_epsilon
        )

        if progressed:
            self.best_goal_distance = goal_distance
            self.frames_without_goal_progress = 0
        else:
            self.frames_without_goal_progress += 1

        if moved:
            self.frames_without_joint_motion = 0
        else:
            self.frames_without_joint_motion += 1

        self.previous_state = state.copy()
        return (
            self.frames_without_goal_progress >= self.progress_patience
            or self.frames_without_joint_motion >= self.motion_patience
        )


class HILController:
    """Enforce POLICY -> ACTIVE -> RELEASE -> POLICY transitions.

    RELEASE lasts exactly one recorded frame. ``rollout_core`` clears the
    policy action queue and requests a fresh action before calling
    :meth:`after_frame`.
    """

    def __init__(self) -> None:
        self.state = HILState.POLICY

    def start(self) -> None:
        if self.state is not HILState.POLICY:
            raise RuntimeError("intervention can only start from POLICY")
        self.state = HILState.ACTIVE

    def finish(self) -> None:
        if self.state is not HILState.ACTIVE:
            raise RuntimeError("only ACTIVE can finish")
        self.state = HILState.RELEASE

    def abort(self) -> None:
        """Return to POLICY when a preplanned correction cannot start."""
        if self.state is not HILState.ACTIVE:
            raise RuntimeError("only ACTIVE can abort")
        self.state = HILState.POLICY

    def after_frame(self) -> None:
        if self.state is HILState.RELEASE:
            self.state = HILState.POLICY

    @property
    def is_intervention(self) -> bool:
        return self.state is HILState.ACTIVE


def policy_placeholder(action_dim: int = 16) -> np.ndarray:
    """Policy output stored during ACTIVE, when policy inference is disabled."""
    return np.zeros(action_dim, dtype=np.float32)
