"""Exact target, distributional regression and ACP labeling math."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    task_index: int
    length: int
    success: bool


def value_targets(
    episodes: np.ndarray,
    frames: np.ndarray,
    info: dict[int, EpisodeInfo],
    task_max: dict[int, int],
    c_fail_coef: float = 1.0,
) -> np.ndarray:
    if episodes.shape != frames.shape:
        raise ValueError("episodes and frames must have equal shape")
    out = np.empty(len(episodes), np.float32)
    for i, (ep_id, frame) in enumerate(zip(episodes, frames, strict=True)):
        ep = info[int(ep_id)]
        maximum = int(task_max[ep.task_index])
        remaining = ep.length - int(frame) - 1
        c_fail = maximum * c_fail_coef
        ret = -float(remaining) - (0.0 if ep.success else c_fail)
        out[i] = np.clip(ret / (maximum + c_fail), -1.0, 0.0)
    return out


def bin_centers(count: int = 201, low: float = -1.0, high: float = 0.0) -> np.ndarray:
    return np.linspace(low, high, count, dtype=np.float32)


def two_hot(values: np.ndarray, centers: np.ndarray | None = None) -> np.ndarray:
    centers = bin_centers() if centers is None else np.asarray(centers, np.float32)
    values = np.clip(
        np.asarray(values, np.float32).reshape(-1), centers[0], centers[-1]
    )
    scaled = (values - centers[0]) / (centers[1] - centers[0])
    lo = np.floor(scaled).astype(np.int64)
    hi = np.minimum(lo + 1, len(centers) - 1)
    hi_w = np.clip(scaled - lo, 0, 1)
    lo_w = 1 - hi_w
    result = np.zeros((len(values), len(centers)), np.float32)
    rows = np.arange(len(values))
    np.add.at(result, (rows, lo), lo_w)
    np.add.at(result, (rows, hi), hi_w)
    return result


def expected_value(logits: np.ndarray, centers: np.ndarray | None = None) -> np.ndarray:
    centers = bin_centers() if centers is None else centers
    logits = np.asarray(logits, np.float32)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs @ np.asarray(centers, np.float32)


def dense_rewards(
    targets: np.ndarray, episodes: np.ndarray, frames: np.ndarray
) -> np.ndarray:
    targets, episodes, frames = map(np.asarray, (targets, episodes, frames))
    out = np.empty(len(targets), np.float32)
    for i in range(len(targets)):
        contiguous = (
            i + 1 < len(targets)
            and episodes[i + 1] == episodes[i]
            and frames[i + 1] == frames[i] + 1
        )
        out[i] = targets[i] - targets[i + 1] if contiguous else targets[i]
    return out


def n_step_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    episodes: np.ndarray,
    frames: np.ndarray,
    n_step: int = 50,
) -> np.ndarray:
    if n_step <= 0:
        raise ValueError("n_step must be positive")
    rewards, values, episodes, frames = map(
        np.asarray, (rewards, values, episodes, frames)
    )
    out = np.zeros(len(rewards), np.float32)
    for i in range(len(rewards)):
        total = 0.0
        steps = 0
        j = i
        while (
            steps < n_step
            and j < len(rewards)
            and episodes[j] == episodes[i]
            and frames[j] == frames[i] + steps
        ):
            total += float(rewards[j])
            steps += 1
            j += 1
        bootstrap = (
            float(values[j])
            if steps == n_step
            and j < len(values)
            and episodes[j] == episodes[i]
            and frames[j] == frames[i] + n_step
            else 0.0
        )
        out[i] = total + bootstrap - float(values[i])
    return out


def acp_labels(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    interventions: np.ndarray,
    positive_ratio: float = 0.3,
    force_intervention_positive: bool = True,
) -> tuple[np.ndarray, dict[int, float]]:
    if not 0 <= positive_ratio <= 1:
        raise ValueError("positive_ratio must be in [0,1]")
    task_indices, advantages, interventions = map(
        np.asarray, (task_indices, advantages, interventions)
    )
    thresholds = {
        int(t): float(np.quantile(advantages[task_indices == t], 1 - positive_ratio))
        for t in np.unique(task_indices)
    }
    labels = np.array(
        [
            int(a >= thresholds[int(t)])
            for t, a in zip(task_indices, advantages, strict=True)
        ],
        np.int64,
    )
    if force_intervention_positive:
        labels[interventions.astype(np.float32) > 0.5] = 1
    return labels, thresholds
