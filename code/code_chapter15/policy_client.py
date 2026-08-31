"""pi0.5 websocket client with an explicit 50-action queue."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import numpy as np

from acp import tagged_task
from config import ACTION_DIM, PI05_ACTION_STEPS
from openpi_runtime import prepare_openpi_client


def policy_observation(images, state, task: str) -> dict:
    """Build the native 16D G2 observation expected by ``pi05_adapter.Inputs``."""
    head, left, right = images
    return {
        "images": {
            "head": np.asarray(head, dtype=np.uint8),
            "left_wrist": np.asarray(left, dtype=np.uint8),
            "right_wrist": np.asarray(right, dtype=np.uint8),
        },
        "state": np.asarray(state, dtype=np.float32).reshape(ACTION_DIM),
        # Evo-RL inference requests the positive branch. No CFG is used.
        "prompt": tagged_task(task, positive=True),
    }



@dataclass
class RemotePolicy:
    host: str = "127.0.0.1"
    port: int = 8000
    execute: int = PI05_ACTION_STEPS
    connect_timeout: float = 60.0
    inference_timeout: float = 300.0
    verbose: bool = False
    _queue: deque[np.ndarray] = field(default_factory=deque, init=False)
    fresh_inferences: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.execute <= PI05_ACTION_STEPS:
            raise ValueError(f"execute must be in [1, {PI05_ACTION_STEPS}]")
        if self.connect_timeout <= 0 or self.inference_timeout <= 0:
            raise ValueError("policy timeouts must be positive")
        prepare_openpi_client()
        from openpi_client import msgpack_numpy
        import websockets.sync.client

        uri = self.host if self.host.startswith("ws") else f"ws://{self.host}:{self.port}"
        print(
            f"connecting to policy server {uri} "
            f"(timeout={self.connect_timeout:g}s)...",
            flush=True,
        )
        deadline = time.monotonic() + self.connect_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                self._ws = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    open_timeout=min(5.0, remaining),
                )
                self.server_metadata = msgpack_numpy.unpackb(
                    self._ws.recv(timeout=min(5.0, remaining))
                )
                break
            except (ConnectionRefusedError, OSError, TimeoutError) as exc:
                last_error = exc
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        else:
            raise TimeoutError(
                f"policy server did not become ready at {uri} within "
                f"{self.connect_timeout:g}s"
            ) from last_error
        self._packer = msgpack_numpy.Packer()
        print(f"connected to policy server {uri}", flush=True)

    def reset(self) -> None:
        """Discard stale chunk actions; RELEASE always calls this before inference."""
        self._queue.clear()

    def _infer_with_timeout(self, observation: dict) -> dict:
        """Use the pinned client's wire format, but bound the blocking receive."""
        from openpi_client import msgpack_numpy

        data = self._packer.pack(observation)
        self._ws.send(data)
        try:
            response = self._ws.recv(timeout=self.inference_timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"policy inference exceeded {self.inference_timeout:g}s; "
                "check the serve_policy.py terminal"
            ) from exc
        if isinstance(response, str):
            raise RuntimeError(f"error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        if hasattr(self, "_ws"):
            self._ws.close()


    def infer_fresh(self, images, state, task: str) -> np.ndarray:
        number = self.fresh_inferences + 1
        started = time.monotonic()
        if self.verbose:
            print(f"policy inference {number} started...", flush=True)
        result = self._infer_with_timeout(policy_observation(images, state, task))
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(
                f"server returned actions with shape {actions.shape}, expected (N, {ACTION_DIM})"
            )
        if len(actions) < self.execute:
            raise ValueError(
                f"server returned only {len(actions)} actions, execute={self.execute}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("server returned NaN/Inf actions")
        self._queue.extend(actions[: self.execute])
        self.fresh_inferences += 1
        if self.verbose:
            server_ms = result.get("server_timing", {}).get("infer_ms")
            timing = (
                f", server={server_ms / 1000:.2f}s"
                if isinstance(server_ms, (int, float))
                else ""
            )
            print(
                f"policy inference {number} finished in "
                f"{time.monotonic() - started:.2f}s{timing}",
                flush=True,
            )
        return self._queue.popleft().copy()

    def next_action(
        self, images, state, task: str, *, force_fresh: bool = False
    ) -> np.ndarray:
        if force_fresh:
            self.reset()
        if not self._queue:
            return self.infer_fresh(images, state, task)
        return self._queue.popleft().copy()


def unsafe_action(
    action: np.ndarray, state: np.ndarray, max_joint_jump: float = 0.65
) -> bool:
    """Simple automatic trigger for numerically unsafe policy outputs."""
    action = np.asarray(action, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
        return True
    return bool(np.max(np.abs(action[:14] - state[:14])) > max_joint_jump)
