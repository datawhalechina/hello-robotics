"""Isaac Sim 端的独立 OpenPI WebSocket 客户端与动作执行器。"""

from __future__ import annotations

import importlib
import sys

import numpy as np
from config import OPENPI_ROOT, ControlConfig
from robot import gripper_closed_amount, quintic_blend


def add_client_dependencies() -> None:
    source = OPENPI_ROOT / "packages/openpi-client/src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    for package in ("msgpack", "websockets"):
        try:
            importlib.import_module(package)
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"Isaac Sim Python 缺少 {package}，请为 Isaac Python 安装该包"
            ) from error


class VLAClient:
    def __init__(self, host: str, port: int):
        add_client_dependencies()
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.policy.get_server_metadata()
        self.state_dim = int(self.metadata.get("state_dim", 16))
        self.action_dim = int(self.metadata.get("action_dim", 16))
        print(f"[VLA] 已连接 ws://{host}:{port}，{self.metadata}", flush=True)

    def _request(self, state, images, prompt: str, **extra) -> np.ndarray:
        head, left, right = images
        result = self.policy.infer(
            {
                "images": {"top_head": head, "hand_left": left, "hand_right": right},
                "state": np.asarray(state, dtype=np.float32),
                "prompt": prompt,
                **extra,
            }
        )
        actions = np.asarray(result["actions"], dtype=np.float32)[:, : self.action_dim]
        if actions.ndim != 2 or not len(actions) or not np.isfinite(actions).all():
            raise RuntimeError(f"模型动作无效：{actions.shape}")
        return actions

    def infer(self, state, images, prompt: str) -> np.ndarray:
        return self._request(state, images, prompt)


class ChunkRunner:
    def __init__(self, robot, simulation, config: ControlConfig, control_waist=False):
        self.robot, self.sim, self.config, self.control_waist = (
            robot,
            simulation,
            config,
            control_waist,
        )

    def _step(self, action):
        self.sim.task.update(gripper_closed_amount(float(action[15])))
        self.sim.step(render=True)

    def move_home(self):
        start, target = (
            self.robot.state16().astype(float),
            self.robot.home_action.astype(float),
        )
        steps = max(1, round(self.config.home_duration_s * self.sim.config.physics_hz))
        for index in range(1, steps + 1):
            applied = self.robot.apply_absolute(
                start + quintic_blend(index / steps) * (target - start)
            )
            self._step(applied)

    def hold(self):
        action = self.robot.state(self.control_waist)
        for _ in range(self.config.physics_steps_per_action):
            self._step(self.robot.apply_absolute(action, self.control_waist))

    def execute(self, actions: np.ndarray):
        for target in actions[: self.config.execute_chunk or len(actions)]:
            self.execute_target(target)

    def execute_target(self, target: np.ndarray) -> np.ndarray:
        """专家与模型共用：下发绝对关节目标，并保持一个动作周期。"""
        if self.config.physics_steps_per_action < 1:
            raise ValueError("每个动作至少执行一个物理步")
        applied = self.robot.apply_absolute(target, self.control_waist)
        for _ in range(self.config.physics_steps_per_action):
            self._step(applied)
        return applied
