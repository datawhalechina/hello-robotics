"""Isaac Sim侧的轻量WebSocket客户端与动作块执行器。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

try:
    from .config import ControlConfig, OPENPI_ROOT
    from .robot import gripper_closed_amount, quintic_blend
except ImportError:
    from config import ControlConfig, OPENPI_ROOT
    from robot import gripper_closed_amount, quintic_blend


def add_client_dependencies() -> None:
    client_src = OPENPI_ROOT / "packages/openpi-client/src"
    if str(client_src) not in sys.path:
        sys.path.insert(0, str(client_src))
    missing = []
    for package in ("msgpack", "websockets"):
        try:
            importlib.import_module(package)
        except ModuleNotFoundError:
            missing.append(package)
    if missing:
        site = OPENPI_ROOT / ".venv/lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if site.is_dir() and str(site) not in sys.path:
            sys.path.append(str(site))
    for package in missing:
        try:
            importlib.import_module(package)
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"Isaac Sim Python缺少{package}。请执行："
                f"/home/robot/isaac-sim/python.sh -m pip install {package}"
            ) from error


class Pi05Client:
    def __init__(self, host: str, port: int) -> None:
        add_client_dependencies()
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        import openpi_client

        module_path = getattr(openpi_client, "__file__", "")
        if not module_path or not Path(module_path).resolve().is_relative_to(OPENPI_ROOT.resolve()):
            raise RuntimeError(f"检测到外部openpi_client：{module_path}")

        print(f"[VLA] 使用本章客户端：{module_path}", flush=True)
        print(f"[VLA] 正在连接 ws://{host}:{port} ...", flush=True)
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.policy.get_server_metadata()
        self.state_dim = int(self.metadata.get("state_dim", 16))
        self.action_dim = int(self.metadata.get("action_dim", 16))
        print(f"[VLA] 已连接，服务信息：{self.metadata}", flush=True)

    def infer(self, state, images, prompt: str) -> np.ndarray:
        head, left, right = images
        result = self.policy.infer({
            "images": {"top_head": head, "hand_left": left, "hand_right": right},
            "state": np.asarray(state, dtype=np.float32),
            "prompt": prompt,
        })
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] < self.action_dim:
            raise RuntimeError(f"模型返回动作形状错误：{actions.shape}")
        actions = actions[:, : self.action_dim]
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("模型动作包含NaN或无穷大")
        return actions


class ChunkRunner:
    """按顺序执行模型给出的绝对关节目标，不做缩放或低通滤波。"""

    def __init__(self, robot, simulation, config: ControlConfig, control_waist=False) -> None:
        self.robot = robot
        self.sim = simulation
        self.config = config
        self.control_waist = control_waist

    def _task_step(self, action) -> None:
        self.sim.task.update(gripper_closed_amount(float(action[15])))
        self.sim.step(render=True)

    def move_home(self) -> None:
        start = self.robot.state16().astype(np.float64)
        target = self.robot.home_action.astype(np.float64)
        steps = max(1, round(self.config.home_duration_s * self.sim.config.physics_hz))
        for index in range(1, steps + 1):
            action = start + quintic_blend(index / steps) * (target - start)
            applied = self.robot.apply_absolute(action)
            self._task_step(applied)

    def hold(self) -> None:
        action = self.robot.state(self.control_waist)
        for _ in range(self.config.physics_steps_per_action):
            applied = self.robot.apply_absolute(action, self.control_waist)
            self._task_step(applied)

    def execute(self, actions: np.ndarray) -> None:
        count = self.config.execute_chunk or len(actions)
        for target in actions[:count]:
            applied = self.robot.apply_absolute(target, self.control_waist)
            for _ in range(self.config.physics_steps_per_action):
                self._task_step(applied)

    def print_grippers(self, state: np.ndarray, actions: np.ndarray) -> None:
        """打印夹爪原始输出，用于区分模型切换和物理关节抖动。"""
        count = min(self.config.execute_chunk or len(actions), len(actions))
        values = np.asarray(actions[:count, 14:16])
        text = np.array2string(values, precision=3, suppress_small=True, max_line_width=160)
        print(
            f"[夹爪调试] 当前=[{state[14]:.3f}, {state[15]:.3f}] "
            f"即将执行{count}步 [左, 右]=\n{text}",
            flush=True,
        )
