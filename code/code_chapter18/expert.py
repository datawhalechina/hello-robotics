"""自动脚本专家：替代真机实验中的人工接管。"""

from __future__ import annotations

import numpy as np

try:
    from .config import ControlConfig
    from .kinematics import RightArmKinematics
    from .robot import quintic_blend
    from .vla_client import ChunkRunner
except ImportError:
    from config import ControlConfig
    from kinematics import RightArmKinematics
    from robot import quintic_blend
    from vla_client import ChunkRunner


class ScriptedExpert:
    def __init__(
        self, simulation, robot, recorder, *, correction: bool = False
    ) -> None:
        self.sim = simulation
        self.robot = robot
        self.recorder = recorder
        self.correction = bool(correction)
        self.kinematics = RightArmKinematics()
        self.grasp_rotation = self.kinematics.forward(robot.home_action[7:14]).rotation
        if (
            recorder.dataset_fps < 1
            or simulation.config.physics_hz % recorder.dataset_fps
        ):
            raise ValueError("数据频率必须整除物理频率")
        self.substeps = simulation.config.physics_hz // recorder.dataset_fps
        self.runner = ChunkRunner(
            robot, simulation, ControlConfig(physics_steps_per_action=self.substeps)
        )
        self._scheduled_physics_steps = 0
        self._target_count = 0

    def _step(self, action: np.ndarray) -> None:
        state, images = self.sim.observe(self.robot)
        observation_time = self.sim.world.current_time
        image_times = [
            c.get_current_frame()["rendering_time"]
            for c in (
                self.sim.cameras.head,
                self.sim.cameras.left,
                self.sim.cameras.right,
            )
        ]
        applied = self.runner.execute_target(action)
        self.recorder.record(
            images,
            state,
            applied,
            observation_time=observation_time,
            image_times=image_times,
            correction=self.correction,
            source="auto_corrector" if self.correction else "demonstration",
        )

    def move_action(self, target, duration: float) -> None:
        start = self.robot.state16().astype(float)
        target = np.asarray(target, dtype=float).reshape(16)
        if duration <= 0:
            raise ValueError("动作阶段时长必须大于 0")
        # 按累计时长分配动作数，避免各阶段独立取整累积出额外帧。
        self._scheduled_physics_steps += round(duration * self.sim.config.physics_hz)
        target_count = self._scheduled_physics_steps // self.substeps
        steps = target_count - self._target_count
        if steps < 1:
            raise ValueError("动作阶段短于一个控制周期")
        for index in range(1, steps + 1):
            self._step(start + quintic_blend(index / steps) * (target - start))
        self._target_count = target_count

    def move(self, right_target, right_gripper: float, duration: float) -> None:
        target = self.robot.state16().astype(float)
        target[7:14] = np.asarray(right_target)
        target[15] = float(right_gripper)
        self.move_action(target, duration)

    def solve(self, position, seed=None) -> np.ndarray:
        initial = self.robot.state16()[7:14] if seed is None else seed
        result = self.kinematics.inverse(position, self.grasp_rotation, initial)
        if not result.success and result.position_error > 0.01:
            raise RuntimeError(
                f"自动专家 IK 失败：target={np.round(position, 3)}, error={result.position_error:.4f}"
            )
        return result.joints

    def run(self, color: str) -> None:
        home = self.robot.home_action
        self.move_action(home, 1.2)
        block = self.sim.task.block_arm_positions[color].copy()
        above = block.copy()
        above[1] -= self.sim.task.config.pregrasp_clearance
        grasp = block.copy()
        index = tuple(self.sim.task.blocks).index(color)
        grasp[1] -= self.sim.task.config.grasp_clearances[index]
        box = np.asarray(self.sim.task.config.box_arm_position, dtype=float)
        above_box = box.copy()
        above_box[1] = self.sim.task.config.table_top_arm_y - 0.30
        place = box.copy()
        place[1] = (
            self.sim.task.config.table_top_arm_y - self.sim.task.config.place_clearance
        )

        q_above = self.solve(above)
        q_grasp = self.solve(grasp, q_above)
        q_above_box = self.solve(above_box, q_above)
        q_place = self.solve(place, q_above_box)
        self.move(q_above, -0.785, 1.1)
        self.move(q_grasp, -0.785, 0.7)
        self.move(q_grasp, 0.0, 0.65)
        self.move(q_above, 0.0, 1.0)
        self.move(q_above_box, 0.0, 1.1)
        self.move(q_place, 0.0, 0.65)
        self.move(q_place, -0.785, 0.4)
        self.move(q_above_box, -0.785, 0.7)
        self.move_action(home, 1.0)
