"""脚本专家：独立生成单个颜色物块入盒示教。"""

from __future__ import annotations

import numpy as np

try:
    from .kinematics import RightArmKinematics
    from .robot import gripper_closed_amount, quintic_blend
except ImportError:
    from kinematics import RightArmKinematics
    from robot import gripper_closed_amount, quintic_blend


class ScriptedExpert:
    """左臂保持不动，右臂用本章数值IK完成抓取；记录的仍是完整16维G2动作。"""

    def __init__(self, simulation, robot, recorder) -> None:
        self.sim = simulation
        self.robot = robot
        self.recorder = recorder
        self.kinematics = RightArmKinematics()
        self.grasp_rotation = self.kinematics.forward(robot.home_action[7:14]).rotation

    def _step(self, action: np.ndarray) -> None:
        record = self.recorder.should_record()
        if record:
            images = self.sim.cameras.capture()
            state = self.robot.state16()
        applied = self.robot.apply_absolute(action)
        self.sim.task.update(gripper_closed_amount(float(applied[15])))
        self.sim.step(render=True)
        if record:
            self.recorder.record(images, state, applied)

    def move_action(self, target, duration: float) -> None:
        """平滑执行一个完整16维目标，同时把实际下发动作写入数据集。"""
        start = self.robot.state16().astype(float)
        target = np.asarray(target, dtype=float).reshape(16)
        steps = max(2, round(duration * self.sim.config.physics_hz))
        for i in range(1, steps + 1):
            self._step(start + quintic_blend(i / steps) * (target - start))

    def move(self, right_target, right_gripper: float, duration: float) -> None:
        target = self.robot.state16().astype(float)
        target[7:14] = np.asarray(right_target)
        target[15] = float(right_gripper)
        self.move_action(target, duration)

    def solve(self, position, seed=None) -> np.ndarray:
        initial = self.robot.state16()[7:14] if seed is None else seed
        result = self.kinematics.inverse(position, self.grasp_rotation, initial)
        if not result.success and result.position_error > 0.01:
            raise RuntimeError(f"示教IK失败：target={np.round(position, 3)}, error={result.position_error:.4f}")
        return result.joints

    def run(self, color: str) -> None:
        home = self.robot.home_action
        # 每个回合先将双臂和双夹爪统一到公开G2数据的初始姿态。
        self.move_action(home, 1.5)
        block = self.sim.task.block_arm_positions[color].copy()
        above = block.copy()
        above[1] -= self.sim.task.config.pregrasp_clearance
        grasp = block.copy()
        color_index = tuple(self.sim.task.blocks).index(color)
        grasp[1] -= self.sim.task.config.grasp_clearances[color_index]
        box = np.asarray(self.sim.task.config.box_arm_position, dtype=float)
        above_box = box.copy()
        above_box[1] = self.sim.task.config.table_top_arm_y - 0.30
        place = box.copy()
        place[1] = self.sim.task.config.table_top_arm_y - self.sim.task.config.place_clearance

        q_above = self.solve(above)
        q_grasp = self.solve(grasp, q_above)
        q_above_box = self.solve(above_box, q_above)
        q_place = self.solve(place, q_above_box)
        self.move(q_above, -0.785, 1.3)
        self.move(q_grasp, -0.785, 0.8)
        self.move(q_grasp, 0.0, 0.75)
        self.move(q_above, 0.0, 1.2)
        self.move(q_above_box, 0.0, 1.3)
        self.move(q_place, 0.0, 0.7)
        self.move(q_place, -0.785, 0.45)
        self.move(q_above_box, -0.785, 0.8)
        self.move_action(home, 1.3)
