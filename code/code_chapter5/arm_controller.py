"""G2 单臂关节控制器。
"""

from typing import Callable, Sequence

import numpy as np

try:
    from .config import (
        ARM_JOINT_NAMES,
        JOINT_LOWER_LIMITS,
        JOINT_UPPER_LIMITS,
        JOINT_VELOCITY_LIMITS,
        MIRRORED_JOINT_DELTA_SIGNS,
    )
    from .kinematics import G2ArmKinematics, IKResult, Pose
except ImportError:
    from config import (
        ARM_JOINT_NAMES,
        JOINT_LOWER_LIMITS,
        JOINT_UPPER_LIMITS,
        JOINT_VELOCITY_LIMITS,
        MIRRORED_JOINT_DELTA_SIGNS,
    )
    from kinematics import G2ArmKinematics, IKResult, Pose


class G2ArmController:
    """将教学运动学连接到 Isaac Sim articulation。"""

    def __init__(self, articulation, arm: str = "right") -> None:
        if arm not in ARM_JOINT_NAMES:
            raise ValueError("arm 只能是 'left' 或 'right'")
        self.articulation = articulation
        self.arm = arm
        self.joint_names = ARM_JOINT_NAMES[arm]
        self.joint_indices = self._find_joint_indices()
        self.lower_limits = JOINT_LOWER_LIMITS.copy()
        self.upper_limits = JOINT_UPPER_LIMITS.copy()
        self.velocity_limits = JOINT_VELOCITY_LIMITS.copy()
        self.kinematics = G2ArmKinematics(arm)

    def _find_joint_indices(self) -> np.ndarray:
        dof_names = list(self.articulation.dof_names)
        missing = [name for name in self.joint_names if name not in dof_names]
        if missing:
            raise RuntimeError(f"G2 模型缺少机械臂关节：{missing}")
        return np.array([dof_names.index(name) for name in self.joint_names], dtype=np.int64)

    @staticmethod
    def _check_positions(positions: Sequence[float]) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError(f"机械臂目标必须是 7 个关节角，实际形状为 {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("关节目标中不能包含 NaN 或无穷大")
        return values

    def clamp_positions(self, positions: Sequence[float]) -> np.ndarray:
        """把关节目标裁剪到 G2 的物理限位内。"""
        return np.clip(self._check_positions(positions), self.lower_limits, self.upper_limits)

    def get_joint_positions(self) -> np.ndarray:
        """读取当前 7 个关节角。"""
        all_positions = np.asarray(self.articulation.get_joint_positions())
        return all_positions[self.joint_indices].astype(np.float64, copy=True)

    def get_joint_velocities(self) -> np.ndarray:
        """读取当前 7 个关节角速度。"""
        all_velocities = np.asarray(self.articulation.get_joint_velocities())
        return all_velocities[self.joint_indices].astype(np.float64, copy=True)

    def command_positions(self, positions: Sequence[float]) -> np.ndarray:
        """发送位置目标；返回经过关节限位后的实际命令。"""
        from isaacsim.core.utils.types import ArticulationAction

        target = self.clamp_positions(positions)
        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=target,
                joint_indices=self.joint_indices,
            )
        )
        return target

    def make_trajectory(
        self,
        target_positions: Sequence[float],
        duration: float,
        dt: float,
        start_positions: Sequence[float] | None = None,
    ) -> np.ndarray:
        """生成三次时间缩放轨迹 q=q0+(3t²-2t³)(q1-q0)。"""
        if duration <= 0.0 or dt <= 0.0:
            raise ValueError("duration 和 dt 必须大于 0")
        start = self.get_joint_positions() if start_positions is None else self._check_positions(start_positions)
        target = self.clamp_positions(target_positions)

        # 自动保证平均速度不超过 URDF 速度上限的 60%，给物理 drive 留出余量。
        minimum_duration = float(np.max(np.abs(target - start) / (0.6 * self.velocity_limits)))
        duration = max(float(duration), minimum_duration)
        steps = max(2, int(np.ceil(duration / dt)))
        tau = np.linspace(0.0, 1.0, steps + 1)[1:]
        scale = 3.0 * tau**2 - 2.0 * tau**3
        return start[None, :] + scale[:, None] * (target - start)[None, :]

    def move_to(
        self,
        target_positions: Sequence[float],
        duration: float,
        dt: float,
        step_callback: Callable[[], None],
    ) -> np.ndarray:
        """沿平滑轨迹运动；step_callback 通常传入 simulation.step。"""
        trajectory = self.make_trajectory(target_positions, duration, dt)
        for waypoint in trajectory:
            self.command_positions(waypoint)
            step_callback()
        final_target = trajectory[-1]
        # 额外保持少量时间，让物理位置闭环收敛。
        for _ in range(max(1, int(0.25 / dt))):
            self.command_positions(final_target)
            step_callback()
        return self.get_joint_positions()

    def forward_kinematics(self, positions: Sequence[float] | None = None) -> Pose:
        """计算指定或当前关节角对应的末端位姿。"""
        q = self.get_joint_positions() if positions is None else positions
        return self.kinematics.forward(q)

    def solve_ik(
        self,
        target_position: Sequence[float],
        target_rotation: np.ndarray | None = None,
        initial_positions: Sequence[float] | None = None,
    ) -> IKResult:
        """调用本章自行实现的阻尼最小二乘 IK。"""
        seed = self.get_joint_positions() if initial_positions is None else initial_positions
        return self.kinematics.inverse(target_position, target_rotation, seed)


class G2DualArmController:
    """左右机械臂同步控制器，一次发送 14 个关节的位置目标。"""

    def __init__(self, articulation) -> None:
        self.articulation = articulation
        self.left = G2ArmController(articulation, arm="left")
        self.right = G2ArmController(articulation, arm="right")
        self.joint_indices = np.concatenate(
            (self.left.joint_indices, self.right.joint_indices)
        )

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        return {
            "left": self.left.get_joint_positions(),
            "right": self.right.get_joint_positions(),
        }

    def mirrored_targets(
        self,
        center_positions: Sequence[float],
        left_offsets: Sequence[float],
    ) -> dict[str, np.ndarray]:
        """把左臂动作增量转换成视觉同向的右臂增量。

        注意：这里只用于人工设计的对称关节动作。FK/IK 返回的是真实关节角，
        不应再经过该符号转换。
        """
        center = self.left._check_positions(center_positions)
        offsets = self.left._check_positions(left_offsets)
        return {
            "left": self.left.clamp_positions(center + offsets),
            "right": self.right.clamp_positions(
                center + MIRRORED_JOINT_DELTA_SIGNS * offsets
            ),
        }

    def move_mirrored(
        self,
        center_positions: Sequence[float],
        left_offsets: Sequence[float],
        duration: float,
        dt: float,
        step_callback: Callable[[], None],
    ) -> dict[str, np.ndarray]:
        """执行视觉同向的双臂镜像关节动作。"""
        targets = self.mirrored_targets(center_positions, left_offsets)
        return self.move_to(
            targets["left"],
            targets["right"],
            duration,
            dt,
            step_callback,
        )

    def command_positions(
        self,
        left_positions: Sequence[float],
        right_positions: Sequence[float],
    ) -> dict[str, np.ndarray]:
        """在同一个 ArticulationAction 中发送左右臂目标。"""
        from isaacsim.core.utils.types import ArticulationAction

        left_target = self.left.clamp_positions(left_positions)
        right_target = self.right.clamp_positions(right_positions)
        targets = np.concatenate((left_target, right_target))
        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=targets,
                joint_indices=self.joint_indices,
            )
        )
        return {"left": left_target, "right": right_target}

    def make_trajectories(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成步数完全相同的左右臂轨迹，保证同时开始、同时结束。"""
        if duration <= 0.0 or dt <= 0.0:
            raise ValueError("duration 和 dt 必须大于 0")

        starts = self.get_joint_positions()
        left_goal = self.left.clamp_positions(left_target)
        right_goal = self.right.clamp_positions(right_target)

        left_minimum = np.max(
            np.abs(left_goal - starts["left"]) / (0.6 * self.left.velocity_limits)
        )
        right_minimum = np.max(
            np.abs(right_goal - starts["right"]) / (0.6 * self.right.velocity_limits)
        )
        duration = max(float(duration), float(left_minimum), float(right_minimum))
        steps = max(2, int(np.ceil(duration / dt)))
        tau = np.linspace(0.0, 1.0, steps + 1)[1:]
        scale = 3.0 * tau**2 - 2.0 * tau**3

        left_trajectory = starts["left"][None, :] + scale[:, None] * (
            left_goal - starts["left"]
        )[None, :]
        right_trajectory = starts["right"][None, :] + scale[:, None] * (
            right_goal - starts["right"]
        )[None, :]
        return left_trajectory, right_trajectory

    def move_to(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        duration: float,
        dt: float,
        step_callback: Callable[[], None],
    ) -> dict[str, np.ndarray]:
        """同步执行左右臂轨迹。"""
        left_trajectory, right_trajectory = self.make_trajectories(
            left_target, right_target, duration, dt
        )
        for left_waypoint, right_waypoint in zip(
            left_trajectory, right_trajectory
        ):
            self.command_positions(left_waypoint, right_waypoint)
            step_callback()

        for _ in range(max(1, int(0.25 / dt))):
            self.command_positions(left_trajectory[-1], right_trajectory[-1])
            step_callback()
        return self.get_joint_positions()

    def forward_kinematics(self) -> dict[str, Pose]:
        return {
            "left": self.left.forward_kinematics(),
            "right": self.right.forward_kinematics(),
        }

    def solve_ik(
        self,
        left_position: Sequence[float],
        right_position: Sequence[float],
        left_rotation: np.ndarray | None = None,
        right_rotation: np.ndarray | None = None,
    ) -> dict[str, IKResult]:
        """分别求解左右臂 IK；两边都成功后再同步执行。"""
        current = self.get_joint_positions()
        return {
            "left": self.left.solve_ik(
                left_position, left_rotation, current["left"]
            ),
            "right": self.right.solve_ik(
                right_position, right_rotation, current["right"]
            ),
        }
