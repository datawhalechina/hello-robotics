"""G2 右夹爪的最小位置控制器。"""

import numpy as np

try:
    from .config import GRIPPER_JOINT_NAMES
except ImportError:
    from config import GRIPPER_JOINT_NAMES


class G2GripperController:
    def __init__(self, articulation) -> None:
        self.articulation = articulation
        names = list(articulation.dof_names)
        self.indices = np.asarray([names.index(name) for name in GRIPPER_JOINT_NAMES], dtype=np.int64)

    def command(self, opened: bool) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        # DOF 顺序为 outer、inner；两侧镜像转动。
        target = np.array([0.68, -0.68]) if opened else np.array([0.03, -0.03])
        self.articulation.apply_action(
            ArticulationAction(joint_positions=target, joint_indices=self.indices)
        )

    def hold(self, opened: bool, steps: int, step_callback) -> None:
        for _ in range(steps):
            self.command(opened)
            step_callback()
