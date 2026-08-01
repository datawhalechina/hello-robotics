"""G2 右臂几何模型。

运动学直接复用第五章的自编 FK/IK；本文件补充规划所需的连杆采样点。
"""

from pathlib import Path
import sys
from typing import Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from code_chapter5.kinematics import G2ArmKinematics, IKResult, Pose, axis_rotation


class G2PlanningModel:
    """把第五章运动学包装为碰撞检测和规划可用的接口。"""

    def __init__(self) -> None:
        self.kinematics = G2ArmKinematics("right")

    @property
    def lower_limits(self) -> np.ndarray:
        return self.kinematics.lower_limits

    @property
    def upper_limits(self) -> np.ndarray:
        return self.kinematics.upper_limits

    def forward(self, joint_positions: Sequence[float]) -> Pose:
        return self.kinematics.forward(joint_positions)

    def inverse(
        self,
        target_position: Sequence[float],
        initial_positions: Sequence[float] | None = None,
    ) -> IKResult:
        # 抓取教学只约束位置，让 7 自由度冗余空间用于避开关节极限。
        return self.kinematics.inverse(
            target_position=target_position,
            target_rotation=None,
            initial_positions=initial_positions,
        )

    def link_points(self, joint_positions: Sequence[float]) -> np.ndarray:
        """返回肩部、七个关节点和夹爪中心，共 9 个点。"""
        q = np.asarray(joint_positions, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError("joint_positions 必须包含 7 个关节角")
        current = np.eye(4, dtype=np.float64)
        points = [np.zeros(3, dtype=np.float64)]
        for origin, axis, angle in zip(
            self.kinematics.joint_origins, self.kinematics.joint_axes, q
        ):
            current = current @ origin
            points.append(current[:3, 3].copy())
            current = current @ axis_rotation(axis, float(angle))
        end = current @ self.kinematics.tool_transform
        points.append(end[:3, 3].copy())
        return np.asarray(points)
