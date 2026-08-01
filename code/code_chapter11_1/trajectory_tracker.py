"""Isaac Sim 关节轨迹跟踪与跟踪误差保护。"""

from typing import Callable, Sequence

import numpy as np

try:
    from .config import RIGHT_ARM_JOINT_NAMES, TrajectoryConfig
    from .trajectory_optimizer import TrajectoryPoint
except ImportError:
    from config import RIGHT_ARM_JOINT_NAMES, TrajectoryConfig
    from trajectory_optimizer import TrajectoryPoint


class G2TrajectoryTracker:
    def __init__(self, articulation, config: TrajectoryConfig | None = None) -> None:
        self.articulation = articulation
        self.config = config or TrajectoryConfig()
        names = list(articulation.dof_names)
        missing = [name for name in RIGHT_ARM_JOINT_NAMES if name not in names]
        if missing:
            raise RuntimeError(f"G2 缺少右臂关节：{missing}")
        self.indices = np.asarray([names.index(name) for name in RIGHT_ARM_JOINT_NAMES], dtype=np.int64)

    def get_positions(self) -> np.ndarray:
        return np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)[self.indices].copy()

    def command(self, positions: Sequence[float]) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(positions, dtype=np.float64),
                joint_indices=self.indices,
            )
        )

    def execute(
        self,
        trajectory: Sequence[TrajectoryPoint],
        step_callback: Callable[[], None],
        publish_callback: Callable[[np.ndarray, np.ndarray], None] | None = None,
    ) -> float:
        """逐点跟踪，返回最大关节误差。"""
        maximum_error = 0.0
        for point in trajectory[1:]:
            actual = self.get_positions()
            error = float(np.max(np.abs(point.positions - actual)))
            maximum_error = max(maximum_error, error)
            if error > self.config.abort_tolerance:
                raise RuntimeError(f"轨迹跟踪误差过大：{error:.3f} rad")
            # 误差偏大时保持当前目标一帧，让底层 drive 追上，而不是继续超前。
            if error <= self.config.tracking_tolerance:
                self.command(point.positions)
            if publish_callback is not None:
                publish_callback(point.positions, actual)
            step_callback()
        final = np.asarray(trajectory[-1].positions)
        for _ in range(30):
            self.command(final)
            step_callback()
        return maximum_error
