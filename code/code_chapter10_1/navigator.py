"""把规划、优化、局部避障、跟踪和恢复组织成完整导航状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Sequence

try:
    from .costmap import OccupancyGrid2D
    from .geometry import CircleObstacle, Pose2D, Velocity2D, distance
    from .global_planner import AStarPlanner, PlanningError
    from .local_planner import LocalPlanner
    from .path_tracker import HolonomicPathTracker
    from .trajectory import TrajectoryOptimizer
except ImportError:
    from costmap import OccupancyGrid2D
    from geometry import CircleObstacle, Pose2D, Velocity2D, distance
    from global_planner import AStarPlanner, PlanningError
    from local_planner import LocalPlanner
    from path_tracker import HolonomicPathTracker
    from trajectory import TrajectoryOptimizer


class NavigationState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    CONTROLLING = "CONTROLLING"
    RECOVERY = "RECOVERY"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class NavigationOutput:
    command: Velocity2D
    state: NavigationState
    global_path: list[Pose2D]
    local_path: list[Pose2D]
    target: Pose2D | None
    message: str = ""


class CollisionMonitor:
    """在控制器之外增加最后一道短时域碰撞保护。"""

    def __init__(self, costmap: OccupancyGrid2D, horizon: float = 0.12, step: float = 0.05) -> None:
        self.costmap = costmap
        self.horizon = horizon
        self.step = step

    def safe_command(self, pose: Pose2D, command: Velocity2D) -> Velocity2D:
        x, y, yaw = pose.x, pose.y, pose.yaw
        elapsed = 0.0
        while elapsed < self.horizon:
            cosine, sine = math.cos(yaw), math.sin(yaw)
            x += (cosine * command.vx - sine * command.vy) * self.step
            y += (sine * command.vx + cosine * command.vy) * self.step
            yaw += command.wz * self.step
            if self.costmap.is_occupied_world(x, y):
                return Velocity2D()
            elapsed += self.step
        return command


class ProgressChecker:
    def __init__(self, minimum_progress: float = 0.12, timeout: float = 7.0) -> None:
        self.minimum_progress = minimum_progress
        self.timeout = timeout
        self.anchor = Pose2D()
        self.anchor_time = 0.0

    def reset(self, pose: Pose2D, now: float) -> None:
        self.anchor = Pose2D(pose.x, pose.y, pose.yaw)
        self.anchor_time = now

    def is_stuck(self, pose: Pose2D, now: float) -> bool:
        if distance(pose, self.anchor) >= self.minimum_progress:
            self.reset(pose, now)
            return False
        return now - self.anchor_time >= self.timeout


class TeachingNavigator:
    """不依赖 Nav2 的最小完整导航器。"""

    def __init__(
        self,
        costmap: OccupancyGrid2D,
        planner: AStarPlanner,
        optimizer: TrajectoryOptimizer,
        local_planner: LocalPlanner,
        tracker: HolonomicPathTracker,
        collision_horizon: float = 0.12,
        collision_step: float = 0.05,
        progress_timeout: float = 7.0,
        minimum_progress: float = 0.12,
        max_recovery_attempts: int = 3,
        local_replan_period: float = 0.40,
        safety_costmap: OccupancyGrid2D | None = None,
    ) -> None:
        self.costmap = costmap
        self.planner = planner
        self.optimizer = optimizer
        self.local_planner = local_planner
        self.tracker = tracker
        self.collision_monitor = CollisionMonitor(
            safety_costmap or costmap, horizon=collision_horizon, step=collision_step
        )
        self.progress_checker = ProgressChecker(minimum_progress, progress_timeout)
        self.max_recovery_attempts = max_recovery_attempts
        self.local_replan_period = max(0.05, local_replan_period)

        self.state = NavigationState.IDLE
        self.goal: Pose2D | None = None
        self.global_path: list[Pose2D] = []
        self.local_path: list[Pose2D] = []
        self.recovery_attempts = 0
        self._last_local_plan_time = -math.inf
        self._using_detour = False
        self._message = "等待目标"

    def set_goal(self, goal: Pose2D, current: Pose2D, now: float | None = None) -> None:
        self.goal = Pose2D(goal.x, goal.y, goal.yaw)
        self.global_path = []
        self.local_path = []
        self.local_planner.reset()
        self.recovery_attempts = 0
        self._last_local_plan_time = -math.inf
        self._using_detour = False
        self.state = NavigationState.PLANNING
        self._message = "收到新目标"
        self.progress_checker.reset(current, time.monotonic() if now is None else now)

    def cancel(self) -> None:
        self.state = NavigationState.IDLE
        self.goal = None
        self.global_path = []
        self.local_path = []
        self._using_detour = False
        self._message = "导航已取消"

    def update(
        self,
        current: Pose2D,
        dynamic_obstacles: Sequence[CircleObstacle] = (),
        now: float | None = None,
    ) -> NavigationOutput:
        now = time.monotonic() if now is None else now
        if self.goal is None or self.state in (NavigationState.IDLE, NavigationState.SUCCEEDED, NavigationState.FAILED):
            return self._output(Velocity2D(), None)

        if self.state == NavigationState.PLANNING:
            try:
                raw = self.planner.plan(current, self.goal)
                self.global_path = self.optimizer.optimize(raw.path)
                self.local_planner.reset()
                self.state = NavigationState.CONTROLLING
                self._message = f"全局规划完成：扩展 {raw.expanded_nodes} 个栅格"
                self.progress_checker.reset(current, now)
            except PlanningError as exc:
                self.state = NavigationState.FAILED
                self._message = str(exc)
                return self._output(Velocity2D(), None)

        if self.state == NavigationState.RECOVERY:
            # 教学版恢复行为：停止一小段时间、清除局部历史并重新做全局规划。
            self.local_planner.reset()
            self._last_local_plan_time = -math.inf
            self._using_detour = False
            self.state = NavigationState.PLANNING
            self._message = f"恢复行为 {self.recovery_attempts}/{self.max_recovery_attempts}：重新规划"
            return self._output(Velocity2D(), None)

        if now - self._last_local_plan_time >= self.local_replan_period or not self.local_path:
            local_result = self.local_planner.plan(current, self.global_path, dynamic_obstacles)
            self.local_path = local_result.path
            self._using_detour = local_result.used_detour
            self._last_local_plan_time = now
            if local_result.blocked:
                return self._request_recovery("局部路径被阻塞")
            self._message = (
                "检测到障碍，正在稳定绕行"
                if self._using_detour
                else "局部路径畅通，正在跟踪全局路径"
            )

        tracking = self.tracker.compute(current, self.local_path, self.goal)
        if tracking.reached:
            self.state = NavigationState.SUCCEEDED
            self._message = "目标已到达"
            return self._output(Velocity2D(), self.goal)

        if self.progress_checker.is_stuck(current, now):
            return self._request_recovery("进度检查失败：机器人可能被卡住")

        desired_command = tracking.command
        if self._using_detour:
            desired_command = self._limit_linear_speed(
                desired_command, self.tracker.detour_speed_limit
            )
        command = self.collision_monitor.safe_command(current, desired_command)
        if command == Velocity2D() and math.hypot(desired_command.vx, desired_command.vy) > 0.05:
            return self._request_recovery("碰撞监控触发急停")
        return self._output(command, tracking.target)

    @staticmethod
    def _limit_linear_speed(command: Velocity2D, speed_limit: float) -> Velocity2D:
        speed = math.hypot(command.vx, command.vy)
        if speed <= speed_limit or speed < 1e-12:
            return command
        scale = speed_limit / speed
        return Velocity2D(command.vx * scale, command.vy * scale, command.wz)

    def _request_recovery(self, message: str) -> NavigationOutput:
        self.recovery_attempts += 1
        if self.recovery_attempts > self.max_recovery_attempts:
            self.state = NavigationState.FAILED
            self._message = f"{message}；超过最大恢复次数"
        else:
            self.state = NavigationState.RECOVERY
            self._message = message
        return self._output(Velocity2D(), None)

    def _output(self, command: Velocity2D, target: Pose2D | None) -> NavigationOutput:
        return NavigationOutput(
            command=command,
            state=self.state,
            global_path=list(self.global_path),
            local_path=list(self.local_path),
            target=target,
            message=self._message,
        )
