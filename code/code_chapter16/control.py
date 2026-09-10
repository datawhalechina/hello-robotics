"""Pure NumPy joint validation for Chapter-16 G2 policy execution."""

from __future__ import annotations

import numpy as np

from settings import ACTION_DIM

GRIPPER_OPEN_MAGNITUDE = 0.785


def _closed_fraction(value: float) -> float:
    """Convert the model gripper convention (-0.785=open, 0=closed)."""
    joint_like = float(np.clip(-float(value), 0.0, GRIPPER_OPEN_MAGNITUDE))
    return float((GRIPPER_OPEN_MAGNITUDE - joint_like) / GRIPPER_OPEN_MAGNITUDE)

ARM_LOWER_7 = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708], np.float32)
ARM_UPPER_7 = np.array([3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708], np.float32)
LOWER = np.concatenate([np.tile(ARM_LOWER_7, 2), [-0.785, -0.785]]).astype(np.float32)
UPPER = np.concatenate([np.tile(ARM_UPPER_7, 2), [0.0, 0.0]]).astype(np.float32)


def _vectors(raw: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, np.float32).reshape(-1)
    current = np.asarray(current, np.float32).reshape(-1)
    if raw.shape != (ACTION_DIM,) or current.shape != (ACTION_DIM,):
        raise ValueError(f"action/current must be {ACTION_DIM}D")
    if not np.isfinite(raw).all() or not np.isfinite(current).all():
        raise ValueError("action/current must be finite")
    return raw, current


def clip_joint_limits(raw: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, dict]:
    """Official-like execution: absolute qpos targets with physical clipping only."""
    raw, current = _vectors(raw, current)
    result = np.clip(raw, LOWER, UPPER).astype(np.float32)
    return result, {
        "limited": bool(not np.allclose(result, raw, atol=1e-6)),
        "raw_max_arm_delta": float(np.max(np.abs(raw[:14] - current[:14]))),
        "raw_max_gripper_delta": float(np.max(np.abs(raw[14:] - current[14:]))),
    }


def rate_limit_target(
    raw: np.ndarray,
    current: np.ndarray,
    max_arm_step: float = 0.40,
    max_gripper_step: float = 0.80,
) -> tuple[np.ndarray, dict]:
    """Optional conservative receding-horizon limiter, not used in official mode."""
    bounded, report = clip_joint_limits(raw, current)
    _, current = _vectors(raw, current)
    max_step = np.concatenate(
        [np.full(14, float(max_arm_step)), np.full(2, float(max_gripper_step))]
    ).astype(np.float32)
    result = np.clip(current + np.clip(bounded - current, -max_step, max_step), LOWER, UPPER)
    report["limited"] = bool(not np.allclose(result, raw, atol=1e-6))
    return result.astype(np.float32), report


def safe_target(raw: np.ndarray, current: np.ndarray, max_arm_step: float = 0.40, max_gripper_step: float = 0.80):
    """Backward-compatible alias for the optional rate-limited mode."""
    return rate_limit_target(raw, current, max_arm_step, max_gripper_step)


class GraspReleaseGuard:
    """Task-specific gripper state machine for the G2 pick-and-place task.

    Motus predicts absolute qpos targets and has no force/contact controller.  In
    a closed-loop rollout it can therefore reopen the gripper immediately after
    the object is lifted.  This guard only changes the right-gripper target:

    approach/closing -> model controls the gripper;
    holding          -> force the right gripper closed;
    release_allowed  -> model may open it near the box.

    The left gripper and all arm joints are untouched.  It is deliberately kept
    outside the Motus model so the official checkpoint and action format remain
    unchanged.
    """

    APPROACH = "approach"
    CLOSING = "closing"
    HOLDING = "holding"
    RELEASE_ALLOWED = "release_allowed"
    RELEASED = "released"

    def __init__(
        self,
        color: str,
        *,
        closed_fraction_threshold: float = 0.45,
        close_command_fraction: float = 0.30,
        lifted_clearance: float = 0.06,
        grasp_distance: float = 0.28,
        box_horizontal_distance: float = 0.14,
        box_vertical_distance: float = 0.30,
    ):
        self.color = str(color)
        self.closed_fraction_threshold = float(closed_fraction_threshold)
        self.close_command_fraction = float(close_command_fraction)
        self.lifted_clearance = float(lifted_clearance)
        self.grasp_distance = float(grasp_distance)
        self.box_horizontal_distance = float(box_horizontal_distance)
        self.box_vertical_distance = float(box_vertical_distance)
        self.phase = self.APPROACH
        self.locked_actions = 0
        self.release_actions = 0
        self._ik = None

    def reset(self, color: str | None = None) -> None:
        if color is not None:
            self.color = str(color)
        self.phase = self.APPROACH
        self.locked_actions = 0
        self.release_actions = 0

    @staticmethod
    def _ee_position(robot) -> np.ndarray:
        # Import lazily: environment.py puts task_runtime on sys.path after
        # evaluate.py imports this module.
        from kinematics import RightArmIK

        ik = getattr(robot, "_chapter16_grasp_ik", None)
        if ik is None:
            ik = RightArmIK()
            robot._chapter16_grasp_ik = ik
        return np.asarray(ik.chain(robot.state()[7:14])[0][:3, 3], dtype=np.float32)

    def _is_closed(self, robot) -> bool:
        return _closed_fraction(float(robot.state()[15])) >= self.closed_fraction_threshold

    def _is_lifted(self, sim) -> bool:
        position = np.asarray(sim.task.block_position(self.color), dtype=np.float32)
        table_top = float(sim.task_cfg.table_top_y)
        required = max(self.lifted_clearance, float(sim.task_cfg.block_size))
        return bool(position[1] < table_top - required)

    def _is_near_block(self, sim, robot) -> bool:
        block = np.asarray(sim.task.block_position(self.color), dtype=np.float32)
        ee = self._ee_position(robot)
        return bool(np.linalg.norm(ee - block) <= self.grasp_distance)

    def object_is_grasped(self, sim, robot) -> bool:
        # Once the block has visibly left the table, the extra FK distance check
        # is intentionally not required.  Contact dynamics can leave a small
        # transient gap between the gripper tool frame and the block.
        return self._is_closed(robot) and self._is_lifted(sim)

    def _set_phase(self, phase: str, *, reason: str) -> None:
        if self.phase != phase:
            print(
                f"[grasp-guard] {self.color}: {self.phase} -> {phase} ({reason})",
                flush=True,
            )
            self.phase = phase

    def box_is_near(self, sim, robot) -> bool:
        ee = self._ee_position(robot)
        box = np.asarray(sim.task_cfg.box_position, dtype=np.float32)
        horizontal = float(np.linalg.norm(ee[[0, 2]] - box[[0, 2]]))
        vertical = float(abs(ee[1] - box[1]))
        return (
            horizontal <= self.box_horizontal_distance
            and vertical <= self.box_vertical_distance
        )

    def filter_target(self, sim, robot, target: np.ndarray) -> tuple[np.ndarray, dict]:
        """Apply the state machine to one absolute G2 action target."""
        target = np.asarray(target, dtype=np.float32).reshape(ACTION_DIM).copy()
        current = robot.state()
        current_closed = _closed_fraction(float(current[15]))
        target_closed = _closed_fraction(float(target[15]))

        near_block = self._is_near_block(sim, robot)

        # A model target moving from open toward closed enters the closing phase.
        if self.phase == self.APPROACH and target_closed > current_closed + 0.05:
            self._set_phase(self.CLOSING, reason="model started closing")

        # Latch the grasp as soon as a closing command is made near the block.
        # Waiting for a visibly lifted block was too late for some G2 contact
        # rollouts: the next action could already reopen the gripper.  This is
        # still conservative because it requires both a close command/current
        # closed state and proximity to the selected block.
        close_attempt = (
            target_closed >= self.close_command_fraction
            or current_closed >= self.close_command_fraction
        )
        if self.phase in (self.APPROACH, self.CLOSING) and near_block and close_attempt:
            self._set_phase(self.HOLDING, reason="close attempt near block")

        # Fallback for a contact transient where FK distance is briefly large:
        # a closed gripper plus a lifted selected block is enough to keep it
        # closed.  This prevents a release during the first lift waypoint.
        if self.phase in (self.APPROACH, self.CLOSING) and self.object_is_grasped(sim, robot):
            self._set_phase(self.HOLDING, reason="closed gripper and block lifted")

        box_near = self.box_is_near(sim, robot)
        if self.phase == self.HOLDING and box_near:
            self._set_phase(self.RELEASE_ALLOWED, reason="end effector near box")

        before = float(target[15])
        locked = False
        if self.phase == self.HOLDING:
            target[15] = 0.0
            locked = True
            self.locked_actions += 1
        elif self.phase == self.RELEASE_ALLOWED:
            self.release_actions += 1
            if target_closed < 0.45:
                self._set_phase(self.RELEASED, reason="model opened near box")

        return target, {
            "phase": self.phase,
            "right_gripper_locked": locked,
            "raw_right_gripper": before,
            "applied_right_gripper": float(target[15]),
            "object_grasped": self.object_is_grasped(sim, robot),
            "near_block": near_block,
            "box_near": box_near,
        }
