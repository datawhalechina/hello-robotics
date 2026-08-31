"""Scripted demonstrations and short automatic correction segments."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from config import ARM_LOWER_7, ARM_UPPER_7, COLORS, HOME_ARMS_14
from hil import HILState
from kinematics import RightArmIK
from robot import closed_fraction, smoothstep


IK_TOLERANCE = 0.012


class IKError(RuntimeError):
    """Raised only after every deterministic IK initial guess has failed."""

    def __init__(self, point, best_error, attempts):
        self.point = np.asarray(point, float).copy()
        self.best_error = float(best_error)
        self.attempts = int(attempts)
        super().__init__(
            f"IK failed after {self.attempts} attempts: "
            f"best_error={self.best_error:.4f}, point={self.point.round(4).tolist()}"
        )


@dataclass(frozen=True)
class CorrectionState:
    block_lifted: bool
    gripper_closed: bool
    inside_box: bool
    near_pregrasp: bool


def choose_segment(s):
    if s.inside_box or (s.block_lifted and s.gripper_closed):
        return "place_and_release"
    return "grasp_and_lift" if s.near_pregrasp else "pregrasp"


class AutoExpert:
    def __init__(self, sim, robot, recorder=None, intervention=False):
        self.sim, self.robot, self.recorder, self.intervention, self.ik = (
            sim,
            robot,
            recorder,
            bool(intervention),
            RightArmIK(),
        )
        self.tick = 0
        self.source = "auto_correction" if intervention else "auto_demonstration"

    def _step(self, action):
        record = (
            self.recorder is not None and self.tick % self.sim.cfg.record_every == 0
        )
        if record:
            images, state = self.sim.cameras.capture(), self.robot.state()
        applied = self.robot.apply(action)
        self.sim.task.update(closed_fraction(applied[15]))
        self.sim.step(True)
        if record:
            self.recorder.add(
                images,
                state,
                applied,
                intervention_state=HILState.ACTIVE
                if self.intervention
                else HILState.POLICY,
                source=self.source,
            )
        self.tick += 1

    def move(self, target, seconds):
        start = self.robot.state().astype(float)
        target = np.asarray(target, float)
        steps = max(2, round(seconds * self.sim.cfg.physics_hz))
        for i in range(1, steps + 1):
            self._step(start + smoothstep(i / steps) * (target - start))

    def move_right(self, joints, gripper, seconds):
        action = self.robot.state().astype(float)
        action[7:14] = joints
        action[15] = gripper
        self.move(action, seconds)

    @staticmethod
    def _ik_seeds(seed):
        """Use a few fixed initial guesses to avoid numerical IK local minima."""
        current = np.clip(np.asarray(seed, float), ARM_LOWER_7, ARM_UPPER_7)
        home = HOME_ARMS_14[7:14]
        center = (ARM_LOWER_7 + ARM_UPPER_7) / 2
        candidates = (current, home, (current + home) / 2, center)
        unique = []
        for candidate in candidates:
            if not any(np.allclose(candidate, saved) for saved in unique):
                unique.append(np.asarray(candidate, float).copy())
        return unique

    def solve(self, point, seed):
        answers = [self.ik.solve(point, guess) for guess in self._ik_seeds(seed)]
        best = min(answers, key=lambda answer: answer.error)
        if not best.success and best.error > IK_TOLERANCE:
            raise IKError(point, best.error, len(answers))
        return best.joints

    def targets(self, color):
        block = self.sim.task.block_position(color)
        above = block.copy()
        above[1] -= self.sim.task_cfg.pregrasp_clearance
        grasp = block.copy()
        grasp[1] -= self.sim.task_cfg.grasp_offsets[COLORS.index(color)]
        box = np.asarray(self.sim.task_cfg.box_position, float)
        above_box = box.copy()
        above_box[1] = self.sim.task_cfg.table_top_y - 0.30
        place = box.copy()
        place[1] = self.sim.task_cfg.table_top_y - self.sim.task_cfg.place_clearance
        return above, grasp, above_box, place

    def state(self, color):
        q = self.robot.state()
        block = self.sim.task.block_position(color)
        above = self.targets(color)[0]
        ee = self.ik.chain(q[7:14])[0][:3, 3]
        return CorrectionState(
            block[1]
            < self.sim.task_cfg.table_top_y - max(0.06, self.sim.task_cfg.block_size),
            closed_fraction(q[15]) >= 0.45,
            self.sim.task.inside_box(color),
            np.linalg.norm(ee - above) <= 0.11,
        )

    def correct(self, color):
        if not self.intervention:
            raise RuntimeError("correction expert must use intervention=True")
        segment = choose_segment(self.state(color))
        self.source = f"auto_correction:{segment}"
        above, grasp, above_box, place = self.targets(color)
        seed = self.robot.state()[7:14]
        if segment == "pregrasp":
            # Plan before moving. If IK fails, the ACTIVE segment remains empty and
            # the collector can safely discard only this episode.
            qa = self.solve(above, seed)
            self.move_right(seed, -0.785, 0.25)
            self.move_right(qa, -0.785, 0.75)
        elif segment == "grasp_and_lift":
            qa = self.solve(above, seed)
            qg = self.solve(grasp, qa)
            self.move_right(qa, -0.785, 0.4)
            self.move_right(qg, -0.785, 0.6)
            self.move_right(qg, 0, 0.5)
            self.move_right(qa, 0, 0.8)
        else:
            qa = self.solve(above_box, seed)
            qp = self.solve(place, qa)
            self.move_right(qa, 0, 0.9)
            self.move_right(qp, 0, 0.5)
            self.move_right(qp, -0.785, 0.4)
            self.move_right(qa, -0.785, 0.5)
        return segment

    def demonstrate(self, color):
        self.move(self.robot.home, 1)
        above, grasp, above_box, place = self.targets(color)
        qa = self.solve(above, self.robot.state()[7:14])
        qg = self.solve(grasp, qa)
        qb = self.solve(above_box, qa)
        qp = self.solve(place, qb)
        for q, g, s in (
            (qa, -0.785, 1),
            (qg, -0.785, 0.7),
            (qg, 0, 0.6),
            (qa, 0, 0.9),
            (qb, 0, 1),
            (qp, 0, 0.6),
            (qp, -0.785, 0.4),
            (qb, -0.785, 0.6),
        ):
            self.move_right(q, g, s)
        self.move(self.robot.home, 0.9)
