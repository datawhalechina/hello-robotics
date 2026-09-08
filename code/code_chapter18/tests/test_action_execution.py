"""专家与推理使用同一目标保持策略，阶段取整不增加尾帧。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ControlConfig, SimulationConfig
from expert import ScriptedExpert
from vla_client import ChunkRunner


class ActionExecutionTest(unittest.TestCase):
    def test_one_target_is_applied_once_and_held_four_steps(self):
        robot, sim = Mock(), Mock()
        target = np.zeros(16)
        robot.apply_absolute.return_value = target
        runner = ChunkRunner(robot, sim, ControlConfig())
        result = runner.execute_target(target)
        robot.apply_absolute.assert_called_once_with(target, False)
        self.assertEqual(sim.step.call_count, 4)
        np.testing.assert_array_equal(result, target)

    def test_stage_rounding_preserves_total_duration(self):
        expert = ScriptedExpert.__new__(ScriptedExpert)
        expert.robot = Mock()
        expert.robot.state16.return_value = np.zeros(16)
        expert.sim = Mock(config=SimulationConfig())
        expert.substeps = 4
        expert._scheduled_physics_steps = 0
        expert._target_count = 0
        expert._step = Mock()
        for duration in (1.2, 1.1, 0.7, 0.65, 1.0, 1.1, 0.65, 0.4, 0.7, 1.0):
            expert.move_action(np.ones(16), duration)
        self.assertEqual(expert._step.call_count, 255)
        self.assertEqual(expert._target_count * expert.substeps, 1020)


if __name__ == "__main__":
    unittest.main()
