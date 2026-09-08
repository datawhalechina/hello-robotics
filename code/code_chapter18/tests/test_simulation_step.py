"""无需启动 Isaac Sim，防止把渲染帧误当成物理步。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation import G2Simulation


class SimulationStepTest(unittest.TestCase):
    def test_render_does_not_advance_extra_physics(self):
        sim = G2Simulation.__new__(G2Simulation)
        sim.world = Mock()
        sim.step(render=True)
        self.assertEqual(sim.world.mock_calls, [call.step(render=False), call.render()])

    def test_physics_only(self):
        sim = G2Simulation.__new__(G2Simulation)
        sim.world = Mock()
        sim.step(render=False)
        self.assertEqual(sim.world.mock_calls, [call.step(render=False)])


if __name__ == "__main__":
    unittest.main()
