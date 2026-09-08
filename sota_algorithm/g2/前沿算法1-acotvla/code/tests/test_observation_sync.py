"""观测必须来自同一物理时刻，且不引用会被后续渲染覆盖的缓冲区。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import EpisodeRecorder
from robot import G2Cameras
from simulation import G2Simulation
from training.convert_dataset import validate_timing


class ObservationSyncTest(unittest.TestCase):
    def cameras(self, times):
        cameras = G2Cameras.__new__(G2Cameras)
        for name, time in zip(("head", "left", "right"), times):
            camera = Mock()
            camera.get_current_frame.return_value = {
                "rgb": np.ones((2, 3, 4), dtype=np.uint8),
                "rendering_time": time,
            }
            setattr(cameras, name, camera)
        return cameras

    def test_aligned_frames_are_copied(self):
        cameras = self.cameras([1.0, 1.0, 1.0])
        images = cameras.capture(1.0)
        cameras.head.get_current_frame()["rgb"][:] = 0
        self.assertTrue(np.all(images[0] == 1))
        cameras.head.get_rgba.assert_not_called()

    def test_stale_or_invalid_frame_is_rejected(self):
        for time in (0.99, float("nan"), float("inf")):
            with self.subTest(time=time), self.assertRaises(RuntimeError):
                self.cameras([1.0, time, 1.0]).capture(1.0)

    def test_observe_renders_without_physics(self):
        sim = G2Simulation.__new__(G2Simulation)
        sim.world = Mock(current_time=1.0, current_time_step_index=120)
        sim.cameras = self.cameras([1.0] * 3)
        robot = Mock()
        robot.state16.return_value = np.zeros(16)
        state, images = sim.observe(robot)
        self.assertEqual(state.shape, (16,))
        self.assertEqual(len(images), 3)
        sim.world.render.assert_called_once()
        sim.world.step.assert_not_called()

    def test_observe_rejects_physics_advancement(self):
        sim = G2Simulation.__new__(G2Simulation)
        sim.world = Mock(current_time=1.0, current_time_step_index=120)
        sim.world.render.side_effect = lambda: setattr(
            sim.world, "current_time_step_index", 121
        )
        with self.assertRaises(RuntimeError):
            sim.observe(Mock())

    def test_observe_flushes_delayed_render_without_stepping(self):
        sim = G2Simulation.__new__(G2Simulation)
        sim.world = Mock(current_time=1.0, current_time_step_index=120)
        sim.cameras = Mock()
        sim.cameras.capture.side_effect = [RuntimeError("旧帧"), (None, None, None)]
        robot = Mock()
        robot.state16.return_value = np.zeros(16)
        sim.observe(robot)
        self.assertEqual(sim.world.render.call_count, 2)
        sim.world.step.assert_not_called()

    def test_saved_timestamps_match_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = EpisodeRecorder(Path(tmp), 0, "test", "red")
            images = [np.zeros((2, 3, 3), dtype=np.uint8)] * 3
            rec.record(
                images,
                np.zeros(16),
                np.zeros(16),
                observation_time=1.0,
                image_times=[1.0] * 3,
            )
            with np.load(rec.save(True), allow_pickle=False) as episode:
                self.assertEqual(episode["image_time"].shape, (1, 3))
                np.testing.assert_allclose(
                    episode["image_time"] - episode["observation_time"][:, None], 0
                )
            with self.assertRaises(ValueError):
                rec.record(
                    images,
                    np.zeros(16),
                    np.zeros(16),
                    observation_time=2.0,
                    image_times=[1.0] * 3,
                )

    def test_converter_checks_real_interval(self):
        times = np.arange(3) / 30
        data = {
            "state": np.zeros((3, 16)),
            "observation_time": times,
            "image_time": np.repeat(times[:, None], 3, axis=1),
        }
        validate_timing(data, 30)
        data["observation_time"] = times * 4
        data["image_time"] *= 4
        with self.assertRaises(ValueError):
            validate_timing(data, 30)

    def test_converter_rejects_missing_timestamps(self):
        with self.assertRaises(ValueError):
            validate_timing({"state": np.zeros((3, 16))}, 30)


if __name__ == "__main__":
    unittest.main()
