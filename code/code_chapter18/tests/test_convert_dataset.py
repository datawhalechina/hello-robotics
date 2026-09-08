"""LeRobot 转换前必须检查每条轨迹，并限制覆盖目录。"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.convert_dataset import resolve_dataset_output, validate_episode


class ConvertDatasetTest(unittest.TestCase):
    @staticmethod
    def episode(frames: int = 3) -> dict:
        times = np.arange(frames, dtype=np.float64) / 30
        images = np.zeros((frames, 240, 320, 3), dtype=np.uint8)
        return {
            "head_image": images.copy(),
            "left_image": images.copy(),
            "right_image": images.copy(),
            "state": np.zeros((frames, 16), dtype=np.float32),
            "actions": np.zeros((frames, 16), dtype=np.float32),
            "observation_time": times,
            "image_time": np.repeat(times[:, None], 3, axis=1),
            "prompt": np.asarray("Pick up the red block."),
            "target_color": np.asarray("red"),
            "success": np.asarray(True),
            "fps": np.asarray(30),
        }

    def test_valid_episode(self):
        validate_episode(
            self.episode(), path=Path("episode_0000.npz"), fps=30, expected_frames=3
        )

    def test_every_episode_dimension_is_checked(self):
        data = self.episode()
        data["actions"] = np.zeros((3, 15), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_episode(
                data, path=Path("episode_0001.npz"), fps=30, expected_frames=3
            )

    def test_output_cannot_escape_lerobot_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                resolve_dataset_output(root, "org/name"), root / "org/name"
            )
            for repo_id in ("/tmp/outside", "../../outside", "."):
                with self.subTest(repo_id=repo_id), self.assertRaises(ValueError):
                    resolve_dataset_output(root, repo_id)


if __name__ == "__main__":
    unittest.main()
