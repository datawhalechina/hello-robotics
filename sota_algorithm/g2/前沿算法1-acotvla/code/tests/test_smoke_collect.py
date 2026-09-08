import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import smoke_collect


class SmokeCollectTest(unittest.TestCase):
    def data(self):
        times = np.arange(2) / 30
        images = np.zeros((2, 240, 320, 3), dtype=np.uint8)
        images[1] = 1
        return {
            "state": np.zeros((2, 16)),
            "actions": np.zeros((2, 16)),
            "observation_time": times,
            "image_time": np.repeat(times[:, None], 3, axis=1),
            **{k: images.copy() for k in ("head_image", "left_image", "right_image")},
        }

    def test_valid_data(self):
        self.assertEqual(smoke_collect.check_arrays(self.data()), 2)

    def test_bad_wrist_or_action_is_rejected(self):
        for key in ("left_image", "right_image", "actions"):
            data = self.data()
            if key == "actions":
                data[key][0, 0] = np.nan
            else:
                data[key][:] = 0
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke_collect.check_arrays(data)

    def test_worker_report_controls_exit_status(self):
        for passed, child_code, expected in ((True, 0, 0), (False, 0, 1), (True, 1, 1)):
            with (
                self.subTest(passed=passed, code=child_code),
                tempfile.TemporaryDirectory() as tmp,
            ):

                def worker(
                    command,
                    *,
                    passed=passed,
                    child_code=child_code,
                    **kwargs,
                ):
                    run_dir = Path(command[command.index("--run-dir") + 1])
                    (run_dir / "report.json").write_text(json.dumps({"passed": passed}))
                    return Mock(returncode=child_code)

                with (
                    patch.object(sys, "argv", ["smoke_collect.py", "--output", tmp]),
                    patch.object(smoke_collect.subprocess, "run", side_effect=worker),
                    patch("builtins.print"),
                    self.assertRaises(SystemExit) as raised,
                ):
                    smoke_collect.main()
                self.assertEqual(raised.exception.code, expected)


if __name__ == "__main__":
    unittest.main()
