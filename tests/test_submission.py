from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class SubmissionTests(unittest.TestCase):
    def test_submission_uses_an_immutable_config_snapshot(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        script = repository / "jobs/submit_training.sh"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"value": "at submission"}))
            environment = {
                **os.environ,
                "SBATCH_BIN": "/bin/echo",
                "SUBMISSION_ROOT": str(root / "submissions"),
            }
            result = subprocess.run(
                [
                    script,
                    "dw4",
                    "--config",
                    config,
                    "--run-id",
                    "queued-run",
                    "--",
                    "--seed",
                    "17",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                cwd=root,
            )
            snapshot = root / "submissions/dw4/queued-run/config.json"
            config.write_text(json.dumps({"value": "edited later"}))
            subprocess.run(
                [script, "dw4", "--config", config, "--run-id", "second-run"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                cwd=root,
            )
            second_snapshot = root / "submissions/dw4/second-run/config.json"

            self.assertEqual(json.loads(snapshot.read_text()), {"value": "at submission"})
            self.assertEqual(json.loads(second_snapshot.read_text()), {"value": "edited later"})
            self.assertEqual(snapshot.stat().st_mode & 0o222, 0)
            self.assertIn(f"CONFIG_SNAPSHOT={snapshot}", result.stdout)
            self.assertIn("train_dw4.sh --seed 17", result.stdout)


if __name__ == "__main__":
    unittest.main()
