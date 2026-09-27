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
            snapshots = sorted((root / "submissions/dw4/queued-run").glob("*.json"))
            config.write_text(json.dumps({"value": "edited later"}))
            subprocess.run(
                [script, "dw4", "--config", config, "--run-id", "second-run"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                cwd=root,
            )
            second_snapshots = sorted((root / "submissions/dw4/second-run").glob("*.json"))

            self.assertEqual(
                [path.stem for path in snapshots],
                ["egnn_norm", "egnn_unnorm", "gnn_norm", "gnn_unnorm"],
            )
            combinations = {
                (
                    json.loads(path.read_text())["model"]["architecture"],
                    json.loads(path.read_text())["drift"]["normalized"],
                )
                for path in snapshots
            }
            self.assertEqual(
                combinations,
                {("egnn", False), ("egnn", True), ("gnn", False), ("gnn", True)},
            )
            self.assertTrue(
                all(json.loads(path.read_text())["value"] == "at submission" for path in snapshots)
            )
            self.assertTrue(
                all(
                    json.loads(path.read_text())["value"] == "edited later"
                    for path in second_snapshots
                )
            )
            self.assertTrue(all(path.stat().st_mode & 0o222 == 0 for path in snapshots))
            self.assertEqual(result.stdout.count("CONFIG_SNAPSHOT="), 4)
            self.assertEqual(result.stdout.count("train_dw4.sh --seed 17"), 4)


if __name__ == "__main__":
    unittest.main()
