import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from plot_training_history import load_history, plot_history
from train import main as train_main


class TrainingHistoryPlotTests(unittest.TestCase):
    def record(self, epoch, train_loss, test_loss, train_wasserstein, test_wasserstein):
        return {
            "epoch": epoch,
            "global_step": epoch * 10,
            "train": {
                "drift_loss": train_loss,
                "energy_wasserstein_1": train_wasserstein,
            },
            "test": {
                "drift_loss": test_loss,
                "energy_wasserstein_1": test_wasserstein,
            },
        }

    def test_loads_history_and_writes_both_plots(self):
        records = [
            self.record(20, 2e-6, 3e-6, 4.0, 4.2),
            self.record(40, 1e-6, 2e-6, 3.0, 3.2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.jsonl"
            history.write_text("".join(json.dumps(record) + "\n" for record in records))
            loaded = load_history(history)
            plot_history(loaded, root / "plots")
            self.assertEqual([record["epoch"] for record in loaded], [20, 40])
            self.assertGreater((root / "plots/loss_over_epochs.png").stat().st_size, 0)
            self.assertGreater(
                (root / "plots/energy_wasserstein_over_epochs.png").stat().st_size, 0
            )

    def test_rejects_nonincreasing_epochs(self):
        records = [
            self.record(20, 2e-6, 3e-6, 4.0, 4.2),
            self.record(20, 1e-6, 2e-6, 3.0, 3.2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            history.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_history(history)

    def test_training_records_fixed_train_and_test_metrics(self):
        config = {
            "system": "dw4",
            "drift": {
                "space": "particle_coordinates",
                "kernel": "gaussian",
                "normalized": False,
            },
            "data": "unused.npz",
            "seed": 42,
            "model": {
                "architecture": "egnn",
                "feature_dim": 4,
                "hidden_dim": 8,
                "layers": 1,
                "radial_basis": 4,
                "max_distance": 8,
            },
            "training": {
                "epochs": 2,
                "batch_size": 2,
                "positive_references": 2,
                "coordinate_noise_scale": 1,
                "bandwidth": [1.0],
                "eta": 0.1,
                "repulsion": 1,
                "min_radius_fraction": 0,
                "learning_rate": 1e-4,
                "weight_decay": 0,
                "gradient_clip": 1,
                "ema_decay": None,
                "log_every": 1,
                "checkpoint_every": 1,
                "track_train_test_metrics": True,
                "tracking_every": 1,
                "tracking_generated_samples": 4,
                "tracking_batch_size": 2,
                "tracking_positive_references": 2,
            },
        }
        train = torch.randn(6, 4, 2)
        test = torch.randn(5, 4, 2)

        def load(_path, split):
            return (test if split == "test" else train), {"system": "dw4"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            arguments = [
                "train",
                "--config",
                str(config_path),
                "--device",
                "cpu",
                "--output",
                str(root / "run"),
            ]
            with (
                patch("train.load_dataset", side_effect=load),
                patch("sys.argv", arguments),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train_main()
            records = load_history(root / "run/train_test_history.jsonl")
            self.assertEqual([record["epoch"] for record in records], [1, 2])
            for record in records:
                for split in ("train", "test"):
                    self.assertGreaterEqual(record[split]["drift_loss"], 0)
                    self.assertGreaterEqual(record[split]["energy_wasserstein_1"], 0)


if __name__ == "__main__":
    unittest.main()
