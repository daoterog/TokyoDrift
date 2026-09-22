import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from train import EarlyStopping
from train import main as train_main


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_patience_epochs_without_epsilon_improvement(self):
        stopping = EarlyStopping(epsilon=0.1, patience=3)

        self.assertFalse(stopping.update(1.0))
        self.assertFalse(stopping.update(0.95))
        self.assertFalse(stopping.update(0.91))
        self.assertTrue(stopping.update(0.90))
        self.assertEqual(stopping.best_loss, 1.0)

    def test_meaningful_improvement_resets_patience(self):
        stopping = EarlyStopping(epsilon=0.1, patience=2)

        self.assertFalse(stopping.update(1.0))
        self.assertFalse(stopping.update(0.95))
        self.assertFalse(stopping.update(0.89))
        self.assertEqual(stopping.epochs_without_improvement, 0)
        self.assertFalse(stopping.update(0.88))
        self.assertTrue(stopping.update(0.87))

    def test_rejects_invalid_settings(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            EarlyStopping(epsilon=-1.0, patience=1)
        with self.assertRaisesRegex(ValueError, "positive"):
            EarlyStopping(epsilon=0.0, patience=0)

    def test_flat_training_loss_writes_only_the_stopping_epoch(self):
        config = {
            "system": "dw4",
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
                "epochs": 10,
                "batch_size": 2,
                "positive_references": 2,
                "coordinate_noise_scale": 1,
                "bandwidth": [1.0],
                "eta": 0,
                "repulsion": 1,
                "min_radius_fraction": 0,
                "learning_rate": 1e-4,
                "weight_decay": 0,
                "gradient_clip": 1,
                "ema_decay": None,
                "log_every": 1,
                "early_stopping_epsilon": 1e-6,
                "early_stopping_patience": 2,
            },
        }
        samples = torch.randn(4, 4, 2)

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
                patch("train.load_dataset", return_value=(samples, {"system": "dw4"})),
                patch("sys.argv", arguments),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train_main()

            checkpoints = list((root / "run").glob("*.pt"))
            self.assertEqual([path.name for path in checkpoints], ["final.pt"])
            checkpoint = torch.load(checkpoints[0], weights_only=False)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["early_stopping"]["epochs_without_improvement"], 2)


if __name__ == "__main__":
    unittest.main()
