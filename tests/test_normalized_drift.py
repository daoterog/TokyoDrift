from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from drifting import Drifting
from train import main as train_main
from train import resolve_drift_definition
from utils.descriptors import DescriptorDrift, particle_descriptors


class NormalizedDriftTests(unittest.TestCase):
    def test_fields_match_gradient_of_log_kernel_means(self):
        torch.manual_seed(97)
        samples = torch.randn(3, 4, 2, dtype=torch.float64)
        positive = torch.randn(6, 4, 2, dtype=torch.float64)
        negative = torch.randn(5, 4, 2, dtype=torch.float64)
        self_indices = torch.tensor([4, -1, 1])
        negative[4], negative[1] = samples[0], samples[2]
        positive_weights = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float64) / 15
        for drift_class in (Drifting, DescriptorDrift):
            for kernel in ("gaussian", "laplacian"):
                with self.subTest(drift_class=drift_class.__name__, kernel=kernel):
                    query = samples.clone().requires_grad_()
                    transform = (
                        particle_descriptors
                        if drift_class is DescriptorDrift
                        else lambda value: value.flatten(start_dim=1)
                    )
                    z = transform(query)
                    pos_distance = (z[:, None] - transform(positive)[None]).norm(dim=-1)
                    neg_distance = (z[:, None] - transform(negative)[None]).norm(dim=-1)
                    mask = torch.ones_like(neg_distance)
                    mask[0, 4] = mask[2, 1] = 0
                    objective = 0
                    for bandwidth in (0.7, 1.9):
                        if kernel == "gaussian":
                            pos_kernel = torch.exp(-pos_distance.square() / (2 * bandwidth**2))
                            neg_kernel = torch.exp(-neg_distance.square() / (2 * bandwidth**2))
                        else:
                            pos_kernel = torch.exp(-pos_distance / bandwidth)
                            neg_kernel = torch.exp(-neg_distance / bandwidth)
                        pos_mass = pos_kernel @ positive_weights
                        neg_mass = (neg_kernel * mask).mean(dim=1)
                        objective = objective + (pos_mass.log() - 0.6 * neg_mass.log()).sum() / 2
                    expected = torch.autograd.grad(objective, query)[0]
                    actual, metrics = drift_class([0.7, 1.9], kernel=kernel, normalized=True)(
                        query,
                        positive,
                        negative,
                        self_indices=self_indices,
                        positive_weights=positive_weights,
                        repulsion=0.6,
                    )
                    torch.testing.assert_close(actual, expected)
                    self.assertFalse(actual.requires_grad)
                    _, raw_metrics = drift_class([0.7, 1.9], kernel=kernel)(
                        query,
                        positive,
                        negative,
                        self_indices=self_indices,
                        positive_weights=positive_weights,
                        repulsion=0.6,
                    )
                    for name in ("positive_kernel_mass", "negative_kernel_mass"):
                        torch.testing.assert_close(metrics[name], raw_metrics[name])

    def test_small_kernel_mass_is_normalized_and_empty_repulsion_is_zero(self):
        query = torch.tensor([[[0.0]]])
        reference = torch.tensor([[[10.0]]])
        # exp(-50) is representable but much smaller than float32 eps.
        field, metrics = Drifting(1.0, normalized=True)(query, reference)
        torch.testing.assert_close(field, reference - query)
        self.assertEqual(float(metrics["negative_kernel_mass"]), 0)
        raw, _ = Drifting(1.0)(query, reference)
        self.assertLess(float(raw.abs().max()), 1e-19)

    def test_underflow_and_coincident_singletons_remain_finite(self):
        for drift_class in (Drifting, DescriptorDrift):
            for kernel in ("gaussian", "laplacian"):
                query = torch.zeros(1, 4, 3)
                references = torch.arange(12, dtype=query.dtype).reshape_as(query)
                drift = drift_class(1e-6, kernel=kernel, normalized=True)
                for positive in (query, references):
                    field, _ = drift(query, positive)
                    torch.testing.assert_close(field, torch.zeros_like(field))

    def test_normalized_descriptor_field_still_preserves_symmetry(self):
        torch.manual_seed(12)
        query = torch.randn(3, 4, 3, dtype=torch.float64)
        references = torch.randn(5, 4, 3, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=query.dtype))
        permutation = [2, 0, 3, 1]
        drift = DescriptorDrift([0.3, 1.1], normalized=True)
        field, _ = drift(query, references)
        transformed, _ = drift(query[:, permutation] @ rotation + 12, references)
        torch.testing.assert_close(transformed, field[:, permutation] @ rotation)
        torch.testing.assert_close(field.sum(dim=1), torch.zeros_like(field[:, 0]))

    def test_normalized_switch_is_strict_and_opt_in(self):
        self.assertFalse(resolve_drift_definition({})["normalized"])
        for value in (True, False):
            definition = resolve_drift_definition({"drift": {"normalized": value}})
            self.assertIs(definition["normalized"], value)
        for value in ("true", "false", 1, None):
            with self.assertRaises(ValueError):
                resolve_drift_definition({"drift": {"normalized": value}})
            with self.assertRaises(ValueError):
                Drifting(1.0, normalized=value)

    def test_lj55_training_override_checkpoint_and_resume(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/lj55_config.json"
        config = json.loads(config_path.read_text())
        original_model = config["model"].copy()
        # Exercise the real LJ55 architecture with a tiny synthetic reference split.
        config["training"].update(epochs=1, batch_size=2, positive_references=2)
        samples = torch.randn(3, 55, 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_config = root / "config.json"
            test_config.write_text(json.dumps(config))
            argv = [
                "train",
                "--config",
                str(test_config),
                "--device",
                "cpu",
                "--normalized",
                "--output",
                str(root / "run"),
            ]
            with (
                patch(
                    "train.load_dataset",
                    return_value=(samples, {"system": "lj55"}),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with patch("sys.argv", argv):
                    train_main()
                checkpoint_path = root / "run/latest.pt"
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                self.assertTrue(checkpoint["config"]["drift"]["normalized"])
                self.assertEqual(checkpoint["config"]["model"], original_model)
                self.assertEqual(checkpoint["global_step"], 2)
                self.assertTrue(all(torch.isfinite(t).all() for t in checkpoint["model"].values()))
                parameters = json.loads((root / "run/parameters.json").read_text())
                self.assertTrue(parameters["drift"]["normalized"])
                with patch("sys.argv", [*argv, "--resume", str(checkpoint_path), "--epochs", "2"]):
                    train_main()
                resumed = torch.load(checkpoint_path, weights_only=False)
                self.assertEqual(resumed["epoch"], 2)
                self.assertEqual(resumed["global_step"], 4)
                with (
                    patch("sys.argv", [*argv, "--resume", str(checkpoint_path), "--no-normalized"]),
                    self.assertRaisesRegex(ValueError, "drift definitions differ"),
                ):
                    train_main()


if __name__ == "__main__":
    unittest.main()
