from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data.alanine_dipeptide.metrics import (
    exact_wasserstein_1,
    probability_metrics,
    torsion_comparison,
)
from data.alanine_dipeptide.system import (
    PHI,
    PSI,
    AlanineDrift,
    backbone_angles,
    chirality,
    geometry_masks,
    geometry_rules,
)
from evaluate_alanine import energy_report
from evaluate_alanine import main as evaluate_main
from train import main as train_main
from utils.io import build_model


def molecular_frames(count: int = 128) -> torch.Tensor:
    """Build nondegenerate labeled test frames with fixed topology geometry."""
    generator = torch.Generator().manual_seed(731)
    # A random walk supplies nondegenerate local geometry for pure unit tests.
    increments = torch.randn(count, 22, 3, generator=generator)
    increments /= increments.norm(dim=-1, keepdim=True)
    frames = increments.cumsum(dim=1)
    frames -= frames.mean(dim=1, keepdim=True)
    # Retain one chirality consistently, as the physical dataset does.
    sign = chirality(frames).sign()
    frames[sign < 0, :, 0] *= -1
    return frames


class AlanineTests(unittest.TestCase):
    def test_torsions_match_mdtraj_formula(self) -> None:
        positions = molecular_frames(8)
        expected = []
        for indices in (PHI, PSI):
            x = positions[:, indices]
            b1, b2, b3 = x[:, 1] - x[:, 0], x[:, 2] - x[:, 1], x[:, 3] - x[:, 2]
            c1, c2 = torch.linalg.cross(b2, b3), torch.linalg.cross(b1, b2)
            expected.append(torch.atan2((b1 * c1).sum(-1) * b2.norm(dim=-1), (c1 * c2).sum(-1)))
        torch.testing.assert_close(backbone_angles(positions), torch.stack(expected, dim=-1))

    def test_labeled_descriptor_is_rigid_motion_invariant_but_not_permutation_invariant(
        self,
    ) -> None:
        positions = molecular_frames(4).double()
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        expected = AlanineDrift.descriptors(positions)
        torch.testing.assert_close(AlanineDrift.descriptors(positions @ rotation + 13), expected)
        self.assertFalse(
            torch.allclose(AlanineDrift.descriptors(positions[:, [1, 0, *range(2, 22)]]), expected)
        )

    def test_geometry_rules_detect_mirror_chirality_and_broken_bond(self) -> None:
        frames = molecular_frames()
        rules = geometry_rules(frames)
        original = geometry_masks(frames, rules)
        mirrored = frames.clone()
        mirrored[:, :, 0] *= -1
        masks = geometry_masks(mirrored, rules)
        torch.testing.assert_close(masks["geometry_valid"], original["geometry_valid"])
        self.assertTrue(original["correct_chirality"].all())
        self.assertFalse(masks["correct_chirality"].any())
        broken = frames[:1].clone()
        broken[:, 0] += 5
        self.assertFalse(geometry_masks(broken, rules)["bonds_in_range"].item())

    def test_periodic_torsion_metrics_are_finite_and_identity_is_zero(self) -> None:
        frames = molecular_frames()
        result = torsion_comparison(frames, frames)
        for metric in result["phi_psi"].values():
            self.assertEqual(metric, 0.0)
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        separated = probability_metrics(p, q)
        self.assertAlmostEqual(separated["total_variation"], 1.0)
        self.assertTrue(math.isfinite(separated["kl_reference_to_generated"]))
        self.assertEqual(exact_wasserstein_1(np.arange(4), np.arange(4) + 2), 2.0)

    def test_energy_report_preserves_nonfinite_and_tail_failures(self) -> None:
        frames = molecular_frames(8)
        rules = geometry_rules(frames)
        reference = np.arange(8, dtype=float)
        generated = reference.copy()
        generated[0] = np.nan
        generated[-1] = 100
        report = energy_report(generated, reference, frames, frames, rules)
        self.assertEqual(report["generated"]["finite_fraction"], 7 / 8)
        self.assertGreater(report["generated_above_reference_q99_fraction"], 0)
        self.assertLess(report["generated_geometry_chirality_and_energy_valid_fraction"], 1)
        self.assertTrue(math.isfinite(report["histogram"]["js"]))

    def test_fixed_atom_identity_is_required_and_replaces_feature_noise(self) -> None:
        coordinates = molecular_frames(2)
        for architecture in ("egnn", "gnn"):
            with self.subTest(architecture=architecture):
                config = {
                    "system": "aldp",
                    "model": {
                        "architecture": architecture,
                        "feature_dim": 22,
                        "fixed_atom_identity": True,
                        "hidden_dim": 8,
                        "layers": 1,
                        "radial_basis": 4,
                        "max_distance": 8,
                    },
                }
                model = build_model(config)
                first = model(coordinates, torch.randn(2, 22, 22))
                second = model(coordinates, torch.randn(2, 22, 22))
                torch.testing.assert_close(first, second)
                config["model"]["fixed_atom_identity"] = False
                with self.assertRaisesRegex(ValueError, "fixed_atom_identity"):
                    build_model(config)

    def test_training_uses_official_validation_split_and_writes_best_checkpoint(self) -> None:
        train, validation = molecular_frames(4), molecular_frames(5)
        metadata = {
            "system": "aldp",
            "coordinate_units": "angstrom",
            "geometry_rules": geometry_rules(train),
            "record": "test-record",
        }
        config = {
            "system": "aldp",
            "drift": {
                "space": "particle_coordinates",
                "kernel": "gaussian",
                "normalized": False,
                "descriptors": True,
            },
            "data": "unused.npz",
            "seed": 42,
            "model": {
                "architecture": "egnn",
                "feature_dim": 22,
                "fixed_atom_identity": True,
                "hidden_dim": 8,
                "layers": 1,
                "radial_basis": 4,
                "max_distance": 8,
            },
            "training": {
                "epochs": 1,
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
                "validation_split": "validation",
                "validation_every": 1,
                "validation_generated_samples": 2,
                "validation_batch_size": 2,
                "validation_positive_references": 2,
                "validation_metric": "selection_score",
            },
        }

        def load(_path: Path, split: str):
            return (validation if split == "validation" else train), metadata

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
            self.assertTrue((root / "run/best_validation.pt").is_file())
            record = json.loads((root / "run/best_validation.json").read_text())
            self.assertIn("selection_score", record["validation"])
            evaluation_arguments = [
                "evaluate_alanine",
                "--checkpoint",
                str(root / "run/best_validation.pt"),
                "--output",
                str(root / "evaluation"),
                "--device",
                "cpu",
                "--num-samples",
                "8",
                "--batch-size",
                "4",
                "--energy-samples",
                "2",
                "--metric-sample-size",
                "4",
                "--skip-energy",
            ]
            with (
                patch(
                    "evaluate_alanine.load_dataset",
                    return_value=(validation, metadata),
                ),
                patch("sys.argv", evaluation_arguments),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                evaluate_main()
            report = json.loads((root / "evaluation/metrics.json").read_text())
            self.assertEqual(report["num_generated_samples"], 8)
            self.assertFalse(report["energy"]["available"])
            self.assertTrue((root / "evaluation/ramachandran.png").is_file())


if __name__ == "__main__":
    unittest.main()
