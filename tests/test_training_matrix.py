from __future__ import annotations

import unittest

from models import EGNN, GNN
from utils.io import build_model
from utils.training_matrix import trainable_parameter_count, training_variants


class TrainingMatrixTests(unittest.TestCase):
    def test_crosses_architecture_and_normalization_without_mutating_base(self) -> None:
        config = {
            "system": "lj13",
            "seed": 42,
            "model": {
                "architecture": "egnn",
                "variant": "bounded",
                "feature_dim": 8,
                "hidden_dim": 64,
                "layers": 4,
                "radial_basis": 16,
            },
            "drift": {"normalized": False, "kernel": "gaussian"},
        }

        variants = dict(training_variants(config))

        self.assertEqual(
            list(variants),
            ["egnn_unnorm", "egnn_norm", "gnn_unnorm", "gnn_norm"],
        )
        self.assertEqual(
            {
                name: (variant["model"]["architecture"], variant["drift"]["normalized"])
                for name, variant in variants.items()
            },
            {
                "egnn_unnorm": ("egnn", False),
                "egnn_norm": ("egnn", True),
                "gnn_unnorm": ("gnn", False),
                "gnn_norm": ("gnn", True),
            },
        )
        self.assertTrue(all(variant["seed"] == 42 for variant in variants.values()))
        self.assertTrue(all(variant["model"]["hidden_dim"] == 64 for variant in variants.values()))
        self.assertEqual(variants["egnn_unnorm"]["model"]["radial_basis"], 16)
        self.assertEqual(variants["egnn_norm"]["model"]["radial_basis"], 16)
        self.assertEqual(variants["gnn_unnorm"]["model"]["radial_basis"], 10)
        self.assertEqual(variants["gnn_norm"]["model"]["radial_basis"], 10)
        reference_count = trainable_parameter_count(variants["egnn_unnorm"])
        for name in ("egnn_norm", "gnn_unnorm", "gnn_norm"):
            with self.subTest(name=name):
                self.assertLessEqual(
                    abs(trainable_parameter_count(variants[name]) - reference_count),
                    8,
                )
        self.assertEqual(config["model"]["architecture"], "egnn")
        self.assertEqual(config["model"]["radial_basis"], 16)
        self.assertFalse(config["drift"]["normalized"])

    def test_two_dimensional_gnn_drops_three_radial_channels(self) -> None:
        config = {
            "system": "dw4",
            "model": {
                "architecture": "egnn",
                "variant": "bounded",
                "feature_dim": 8,
                "hidden_dim": 128,
                "layers": 6,
                "radial_basis": 24,
            },
        }

        variants = dict(training_variants(config))

        self.assertEqual(variants["gnn_unnorm"]["model"]["radial_basis"], 21)
        self.assertEqual(
            trainable_parameter_count(variants["gnn_unnorm"])
            - trainable_parameter_count(variants["egnn_unnorm"]),
            6,
        )

    def test_analytical_count_matches_built_models(self) -> None:
        base = {"feature_dim": 5, "hidden_dim": 12, "layers": 3, "radial_basis": 7, "max_distance": 8}
        models = (
            {"architecture": "egnn", "variant": "block"},
            {"architecture": "egnn", "variant": "block", "layers_per_block": 2, "attention": False},
            {"architecture": "egnn", "variant": "legacy"},
            {"architecture": "egnn", "variant": "bounded"},
            {"architecture": "gnn"},
        )
        for system in ("dw4", "lj13"):
            for model in models:
                with self.subTest(system=system, model=model):
                    config = {"system": system, "model": {**base, **model}}
                    built = build_model(config)
                    actual = sum(p.numel() for p in built.parameters() if p.requires_grad)
                    self.assertEqual(trainable_parameter_count(config), actual)

    def test_block_egnn_gnn_is_parameter_matched(self) -> None:
        config = {
            "system": "dw4",
            "model": {
                "architecture": "egnn",
                "variant": "block",
                "feature_dim": 8,
                "hidden_dim": 128,
                "layers": 6,
                "max_distance": 8.0,
            },
        }

        variants = dict(training_variants(config))

        gnn = variants["gnn_unnorm"]["model"]
        self.assertNotIn("variant", gnn)
        self.assertNotIn("aggregation", gnn)
        self.assertGreater(gnn["radial_basis"], 0)
        self.assertLessEqual(
            abs(
                trainable_parameter_count(variants["gnn_unnorm"])
                - trainable_parameter_count(variants["egnn_unnorm"])
            ),
            6 * 128 // 2,
        )
        self.assertIsInstance(build_model(variants["gnn_unnorm"]), GNN)
        self.assertIsInstance(build_model(variants["egnn_unnorm"]), EGNN)


if __name__ == "__main__":
    unittest.main()
