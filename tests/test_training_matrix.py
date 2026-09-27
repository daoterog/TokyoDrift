from __future__ import annotations

import unittest

from utils.training_matrix import training_variants


class TrainingMatrixTests(unittest.TestCase):
    def test_crosses_architecture_and_normalization_without_mutating_base(self) -> None:
        config = {
            "seed": 42,
            "model": {"architecture": "egnn", "hidden_dim": 64},
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
        self.assertEqual(config["model"]["architecture"], "egnn")
        self.assertFalse(config["drift"]["normalized"])


if __name__ == "__main__":
    unittest.main()
