from __future__ import annotations

import math
import unittest

import torch

from particle_systems.kde import (
    descriptor_kde_nll,
    gaussian_kde_log_prob,
    raw_distance_kde_report,
)
from particle_systems.systems import center


class KDETests(unittest.TestCase):
    def test_single_center_matches_normalized_gaussian_density(self) -> None:
        queries = torch.zeros(1, 2)
        centers = torch.zeros(1, 2)
        bandwidth = 0.5
        expected = -2 * math.log(bandwidth) - math.log(2 * math.pi)
        torch.testing.assert_close(
            gaussian_kde_log_prob(queries, centers, bandwidth),
            torch.tensor([expected]),
        )

    def test_invalid_bandwidth_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "bandwidth"):
            gaussian_kde_log_prob(torch.zeros(1, 2), torch.zeros(1, 2), 0)

    def test_manifold_dimension_controls_kde_normalization(self) -> None:
        queries = torch.zeros(1, 2)
        centers = torch.zeros(1, 2)
        bandwidth = 0.5
        expected = -math.log(bandwidth) - 0.5 * math.log(2 * math.pi)
        torch.testing.assert_close(
            gaussian_kde_log_prob(queries, centers, bandwidth, density_dimension=1),
            torch.tensor([expected]),
        )

    def test_descriptor_report_is_deterministic_and_symmetry_invariant(self) -> None:
        generator = torch.Generator().manual_seed(73)
        training = center(torch.randn(40, 4, 2, generator=generator))
        test = center(torch.randn(20, 4, 2, generator=generator))
        generated = center(torch.randn(30, 4, 2, generator=generator))
        options = {
            "center_count": 10,
            "tuning_query_count": 5,
            "test_query_count": 8,
            "query_batch_size": 3,
            "bandwidth_candidates": [0.1, 0.5, 1.0],
        }
        report = descriptor_kde_nll(generated, training, test, **options)
        repeated = descriptor_kde_nll(generated, training, test, **options)
        self.assertEqual(report, repeated)
        self.assertTrue(math.isfinite(report["generated_kde"]["mean"]))
        self.assertEqual(report["descriptor_dimension"], 6)
        self.assertEqual(report["density_dimension"], 5)
        self.assertIn(
            report["bandwidth"]["generated_kde"]["selected"],
            options["bandwidth_candidates"],
        )
        self.assertIn(
            report["bandwidth"]["reference_kde_baseline"]["selected"],
            options["bandwidth_candidates"],
        )

        raw_report = raw_distance_kde_report(report)
        scale = math.sqrt(6)
        shift = 5 * math.log(scale)
        self.assertAlmostEqual(
            raw_report["generated_kde"]["mean"],
            report["generated_kde"]["mean"] + shift,
        )
        self.assertAlmostEqual(
            raw_report["bandwidth"]["generated_kde"]["selected"],
            report["bandwidth"]["generated_kde"]["selected"] * scale,
        )
        self.assertEqual(
            raw_report["excess_nll_generated_minus_reference"],
            report["excess_nll_generated_minus_reference"],
        )

        permutation = torch.tensor([2, 0, 3, 1])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        transformed = generated[:, permutation] @ rotation.T
        invariant = descriptor_kde_nll(transformed, training, test, **options)
        for statistic, value in report["generated_kde"].items():
            self.assertAlmostEqual(value, invariant["generated_kde"][statistic], places=6)


if __name__ == "__main__":
    unittest.main()
