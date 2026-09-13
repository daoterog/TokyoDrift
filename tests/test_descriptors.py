from __future__ import annotations

import unittest

import torch

from train import resolve_drift_definition
from utils.descriptors import (
    DescriptorDrift,
    descriptor_bandwidth,
    particle_descriptors,
)


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.query = torch.randn(5, 4, 3, dtype=torch.float64)
        self.references = torch.randn(7, 4, 3, dtype=torch.float64)

    def test_invariance_and_field_equivariance(self):
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        permutation = [2, 0, 3, 1]
        transformed = self.query[:, permutation] @ rotation + 12
        torch.testing.assert_close(
            particle_descriptors(transformed), particle_descriptors(self.query)
        )
        drift = DescriptorDrift([0.3, 1.1])
        field, _ = drift(self.query, self.references)
        # References may be transformed independently from the queries.
        actual, _ = drift(transformed, self.references[:, permutation].flip(-1) - 4)
        torch.testing.assert_close(actual, field[:, permutation] @ rotation)
        torch.testing.assert_close(field.sum(dim=1), torch.zeros_like(field[:, 0]))

    def test_field_matches_full_coordinate_autograd(self):
        for kernel in ("gaussian", "laplacian"):
            query = self.query.clone().requires_grad_()
            z = particle_descriptors(query)
            pos = particle_descriptors(self.references)
            neg = particle_descriptors(self.query)
            positive_weights = torch.arange(1, 8, dtype=query.dtype)
            positive_weights /= positive_weights.sum()
            density = 0
            for bandwidth in (0.3, 1.1):
                positive_distance = (z[:, None] - pos[None]).norm(dim=-1)
                negative_distance = (z[:, None] - neg[None]).norm(dim=-1)
                if kernel == "gaussian":
                    positive = torch.exp(-positive_distance.square() / (2 * bandwidth**2))
                    negative = torch.exp(-negative_distance.square() / (2 * bandwidth**2))
                else:
                    positive = torch.exp(-positive_distance / bandwidth)
                    negative = torch.exp(-negative_distance / bandwidth)
                negative = negative * (1 - torch.eye(len(query)))
                density = (
                    density + (positive @ positive_weights - 0.7 * negative.mean(dim=1)).sum() / 2
                )
            expected = torch.autograd.grad(density, query)[0]
            with torch.no_grad():
                actual, _ = DescriptorDrift([0.3, 1.1], kernel=kernel)(
                    query, self.references, repulsion=0.7, positive_weights=positive_weights
                )
            torch.testing.assert_close(actual, expected)
            self.assertFalse(actual.requires_grad)

    def test_pair_attraction_reduces_distance_mismatch(self):
        query = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])
        references = query * 2
        updated, _ = DescriptorDrift(2.0).step(query, references, 0.1, repulsion=0)
        before = (particle_descriptors(query) - particle_descriptors(references)).norm()
        after = (particle_descriptors(updated) - particle_descriptors(references)).norm()
        self.assertLess(after, before)

    def test_self_exclusion_and_collisions_are_finite(self):
        for kernel in ("gaussian", "laplacian"):
            query = torch.zeros(1, 4, 3)
            field, metrics = DescriptorDrift([0.2, 1.0], kernel=kernel)(query, query)
            self.assertTrue(torch.isfinite(field).all())
            self.assertEqual(float(metrics["negative_kernel_mass"]), 0)

    def test_auto_bandwidth_has_rms_pair_distance_units(self):
        samples = torch.tensor([[[0.0], [1.0]], [[0.0], [4.0]]])
        self.assertEqual(descriptor_bandwidth(samples), 3.0)
        with self.assertRaises(ValueError):
            descriptor_bandwidth(samples[:1])
        with self.assertRaises(ValueError):
            descriptor_bandwidth(samples[:1].expand(3, -1, -1))

    def test_toggle_is_strict_and_legacy_config_is_compatible(self):
        legacy = {
            "drift": {"space": "particle_coordinates", "kernel": "gaussian", "normalized": False}
        }
        self.assertFalse(resolve_drift_definition(legacy)["descriptors"])
        for value in (True, False):
            self.assertIs(
                resolve_drift_definition({"drift": {"descriptors": value}})["descriptors"], value
            )
        for value in ("true", "false", 1, None):
            with self.assertRaises(ValueError):
                resolve_drift_definition({"drift": {"descriptors": value}})


if __name__ == "__main__":
    unittest.main()
