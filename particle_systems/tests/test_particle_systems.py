from __future__ import annotations

import unittest

import torch

from particle_systems.drift import UnnormalizedDrift
from particle_systems.model import ParticleGenerator
from particle_systems.systems import center, dw4_energy, lj_energy


class PotentialTests(unittest.TestCase):
    def test_dw_pair_at_minimum(self) -> None:
        positions = torch.tensor([[[0.0, 0.0], [4.0, 0.0]]])
        self.assertTrue(torch.allclose(dw4_energy(positions), torch.tensor([0.0])))

    def test_lj_pair_at_minimum(self) -> None:
        positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        self.assertTrue(torch.allclose(lj_energy(positions), torch.tensor([-1.0])))


class DriftTests(unittest.TestCase):
    def test_raw_field_is_not_kernel_mass_normalized(self) -> None:
        query = torch.zeros(1, 2, 2)
        near = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        far = 5.0 * near
        drift = UnnormalizedDrift(1.0)
        near_field, _ = drift(query, near, repulsion=0.0)
        far_field, _ = drift(query, far, repulsion=0.0)
        self.assertLess(far_field.norm(), near_field.norm())

    def test_drift_preserves_center(self) -> None:
        query = center(torch.randn(3, 4, 2))
        references = center(torch.randn(5, 4, 2))
        field, _ = UnnormalizedDrift(2.0)(query, references, repulsion=0.0)
        self.assertTrue(torch.allclose(field.mean(dim=1), torch.zeros(3, 2), atol=1e-6))


class ModelTests(unittest.TestCase):
    def test_permutation_and_rotation_equivariance(self) -> None:
        torch.manual_seed(4)
        model = ParticleGenerator(feature_dim=3, hidden_dim=16, layers=2, radial_basis=6)
        positions = center(torch.randn(2, 4, 2))
        features = torch.randn(2, 4, 3)
        permutation = torch.tensor([2, 0, 3, 1])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        expected = model(positions, features)[:, permutation] @ rotation.T
        actual = model(positions[:, permutation] @ rotation.T, features[:, permutation])
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
