from __future__ import annotations

import unittest

import torch

from particle_systems.drift import UnnormalizedDrift, descriptor_whitening, invariant_descriptor
from particle_systems.evaluate import valid_sample_observables, validity_metrics
from particle_systems.model import ParticleGenerator
from particle_systems.systems import center, dw4_energy, get_system, lj_energy
from particle_systems.toys.gmm40 import GMM40, metrics
from particle_systems.train import EnergyStratifiedReferenceSampler, bandwidth_scale, learning_rate


class PotentialTests(unittest.TestCase):
    def test_dw_pair_at_minimum(self) -> None:
        positions = torch.tensor([[[0.0, 0.0], [4.0, 0.0]]])
        self.assertTrue(torch.allclose(dw4_energy(positions), torch.tensor([0.0])))

    def test_lj_pair_at_minimum(self) -> None:
        positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        self.assertTrue(torch.allclose(lj_energy(positions), torch.tensor([-1.0])))

    def test_dw4_validity_is_a_broad_threshold_not_a_distribution_distance(self) -> None:
        samples = torch.tensor(
            [
                [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
                [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
            ]
        )
        report = validity_metrics(samples, samples, "dw4", energy_quantile=0.99, minimum_pair_distance=0.5)
        self.assertEqual(report["generated"]["valid_fraction"], 1.0)
        self.assertEqual(report["reference"]["valid_fraction"], 1.0)
        observables = valid_sample_observables(
            samples, samples, "dw4", energy_quantile=0.99, minimum_pair_distance=0.5
        )
        self.assertEqual(observables["generated_retained_fraction"], 1.0)
        self.assertEqual(observables["energy_wasserstein_1"], 0.0)


class GMMTests(unittest.TestCase):
    def test_imbalanced_gmm_has_the_intended_long_tail(self) -> None:
        target = GMM40(torch.device("cpu"), distribution="imbalanced")
        self.assertTrue(torch.isclose(target.weights.sum(), torch.tensor(1.0)))
        self.assertEqual(int((target.tiers == 0).sum()), 8)
        self.assertEqual(int((target.tiers == 1).sum()), 12)
        self.assertEqual(int((target.tiers == 2).sum()), 20)
        self.assertAlmostEqual(float(target.weights[target.tiers == 2].sum()), 0.12)

    def test_validity_metric_accepts_exact_mode_centers(self) -> None:
        target = GMM40(torch.device("cpu"), distribution="imbalanced")
        report = metrics(target, target.centers, target.centers)
        self.assertEqual(report["validity_rate"], 1.0)
        self.assertEqual(report["valid_mode_coverage"], 0.0)  # requires 10 valid samples/mode


class ReferenceSamplingTests(unittest.TestCase):
    def test_stratified_sampler_returns_unbiased_normalized_weights(self) -> None:
        torch.manual_seed(13)
        samples = center(torch.randn(40, 4, 2))
        sampler = EnergyStratifiedReferenceSampler.build(samples, get_system("dw4"), [0.5, 0.9])
        references, weights = sampler.sample(12, torch.Generator().manual_seed(9))
        self.assertEqual(len(references), 12)
        self.assertTrue(torch.isclose(weights.sum(), torch.tensor(1.0)))
        self.assertTrue(torch.all(weights > 0))

    def test_uniform_positive_weights_match_the_empirical_mean_field(self) -> None:
        query = center(torch.randn(2, 4, 2))
        references = center(torch.randn(5, 4, 2))
        drift = UnnormalizedDrift(1.0)
        ordinary, _ = drift(query, references, repulsion=0.0)
        weighted, _ = drift(
            query,
            references,
            repulsion=0.0,
            positive_weights=torch.full((5,), 0.2),
        )
        self.assertTrue(torch.allclose(ordinary, weighted, atol=1e-6, rtol=1e-6))


class DriftTests(unittest.TestCase):
    def test_descriptor_is_invariant_to_rigid_motion_and_permutation(self) -> None:
        positions = center(torch.randn(3, 4, 2))
        permutation = torch.tensor([2, 0, 3, 1])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        transformed = positions[:, permutation] @ rotation.T + torch.tensor([7.0, -3.0])
        self.assertTrue(
            torch.allclose(
                invariant_descriptor(positions), invariant_descriptor(transformed), atol=1e-6
            )
        )

    def test_energy_augmented_descriptor_is_invariant_to_rigid_motion_and_permutation(self) -> None:
        positions = center(torch.randn(3, 4, 2))
        permutation = torch.tensor([2, 0, 3, 1])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        transformed = positions[:, permutation] @ rotation.T + torch.tensor([7.0, -3.0])
        self.assertTrue(
            torch.allclose(
                invariant_descriptor(
                    positions,
                    energy_function=dw4_energy,
                    energy_mean=torch.tensor(0.0),
                    energy_standard_deviation=torch.tensor(1.0),
                    energy_feature_scale=1.0,
                ),
                invariant_descriptor(
                    transformed,
                    energy_function=dw4_energy,
                    energy_mean=torch.tensor(0.0),
                    energy_standard_deviation=torch.tensor(1.0),
                    energy_feature_scale=1.0,
                ),
                atol=1e-6,
            )
        )

    def test_field_is_not_kernel_mass_normalized(self) -> None:
        query = torch.tensor([[[-0.5, 0.0], [0.5, 0.0]]])
        near = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])
        far = torch.tensor([[[-5.0, 0.0], [5.0, 0.0]]])
        drift = UnnormalizedDrift(1.0)
        near_field, _ = drift(query, near, repulsion=0.0)
        far_field, _ = drift(query, far, repulsion=0.0)
        self.assertLess(far_field.norm(), near_field.norm())

    def test_imq_retains_more_far_field_than_gaussian(self) -> None:
        query = torch.tensor([[[-0.5, 0.0], [0.5, 0.0]]])
        far = torch.tensor([[[-5.0, 0.0], [5.0, 0.0]]])
        gaussian_field, _ = UnnormalizedDrift(1.0, kernel="gaussian")(
            query, far, repulsion=0.0
        )
        imq_field, _ = UnnormalizedDrift(1.0, kernel="imq")(
            query, far, repulsion=0.0
        )
        self.assertGreater(imq_field.norm(), gaussian_field.norm())

    def test_attraction_expands_an_undersized_configuration(self) -> None:
        query = center(
            torch.tensor([[[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]])
        )
        reference = 3.0 * query
        field, _ = UnnormalizedDrift(2.0)(query, reference, repulsion=0.0)
        self.assertGreater(torch.sum(field * query), 0.0)

    def test_field_is_permutation_and_rotation_equivariant(self) -> None:
        torch.manual_seed(7)
        query = center(torch.randn(2, 4, 2))
        references = center(torch.randn(5, 4, 2))
        permutation = torch.tensor([2, 0, 3, 1])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        drift = UnnormalizedDrift(2.0)
        expected, _ = drift(query, references, repulsion=0.0)
        actual, _ = drift(query[:, permutation] @ rotation.T, references, repulsion=0.0)
        self.assertTrue(
            torch.allclose(actual, expected[:, permutation] @ rotation.T, atol=2e-5, rtol=2e-5)
        )

    def test_drift_preserves_center(self) -> None:
        query = center(torch.randn(3, 4, 2))
        references = center(torch.randn(5, 4, 2))
        field, _ = UnnormalizedDrift(2.0)(query, references, repulsion=0.0)
        self.assertTrue(torch.allclose(field.mean(dim=1), torch.zeros(3, 2), atol=1e-6))

    def test_whitened_descriptor_field_remains_equivariant(self) -> None:
        torch.manual_seed(8)
        samples = center(torch.randn(16, 4, 2))
        mean, whitener = descriptor_whitening(samples)
        query, references = samples[:2], samples[2:]
        permutation = torch.tensor([2, 0, 3, 1])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        drift = UnnormalizedDrift(1.0, descriptor_mean=mean, descriptor_whitener=whitener)
        expected, _ = drift(query, references, repulsion=0.0)
        actual, _ = drift(query[:, permutation] @ rotation.T, references, repulsion=0.0)
        self.assertTrue(
            torch.allclose(actual, expected[:, permutation] @ rotation.T, atol=2e-5, rtol=2e-5)
        )


class BandwidthScheduleTests(unittest.TestCase):
    def test_cosine_schedule_has_exact_endpoints_and_midpoint(self) -> None:
        training = {
            "bandwidth_scale_start": 1.5,
            "bandwidth_scale_end": 0.5,
            "bandwidth_anneal_start": 10,
            "bandwidth_anneal_end": 30,
        }
        self.assertEqual(bandwidth_scale(training, 1), 1.5)
        self.assertEqual(bandwidth_scale(training, 10), 1.5)
        self.assertAlmostEqual(bandwidth_scale(training, 20), 1.0)
        self.assertEqual(bandwidth_scale(training, 30), 0.5)
        self.assertEqual(bandwidth_scale(training, 40), 0.5)

    def test_legacy_constant_scale_remains_supported(self) -> None:
        self.assertEqual(bandwidth_scale({"bandwidth_scale": 1.25}, 100), 1.25)

    def test_cosine_learning_rate_schedule(self) -> None:
        training = {
            "learning_rate": 2e-5,
            "learning_rate_end": 2e-6,
            "learning_rate_decay_start": 10,
            "learning_rate_decay_end": 30,
        }
        self.assertEqual(learning_rate(training, 10), 2e-5)
        self.assertAlmostEqual(learning_rate(training, 20), 1.1e-5)
        self.assertEqual(learning_rate(training, 30), 2e-6)


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
