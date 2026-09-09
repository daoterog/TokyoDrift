from __future__ import annotations

import unittest

import numpy as np
import torch

from particle_systems.evaluate import valid_sample_observables, validity_metrics
from particle_systems.model import ParticleGenerator
from particle_systems.prepare_data import select_rows
from particle_systems.systems import center, dw4_energy, get_system, lj_energy
from particle_systems.toys.gmm40 import GMM40, metrics
from particle_systems.train import (
    EnergyStratifiedReferenceSampler,
    bandwidth_scale,
    epoch_reference_batches,
    learning_rate,
    resolve_ema_decay,
)
from particle_systems.unnormalized_drifting import DirectCoordinateDrift, median_bandwidth


class PotentialTests(unittest.TestCase):
    def test_dw_pair_at_minimum(self) -> None:
        positions = torch.tensor([[[0.0, 0.0], [4.0, 0.0]]])
        self.assertTrue(torch.allclose(dw4_energy(positions), torch.tensor([0.0])))

    def test_lj_pair_at_minimum(self) -> None:
        positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        self.assertTrue(torch.allclose(lj_energy(positions), torch.tensor([-1.0])))

    def test_lj55_registry_matches_the_two_part_release(self) -> None:
        system = get_system("lj55")
        self.assertEqual((system.particles, system.dimensions), (55, 3))
        self.assertEqual(len(system.sources), 2)
        self.assertEqual(sum(source.shape[0] for source in system.sources), 10_000_000)
        self.assertTrue(all(source.shape[1] == 165 for source in system.sources))

    def test_global_rows_are_selected_across_source_parts(self) -> None:
        parts = [np.arange(6).reshape(3, 2), np.arange(6, 12).reshape(3, 2)]
        actual = select_rows(parts, np.array([0, 2, 3, 5]))
        expected = np.array([[0, 1], [4, 5], [6, 7], [10, 11]], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_multipart_selection_rejects_an_out_of_range_index(self) -> None:
        with self.assertRaises(IndexError):
            select_rows([np.zeros((2, 3))], np.array([2]))

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
    def test_full_epoch_batches_visit_every_reference_once(self) -> None:
        batches = epoch_reference_batches(10, 4, torch.Generator().manual_seed(9))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
        self.assertEqual(torch.cat(batches).sort().values.tolist(), list(range(10)))

    def test_stratified_epoch_batches_visit_every_reference_once(self) -> None:
        torch.manual_seed(13)
        samples = center(torch.randn(40, 4, 2))
        sampler = EnergyStratifiedReferenceSampler.build(samples, get_system("dw4"), [0.5, 0.9])
        batches = sampler.epoch_batches(12, torch.Generator().manual_seed(9))
        self.assertEqual(torch.cat(batches).sort().values.tolist(), list(range(40)))

    def test_uniform_positive_weights_match_the_empirical_mean_field(self) -> None:
        query = center(torch.randn(2, 4, 2))
        references = center(torch.randn(5, 4, 2))
        drift = DirectCoordinateDrift(1.0)
        ordinary, _ = drift(query, references, repulsion=0.0)
        weighted, _ = drift(
            query,
            references,
            repulsion=0.0,
            positive_weights=torch.full((5,), 0.2),
        )
        self.assertTrue(torch.allclose(ordinary, weighted, atol=1e-6, rtol=1e-6))


class DriftTests(unittest.TestCase):
    def test_field_matches_analytical_gaussian_gradient(self) -> None:
        query = torch.tensor([[[-0.5], [0.5]]])
        reference = torch.tensor([[[-1.5], [1.5]]])
        field, _ = DirectCoordinateDrift(2.0)(query, reference, repulsion=0.0)
        kernel = torch.exp(torch.tensor(-1.0 / 4.0))
        expected = kernel * (reference - query) / 4.0
        self.assertTrue(torch.allclose(field, expected))

    def test_gaussian_kernel_omits_density_normalizer(self) -> None:
        coordinate = torch.linspace(-10.0, 10.0, 20_001)
        query = torch.stack(
            (-coordinate / 2.0**0.5, coordinate / 2.0**0.5), dim=1
        ).unsqueeze(-1)
        reference = torch.zeros(1, 2, 1)
        _, kernel_density = DirectCoordinateDrift(1.0)._fields(query, reference)
        integral = torch.trapezoid(kernel_density.squeeze(0), coordinate)
        expected = torch.sqrt(torch.tensor(2.0 * torch.pi))
        self.assertTrue(torch.allclose(integral, expected, atol=1e-5))

    def test_laplacian_field_matches_analytical_density_gradient(self) -> None:
        query = torch.tensor([[[-0.5], [0.5]]])
        reference = torch.tensor([[[-1.5], [1.5]]])
        field, _ = DirectCoordinateDrift(2.0, kernel="laplacian")(
            query, reference, repulsion=0.0
        )
        distance = torch.sqrt(torch.tensor(2.0))
        kernel = torch.exp(-distance / 2.0)
        expected = kernel * (reference - query) / (2.0 * distance)
        self.assertTrue(torch.allclose(field, expected))

    def test_laplacian_kernel_omits_density_normalizer(self) -> None:
        coordinate = torch.linspace(-20.0, 20.0, 40_001)
        query = torch.stack(
            (-coordinate / 2.0**0.5, coordinate / 2.0**0.5), dim=1
        ).unsqueeze(-1)
        reference = torch.zeros(1, 2, 1)
        _, kernel_density = DirectCoordinateDrift(1.0, kernel="laplacian")._fields(
            query, reference
        )
        integral = torch.trapezoid(kernel_density.squeeze(0), coordinate)
        self.assertTrue(torch.allclose(integral, torch.tensor(2.0), atol=1e-5))

    def test_laplacian_supports_raw_multi_bandwidth_averaging(self) -> None:
        query = torch.tensor([[[-0.5], [0.5]], [[-1.0], [1.0]]])
        references = torch.tensor([[[-1.5], [1.5]], [[-2.0], [2.0]]])
        bandwidths = (0.5, 1.0, 2.0)
        combined, _ = DirectCoordinateDrift(bandwidths, kernel="laplacian")(
            query, references, repulsion=0.0
        )
        individual = torch.stack(
            [
                DirectCoordinateDrift(value, kernel="laplacian")(
                    query, references, repulsion=0.0
                )[0]
                for value in bandwidths
            ]
        )
        expected = individual.mean(dim=0)
        self.assertTrue(torch.allclose(combined, expected))

    def test_unknown_kernel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DirectCoordinateDrift(1.0, kernel="triangular")

    def test_multiple_bandwidths_average_their_raw_fields(self) -> None:
        query = torch.tensor([[[0.0]], [[0.5]]])
        references = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
        bandwidths = (0.5, 1.0, 2.0)
        combined_field, combined_metrics = DirectCoordinateDrift(bandwidths)(
            query, references, repulsion=0.0
        )
        individual = [
            DirectCoordinateDrift(bandwidth)(query, references, repulsion=0.0)
            for bandwidth in bandwidths
        ]
        individual_fields = torch.stack([field for field, _ in individual])
        individual_rms = individual_fields.square().mean(dim=(1, 2, 3)).sqrt()
        expected_field = individual_fields.mean(dim=0)
        expected_mass = torch.stack(
            [metrics["positive_kernel_mass"] for _, metrics in individual]
        ).mean()
        self.assertTrue(torch.allclose(combined_field, expected_field))
        self.assertTrue(
            torch.allclose(combined_metrics["positive_kernel_mass"], expected_mass)
        )
        self.assertTrue(
            torch.allclose(combined_metrics["temperature_field_rms_mean"], individual_rms.mean())
        )

    def test_multiple_bandwidths_must_be_nonempty_and_positive(self) -> None:
        with self.assertRaises(ValueError):
            DirectCoordinateDrift([])
        with self.assertRaises(ValueError):
            DirectCoordinateDrift([0.5, 0.0])

    def test_zero_field_at_one_temperature_does_not_produce_nan(self) -> None:
        query = torch.tensor([[[0.0]]])
        references = torch.tensor([[[10.0]]])
        field, metrics = DirectCoordinateDrift([1e-6, 10.0])(
            query, references, repulsion=0.0
        )
        self.assertTrue(torch.isfinite(field).all())
        self.assertTrue(torch.isfinite(metrics["drift_rms"]))

    def test_auto_bandwidth_uses_flattened_coordinate_distance(self) -> None:
        samples = torch.tensor([[[0.0]], [[3.0]], [[7.0]]])
        self.assertEqual(median_bandwidth(samples), 4.0)

    def test_field_is_not_kernel_mass_normalized(self) -> None:
        query = torch.tensor([[[-0.5, 0.0], [0.5, 0.0]]])
        near = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])
        far = torch.tensor([[[-5.0, 0.0], [5.0, 0.0]]])
        drift = DirectCoordinateDrift(1.0)
        near_field, _ = drift(query, near, repulsion=0.0)
        far_field, _ = drift(query, far, repulsion=0.0)
        self.assertLess(far_field.norm(), near_field.norm())

    def test_attraction_expands_an_undersized_configuration(self) -> None:
        query = center(
            torch.tensor([[[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]])
        )
        reference = 3.0 * query
        field, _ = DirectCoordinateDrift(2.0)(query, reference, repulsion=0.0)
        self.assertGreater(torch.sum(field * query), 0.0)

    def test_field_is_equivariant_when_query_and_references_transform_together(self) -> None:
        torch.manual_seed(7)
        query = center(torch.randn(2, 4, 2))
        references = center(torch.randn(5, 4, 2))
        permutation = torch.tensor([2, 0, 3, 1])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        drift = DirectCoordinateDrift(2.0)
        expected, _ = drift(query, references, repulsion=0.0)
        actual, _ = drift(
            query[:, permutation] @ rotation.T,
            references[:, permutation] @ rotation.T,
            repulsion=0.0,
        )
        self.assertTrue(
            torch.allclose(actual, expected[:, permutation] @ rotation.T, atol=2e-5, rtol=2e-5)
        )

    def test_field_depends_on_reference_particle_order(self) -> None:
        query = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        references = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])
        drift = DirectCoordinateDrift(2.0)
        ordered, _ = drift(query, references, repulsion=0.0)
        permuted, _ = drift(query, references[:, [1, 0]], repulsion=0.0)
        self.assertFalse(torch.allclose(ordered, permuted))

    def test_drift_preserves_center(self) -> None:
        query = center(torch.randn(3, 4, 2))
        references = center(torch.randn(5, 4, 2))
        field, _ = DirectCoordinateDrift(2.0)(query, references, repulsion=0.0)
        self.assertTrue(torch.allclose(field.mean(dim=1), torch.zeros(3, 2), atol=1e-6))

    def test_step_applies_the_requested_field_scale(self) -> None:
        query = torch.tensor([[[0.0]]])
        reference = torch.tensor([[[1.0]]])
        drift = DirectCoordinateDrift(2.0)
        field, _ = drift(query, reference, repulsion=0.0)
        updated, _ = drift.step(query, reference, step_size=0.3, repulsion=0.0)
        self.assertTrue(torch.allclose(updated, query + 0.3 * field))


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

    def test_ema_can_be_disabled_with_null_but_not_decay_one(self) -> None:
        self.assertIsNone(resolve_ema_decay({"ema_decay": None}))
        self.assertEqual(resolve_ema_decay({"ema_decay": 0.999}), 0.999)
        with self.assertRaises(ValueError):
            resolve_ema_decay({"ema_decay": 1.0})


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
