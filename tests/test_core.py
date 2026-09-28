from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from data.prepare import select_rows
from data.systems import center, dw4_energy, get_system, lj_energy
from drifting import Drifting, median_bandwidth
from evaluate import arguments as evaluation_arguments
from evaluate import (
    histogram_js,
    metric_observations,
    minimum_pair_distances,
    pair_distance_observations,
    plot_energy_distributions,
    valid_sample_observables,
    validity_metrics,
    wasserstein_1,
)
from models import EGNN, GNN, LegacyEGNN, LegacyEGNNLayer
from train import (
    EnergyStratifiedReferenceSampler,
    bandwidth_scale,
    epoch_reference_batches,
    learning_rate,
    resolve_ema_decay,
)
from utils.io import build_model


class PotentialTests(unittest.TestCase):
    def test_particle_evaluation_defaults_to_seed_42(self) -> None:
        with patch(
            "sys.argv",
            ["evaluate", "--checkpoint", "checkpoint.pt", "--output", "evaluation"],
        ):
            args = evaluation_arguments()
        self.assertEqual(args.seed, 42)

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
        report = validity_metrics(
            samples, samples, "dw4", energy_quantile=0.99, minimum_pair_distance=0.5
        )
        self.assertEqual(report["generated"]["valid_fraction"], 1.0)
        self.assertEqual(report["reference"]["valid_fraction"], 1.0)
        observables = valid_sample_observables(
            samples, samples, "dw4", energy_quantile=0.99, minimum_pair_distance=0.5
        )
        self.assertEqual(observables["generated_retained_fraction"], 1.0)
        self.assertEqual(observables["energy_wasserstein_1"], 0.0)

    def test_empty_valid_population_is_reported_instead_of_raising(self) -> None:
        generated = torch.zeros(2, 4, 2)
        reference = torch.tensor(
            [
                [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
                [[-2.1, -2.0], [-2.0, 2.1], [2.1, -2.0], [2.0, 2.1]],
            ]
        )
        report = valid_sample_observables(
            generated,
            reference,
            "dw4",
            energy_quantile=0.99,
            minimum_pair_distance=0.5,
        )
        self.assertFalse(report["comparison_available"])
        self.assertEqual(report["retained_counts"]["generated"], 0)
        self.assertIsNone(report["pair_distance_wasserstein_1"])

    def test_large_metric_inputs_are_bounded_deterministically(self) -> None:
        values = torch.arange(10_000, dtype=torch.float32)
        first = metric_observations(values, max_observations=1_000)
        second = metric_observations(values, max_observations=1_000)
        self.assertEqual(first.numel(), 1_000)
        self.assertTrue(torch.equal(first, second))

    def test_exact_wasserstein_preserves_a_constant_shift(self) -> None:
        left = torch.arange(10_000, dtype=torch.float32)
        right = left + 2.0
        distance = wasserstein_1(left, right, max_observations=1_000)
        self.assertAlmostEqual(distance, 2.0)

    def test_exact_wasserstein_gives_an_extreme_its_empirical_mass(self) -> None:
        reference = torch.zeros(10_000)
        generated = reference.clone()
        generated[-1] = 1_000_000
        self.assertEqual(wasserstein_1(generated, reference), 100.0)

    def test_exact_wasserstein_supports_unequal_population_sizes(self) -> None:
        self.assertEqual(wasserstein_1(torch.tensor([0.0, 2.0]), torch.tensor([0.0])), 1.0)

    def test_energy_histogram_js_keeps_extreme_tail_mass(self) -> None:
        reference = torch.linspace(-220, -180, 1_000)
        generated = torch.full((1_000,), 1e12)
        self.assertGreater(histogram_js(generated, reference), 0.67)

    def test_lj13_pair_distance_observations_are_bounded_before_expansion(self) -> None:
        positions = torch.randn(100, 13, 3)
        distances = pair_distance_observations(positions, max_observations=1_000)
        self.assertEqual(distances.numel(), 12 * 78)

    def test_chunked_minimum_pair_distances_include_every_configuration(self) -> None:
        positions = torch.randn(23, 13, 3)
        expected = torch.pdist(positions[0]).min()
        actual = minimum_pair_distances(positions, configuration_batch_size=5)
        self.assertEqual(actual.shape, (23,))
        self.assertTrue(torch.allclose(actual[0], expected))

    def test_energy_plots_separate_all_and_valid_populations(self) -> None:
        reference = torch.tensor(
            [
                [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
                [[-2.1, -2.0], [-2.0, 2.1], [2.1, -2.0], [2.0, 2.1]],
            ]
        )
        generated = torch.cat((reference, torch.zeros(1, 4, 2)))
        with TemporaryDirectory() as directory:
            output = Path(directory)
            plot_energy_distributions(generated, reference, "dw4", output, 0.99, 0.5)
            self.assertTrue((output / "energy_all_samples.png").is_file())
            self.assertTrue((output / "energy_valid_samples.png").is_file())
            with np.load(output / "energy_distributions.npz") as archive:
                self.assertEqual(archive["generated"].shape, (3,))
                self.assertFalse(archive["generated_valid"][-1])


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
        drift = Drifting(1.0)
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
        field, _ = Drifting(2.0)(query, reference, repulsion=0.0)
        kernel = torch.exp(torch.tensor(-1.0 / 4.0))
        expected = kernel * (reference - query) / 4.0
        self.assertTrue(torch.allclose(field, expected))

    def test_gaussian_kernel_omits_density_normalizer(self) -> None:
        coordinate = torch.linspace(-10.0, 10.0, 20_001)
        query = torch.stack((-coordinate / 2.0**0.5, coordinate / 2.0**0.5), dim=1).unsqueeze(-1)
        reference = torch.zeros(1, 2, 1)
        _, kernel_density = Drifting(1.0)._fields(query, reference)
        integral = torch.trapezoid(kernel_density.squeeze(0), coordinate)
        expected = torch.sqrt(torch.tensor(2.0 * torch.pi))
        self.assertTrue(torch.allclose(integral, expected, atol=1e-5))

    def test_laplacian_field_matches_analytical_density_gradient(self) -> None:
        query = torch.tensor([[[-0.5], [0.5]]])
        reference = torch.tensor([[[-1.5], [1.5]]])
        field, _ = Drifting(2.0, kernel="laplacian")(query, reference, repulsion=0.0)
        distance = torch.sqrt(torch.tensor(2.0))
        kernel = torch.exp(-distance / 2.0)
        expected = kernel * (reference - query) / (2.0 * distance)
        self.assertTrue(torch.allclose(field, expected))

    def test_laplacian_kernel_omits_density_normalizer(self) -> None:
        coordinate = torch.linspace(-20.0, 20.0, 40_001)
        query = torch.stack((-coordinate / 2.0**0.5, coordinate / 2.0**0.5), dim=1).unsqueeze(-1)
        reference = torch.zeros(1, 2, 1)
        _, kernel_density = Drifting(1.0, kernel="laplacian")._fields(query, reference)
        integral = torch.trapezoid(kernel_density.squeeze(0), coordinate)
        self.assertTrue(torch.allclose(integral, torch.tensor(2.0), atol=1e-5))

    def test_laplacian_supports_raw_multi_bandwidth_averaging(self) -> None:
        query = torch.tensor([[[-0.5], [0.5]], [[-1.0], [1.0]]])
        references = torch.tensor([[[-1.5], [1.5]], [[-2.0], [2.0]]])
        bandwidths = (0.5, 1.0, 2.0)
        combined, _ = Drifting(bandwidths, kernel="laplacian")(query, references, repulsion=0.0)
        individual = torch.stack(
            [
                Drifting(value, kernel="laplacian")(query, references, repulsion=0.0)[0]
                for value in bandwidths
            ]
        )
        expected = individual.mean(dim=0)
        self.assertTrue(torch.allclose(combined, expected))

    def test_unknown_kernel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Drifting(1.0, kernel="triangular")

    def test_multiple_bandwidths_average_their_raw_fields(self) -> None:
        query = torch.tensor([[[0.0]], [[0.5]]])
        references = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
        bandwidths = (0.5, 1.0, 2.0)
        combined_field, combined_metrics = Drifting(bandwidths)(query, references, repulsion=0.0)
        individual = [
            Drifting(bandwidth)(query, references, repulsion=0.0) for bandwidth in bandwidths
        ]
        individual_fields = torch.stack([field for field, _ in individual])
        individual_rms = individual_fields.square().mean(dim=(1, 2, 3)).sqrt()
        expected_field = individual_fields.mean(dim=0)
        expected_mass = torch.stack(
            [metrics["positive_kernel_mass"] for _, metrics in individual]
        ).mean()
        self.assertTrue(torch.allclose(combined_field, expected_field))
        self.assertTrue(torch.allclose(combined_metrics["positive_kernel_mass"], expected_mass))
        self.assertTrue(
            torch.allclose(combined_metrics["temperature_field_rms_mean"], individual_rms.mean())
        )

    def test_multiple_bandwidths_must_be_nonempty_and_positive(self) -> None:
        with self.assertRaises(ValueError):
            Drifting([])
        with self.assertRaises(ValueError):
            Drifting([0.5, 0.0])

    def test_zero_field_at_one_temperature_does_not_produce_nan(self) -> None:
        query = torch.tensor([[[0.0]]])
        references = torch.tensor([[[10.0]]])
        field, metrics = Drifting([1e-6, 10.0])(query, references, repulsion=0.0)
        self.assertTrue(torch.isfinite(field).all())
        self.assertTrue(torch.isfinite(metrics["drift_rms"]))

    def test_auto_bandwidth_uses_flattened_coordinate_distance(self) -> None:
        samples = torch.tensor([[[0.0]], [[3.0]], [[7.0]]])
        self.assertEqual(median_bandwidth(samples), 4.0)

    def test_field_is_not_kernel_mass_normalized(self) -> None:
        query = torch.tensor([[[-0.5, 0.0], [0.5, 0.0]]])
        near = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])
        far = torch.tensor([[[-5.0, 0.0], [5.0, 0.0]]])
        drift = Drifting(1.0)
        near_field, _ = drift(query, near, repulsion=0.0)
        far_field, _ = drift(query, far, repulsion=0.0)
        self.assertLess(far_field.norm(), near_field.norm())

    def test_attraction_expands_an_undersized_configuration(self) -> None:
        query = center(torch.tensor([[[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]]))
        reference = 3.0 * query
        field, _ = Drifting(2.0)(query, reference, repulsion=0.0)
        self.assertGreater(torch.sum(field * query), 0.0)

    def test_field_is_equivariant_when_query_and_references_transform_together(self) -> None:
        torch.manual_seed(7)
        query = center(torch.randn(2, 4, 2))
        references = center(torch.randn(5, 4, 2))
        permutation = torch.tensor([2, 0, 3, 1])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        drift = Drifting(2.0)
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
        drift = Drifting(2.0)
        ordered, _ = drift(query, references, repulsion=0.0)
        permuted, _ = drift(query, references[:, [1, 0]], repulsion=0.0)
        self.assertFalse(torch.allclose(ordered, permuted))

    def test_drift_preserves_center(self) -> None:
        query = center(torch.randn(3, 4, 2))
        references = center(torch.randn(5, 4, 2))
        field, _ = Drifting(2.0)(query, references, repulsion=0.0)
        self.assertTrue(torch.allclose(field.mean(dim=1), torch.zeros(3, 2), atol=1e-6))

    def test_step_applies_the_requested_field_scale(self) -> None:
        query = torch.tensor([[[0.0]]])
        reference = torch.tensor([[[1.0]]])
        drift = Drifting(2.0)
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
    def test_egnn_is_permutation_and_orthogonal_equivariant(self) -> None:
        models = {
            "block": lambda: EGNN(feature_dim=3, hidden_dim=16, layers=3, layers_per_block=2),
            "block_mean": lambda: EGNN(
                feature_dim=3, hidden_dim=16, layers=3, attention=False, aggregation="mean"
            ),
            "legacy": lambda: LegacyEGNN(
                feature_dim=3, hidden_dim=16, layers=3, radial_basis=6, variant="legacy"
            ),
            "bounded": lambda: LegacyEGNN(
                feature_dim=3, hidden_dim=16, layers=3, radial_basis=6, variant="bounded"
            ),
        }
        for variant, factory in models.items():
            with self.subTest(variant=variant):
                torch.manual_seed(4)
                model = factory().double()
                # Freshly initialized coordinate heads barely move particles, which
                # would make any map look equivariant; perturb every weight first.
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(0.3 * torch.randn_like(parameter))
                positions = center(torch.randn(2, 5, 3, dtype=torch.float64))
                features = torch.randn(2, 5, 3, dtype=torch.float64)
                orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
                if torch.det(orthogonal) > 0:
                    orthogonal[:, 0] *= -1
                permutation = torch.tensor([2, 0, 4, 3, 1])
                output = model(positions, features)
                self.assertGreater(float((output - positions).abs().max()), 1e-2)
                torch.testing.assert_close(output.mean(dim=1), torch.zeros(2, 3, dtype=torch.float64))
                transformed = model(
                    positions[:, permutation] @ orthogonal.T + 3.0, features[:, permutation]
                )
                torch.testing.assert_close(transformed, output[:, permutation] @ orthogonal.T)

    def test_block_egnn_follows_reference_layout(self) -> None:
        model = EGNN(feature_dim=3, hidden_dim=8, layers=2, layers_per_block=3)
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(len(model.blocks[0].type_update), 3)
        coordinate_output = model.blocks[0].coord_update.coord_mlp[-1]
        self.assertIsNone(coordinate_output.bias)
        self.assertLess(float(coordinate_output.weight.detach().abs().max()), 1e-2)
        self.assertEqual(model.blocks[0].type_update[0].message_mlp[0].in_features, 2 * 8 + 2)

    def test_block_egnn_bounds_each_coordinate_message(self) -> None:
        model = EGNN(feature_dim=3, hidden_dim=4, layers=1, coordinate_range=0.5)
        with torch.no_grad():
            model.blocks[0].coord_update.coord_mlp[-1].weight.fill_(1e3)
        positions = center(torch.randn(1, 4, 3) * 1e3)
        displacement = model(positions, torch.randn(1, 4, 3)) - positions
        # Summing three unit-bounded messages moves a particle at most 3 * 0.5;
        # recentering can add at most the same amount again.
        self.assertLessEqual(float(displacement.norm(dim=-1).max().detach()), 2 * 3 * 0.5)
        self.assertTrue(torch.isfinite(displacement).all())

    def test_block_egnn_gradients_are_finite_with_coincident_particles(self) -> None:
        model = EGNN(feature_dim=3, hidden_dim=4, layers=2)
        positions = torch.zeros(1, 3, 2, requires_grad=True)
        model(positions, torch.randn(1, 3, 3)).square().sum().backward()
        self.assertTrue(torch.isfinite(positions.grad).all())
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_bounded_egnn_coordinate_update_does_not_scale_with_distance(self) -> None:
        layer = LegacyEGNNLayer(
            hidden_dim=4,
            radial_basis=4,
            variant="bounded",
            coordinate_range=0.5,
        )
        with torch.no_grad():
            for parameter in layer.coordinate_mlp.parameters():
                parameter.zero_()
            layer.coordinate_mlp[-1].bias.fill_(100)
        positions = center(
            torch.tensor([[[1e6, 0.0], [-1e6, 0.0], [0.0, 1e6], [0.0, -1e6]]])
        )
        features = torch.zeros(1, 4, 4)
        difference = positions[:, :, None] - positions[:, None, :]
        initial_squared_distance = difference.square().sum(dim=-1, keepdim=True)

        _, updated = layer(features, positions, initial_squared_distance)

        displacement = updated - positions
        self.assertLessEqual(float(displacement.norm(dim=-1).max().detach()), 1.0 + 1e-6)

    def test_gnn_is_permutation_equivariant_but_not_rotation_equivariant(self) -> None:
        torch.manual_seed(4)
        model = GNN(
            dimensions=2,
            feature_dim=3,
            hidden_dim=16,
            layers=2,
            radial_basis=6,
        )
        positions = center(torch.randn(2, 4, 2))
        features = torch.randn(2, 4, 3)
        permutation = torch.tensor([2, 0, 3, 1])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        expected = model(positions, features)
        permuted = model(positions[:, permutation], features[:, permutation])
        torch.testing.assert_close(permuted, expected[:, permutation])
        rotated = model(positions @ rotation.T, features)
        equivariant_result = expected @ rotation.T
        self.assertGreater(float((rotated - equivariant_result).abs().max().detach()), 1e-6)
        torch.testing.assert_close(rotated.mean(dim=1), torch.zeros(2, 2), atol=1e-6, rtol=0)

    def test_model_architecture_selector(self) -> None:
        definition = {
            "feature_dim": 3,
            "hidden_dim": 8,
            "layers": 1,
            "radial_basis": 4,
            "max_distance": 8,
        }
        for architecture, expected_type in (("egnn", LegacyEGNN), ("gnn", GNN)):
            with self.subTest(architecture=architecture):
                config = {
                    "system": "dw4",
                    "model": {**definition, "architecture": architecture},
                }
                self.assertIsInstance(build_model(config), expected_type)
        self.assertIsInstance(build_model({"system": "dw4", "model": definition}), LegacyEGNN)
        bounded = build_model(
            {
                "system": "dw4",
                "model": {
                    **definition,
                    "architecture": "egnn",
                    "variant": "bounded",
                    "coordinate_range": 0.5,
                },
            }
        )
        self.assertIsInstance(bounded, LegacyEGNN)
        self.assertEqual(bounded.variant, "bounded")
        self.assertEqual(bounded.coordinate_range, 0.5)
        block = build_model(
            {
                "system": "dw4",
                "model": {
                    "architecture": "egnn",
                    "variant": "block",
                    "feature_dim": 3,
                    "hidden_dim": 8,
                    "layers": 2,
                    "aggregation": "mean",
                },
            }
        )
        self.assertIsInstance(block, EGNN)
        self.assertEqual(block.coordinate_range, 15.0)
        self.assertEqual(block.aggregation, "mean")
        with self.assertRaisesRegex(ValueError, "model.variant"):
            build_model(
                {"system": "dw4", "model": {**definition, "architecture": "egnn", "variant": "x"}}
            )
        with self.assertRaisesRegex(ValueError, "model.architecture"):
            build_model(
                {
                    "system": "dw4",
                    "model": {**definition, "architecture": "transformer"},
                }
            )


if __name__ == "__main__":
    unittest.main()
