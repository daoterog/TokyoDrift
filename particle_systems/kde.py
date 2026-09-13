"""Sample-based, symmetry-invariant KDE likelihood estimates for particle systems."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import torch

from .descriptors import particle_descriptors

KDE_SEED = 41_927


@torch.no_grad()
def gaussian_kde_log_prob(
    queries: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float,
    batch_size: int = 256,
    density_dimension: int | None = None,
) -> torch.Tensor:
    """Evaluate a normalized radial Gaussian KDE in bounded query batches.

    ``density_dimension`` may be smaller than the feature dimension when the
    features embed a lower-dimensional manifold, as pair distances do. The
    default retains the usual Euclidean KDE normalization.
    """
    if queries.ndim != 2 or centers.ndim != 2 or queries.shape[1] != centers.shape[1]:
        raise ValueError("KDE queries and centers must be [samples, matching features]")
    if not len(centers):
        raise ValueError("KDE requires at least one center")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("KDE bandwidth must be finite and positive")
    if batch_size <= 0:
        raise ValueError("KDE batch size must be positive")
    if not torch.isfinite(queries).all() or not torch.isfinite(centers).all():
        raise ValueError("KDE inputs must be finite")
    feature_dimension = queries.shape[1]
    dimension = feature_dimension if density_dimension is None else density_dimension
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or not 1 <= dimension <= feature_dimension
    ):
        raise ValueError("KDE density dimension must be between one and the feature dimension")
    log_normalizer = (
        math.log(len(centers))
        + dimension * math.log(bandwidth)
        + 0.5 * dimension * math.log(2 * math.pi)
    )
    values = []
    for start in range(0, len(queries), batch_size):
        squared_distance = torch.cdist(queries[start : start + batch_size], centers).square()
        values.append(torch.logsumexp(-squared_distance / (2 * bandwidth**2), dim=1))
    return torch.cat(values) - log_normalizer


def _configuration_manifold_dimension(particles: int, dimensions: int) -> int:
    """Dimension remaining after translations and observable rotations are removed."""
    centered_dimension = (particles - 1) * dimensions
    rank = min(particles - 1, dimensions)
    rotation_dimension = dimensions * (dimensions - 1) // 2
    stabilizer_dimension = (dimensions - rank) * (dimensions - rank - 1) // 2
    return centered_dimension - (rotation_dimension - stabilizer_dimension)


@torch.no_grad()
def _mean_nll_by_bandwidth(
    queries: torch.Tensor,
    centers: torch.Tensor,
    bandwidths: Sequence[float],
    density_dimension: int,
    batch_size: int,
) -> dict[float, float]:
    """Score several bandwidths while reusing each expensive distance matrix."""
    totals = torch.zeros(len(bandwidths), dtype=torch.float64, device=queries.device)
    constant = math.log(len(centers)) + 0.5 * density_dimension * math.log(2 * math.pi)
    for start in range(0, len(queries), batch_size):
        squared_distance = torch.cdist(queries[start : start + batch_size], centers).square()
        for index, bandwidth in enumerate(bandwidths):
            log_probability = torch.logsumexp(-squared_distance / (2 * bandwidth**2), dim=1) - (
                constant + density_dimension * math.log(bandwidth)
            )
            totals[index] -= log_probability.double().sum()
    totals = totals.cpu() / len(queries)
    return {bandwidth: float(totals[index]) for index, bandwidth in enumerate(bandwidths)}


def _automatic_bandwidths(features: torch.Tensor) -> tuple[tuple[float, ...], float]:
    """Build a broad half-octave grid from a robust descriptor scale."""
    calibration = features[: min(1024, len(features))]
    distances = torch.pdist(calibration)
    distances = distances[distances > 0]
    if not len(distances):
        raise ValueError("cannot calibrate KDE bandwidth from identical centers")
    median_distance = float(distances.median())
    return tuple(median_distance * 2 ** (step / 2) for step in range(-16, 3)), median_distance


@torch.no_grad()
def _select_bandwidth(
    tuning_queries: torch.Tensor,
    centers: torch.Tensor,
    density_dimension: int,
    batch_size: int,
    candidates: Sequence[float] | None,
    source_name: str,
) -> dict[str, Any]:
    """Cross-validate a bandwidth and refine an interior grid minimum in log space."""
    if candidates is None:
        bandwidths, median_distance = _automatic_bandwidths(centers)
        source = f"{source_name} median descriptor distance grid"
    else:
        bandwidths = tuple(float(value) for value in candidates)
        if not bandwidths or any(not math.isfinite(value) or value <= 0 for value in bandwidths):
            raise ValueError("KDE bandwidth candidates must be finite and positive")
        if len(set(bandwidths)) != len(bandwidths):
            raise ValueError("KDE bandwidth candidates must be unique")
        bandwidths = tuple(sorted(bandwidths))
        median_distance = None
        source = "explicit candidates"

    scores = _mean_nll_by_bandwidth(
        tuning_queries, centers, bandwidths, density_dimension, batch_size
    )
    selected = min(scores, key=scores.get)

    # The automatic candidates are evenly spaced in log bandwidth. A local
    # quadratic gives substantially finer selection for just one extra score.
    selected_index = bandwidths.index(selected)
    refined = None
    if candidates is None and 0 < selected_index < len(bandwidths) - 1:
        left, middle, right = (
            scores[bandwidths[selected_index - 1]],
            scores[selected],
            scores[bandwidths[selected_index + 1]],
        )
        curvature = left - 2 * middle + right
        if curvature > 0:
            offset = 0.5 * (left - right) / curvature
            offset = max(-1.0, min(1.0, offset))
            log_step = math.log(bandwidths[selected_index + 1] / selected)
            refined = selected * math.exp(offset * log_step)
            refined_score = _mean_nll_by_bandwidth(
                tuning_queries,
                centers,
                (refined,),
                density_dimension,
                batch_size,
            )[refined]
            scores[refined] = refined_score
            selected = min(scores, key=scores.get)

    return {
        "selected": selected,
        "selection": "minimum held-out same-distribution tuning NLL",
        "source": source,
        "median_descriptor_distance": median_distance,
        "coarse_candidates": list(bandwidths),
        "quadratic_refinement": refined,
        "tuning_nll": {str(value): score for value, score in sorted(scores.items())},
        "minimum_on_grid_boundary": selected_index in {0, len(bandwidths) - 1},
    }


def _indices(count: int, maximum: int, seed: int) -> torch.Tensor:
    """Select configurations deterministically without replacement."""
    if maximum <= 0 or maximum > count:
        raise ValueError(f"cannot select {maximum} configurations from {count}")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(count, generator=generator)[:maximum]


def _nll_summary(log_probability: torch.Tensor) -> dict[str, float]:
    """Summarize per-configuration negative log likelihoods in nats."""
    values = -log_probability.double().cpu()
    return {
        "mean": float(values.mean()),
        "standard_error": (
            float(values.std(unbiased=True) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        ),
        "median": float(values.median()),
        "q05": float(torch.quantile(values, 0.05)),
        "q95": float(torch.quantile(values, 0.95)),
    }


def raw_distance_kde_report(scaled_report: dict[str, Any]) -> dict[str, Any]:
    """Express a scaled pair-distance KDE report in raw-distance units.

    Particle descriptors divide every pair distance by ``sqrt(pair_count)``.
    Because this is a global linear rescaling, the equivalent raw-distance KDE
    can be obtained exactly by scaling bandwidths and applying the manifold
    change-of-units Jacobian; no second KDE fit is necessary.
    """
    if not scaled_report.get("available"):
        return deepcopy(scaled_report)
    if scaled_report.get("space") != "sorted pair distances divided by sqrt(number of pairs)":
        raise ValueError("expected a scaled pair-distance KDE report")

    report = deepcopy(scaled_report)
    pair_count = int(report["descriptor_dimension"])
    density_dimension = int(report["density_dimension"])
    distance_scale = math.sqrt(pair_count)
    nll_shift = density_dimension * math.log(distance_scale)

    report["space"] = "sorted raw pair distances"
    report["units"] = "nats per configuration on the raw pair-distance manifold"
    report["change_of_units"] = {
        "raw_distance_multiplier": distance_scale,
        "nll_additive_shift": nll_shift,
        "excess_nll_additive_shift": 0.0,
    }
    for density_name in ("generated_kde", "reference_kde_baseline"):
        summary = report[density_name]
        for statistic in ("mean", "median", "q05", "q95"):
            summary[statistic] += nll_shift

        bandwidth = report["bandwidth"][density_name]
        bandwidth["selected"] *= distance_scale
        if bandwidth["median_descriptor_distance"] is not None:
            bandwidth["median_descriptor_distance"] *= distance_scale
        bandwidth["coarse_candidates"] = [
            value * distance_scale for value in bandwidth["coarse_candidates"]
        ]
        if bandwidth["quadratic_refinement"] is not None:
            bandwidth["quadratic_refinement"] *= distance_scale
        bandwidth["tuning_nll"] = {
            str(float(value) * distance_scale): score + nll_shift
            for value, score in bandwidth["tuning_nll"].items()
        }

    report["interpretation"] = (
        "The same intrinsic-manifold KDE expressed using unscaled sorted pair distances. "
        "Its absolute NLL differs from the scaled-distance report only by the stated "
        "change-of-units Jacobian; excess NLL is invariant."
    )
    return report


@torch.no_grad()
def descriptor_kde_nll(
    generated: torch.Tensor,
    training_reference: torch.Tensor,
    test_reference: torch.Tensor,
    *,
    center_count: int = 10_000,
    tuning_query_count: int = 2_000,
    test_query_count: int = 10_000,
    query_batch_size: int = 256,
    bandwidth_candidates: Sequence[float] | None = None,
    device: torch.device | str = "cpu",
    seed: int = KDE_SEED,
) -> dict[str, Any]:
    """Estimate held-out descriptor NLL under generated and reference KDEs."""
    if center_count <= 0 or tuning_query_count <= 0 or test_query_count <= 0:
        raise ValueError("KDE sample counts must be positive")
    if query_batch_size <= 0:
        raise ValueError("KDE query batch size must be positive")
    if (
        generated.ndim != 3
        or training_reference.ndim != 3
        or training_reference.shape[1:] != generated.shape[1:]
    ):
        raise ValueError("generated and training configurations must have matching particle shapes")
    if test_reference.ndim != 3 or test_reference.shape[1:] != generated.shape[1:]:
        raise ValueError("generated and test configurations must have matching particle shapes")
    if center_count + tuning_query_count > len(training_reference):
        raise ValueError("reference KDE centers and tuning queries must be disjoint")
    if center_count + tuning_query_count > len(generated):
        raise ValueError("generated KDE centers and tuning queries must be disjoint")
    if test_query_count > len(test_reference):
        raise ValueError("KDE sample request exceeds an available population")

    reference_indices = _indices(len(training_reference), center_count + tuning_query_count, seed)
    generated_indices = _indices(len(generated), center_count + tuning_query_count, seed + 1)
    test_indices = _indices(len(test_reference), test_query_count, seed + 2)
    reference_centers = training_reference[reference_indices[:center_count]]
    reference_tuning_queries = training_reference[reference_indices[center_count:]]
    generated_centers = generated[generated_indices[:center_count]]
    generated_tuning_queries = generated[generated_indices[center_count:]]
    test_queries = test_reference[test_indices]

    compute_device = torch.device(device)

    def descriptors(values: torch.Tensor) -> torch.Tensor:
        return particle_descriptors(values).to(compute_device)

    reference_features = descriptors(reference_centers)
    reference_tuning_features = descriptors(reference_tuning_queries)
    generated_features = descriptors(generated_centers)
    generated_tuning_features = descriptors(generated_tuning_queries)
    test_features = descriptors(test_queries)
    descriptor_dimension = reference_features.shape[1]
    density_dimension = _configuration_manifold_dimension(generated.shape[1], generated.shape[2])
    generated_bandwidth = _select_bandwidth(
        generated_tuning_features,
        generated_features,
        density_dimension,
        query_batch_size,
        bandwidth_candidates,
        "generated-center",
    )
    reference_bandwidth = _select_bandwidth(
        reference_tuning_features,
        reference_features,
        density_dimension,
        query_batch_size,
        bandwidth_candidates,
        "training-reference",
    )
    generated_log_probability = gaussian_kde_log_prob(
        test_features,
        generated_features,
        generated_bandwidth["selected"],
        query_batch_size,
        density_dimension,
    )
    reference_log_probability = gaussian_kde_log_prob(
        test_features,
        reference_features,
        reference_bandwidth["selected"],
        query_batch_size,
        density_dimension,
    )
    paired_excess = (reference_log_probability - generated_log_probability).double().cpu()
    return {
        "available": True,
        "space": "sorted pair distances divided by sqrt(number of pairs)",
        "descriptor_dimension": descriptor_dimension,
        "density_dimension": density_dimension,
        "density_dimension_basis": (
            "generic centered configuration dimension after quotienting observable rotations"
        ),
        "units": "nats per configuration on the scaled descriptor manifold",
        "generated_kde": _nll_summary(generated_log_probability),
        "reference_kde_baseline": _nll_summary(reference_log_probability),
        "excess_nll_generated_minus_reference": float(paired_excess.mean()),
        "excess_nll_standard_error": (
            float(paired_excess.std(unbiased=True) / math.sqrt(len(paired_excess)))
            if len(paired_excess) > 1
            else 0.0
        ),
        "bandwidth": {
            "generated_kde": generated_bandwidth,
            "reference_kde_baseline": reference_bandwidth,
        },
        "sample_counts": {
            "generated_centers": center_count,
            "generated_tuning_queries": tuning_query_count,
            "training_reference_centers": center_count,
            "training_tuning_queries": tuning_query_count,
            "test_queries": test_query_count,
        },
        "seed": seed,
        "interpretation": (
            "A symmetry-invariant smoothed density estimate on the intrinsic pair-distance "
            "manifold, not the exact coordinate-space model NLL or the flow-paper NLL. "
            "Generated and reference KDE bandwidths are independently cross-validated; "
            "lower excess NLL is better and zero matches the reference KDE baseline."
        ),
    }
