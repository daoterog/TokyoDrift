"""Sample and evaluate a trained particle-system generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import build_model, device_summary, load_dataset, select_device
from .systems import center, get_system, pair_distances

DEFAULT_METRIC_SAMPLE_SIZE = 1_000_000
METRIC_SAMPLE_SEED = 17_903


def arguments() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a particle-system checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument(
        "--metric-sample-size",
        type=int,
        default=DEFAULT_METRIC_SAMPLE_SIZE,
        help=(
            "Maximum observations per distribution used by quantile and histogram "
            "estimators. Larger observable arrays are uniformly subsampled."
        ),
    )
    parser.add_argument(
        "--valid-energy-quantile",
        type=float,
        default=0.99,
        help="Reference-energy quantile defining the useful-energy cutoff.",
    )
    parser.add_argument(
        "--min-pair-distance",
        type=float,
        default=0.5,
        help="Minimum pair separation defining a collision-free configuration.",
    )
    return parser.parse_args()


def metric_observations(
    values: torch.Tensor, max_observations: int = DEFAULT_METRIC_SAMPLE_SIZE
) -> torch.Tensor:
    """Return a bounded, deterministic sample of finite scalar observations."""
    if max_observations <= 0:
        raise ValueError("metric sample size must be positive")
    values = values.detach().flatten().float().cpu()
    values = values[torch.isfinite(values)]
    if values.numel() <= max_observations:
        return values
    generator = torch.Generator().manual_seed(METRIC_SAMPLE_SEED)
    indices = torch.randint(values.numel(), (max_observations,), generator=generator)
    return values.index_select(0, indices)


def pair_distance_observations(
    positions: torch.Tensor, max_observations: int = DEFAULT_METRIC_SAMPLE_SIZE
) -> torch.Tensor:
    """Compute bounded pair-distance observations from uniformly sampled configurations."""
    if max_observations <= 0:
        raise ValueError("metric sample size must be positive")
    particles = positions.shape[-2]
    pairs_per_configuration = particles * (particles - 1) // 2
    maximum_configurations = max(1, max_observations // pairs_per_configuration)
    if len(positions) > maximum_configurations:
        generator = torch.Generator().manual_seed(METRIC_SAMPLE_SEED)
        indices = torch.randperm(len(positions), generator=generator)[:maximum_configurations]
        positions = positions.index_select(0, indices.to(positions.device))
    return metric_observations(pair_distances(positions), max_observations)


def minimum_pair_distances(
    positions: torch.Tensor, configuration_batch_size: int = 65_536
) -> torch.Tensor:
    """Compute each configuration's closest pair with bounded temporary storage."""
    if configuration_batch_size <= 0:
        raise ValueError("configuration batch size must be positive")
    return torch.cat(
        [
            pair_distances(positions[start : start + configuration_batch_size]).min(dim=-1).values
            for start in range(0, len(positions), configuration_batch_size)
        ]
    )


def wasserstein_1(
    left: torch.Tensor,
    right: torch.Tensor,
    points: int = 4096,
    max_observations: int = DEFAULT_METRIC_SAMPLE_SIZE,
) -> float | None:
    """Estimate scalar Wasserstein-1 distance from matched empirical quantiles."""
    left = metric_observations(left, max_observations)
    right = metric_observations(right, max_observations)
    if left.numel() == 0 or right.numel() == 0:
        return None
    quantiles = torch.linspace(0.0, 1.0, min(points, left.numel(), right.numel()))
    return float((torch.quantile(left, quantiles) - torch.quantile(right, quantiles)).abs().mean())


def histogram_js(
    generated: torch.Tensor,
    reference: torch.Tensor,
    bins: int = 200,
    max_observations: int = DEFAULT_METRIC_SAMPLE_SIZE,
) -> float | None:
    """Compare energies in a reference-defined histogram with explicit failure bins."""

    def bounded(values: torch.Tensor) -> torch.Tensor:
        values = values.detach().flatten().float().cpu()
        if values.numel() <= max_observations:
            return values
        generator = torch.Generator().manual_seed(METRIC_SAMPLE_SEED)
        indices = torch.randint(values.numel(), (max_observations,), generator=generator)
        return values.index_select(0, indices)

    generated, reference = bounded(generated), bounded(reference)
    finite_reference = reference[torch.isfinite(reference)]
    if generated.numel() == 0 or finite_reference.numel() == 0:
        return None
    low, high = torch.quantile(finite_reference, torch.tensor([0.005, 0.995])).tolist()
    if not high > low:
        return 0.0

    def counts(values: torch.Tensor) -> torch.Tensor:
        finite = values[torch.isfinite(values)]
        return torch.cat(
            (
                (finite < low).sum().view(1),
                torch.histc(finite, bins=bins, min=low, max=high),
                (finite > high).sum().view(1),
                (~torch.isfinite(values)).sum().view(1),
            )
        ).double()

    p = counts(generated).add(1e-12)
    q = counts(reference).add(1e-12)
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)
    return float(0.5 * ((p * (p / midpoint).log()).sum() + (q * (q / midpoint).log()).sum()))


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    system_name: str,
    count: int,
    batch_size: int,
    coordinate_scale: float,
    feature_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Generate samples and their RMS endpoint displacement per particle."""
    system = get_system(system_name)
    samples: list[torch.Tensor] = []
    squared_transport = 0.0
    produced = 0
    while produced < count:
        batch = min(batch_size, count - produced)
        noise = coordinate_scale * torch.randn(
            batch, system.particles, system.dimensions, device=device
        )
        noise = center(noise)
        features = torch.randn(batch, system.particles, feature_dim, device=device)
        generated = model(noise, features)
        squared_transport += float((generated - noise).square().sum(dim=(-1, -2)).sum())
        samples.append(generated.cpu())
        produced += batch
    # This is an endpoint transport distance, not the CNF arc length in the paper.
    endpoint_distance = (squared_transport / (count * system.particles)) ** 0.5
    return torch.cat(samples), endpoint_distance


def distribution_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
    system_name: str,
    metric_sample_size: int = DEFAULT_METRIC_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Compare generated and held-out samples through invariant observables."""
    system = get_system(system_name)
    generated_energy = system.energy(generated)
    reference_energy = system.energy(reference)
    generated_distances = pair_distance_observations(generated, metric_sample_size)
    reference_distances = pair_distance_observations(reference, metric_sample_size)
    generated_radius = generated.square().sum(dim=-1).mean(dim=-1).sqrt()
    reference_radius = reference.square().sum(dim=-1).mean(dim=-1).sqrt()
    finite_generated_energy = generated_energy[torch.isfinite(generated_energy)]
    reference_q99, reference_q999 = torch.quantile(reference_energy, torch.tensor((0.99, 0.999)))
    return {
        "finite_energy_fraction": float(torch.isfinite(generated_energy).float().mean()),
        "energy_mean": float(finite_generated_energy.mean())
        if finite_generated_energy.numel()
        else None,
        "energy_std": float(finite_generated_energy.std())
        if finite_generated_energy.numel() > 1
        else None,
        "energy_wasserstein_1": wasserstein_1(
            generated_energy, reference_energy, max_observations=metric_sample_size
        ),
        "energy_histogram_js": histogram_js(
            generated_energy, reference_energy, max_observations=metric_sample_size
        ),
        "pair_distance_wasserstein_1": wasserstein_1(
            generated_distances, reference_distances, max_observations=metric_sample_size
        ),
        "radius_wasserstein_1": wasserstein_1(
            generated_radius, reference_radius, max_observations=metric_sample_size
        ),
        "collision_fraction_d_lt_0_5": float(
            (minimum_pair_distances(generated) < 0.5).float().mean()
        ),
        "energy_tail": {
            "reference_q99": float(reference_q99),
            "generated_mass_above_reference_q99": float(
                (generated_energy > reference_q99).float().mean()
            ),
            "reference_q999": float(reference_q999),
            "generated_mass_above_reference_q999": float(
                (generated_energy > reference_q999).float().mean()
            ),
        },
        "reference": {
            "energy_mean": float(reference_energy.mean()),
            "energy_std": float(reference_energy.std()),
        },
    }


def validity_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
    system_name: str,
    energy_quantile: float,
    minimum_pair_distance: float,
) -> dict[str, Any]:
    """Report broad sample usefulness without requiring distribution matching.

    A sample is useful when it is finite, collision-free, and below a declared
    high-energy reference cutoff.  The reference only sets that broad cutoff;
    no density or per-bin matching is required.
    """
    if not 0.0 < energy_quantile < 1.0:
        raise ValueError("valid energy quantile must lie strictly between zero and one")
    if minimum_pair_distance <= 0.0:
        raise ValueError("minimum pair distance must be positive")
    system = get_system(system_name)
    reference_energy = system.energy(reference)
    energy_cutoff = torch.quantile(
        reference_energy[torch.isfinite(reference_energy)], energy_quantile
    )

    def classify(samples: torch.Tensor) -> dict[str, float]:
        energy = system.energy(samples)
        finite = torch.isfinite(energy)
        collision_free = minimum_pair_distances(samples) >= minimum_pair_distance
        useful_energy = finite & (energy <= energy_cutoff)
        valid = collision_free & useful_energy
        return {
            "finite_fraction": float(finite.float().mean()),
            "collision_free_fraction": float(collision_free.float().mean()),
            "energy_under_cutoff_fraction": float(useful_energy.float().mean()),
            "valid_fraction": float(valid.float().mean()),
        }

    return {
        "definition": {
            "minimum_pair_distance": minimum_pair_distance,
            "energy_cutoff_reference_quantile": energy_quantile,
            "energy_cutoff": float(energy_cutoff),
        },
        "generated": classify(generated),
        "reference": classify(reference),
    }


def valid_sample_observables(
    generated: torch.Tensor,
    reference: torch.Tensor,
    system_name: str,
    energy_quantile: float,
    minimum_pair_distance: float,
    metric_sample_size: int = DEFAULT_METRIC_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Compare useful retained samples, rather than demanding raw density equality.

    Both populations are conditioned on the identical collision and energy rule.
    This measures whether the generator is useful as a fast proposal source after
    a transparent quality filter; it is not a replacement for a Boltzmann test.
    """
    system = get_system(system_name)
    reference_energy = system.energy(reference)
    cutoff = torch.quantile(reference_energy[torch.isfinite(reference_energy)], energy_quantile)

    def mask(samples: torch.Tensor) -> torch.Tensor:
        energy = system.energy(samples)
        return (
            torch.isfinite(energy)
            & (energy <= cutoff)
            & (minimum_pair_distances(samples) >= minimum_pair_distance)
        )

    generated_mask, reference_mask = mask(generated), mask(reference)
    retained_generated, retained_reference = generated[generated_mask], reference[reference_mask]
    retained_counts = {
        "generated": len(retained_generated),
        "reference": len(retained_reference),
    }
    retained_fractions = {
        "generated_retained_fraction": float(generated_mask.float().mean()),
        "reference_retained_fraction": float(reference_mask.float().mean()),
    }
    if not len(retained_generated) or not len(retained_reference):
        empty_populations = [name for name, count in retained_counts.items() if count == 0]
        return {
            "definition": {
                "minimum_pair_distance": minimum_pair_distance,
                "energy_cutoff_reference_quantile": energy_quantile,
                "energy_cutoff": float(cutoff),
            },
            **retained_fractions,
            "retained_counts": retained_counts,
            "comparison_available": False,
            "unavailable_reason": (
                "No valid samples were retained for: " + ", ".join(empty_populations)
            ),
            "energy_mean_generated": None,
            "energy_mean_reference": None,
            "energy_wasserstein_1": None,
            "energy_histogram_js": None,
            "pair_distance_wasserstein_1": None,
            "radius_wasserstein_1": None,
        }
    generated_energy, reference_energy = (
        system.energy(retained_generated),
        system.energy(retained_reference),
    )
    generated_distance = pair_distance_observations(retained_generated, metric_sample_size)
    reference_distance = pair_distance_observations(retained_reference, metric_sample_size)
    generated_radius = retained_generated.square().sum(dim=-1).mean(dim=-1).sqrt()
    reference_radius = retained_reference.square().sum(dim=-1).mean(dim=-1).sqrt()
    return {
        "definition": {
            "minimum_pair_distance": minimum_pair_distance,
            "energy_cutoff_reference_quantile": energy_quantile,
            "energy_cutoff": float(cutoff),
        },
        **retained_fractions,
        "retained_counts": retained_counts,
        "comparison_available": True,
        "unavailable_reason": None,
        "energy_mean_generated": float(generated_energy.mean()),
        "energy_mean_reference": float(reference_energy.mean()),
        "energy_wasserstein_1": wasserstein_1(
            generated_energy, reference_energy, max_observations=metric_sample_size
        ),
        "energy_histogram_js": histogram_js(
            generated_energy, reference_energy, max_observations=metric_sample_size
        ),
        "pair_distance_wasserstein_1": wasserstein_1(
            generated_distance, reference_distance, max_observations=metric_sample_size
        ),
        "radius_wasserstein_1": wasserstein_1(
            generated_radius, reference_radius, max_observations=metric_sample_size
        ),
    }


def plot_histograms(
    generated: torch.Tensor,
    reference: torch.Tensor,
    system_name: str,
    output: Path,
    metric_sample_size: int = DEFAULT_METRIC_SAMPLE_SIZE,
) -> None:
    """Write energy and pair-distance marginal histograms."""
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "particle-drift-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    system = get_system(system_name)
    generated_energy = system.energy(generated).numpy()
    reference_energy = system.energy(reference).numpy()
    finite = np.concatenate((generated_energy[np.isfinite(generated_energy)], reference_energy))
    low, high = np.quantile(finite, [0.005, 0.995])
    generated_distance = pair_distance_observations(generated, metric_sample_size).numpy()
    reference_distance = pair_distance_observations(reference, metric_sample_size).numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        reference_energy, bins=150, range=(low, high), density=True, alpha=0.6, label="test"
    )
    axes[0].hist(
        generated_energy, bins=150, range=(low, high), density=True, alpha=0.6, label="generated"
    )
    axes[0].set(
        xlabel="dimensionless energy", ylabel="density", title=f"{system.name.upper()} energy"
    )
    axes[0].legend()
    distance_high = np.quantile(np.concatenate((generated_distance, reference_distance)), 0.995)
    axes[1].hist(
        reference_distance,
        bins=150,
        range=(0, distance_high),
        density=True,
        alpha=0.6,
        label="test",
    )
    axes[1].hist(
        generated_distance,
        bins=150,
        range=(0, distance_high),
        density=True,
        alpha=0.6,
        label="generated",
    )
    axes[1].set(xlabel="pair distance", ylabel="density", title="Pair-distance marginal")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_energy_distributions(
    generated: torch.Tensor,
    reference: torch.Tensor,
    system_name: str,
    output: Path,
    energy_quantile: float,
    minimum_pair_distance: float,
    metric_sample_size: int = DEFAULT_METRIC_SAMPLE_SIZE,
) -> None:
    """Plot raw and validity-conditioned energies without hiding extreme tails."""
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "particle-drift-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    system = get_system(system_name)
    generated_energy = system.energy(generated).detach().cpu()
    reference_energy = system.energy(reference).detach().cpu()
    finite_reference = reference_energy[torch.isfinite(reference_energy)]
    if not len(finite_reference):
        raise ValueError("reference energies contain no finite values")
    cutoff = torch.quantile(finite_reference, energy_quantile)

    def valid_mask(samples: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        return (
            torch.isfinite(energy)
            & (energy <= cutoff)
            & (minimum_pair_distances(samples).cpu() >= minimum_pair_distance)
        )

    generated_valid = valid_mask(generated, generated_energy)
    reference_valid = valid_mask(reference, reference_energy)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "energy_distributions.npz",
        generated=generated_energy.numpy(),
        reference=reference_energy.numpy(),
        generated_valid=generated_valid.numpy(),
        reference_valid=reference_valid.numpy(),
        energy_cutoff=float(cutoff),
        minimum_pair_distance=minimum_pair_distance,
    )

    low, high = float(finite_reference.min()), float(finite_reference.max())
    if not high > low:
        low, high = low - 1.0, high + 1.0
    else:
        padding = 0.02 * (high - low)
        low, high = low - padding, high + padding

    def probability_histogram(values: torch.Tensor, label: str, color: str) -> None:
        finite_values = values[torch.isfinite(values)].numpy()
        axes[0].hist(
            finite_values,
            bins=120,
            range=(low, high),
            weights=np.full(len(finite_values), 1.0 / max(len(values), 1)),
            histtype="step",
            linewidth=1.7,
            label=label,
            color=color,
        )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    probability_histogram(reference_energy, "test", "C0")
    probability_histogram(generated_energy, "generated", "C1")
    axes[0].set_yscale("log")
    axes[0].axvline(float(cutoff), color="black", linestyle="--", label="test q99 cutoff")
    axes[0].set(
        xlabel="dimensionless energy",
        ylabel="probability mass per bin (all samples)",
        title=f"{system.name.upper()} all samples: reference-energy window",
        xlim=(low, high),
    )
    axes[0].legend()
    generated_in_window = ((generated_energy >= low) & (generated_energy <= high)).float().mean()
    axes[0].text(
        0.02,
        0.02,
        f"Generated mass in shown window: {float(generated_in_window):.4%}",
        transform=axes[0].transAxes,
        fontsize=9,
    )

    for values, label, color in (
        (reference_energy, "test", "C0"),
        (generated_energy, "generated", "C1"),
    ):
        observations = metric_observations(values, metric_sample_size).sort().values.numpy()
        cumulative = np.arange(1, len(observations) + 1) / len(observations)
        axes[1].plot(observations, cumulative, label=label, color=color)
    axes[1].set_xscale("symlog", linthresh=max(1.0, (high - low) / 20))
    axes[1].axvline(float(cutoff), color="black", linestyle="--", label="test q99 cutoff")
    axes[1].set(
        xlabel="dimensionless energy (symmetric-log scale)",
        ylabel="empirical CDF",
        title="All finite energies: full range",
        ylim=(0, 1.01),
    )
    axes[1].legend()
    figure.savefig(output / "energy_all_samples.png", dpi=180)
    plt.close(figure)

    retained_generated = generated_energy[generated_valid]
    retained_reference = reference_energy[reference_valid]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    if len(retained_generated) and len(retained_reference):
        combined = torch.cat((retained_generated, retained_reference))
        valid_low, valid_high = float(combined.min()), float(combined.max())
        if not valid_high > valid_low:
            valid_low, valid_high = valid_low - 1.0, valid_high + 1.0
        else:
            padding = 0.02 * (valid_high - valid_low)
            valid_low, valid_high = valid_low - padding, valid_high + padding
        for values, label, color in (
            (retained_reference, "test valid", "C0"),
            (retained_generated, "generated valid", "C1"),
        ):
            axes[0].hist(
                values.numpy(),
                bins=100,
                range=(valid_low, valid_high),
                density=True,
                histtype="step",
                linewidth=1.7,
                label=label,
                color=color,
            )
            observations = metric_observations(values, metric_sample_size).sort().values.numpy()
            cumulative = np.arange(1, len(observations) + 1) / len(observations)
            axes[1].plot(observations, cumulative, label=label, color=color)
        axes[0].set_xlim(valid_low, valid_high)
        axes[0].legend()
        axes[1].legend()
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No valid generated samples", ha="center", va="center")
    generated_fraction = float(generated_valid.float().mean())
    reference_fraction = float(reference_valid.float().mean())
    axes[0].set(
        xlabel="dimensionless energy",
        ylabel="conditional density",
        title=(
            f"Valid samples only: generated {len(retained_generated):,}/{len(generated):,} "
            f"({generated_fraction:.4%})"
        ),
    )
    axes[1].set(
        xlabel="dimensionless energy",
        ylabel="empirical CDF",
        title=(
            f"Test retained {len(retained_reference):,}/{len(reference):,} "
            f"({reference_fraction:.2%})"
        ),
        ylim=(0, 1.01),
    )
    figure.suptitle(
        f"Validity: minimum distance ≥ {minimum_pair_distance:g}, energy ≤ {float(cutoff):.3f}"
    )
    figure.savefig(output / "energy_valid_samples.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Run checkpoint sampling and evaluation."""
    args = arguments()
    if args.metric_sample_size <= 0:
        raise ValueError("metric sample size must be positive")
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    system = get_system(config["system"])
    if system.name == "aldp":
        raise ValueError("use python -m particle_systems.evaluate_alanine for molecular metrics")
    reference, metadata = load_dataset(Path(config["data"]), "test")
    if metadata["system"] != system.name:
        raise ValueError("checkpoint and test dataset systems differ")
    model = build_model(config).to(device).eval()
    ema_state = checkpoint.get("ema")
    model.load_state_dict(ema_state if ema_state is not None else checkpoint["model"])
    evaluated_weights = "ema" if ema_state is not None else "model"
    generated, endpoint_distance = generate(
        model,
        system.name,
        args.num_samples,
        args.batch_size,
        float(config["training"]["coordinate_noise_scale"]),
        int(config["model"]["feature_dim"]),
        device,
    )
    metrics = {
        "system": system.name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_epoch": (int(checkpoint["epoch"]) if "epoch" in checkpoint else None),
        "evaluated_weights": evaluated_weights,
        "num_generated_samples": args.num_samples,
        "num_test_samples": len(reference),
        "metric_estimation": {
            "quantile_points": 4096,
            "maximum_observations_per_distribution": args.metric_sample_size,
            "large_scalar_observation_sampling": "deterministic uniform with replacement",
            "pair_distance_sampling": (
                "deterministic uniform configuration sampling without replacement"
            ),
            "sampling_seed": METRIC_SAMPLE_SEED,
            "energy_histogram": (
                "reference q0.005-q0.995 interior with underflow, overflow, and nonfinite bins"
            ),
        },
        "runtime": device_summary(device),
        "sample_metrics": distribution_metrics(
            generated,
            reference,
            system.name,
            metric_sample_size=args.metric_sample_size,
        ),
        "validity_metrics": validity_metrics(
            generated,
            reference,
            system.name,
            args.valid_energy_quantile,
            args.min_pair_distance,
        ),
        "valid_sample_observables": valid_sample_observables(
            generated,
            reference,
            system.name,
            args.valid_energy_quantile,
            args.min_pair_distance,
            metric_sample_size=args.metric_sample_size,
        ),
        "endpoint_transport_distance_per_particle": endpoint_distance,
        "paper_metrics": {
            "reference_results": system.paper_reference,
            "unnormalized_drift": {
                "nll": None,
                "ess_percent": None,
                "path_length": None,
                "reason": (
                    "The direct unnormalized-drift generator is not an invertible continuous "
                    "normalizing flow, so it has no tractable proposal density or CNF path."
                ),
            },
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
    plot_histograms(
        generated,
        reference,
        system.name,
        args.output / "distributions.png",
        metric_sample_size=args.metric_sample_size,
    )
    plot_energy_distributions(
        generated,
        reference,
        system.name,
        args.output,
        args.valid_energy_quantile,
        args.min_pair_distance,
        metric_sample_size=args.metric_sample_size,
    )
    if args.save_samples:
        np.savez_compressed(args.output / "samples.npz", positions=generated.numpy())
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
