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
    return parser.parse_args()


def wasserstein_1(left: torch.Tensor, right: torch.Tensor, points: int = 4096) -> float | None:
    """Estimate scalar Wasserstein-1 distance from matched empirical quantiles."""
    left = left[torch.isfinite(left)].float().cpu()
    right = right[torch.isfinite(right)].float().cpu()
    if left.numel() == 0 or right.numel() == 0:
        return None
    quantiles = torch.linspace(0.0, 1.0, min(points, left.numel(), right.numel()))
    return float((torch.quantile(left, quantiles) - torch.quantile(right, quantiles)).abs().mean())


def histogram_js(left: torch.Tensor, right: torch.Tensor, bins: int = 200) -> float | None:
    """Compute Jensen-Shannon divergence between robust-range histograms."""
    left = left[torch.isfinite(left)].float().cpu()
    right = right[torch.isfinite(right)].float().cpu()
    if left.numel() == 0 or right.numel() == 0:
        return None
    combined = torch.cat((left, right))
    low, high = torch.quantile(combined, torch.tensor([0.005, 0.995])).tolist()
    if not high > low:
        return 0.0
    p = torch.histc(left, bins=bins, min=low, max=high).double().add(1e-12)
    q = torch.histc(right, bins=bins, min=low, max=high).double().add(1e-12)
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
    generated: torch.Tensor, reference: torch.Tensor, system_name: str
) -> dict[str, Any]:
    """Compare generated and held-out samples through invariant observables."""
    system = get_system(system_name)
    generated_energy = system.energy(generated)
    reference_energy = system.energy(reference)
    generated_distances = pair_distances(generated).flatten()
    reference_distances = pair_distances(reference).flatten()
    generated_radius = generated.square().sum(dim=-1).mean(dim=-1).sqrt()
    reference_radius = reference.square().sum(dim=-1).mean(dim=-1).sqrt()
    finite_generated_energy = generated_energy[torch.isfinite(generated_energy)]
    return {
        "finite_energy_fraction": float(torch.isfinite(generated_energy).float().mean()),
        "energy_mean": float(finite_generated_energy.mean())
        if finite_generated_energy.numel()
        else None,
        "energy_std": float(finite_generated_energy.std())
        if finite_generated_energy.numel() > 1
        else None,
        "energy_wasserstein_1": wasserstein_1(generated_energy, reference_energy),
        "energy_histogram_js": histogram_js(generated_energy, reference_energy),
        "pair_distance_wasserstein_1": wasserstein_1(generated_distances, reference_distances),
        "radius_wasserstein_1": wasserstein_1(generated_radius, reference_radius),
        "collision_fraction_d_lt_0_5": float(
            (pair_distances(generated).min(dim=-1).values < 0.5).float().mean()
        ),
        "reference": {
            "energy_mean": float(reference_energy.mean()),
            "energy_std": float(reference_energy.std()),
        },
    }


def plot_histograms(
    generated: torch.Tensor,
    reference: torch.Tensor,
    system_name: str,
    output: Path,
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
    generated_distance = pair_distances(generated).flatten().numpy()
    reference_distance = pair_distances(reference).flatten().numpy()
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


def main() -> None:
    """Run checkpoint sampling and evaluation."""
    args = arguments()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    system = get_system(config["system"])
    reference, metadata = load_dataset(Path(config["data"]), "test")
    if metadata["system"] != system.name:
        raise ValueError("checkpoint and test dataset systems differ")
    model = build_model(config).to(device).eval()
    model.load_state_dict(checkpoint["ema"])
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
        "num_generated_samples": args.num_samples,
        "num_test_samples": len(reference),
        "runtime": device_summary(device),
        "sample_metrics": distribution_metrics(generated, reference, system.name),
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
    plot_histograms(generated, reference, system.name, args.output / "distributions.png")
    if args.save_samples:
        np.savez_compressed(args.output / "samples.npz", positions=generated.numpy())
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
