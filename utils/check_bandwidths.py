"""Measure descriptor kernel coverage and coordinate fields before training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from data.systems import get_system

from .descriptors import DescriptorDrift, particle_descriptors
from .io import build_model, load_config, load_dataset


def quantiles(values: torch.Tensor) -> dict[str, float]:
    """Summarize a small diagnostic tensor."""
    levels = (0.05, 0.5, 0.95)
    results = torch.quantile(values.flatten(), values.new_tensor(levels))
    return {
        f"q{round(level * 100):02d}": float(value)
        for level, value in zip(levels, results, strict=True)
    }


@torch.no_grad()
def analyze(config: dict, samples: int = 512, seed: int = 42) -> dict:
    """Compare disjoint train subsets and the actual initialized generator on CPU."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    torch.manual_seed(seed)
    system = get_system(config["system"])
    data, metadata = load_dataset(Path(config["data"]), "train")
    if metadata["system"] != system.name:
        raise ValueError("config and dataset systems differ")
    holdout = int(config["training"].get("validation_holdout", 0))
    if holdout:
        data = data[:-holdout]
    if len(data) < 2 * samples:
        raise ValueError("need at least twice samples training configurations")
    indices = torch.randperm(len(data))[: 2 * samples]
    references, queries = data[indices].split(samples)
    # Match training's model initialization seed, independent of subset selection.
    torch.manual_seed(int(config["seed"]))
    model = build_model(config).eval()
    noise_rng = torch.Generator().manual_seed(seed + 1)
    coordinate_noise = float(config["training"]["coordinate_noise_scale"]) * torch.randn(
        samples, system.particles, system.dimensions, generator=noise_rng
    )
    feature_noise = torch.randn(
        samples, system.particles, int(config["model"]["feature_dim"]), generator=noise_rng
    )
    generated = torch.cat(
        [
            model(coordinate_noise[start : start + 16], feature_noise[start : start + 16])
            for start in range(0, samples, 16)
        ]
    )
    reference_features = particle_descriptors(references)
    data_distance = torch.cdist(particle_descriptors(queries), reference_features)
    generated_distance = torch.cdist(particle_descriptors(generated), reference_features)
    median = float(data_distance.median())
    local = min(median / 2, 2 * float(data_distance.min(dim=1).values.median()))
    # For fixed separation d, d/h^2 * exp(-d^2/(2h^2)) peaks at h=d/sqrt(2).
    bridge = max(2 * median, float(generated_distance.median()) / math.sqrt(2))
    suggested = sorted(
        {round(value, 6) for value in (local, median / 2, median, 2 * median, bridge)}
    )
    candidates = suggested.copy()
    configured = config["training"]["bandwidth"]
    if isinstance(configured, (int, float)):
        candidates = sorted({*candidates, float(configured)})
    elif isinstance(configured, list):
        candidates = sorted(set(candidates + [float(value) for value in configured]))
    results = []
    for bandwidth in candidates:
        entry = {"bandwidth": bandwidth}
        for label, distances in (
            ("data", data_distance),
            ("initial_generator", generated_distance),
        ):
            weights = torch.exp(-distances.square() / (2 * bandwidth**2))
            entry[label] = {
                "mean_kernel_mass": float(weights.mean()),
                "kernel_mass": quantiles(weights.mean(dim=1)),
                "fraction_queries_max_kernel_below_1e-6": float(
                    (weights.max(dim=1).values < 1e-6).float().mean()
                ),
                "effective_references_median": float(
                    (
                        weights.sum(dim=1).square() / weights.square().sum(dim=1).clamp_min(1e-30)
                    ).median()
                ),
            }
        field_count = min(samples, 128)
        _, metrics = DescriptorDrift(bandwidth)(
            generated[:field_count],
            references,
            generated,
            self_indices=torch.arange(field_count),
            repulsion=float(config["training"]["repulsion"]),
        )
        entry["coordinate_fields"] = {name: float(value) for name, value in metrics.items()}
        results.append(entry)
    return {
        "system": system.name,
        "seed": seed,
        "model_seed": int(config["seed"]),
        "model": config["model"],
        "coordinate_noise_scale": config["training"]["coordinate_noise_scale"],
        "samples_per_bank": samples,
        "descriptor": "sorted_pair_distances / sqrt(number_of_pairs)",
        "kernel": "gaussian",
        "data_descriptor_distances": quantiles(data_distance),
        "initial_generator_descriptor_distances": quantiles(generated_distance),
        "data_nearest_reference_distance": quantiles(data_distance.min(dim=1).values),
        "initial_generator_nearest_reference_distance": quantiles(
            generated_distance.min(dim=1).values
        ),
        "suggested_multiscale_bandwidths": suggested,
        "candidates": results,
        "interpretation": "Coverage diagnostic, not a trained-model quality or convergence test.",
    }


def main() -> None:
    """Write a reproducible Gaussian descriptor-bandwidth diagnostic report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = analyze(load_config(args.config), args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2)
    )


if __name__ == "__main__":
    main()
