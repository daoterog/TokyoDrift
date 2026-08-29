"""Diagnose high-energy DW4 generator samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .io import load_dataset
from .systems import dw4_energy, pair_distances


def arguments() -> argparse.Namespace:
    """Parse tail-audit arguments."""
    parser = argparse.ArgumentParser(description="Audit the DW4 high-energy tail.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pair_energy(distance: torch.Tensor) -> torch.Tensor:
    """Return the DW4 contribution of every pair distance."""
    displacement = distance - 4.0
    return -4.0 * displacement.square() + 0.9 * displacement.pow(4)


def quantile_groups(energy: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Split samples into bulk and increasingly extreme energy bands."""
    boundaries = torch.quantile(energy, torch.tensor([0.9, 0.99, 0.999]))
    return [
        ("bulk 0-90%", energy <= boundaries[0]),
        ("tail 90-99%", (energy > boundaries[0]) & (energy <= boundaries[1])),
        ("tail 99-99.9%", (energy > boundaries[1]) & (energy <= boundaries[2])),
        ("extreme top 0.1%", energy > boundaries[2]),
    ]


def main() -> None:
    """Create metrics and plots explaining the generated DW4 tail."""
    args = arguments()
    with np.load(args.samples, allow_pickle=False) as archive:
        generated = torch.from_numpy(np.asarray(archive["positions"], dtype=np.float32))
    reference, metadata = load_dataset(args.data, "test")
    if metadata["system"] != "dw4":
        raise ValueError("tail analysis currently supports DW4 only")
    generated_energy = dw4_energy(generated)
    reference_energy = dw4_energy(reference)
    generated_distance = pair_distances(generated)
    reference_distance = pair_distances(reference)
    contributions = pair_energy(generated_distance)
    minimum_distance = generated_distance.min(dim=1).values
    maximum_distance = generated_distance.max(dim=1).values
    reference_max_q999 = float(torch.quantile(reference_distance.max(dim=1).values, 0.999))
    reference_q99 = float(torch.quantile(reference_energy, 0.99))
    reference_q999 = float(torch.quantile(reference_energy, 0.999))
    top_one = generated_energy >= torch.quantile(generated_energy, 0.99)
    quantiles = torch.linspace(0, 1, 4096)
    quantile_error = (
        torch.quantile(generated_energy, quantiles) - torch.quantile(reference_energy, quantiles)
    ).abs()
    top_quantile_count = max(1, int(0.01 * len(quantile_error)))
    metrics = {
        "num_generated": len(generated),
        "reference_energy_q99": reference_q99,
        "reference_energy_q999": reference_q999,
        "fraction_above_reference_q99": float((generated_energy > reference_q99).float().mean()),
        "fraction_above_reference_q999": float((generated_energy > reference_q999).float().mean()),
        "fraction_collision_d_lt_0_5": float((minimum_distance < 0.5).float().mean()),
        "reference_max_distance_q999": reference_max_q999,
        "fraction_overexpanded": float((maximum_distance > reference_max_q999).float().mean()),
        "top_1_percent_fraction_with_collision": float(
            (minimum_distance[top_one] < 0.5).float().mean()
        ),
        "top_1_percent_fraction_overexpanded": float(
            (maximum_distance[top_one] > reference_max_q999).float().mean()
        ),
        "top_1_percent_w1_contribution": float(
            quantile_error[-top_quantile_count:].sum() / quantile_error.sum()
        ),
        "energy_quantiles_generated": {
            str(q): float(torch.quantile(generated_energy, q)) for q in (0.9, 0.99, 0.999, 0.9999)
        },
        "energy_quantiles_reference": {
            str(q): float(torch.quantile(reference_energy, q)) for q in (0.9, 0.99, 0.999, 0.9999)
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "tail_metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for values, label in ((reference_energy, "test"), (generated_energy, "generated")):
        ordered = values.sort().values
        survival = torch.arange(len(ordered), 0, -1) / len(ordered)
        axis.semilogy(ordered.numpy(), survival.numpy(), label=label)
    axis.set(xlabel="energy", ylabel="survival probability", title="DW4 energy tail")
    axis.set_xlim(
        float(torch.quantile(reference_energy, 0.001)),
        float(torch.quantile(generated_energy, 0.9999)),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "energy_survival.png", dpi=200)
    plt.close(figure)

    groups = quantile_groups(generated_energy)
    group_names = [name for name, _ in groups]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, values, title in (
        (axes[0], minimum_distance, "minimum pair distance"),
        (axes[1], maximum_distance, "maximum pair distance"),
        (axes[2], contributions.max(dim=1).values, "largest pair-energy contribution"),
    ):
        axis.boxplot(
            [values[mask].numpy() for _, mask in groups], tick_labels=group_names, showfliers=False
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output / "tail_pair_diagnostics.png", dpi=200)
    plt.close(figure)

    generator = torch.Generator().manual_seed(2023)
    descriptor_count = 20_000
    generated_indices = torch.randperm(len(generated), generator=generator)[:descriptor_count]
    reference_indices = torch.randperm(len(reference), generator=generator)[:descriptor_count]
    generated_descriptor = generated_distance[generated_indices].sort(dim=1).values
    reference_descriptor = reference_distance[reference_indices].sort(dim=1).values
    combined = torch.cat((reference_descriptor, generated_descriptor))
    centered = combined - combined.mean(dim=0)
    _, _, directions = torch.pca_lowrank(centered, q=2)
    reference_projection = centered[:descriptor_count] @ directions
    generated_projection = centered[descriptor_count:] @ directions
    tail_projection = (
        generated_distance[top_one].sort(dim=1).values - combined.mean(dim=0)
    ) @ directions
    figure, axis = plt.subplots(figsize=(7, 5.5))
    axis.scatter(
        reference_projection[:, 0], reference_projection[:, 1], s=3, alpha=0.15, label="test"
    )
    axis.scatter(
        generated_projection[:, 0], generated_projection[:, 1], s=3, alpha=0.15, label="generated"
    )
    axis.scatter(
        tail_projection[:, 0], tail_projection[:, 1], s=5, alpha=0.3, label="generated top 1%"
    )
    axis.set(
        xlabel="descriptor PC1", ylabel="descriptor PC2", title="Joint sorted-distance structure"
    )
    axis.legend(markerscale=3)
    figure.tight_layout()
    figure.savefig(args.output / "descriptor_pca.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(4, 5, figsize=(14, 10))
    rows, columns = torch.triu_indices(4, 4, offset=1)
    for group_index, (name, mask) in enumerate(groups):
        indices = torch.where(mask)[0]
        selected = indices[torch.linspace(0, len(indices) - 1, 5).long()]
        selected = selected[torch.argsort(generated_energy[selected])]
        for column, index in enumerate(selected):
            axis = axes[group_index, column]
            positions = generated[index].numpy()
            segments = np.stack((positions[rows], positions[columns]), axis=1)
            values = contributions[index].numpy()
            collection = LineCollection(segments, cmap="coolwarm", linewidths=1.7)
            collection.set_array(values)
            collection.set_clim(-5, max(5, float(values.max())))
            axis.add_collection(collection)
            axis.scatter(positions[:, 0], positions[:, 1], c="black", s=24, zorder=3)
            padding = 0.5
            axis.set_xlim(positions[:, 0].min() - padding, positions[:, 0].max() + padding)
            axis.set_ylim(positions[:, 1].min() - padding, positions[:, 1].max() + padding)
            axis.set_aspect("equal")
            axis.set_title(
                f"E={float(generated_energy[index]):.1f}\nmin={float(minimum_distance[index]):.2f}"
            )
            axis.set_xticks([])
            axis.set_yticks([])
        axes[group_index, 0].set_ylabel(name)
    figure.suptitle("Representative DW4 configurations by energy quantile")
    figure.tight_layout()
    figure.savefig(args.output / "tail_configurations.png", dpi=200)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
