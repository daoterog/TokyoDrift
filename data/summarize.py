"""Create compact statistics and distribution plots for prepared particle datasets."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .systems import ParticleSystem, get_system, pair_distances

DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = tuple(DATA_ROOT / name / "dataset.npz" for name in ("dw4", "lj13", "lj55"))
SPLIT_ORDER = ("train", "validation", "test")
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)
METRIC_LABELS = {
    "energy": "Dimensionless energy",
    "coordinate": "Centered coordinate marginal",
    "particle_radius": "Particle radius from center",
    "radius_of_gyration": "Radius of gyration",
    "pair_distance": "Pair-distance marginal",
    "minimum_pair_distance": "Minimum pair distance",
}


def arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Write JSON summaries and distribution plots for prepared datasets."
    )
    parser.add_argument(
        "datasets",
        type=Path,
        nargs="*",
        help="Prepared NPZ archives. Defaults to DW4, LJ13, and LJ55 under data/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "reports" / "dataset_summaries",
        help="Root directory for one output folder per system.",
    )
    parser.add_argument(
        "--max-configurations",
        type=int,
        default=50_000,
        help="Maximum evenly spaced configurations summarized from each split.",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=1_000_000,
        help="Maximum scalar observations retained for each plotted distribution.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2_048,
        help="Configuration batch size for energy and pair-distance calculations.",
    )
    parser.add_argument("--bins", type=int, default=120)
    return parser.parse_args()


def evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    """Return deterministic indices spread across an ordered split."""
    if size <= 0 or count <= 0:
        raise ValueError("size and count must be positive")
    count = min(size, count)
    return np.arange(count, dtype=np.int64) * size // count


def bounded(values: np.ndarray, maximum: int) -> np.ndarray:
    """Retain at most ``maximum`` evenly spaced scalar observations."""
    flattened = np.asarray(values).reshape(-1)
    if len(flattened) <= maximum:
        return flattened
    return flattened[evenly_spaced_indices(len(flattened), maximum)]


def describe(values: np.ndarray) -> dict[str, Any]:
    """Return finite scalar summary statistics suitable for JSON output."""
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flattened[np.isfinite(flattened)]
    result: dict[str, Any] = {
        "observations": len(flattened),
        "finite_observations": len(finite),
        "finite_fraction": float(len(finite) / len(flattened)) if len(flattened) else None,
    }
    if not len(finite):
        result.update({"mean": None, "standard_deviation": None, "quantiles": {}})
        return result
    quantile_values = np.quantile(finite, QUANTILES)
    result.update(
        {
            "mean": float(finite.mean()),
            "standard_deviation": float(finite.std()),
            "quantiles": {
                f"q{100 * level:g}": float(value)
                for level, value in zip(QUANTILES, quantile_values, strict=True)
            },
        }
    )
    return result


def chunked_energy(
    samples: np.ndarray, system: ParticleSystem, batch_size: int
) -> np.ndarray | None:
    """Evaluate energies without materializing all pair distances at once."""
    if system.energy is None:
        return None
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            positions = torch.from_numpy(np.ascontiguousarray(samples[start : start + batch_size]))
            chunks.append(system.energy(positions).cpu().numpy())
    return np.concatenate(chunks)


def chunked_pair_metrics(samples: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened pair distances and per-configuration minima."""
    distances = []
    minimum_distances = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            positions = torch.from_numpy(np.ascontiguousarray(samples[start : start + batch_size]))
            batch_distances = pair_distances(positions).cpu().numpy()
            distances.append(batch_distances.reshape(-1))
            minimum_distances.append(batch_distances.min(axis=1))
    return np.concatenate(distances), np.concatenate(minimum_distances)


def split_metrics(
    samples: np.ndarray,
    system: ParticleSystem,
    max_observations: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Compute bounded physical and geometric observations for one split."""
    pair_count = system.particles * (system.particles - 1) // 2
    pair_configuration_count = min(len(samples), max(1, max_observations // pair_count))
    pair_samples = samples[evenly_spaced_indices(len(samples), pair_configuration_count)]
    pair_values, minimum_pair_values = chunked_pair_metrics(pair_samples, batch_size)
    squared_radii = np.square(samples, dtype=np.float64).sum(axis=-1)
    metrics = {
        "coordinate": bounded(samples, max_observations),
        "particle_radius": bounded(np.sqrt(squared_radii), max_observations),
        "radius_of_gyration": np.sqrt(squared_radii.mean(axis=1)),
        "pair_distance": bounded(pair_values, max_observations),
        "minimum_pair_distance": minimum_pair_values,
        "center_of_mass_norm": np.linalg.norm(samples.mean(axis=1), axis=-1),
    }
    energy = chunked_energy(samples, system, batch_size)
    if energy is not None:
        metrics["energy"] = energy
    return metrics


def plotting_range(metric: str, split_values: dict[str, np.ndarray]) -> tuple[float, float] | None:
    """Choose a shared robust plotting range across splits."""
    finite_parts = []
    for values in split_values.values():
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        if len(finite):
            finite_parts.append(finite)
    if not finite_parts:
        return None
    combined = np.concatenate(finite_parts)
    low, high = np.quantile(combined, (0.001, 0.999))
    if metric == "coordinate":
        extent = max(abs(float(low)), abs(float(high)), np.finfo(np.float32).eps)
        return -extent, extent
    low = float(low)
    high = float(high)
    if not high > low:
        padding = max(abs(low) * 0.05, np.finfo(np.float32).eps)
        return low - padding, high + padding
    return low, high


def write_distribution_plot(
    metrics_by_split: dict[str, dict[str, np.ndarray]],
    output: Path,
    title: str,
    bins: int,
) -> None:
    """Write a six-panel histogram comparison for one or more splits."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "particle-drift-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plotted_metrics = [
        name for name in METRIC_LABELS if any(name in m for m in metrics_by_split.values())
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    flat_axes = axes.reshape(-1)
    for axis, metric in zip(flat_axes, plotted_metrics, strict=False):
        values_by_split = {
            split: values[metric] for split, values in metrics_by_split.items() if metric in values
        }
        value_range = plotting_range(metric, values_by_split)
        if value_range is None:
            axis.text(0.5, 0.5, "No finite observations", ha="center", va="center")
            continue
        for split, values in values_by_split.items():
            finite = values[np.isfinite(values)]
            axis.hist(
                finite,
                bins=bins,
                range=value_range,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=split,
            )
        axis.set(xlabel=METRIC_LABELS[metric], ylabel="density")
        axis.set_title("central 99.8% range")
        if len(values_by_split) > 1:
            axis.legend()
    for axis in flat_axes[len(plotted_metrics) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def summarize_dataset(
    dataset: Path,
    output_root: Path,
    max_configurations: int,
    max_observations: int,
    batch_size: int,
    bins: int,
) -> Path:
    """Summarize one prepared archive and return its output directory."""
    if min(max_configurations, max_observations, batch_size, bins) <= 0:
        raise ValueError("sampling limits, batch size, and bins must be positive")
    if not dataset.is_file():
        raise FileNotFoundError(f"prepared dataset does not exist: {dataset}")

    with np.load(dataset, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError(f"{dataset} has no metadata entry")
        metadata = json.loads(str(archive["metadata"].item()))
        system = get_system(metadata["system"])
        split_names = [name for name in SPLIT_ORDER if name in archive.files]
        split_names.extend(
            sorted(name for name in archive.files if name not in {*SPLIT_ORDER, "metadata"})
        )
        if not split_names:
            raise ValueError(f"{dataset} has no data splits")

        metrics_by_split: dict[str, dict[str, np.ndarray]] = {}
        split_summaries: dict[str, Any] = {}
        expected_shape = (system.particles, system.dimensions)
        for split in split_names:
            values = np.asarray(archive[split], dtype=np.float32)
            if values.ndim != 3 or tuple(values.shape[1:]) != expected_shape:
                raise ValueError(
                    f"expected {split} to have shape [N, {system.particles}, "
                    f"{system.dimensions}], got {values.shape}"
                )
            sample_count = min(len(values), max_configurations)
            samples = np.ascontiguousarray(values[evenly_spaced_indices(len(values), sample_count)])
            metrics = split_metrics(samples, system, max_observations, batch_size)
            metrics_by_split[split] = metrics
            split_summaries[split] = {
                "configurations": len(values),
                "sampled_configurations": int(sample_count),
                "metrics": {name: describe(observations) for name, observations in metrics.items()},
            }

    output = output_root / system.name
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": str(dataset.resolve()),
        "system": system.name,
        "particles": system.particles,
        "dimensions": system.dimensions,
        "metadata": metadata,
        "sampling": {
            "method": "evenly spaced indices in each ordered split",
            "max_configurations": max_configurations,
            "max_observations_per_distribution": max_observations,
        },
        "splits": split_summaries,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_distribution_plot(
        metrics_by_split,
        output / "all_splits_distributions.png",
        f"{system.name.upper()} split distributions",
        bins,
    )
    for split, metrics in metrics_by_split.items():
        write_distribution_plot(
            {split: metrics},
            output / f"{split}_distributions.png",
            f"{system.name.upper()} {split} distributions",
            bins,
        )
    return output


def main() -> None:
    """Summarize all requested datasets."""
    args = arguments()
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "particle-drift-matplotlib")
    )
    datasets = tuple(args.datasets) or DEFAULT_DATASETS
    for dataset in datasets:
        output = summarize_dataset(
            dataset,
            args.output,
            args.max_configurations,
            args.max_observations,
            args.batch_size,
            args.bins,
        )
        print(f"wrote dataset summary to {output}")


if __name__ == "__main__":
    main()
