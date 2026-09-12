"""Evaluate alanine samples: topology, chirality, periodic torsions and ff96/OBC1 energy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .alanine import backbone_angles, geometry_masks
from .alanine_energy import AlanineEnergy
from .alanine_metrics import (
    exact_wasserstein_1,
    geometry_report,
    observable_report,
    probability_metrics,
    subset_indices,
    torsion_comparison,
    torsion_histogram,
)
from .evaluate import generate
from .io import build_model, device_summary, load_dataset, select_device


def energy_summary(values: np.ndarray) -> dict:
    """Summarize physical energies without silently dropping failures from validity."""
    finite = values[np.isfinite(values)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "finite_fraction": len(finite) / len(values) if len(values) else None,
        "mean_finite": float(finite.mean()) if len(finite) else None,
        "quantiles_finite": dict(
            zip(
                ("q01", "q50", "q95", "q99", "q999"),
                np.quantile(finite, [0.01, 0.5, 0.95, 0.99, 0.999]).tolist()
                if len(finite)
                else [None] * 5,
                strict=True,
            )
        ),
    }


def energy_report(
    gen_energy: np.ndarray,
    ref_energy: np.ndarray,
    generated: torch.Tensor,
    reference: torch.Tensor,
    rules: dict,
) -> dict:
    """Measure raw tails and validity on the explicitly reported energy subset."""
    finite_ref = ref_energy[np.isfinite(ref_energy)]
    if not len(finite_ref):
        raise ValueError("no finite reference energy; check topology, units and force field")
    cutoff = float(np.quantile(finite_ref, 0.99))
    low, high = np.quantile(finite_ref, [0.005, 0.995])
    if high <= low:
        low, high = low - 1, high + 1
    edges = np.linspace(low, high, 101)

    def histogram(values: np.ndarray) -> np.ndarray:
        # Explicit underflow, overflow and nonfinite bins preserve bad tail mass.
        finite = values[np.isfinite(values)]
        return np.r_[
            np.sum(finite < low),
            np.histogram(finite, bins=edges)[0],
            np.sum(finite > high),
            np.sum(~np.isfinite(values)),
        ]

    gen_mask = geometry_masks(generated, rules)["valid"].numpy()
    ref_mask = geometry_masks(reference, rules)["valid"].numpy()
    gen_under = np.isfinite(gen_energy) & (gen_energy <= cutoff)
    ref_under = np.isfinite(ref_energy) & (ref_energy <= cutoff)
    gen_valid, ref_valid = gen_mask & gen_under, ref_mask & ref_under
    return {
        "units": "U/(R*T), dimensionless",
        "temperature_kelvin": 300,
        "generated": energy_summary(gen_energy),
        "reference": energy_summary(ref_energy),
        "wasserstein_1_finite": exact_wasserstein_1(ref_energy, gen_energy),
        "histogram": {
            "range": [float(low), float(high)],
            "interior_bins": 100,
            "includes_underflow_overflow_nonfinite": True,
            **probability_metrics(histogram(ref_energy), histogram(gen_energy)),
        },
        "energy_cutoff_reference_q99": cutoff,
        "generated_above_reference_q99_fraction": float((gen_energy > cutoff).mean()),
        "generated_energy_under_cutoff_fraction": float(gen_under.mean()),
        "generated_geometry_chirality_and_energy_valid_fraction": float(gen_valid.mean()),
        "reference_geometry_chirality_and_energy_valid_fraction": float(ref_valid.mean()),
        "valid_counts": {"generated": int(gen_valid.sum()), "reference": int(ref_valid.sum())},
        "valid_energy_wasserstein_1": exact_wasserstein_1(
            ref_energy[ref_valid], gen_energy[gen_valid]
        ),
        "valid_torsions": torsion_comparison(generated[gen_valid], reference[ref_valid]),
    }


def plot_reports(
    generated: torch.Tensor, reference: torch.Tensor, rules: dict, output: Path
) -> None:
    """Save shared-scale Ramachandran maps and phi/psi free-energy profiles."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    valid = geometry_masks(generated, rules)["valid"]
    populations = [reference, generated, generated[valid]]
    counts = [torsion_histogram(backbone_angles(values)) for values in populations]
    probabilities = [values / max(values.sum(), 1) for values in counts]
    np.savez_compressed(
        output / "torsion_histograms.npz",
        reference=counts[0],
        generated=counts[1],
        generated_valid=counts[2],
        edges_radians=np.linspace(-np.pi, np.pi, 65),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    vmax = max(1e-5, max(float(p.max()) for p in probabilities))
    for axis, probability, title in zip(
        axes,
        probabilities,
        ("Reference", "Generated: all defined torsions", "Generated: geometry + chirality valid"),
        strict=True,
    ):
        picture = axis.imshow(
            np.ma.masked_less_equal(probability.T, 0),
            origin="lower",
            extent=(-180, 180, -180, 180),
            norm=LogNorm(1e-6, vmax),
            cmap="viridis",
        )
        axis.set(xlabel="phi (degrees)", ylabel="psi (degrees)", title=title)
    figure.colorbar(picture, ax=axes, label="Probability per bin")
    figure.savefig(output / "ramachandran.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    centers = np.linspace(-180, 180, 65)
    centers = (centers[:-1] + centers[1:]) / 2
    for axis, dimension, label in zip(axes, (1, 0), ("phi", "psi"), strict=True):
        for probability, name in zip(
            probabilities, ("reference", "generated", "generated valid"), strict=True
        ):
            marginal = probability.sum(axis=dimension)
            if marginal.sum() == 0:
                continue
            free_energy = -np.log(np.maximum(marginal, 1e-10))
            free_energy -= free_energy.min()
            axis.plot(centers, free_energy, label=name)
        axis.set(
            xlabel=f"{label} (degrees)",
            ylabel="Relative free energy / kBT",
            title="Empirical probabilities; floor 1e-10",
        )
        axis.legend()
    figure.savefig(output / "free_energy.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Sample a checkpoint and write reproducible molecular evaluation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=None, help="Override a relocated prepared dataset."
    )
    parser.add_argument("--num-samples", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--energy-samples", type=int, default=10_000)
    parser.add_argument("--metric-sample-size", type=int, default=20_000)
    parser.add_argument(
        "--skip-energy", action="store_true", help="Explicit geometry-only evaluation."
    )
    parser.add_argument("--energy-platform", choices=("Reference", "CPU"), default="Reference")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.num_samples, args.batch_size, args.energy_samples, args.metric_sample_size) <= 0:
        parser.error("all sample counts and batch sizes must be positive")
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["system"] != "aldp":
        raise ValueError("this evaluator requires an alanine checkpoint")
    data_path = args.data or Path(config["data"])
    reference, metadata = load_dataset(data_path, "test")
    if metadata["system"] != "aldp" or metadata.get("coordinate_units") != "angstrom":
        raise ValueError("expected prepared alanine data in angstroms")
    rules = metadata["geometry_rules"]
    energy_fn = (
        None
        if args.skip_energy
        else AlanineEnergy(data_path.parent / metadata["topology_file"], args.energy_platform)
    )
    model = build_model(config).to(device).eval()
    ema = checkpoint.get("ema")
    model.load_state_dict(ema if ema is not None else checkpoint["model"])
    generated, endpoint = generate(
        model,
        "aldp",
        args.num_samples,
        args.batch_size,
        float(config["training"]["coordinate_noise_scale"]),
        int(config["model"]["feature_dim"]),
        device,
    )
    report = {
        "system": "aldp",
        "dataset": metadata["record"],
        "coordinate_units": "angstrom",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "evaluated_weights": "ema" if ema is not None else "model",
        "drift": config["drift"],
        "num_generated_samples": len(generated),
        "num_test_samples": len(reference),
        "runtime": device_summary(device),
        "seed": args.seed,
        "metric_definition": {
            "torsion_bins_per_axis": 64,
            "torsion_range_radians": [-np.pi, np.pi],
            "histogram_probability_floor": 1e-10,
            "divergence_log_base": "e",
            "geometry_calibration_split": "train",
            "observable_sample_cap": args.metric_sample_size,
            "subset_seed": 17903,
            "sampling": "whole configurations without replacement",
            "selection_score": "validation phi/psi JS + (1 - geometry/chirality valid fraction)",
        },
        "geometry": geometry_report(generated, reference, rules),
        "observables": observable_report(generated, reference, args.metric_sample_size),
        "endpoint_transport_distance_per_particle_angstrom": endpoint,
        "energy": {"available": False, "reason": "explicitly disabled by --skip-energy"},
        "density_metrics": {
            "nll": None,
            "importance_weight_ess": None,
            "reason": "Direct drift generator has no tractable proposal density.",
        },
        "limitations": [
            "Pair-distance descriptors and the E(n) generator do not resolve mirror chirality.",
            "Filtering conditions the generated distribution; it does not reweight it.",
            "FAB ff96/OBC1 data differs from the GFN2-xTB Equivariant Flow Matching data.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if energy_fn is not None:
        gen_indices = subset_indices(len(generated), args.energy_samples)
        ref_indices = subset_indices(len(reference), args.energy_samples)
        gen_subset, ref_subset = generated[gen_indices], reference[ref_indices]
        print(
            f"Evaluating energies: {len(gen_subset)} generated, {len(ref_subset)} reference",
            flush=True,
        )
        gen_energy, ref_energy = energy_fn(gen_subset.numpy()), energy_fn(ref_subset.numpy())
        report["energy"] = {
            "available": True,
            "platform": energy_fn.platform,
            "openmm_version": energy_fn.version,
            "forcefield": metadata["forcefield"],
            "topology_sha256": metadata["topology_sha256"],
            "scope": "deterministic subset; these fractions do not cover all generated samples",
            **energy_report(gen_energy, ref_energy, gen_subset, ref_subset, rules),
        }
        np.savez_compressed(
            args.output / "energies.npz",
            generated=gen_energy,
            reference=ref_energy,
            generated_indices=gen_indices,
            reference_indices=ref_indices,
        )
    with (args.output / "metrics.json").open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    plot_reports(generated, reference, rules, args.output)
    if args.save_samples:
        np.save(args.output / "samples.npy", generated.numpy())
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
