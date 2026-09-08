"""Reproducible structure and geometry inspection of a single-molecule checkpoint."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolAlign, rdMolTransforms

from .data import load_split
from .drift import angle_indices, descriptor, torsion_indices
from .metrics import _molecule_with_conformers, geometry_validity
from .model import ConformerGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    r = next(r for r in load_split(Path(config["data"]), "train", 50) if r.identifier == config["molecule"])
    model = ConformerGenerator(**config["model"]).to("mps").eval()
    model.load_state_dict(checkpoint["ema"])
    with torch.no_grad():
        # Bound peak memory; retain independent latent draws across chunks.
        generated = torch.cat([model(torch.randn(min(100, args.samples - start), len(r.atomic_numbers), 3, device="mps") * config["training"]["coordinate_noise_scale"],
                                     r.atomic_numbers.to("mps"), r.bond_index.to("mps"), r.bond_order.to("mps")).cpu()
                               for start in range(0, args.samples, 100)])
    torch.save({"generated": generated, "reference": r.conformers, "seed": args.seed,
                "checkpoint": str(args.checkpoint)}, args.output / "samples.pt")
    gmol = _molecule_with_conformers(r.atomic_numbers, r.bond_index, r.bond_order, generated)
    rmol = _molecule_with_conformers(r.atomic_numbers, r.bond_index, r.bond_order, r.conformers)
    Chem.SanitizeMol(gmol)
    Chem.SanitizeMol(rmol)
    rms = np.empty((len(generated), len(r.conformers)))
    for i in range(len(generated)):
        for j in range(len(r.conformers)):
            rms[i, j] = rdMolAlign.GetBestRMS(gmol, rmol, i, j)
    np.save(args.output / "rmsd_matrix.npy", rms)
    precision, recall = rms.min(1), rms.min(0)
    blue, orange = "#2878A5", "#E68132"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
    arrays = []
    angles = angle_indices(r.bond_index, len(r.atomic_numbers))
    for positions in (r.conformers, generated):
        bonds = (positions[:, r.bond_index[0]] - positions[:, r.bond_index[1]]).norm(dim=-1).numpy()
        values = descriptor(positions, r.bond_index, angles, torch.empty(4, 0, dtype=torch.long), torch.empty(2, 0, dtype=torch.long))
        degrees = values[:, r.bond_index.shape[1]:].numpy() * 180 / np.pi
        radius = positions.square().sum(-1).mean(-1).sqrt().numpy()
        arrays.append((bonds, degrees, radius))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, k, title, unit in zip(axes.flat, range(3), ("Bond lengths", "Bond angles", "Radius of gyration"), ("Å", "degrees", "Å")):
        edges = np.histogram_bin_edges(np.concatenate([a[k].ravel() for a in arrays]), bins=45)
        for a, label, color in zip(arrays, ("Reference (50)", f"Generated ({len(generated)})"), (blue, orange)):
            ax.hist(a[k].ravel(), bins=edges, density=True, histtype="step", linewidth=2, label=label, color=color)
        ax.set(title=title, xlabel=unit, ylabel="Density")
    axes[0, 0].legend(frameon=False)
    ax = axes[1, 1]
    for values, label, color in ((precision, "Generated → reference", orange), (recall, "Reference → generated", blue)):
        ax.plot(np.sort(values), np.arange(1, len(values)+1)/len(values), label=label, color=color)
    ax.axvline(.5, color="grey", linestyle="--")
    ax.set(title="Nearest symmetry-aware RMSD", xlabel="RMSD (Å)", ylabel="Fraction ≤ RMSD", ylim=(0, 1.02))
    ax.legend(frameon=False)
    fig.suptitle(f"Single-molecule drift · {checkpoint['step']:,} steps · {r.identifier}\nTraining-ensemble comparison; raw samples, no geometry relaxation", fontsize=14)
    fig.savefig(args.output / "geometry.png")
    plt.close(fig)

    rotatable = {tuple(sorted(pair)) for pair in rmol.GetSubstructMatches(Lipinski.RotatableBondSmarts)}
    chosen = {}
    for indices in torsion_indices(r.bond_index, len(r.atomic_numbers)).T.tolist():
        central = tuple(sorted(indices[1:3]))
        if central in rotatable:
            chosen.setdefault(central, indices)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    torsion_summary = []
    for ax, (central, indices) in zip(axes.flat, chosen.items()):
        values = []
        for molecule, label, color in ((rmol, "Reference", blue), (gmol, "Generated", orange)):
            degrees = np.array([rdMolTransforms.GetDihedralDeg(c, *indices) for c in molecule.GetConformers()])
            ax.hist(degrees, bins=np.linspace(-180, 180, 25), density=True, histtype="step", linewidth=2, label=label, color=color)
            values.append(degrees)
        # Observed reference support is finite, not a physical-validity definition.
        angular_distance = np.abs((values[1][:, None] - values[0][None] + 180) % 360 - 180).min(1)
        torsion_summary.append({"atoms_1based": [i+1 for i in indices], "generated_farther_than_30deg_from_observed_reference_fraction": float((angular_distance > 30).mean())})
        ax.set(title="Torsion " + "–".join(str(i+1) for i in indices), xlabel="Dihedral (degrees)", ylabel="Density", xlim=(-180, 180), xticks=[-180, -90, 0, 90, 180])
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Rotatable-bond marginals · labelled atoms · periodic angles\nMarginal agreement does not establish joint conformer coverage", fontsize=14)
    fig.savefig(args.output / "torsions.png")
    plt.close(fig)

    fig = plt.figure(figsize=(13, 8), layout="constrained")
    cases = []
    ordered = np.argsort(precision)
    for q in (.1, .5, .9, .99):
        i = int(ordered[round(q*(len(ordered)-1))])
        cases.append((i, int(rms[i].argmin()), f"Generated RMSD percentile {q:.0%}"))
    for j in np.argsort(recall)[-2:]:
        cases.append((int(rms[:, j].argmin()), int(j), f"Poorly covered reference #{j+1}"))
    for panel, (i, j, title) in enumerate(cases):
        rdMolAlign.GetBestRMS(gmol, rmol, i, j)
        xyz_g = np.array(gmol.GetConformer(i).GetPositions())
        xyz_r = np.array(rmol.GetConformer(j).GetPositions())
        # Common camera frame for each aligned pair, using reference principal axes.
        _, _, basis = np.linalg.svd(xyz_r - xyz_r.mean(0), full_matrices=False)
        pair = [(xyz_r - xyz_r.mean(0)) @ basis.T, (xyz_g - xyz_r.mean(0)) @ basis.T]
        ax = fig.add_subplot(2, 3, panel+1, projection="3d")
        for xyz, color, label in zip(pair, (blue, orange), ("Reference", "Generated")):
            for left, right in r.bond_index.T.tolist():
                ax.plot(*xyz[[left, right]].T, color=color, linewidth=2, alpha=.8)
            ax.scatter(*xyz.T, color=color, s=28)
            for atom in (r.atomic_numbers == 8).nonzero().flatten().tolist():
                ax.text(*xyz[atom], " O", color=color, fontsize=10)
        limit = max(np.abs(np.concatenate(pair)).max(), 1) * 1.1
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit), title=f"{title}\nRMSD {rms[i,j]:.3f} Å")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-65)
        ax.set_axis_off()
    fig.suptitle("Aligned structure overlays · blue reference / orange generated\nSelected by RMSD rank, including failures; oxygen marked O", fontsize=14)
    fig.savefig(args.output / "structures.png")
    plt.close(fig)
    summary = {"checkpoint_step": checkpoint["step"], "generated_count": len(generated), "reference_count": len(r.conformers), "seed": args.seed,
               "cov_precision": float((precision<=.5).mean()), "cov_recall": float((recall<=.5).mean()),
               "mat_precision": float(precision.mean()), "mat_recall": float(recall.mean()),
               "generated_geometry": geometry_validity(generated, r.bond_index),
               "reference_geometry": geometry_validity(r.conformers, r.bond_index), "torsions": torsion_summary,
               "rmsd_precision_quantiles": dict(zip(("p10", "p50", "p90", "p99"), np.quantile(precision, [.1,.5,.9,.99]).tolist()))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
