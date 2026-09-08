"""Audit then run a single-graph overfitting experiment with automatic evaluations.

Run from the repository root: python -m geom_qm9.single_molecule
Metrics compare to training conformers: this is a capacity diagnostic, not a
held-out generalization benchmark. No alignment is used in training.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .data import load_split
from .drift import descriptor, descriptor_statistics, field, median_bandwidth, angle_indices, torsion_indices, three_hop_pair_indices, unnormalised_drift
from .metrics import geometry_validity, symmetry_aware_cov_mat
from .model import ConformerGenerator
from .prepare_geom import map_records, molecule_record


def prepare(config: dict) -> None:
    """Rebuild this record from raw XYZ using the corrected source-to-XYZ mapping."""
    destination = Path(config["data"]) / "train.pt"
    if destination.exists():
        return
    unpacker = map_records(Path("geom_qm9/data/raw/qm9_crude.msgpack.tar.gz"))
    for _ in range(unpacker.read_map_header()):
        identifier, source = unpacker.unpack(), unpacker.unpack()
        if identifier == config["molecule"]:
            record = molecule_record(identifier, source, 50)
            if record is None:
                raise ValueError("Raw record could not be reconstructed")
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"molecules": [record]}, destination)
            return
    raise ValueError("Selected molecule missing from raw archive")


def audit(config: dict) -> dict:
    """Check reference geometry, metric identity, kernel gradients and initialization."""
    torch.manual_seed(config["seed"])
    records = [r for r in load_split(Path(config["data"]), "train", 50) if r.identifier == config["molecule"]]
    assert len(records) == 1
    r = records[0]
    mol = Chem.MolFromSmiles(r.identifier)
    assert mol is not None and "@" not in r.identifier
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == r.atomic_numbers.tolist()
    expected = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())): b.GetBondTypeAsDouble() for b in mol.GetBonds()}
    actual = {tuple(sorted(pair)): order for pair, order in zip(r.bond_index.T.tolist(), r.bond_order.tolist())}
    assert expected == actual, "SMILES/coordinate graph labels disagree"
    assert torch.isfinite(r.conformers).all()
    validity = geometry_validity(r.conformers, r.bond_index)
    assert validity["no_clash_fraction"] == 1.0
    bond_lengths = (r.conformers[:, r.bond_index[0]] - r.conformers[:, r.bond_index[1]]).norm(dim=-1)
    assert bond_lengths.min() > 1.0 and bond_lengths.max() < 1.9
    metric_args = (r.atomic_numbers, r.bond_index, r.bond_order)
    identity = symmetry_aware_cov_mat(r.conformers, r.conformers, *metric_args)
    assert identity["mat_recall"] < 1e-4 and identity["cov_recall"] == 1.0
    split_reference = symmetry_aware_cov_mat(r.conformers[::2], r.conformers[1::2], *metric_args)
    a = angle_indices(r.bond_index, len(r.atomic_numbers))
    t = torsion_indices(r.bond_index, len(r.atomic_numbers))
    p = three_hop_pair_indices(r.bond_index, len(r.atomic_numbers))
    mean, scale = descriptor_statistics(r.conformers, r.bond_index, a, t, p)
    kw = dict(bond_index=r.bond_index, angles=a, torsions=t, three_hop_pairs=p, mean=mean, scale=scale)
    bandwidth = median_bandwidth(r.conformers, **kw) * config["training"]["bandwidth_scale"]
    model = ConformerGenerator(**config["model"]).eval()
    z = torch.randn(100, len(r.atomic_numbers), 3)
    with torch.no_grad():
        generated = model(z, *metric_args)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        rotation[:, 0] *= torch.linalg.det(rotation)
        rotated = model(z @ rotation, *metric_args)
    rotation_error = float((rotated - generated @ rotation).abs().max())
    assert rotation_error < 1e-4
    drift, diagnostics = unnormalised_drift(generated, r.conformers, bandwidth, 1.0, **kw)
    assert torch.isfinite(drift).all() and diagnostics["positive_kernel_mass"] > 1e-5
    assert diagnostics["drift_rms"] > 1e-6
    # Central finite differences of the exact potential used by field().
    x, refs = generated[:4].double(), r.conformers.double()
    direction = torch.randn_like(x)
    direction /= direction.norm()
    def potential(q, reference, self_field):
        qd = (descriptor(q, r.bond_index, a, t, p) - mean) / scale
        rd = (descriptor(reference, r.bond_index, a, t, p) - mean) / scale
        kernel = torch.exp(-(qd[:, None] - rd[None]).square().mean(-1) / (2 * bandwidth**2))
        if self_field:
            kernel = kernel * (1 - torch.eye(len(q), dtype=q.dtype))
        return qd.shape[-1] * kernel.mean(1).sum()
    derivative_errors = {}
    for label, reference, is_self in (("attraction", refs, False), ("repulsion_potential", x.clone(), True)):
        gradient, _ = field(x, reference, bandwidth, is_self, **kw)
        numerical = (potential(x + 1e-5 * direction, reference, is_self) - potential(x - 1e-5 * direction, reference, is_self)) / 2e-5
        analytic = (gradient * direction).sum()
        error = float((numerical - analytic).abs())
        assert error < 1e-5, (label, error)
        derivative_errors[label] = error
    return {"molecule": r.identifier, "atoms": len(r.atomic_numbers), "conformers": len(refs),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "evaluation_scope": "overfit to the same training conformer ensemble, no generalization claim",
            "reference_geometry": validity, "reference_self_metrics": identity,
            "alternating_reference_halves_metrics": split_reference,
            "initial_metrics": symmetry_aware_cov_mat(generated, r.conformers, *metric_args),
            "initial_geometry": geometry_validity(generated, r.bond_index),
            "initial_drift": diagnostics, "bandwidth": bandwidth,
            "rotation_max_error": rotation_error, "finite_difference_errors": derivative_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("geom_qm9/configs/single_flexible.json"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--milestones", type=int, nargs="+", default=[1000, 5000, 10000])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        start_step = int(checkpoint["step"])
        del checkpoint
    if args.milestones != sorted(set(args.milestones)) or args.milestones[0] <= start_step:
        raise ValueError("Milestones must be strictly increasing and later than the resume step")
    prepare(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    report = audit(config)
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if args.audit_only:
        return
    if (output / "latest.pt").exists() and args.resume is None:
        raise RuntimeError("Run already has checkpoints; resume explicitly instead of overwriting")
    previous = args.resume
    for step in args.milestones:
        command = [sys.executable, "-u", "-m", "geom_qm9.train", "--config", str(args.config), "--steps", str(step), "--device", "mps"]
        if previous is not None:
            command += ["--resume", str(previous)]
        subprocess.run(command, check=True)
        previous = output / f"checkpoint_{step:07d}.pt"
        subprocess.run([sys.executable, "-u", "-m", "geom_qm9.evaluate", "--checkpoint", str(previous), "--split", "train",
                        "--samples-per-molecule", "100", "--seed", "20260908", "--device", "mps",
                        "--output", str(output / f"evaluation_{step:07d}.json")], check=True)


if __name__ == "__main__":
    main()
