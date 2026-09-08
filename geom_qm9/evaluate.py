"""Evaluate one-shot GEOM-QM9 conformers on held-out molecular graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import load_split
from .metrics import ensemble_metrics, geometry_validity, symmetry_aware_cov_mat
from .model import ConformerGenerator
from .train import select_device


def main() -> None:
    """Sample held-out graph-conditioned ensembles and aggregate validity/COV/MAT proxies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--samples-per-molecule", type=int, default=100)
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = select_device(args.device)
    model = ConformerGenerator(**config["model"]).to(device).eval()
    model.load_state_dict(checkpoint["ema"])
    records = load_split(Path(config["data"]), args.split, config["training"].get("max_conformers"))
    if config.get("molecule") is not None:
        records = [record for record in records if record.identifier == config["molecule"]]
        if len(records) != 1:
            raise ValueError("Single-molecule evaluation must match exactly one record in the selected split")
    if args.max_molecules is not None:
        records = records[: args.max_molecules]
    results = []
    with torch.no_grad():
        for record in records:
            positions = model(
                float(config["training"]["coordinate_noise_scale"]) * torch.randn(args.samples_per_molecule, len(record.atomic_numbers), 3, device=device),
                record.atomic_numbers.to(device), record.bond_index.to(device), record.bond_order.to(device),
            ).cpu()
            metrics = ensemble_metrics(positions, record.conformers)
            metrics.update(geometry_validity(positions, record.bond_index))
            metrics.update(
                symmetry_aware_cov_mat(
                    positions,
                    record.conformers,
                    record.atomic_numbers,
                    record.bond_index,
                    record.bond_order,
                )
            )
            results.append(metrics)
    # Some valid archive entries contain no heavy-atom bond, for which local
    # bond-length statistics are undefined. Aggregate every reported metric
    # over precisely the molecules that define it.
    keys = set().union(*(item.keys() for item in results))
    aggregate = {
        key: sum(item[key] for item in results if key in item) / sum(key in item for item in results)
        for key in keys
    }
    payload = {"checkpoint_step": checkpoint["step"], "molecules": len(results), "metrics": aggregate, "per_molecule": results,
               "split": args.split, "sampling_seed": args.seed, "samples_per_molecule": args.samples_per_molecule,
               "identifiers": [record.identifier for record in records]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["metrics"]))


if __name__ == "__main__":
    main()
