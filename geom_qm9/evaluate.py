"""Evaluate one-shot GEOM-QM9 conformers on held-out molecular graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import load_split
from .metrics import ensemble_metrics, geometry_validity
from .model import ConformerGenerator
from .train import select_device


def main() -> None:
    """Sample held-out graph-conditioned ensembles and aggregate validity/COV/MAT proxies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--samples-per-molecule", type=int, default=100)
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = select_device(args.device)
    model = ConformerGenerator(**config["model"]).to(device).eval()
    model.load_state_dict(checkpoint["ema"])
    records = load_split(Path(config["data"]), args.split, config["training"].get("max_conformers"))
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
            results.append(metrics)
    aggregate = {key: sum(item[key] for item in results) / len(results) for key in results[0]}
    payload = {"checkpoint_step": checkpoint["step"], "molecules": len(results), "metrics": aggregate, "per_molecule": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["metrics"]))


if __name__ == "__main__":
    main()
