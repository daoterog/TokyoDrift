"""Train a graph-conditioned, one-shot GEOM-QM9 conformer generator."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import MoleculeEnsemble, load_split
from .drift import median_bandwidth, unnormalised_drift
from .model import ConformerGenerator


def arguments() -> argparse.Namespace:
    """Parse the minimal experiment interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    """Choose CUDA, MPS, then CPU unless a device is requested explicitly."""
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_graph(records: list[MoleculeEnsemble], generator: torch.Generator) -> MoleculeEnsemble:
    """Choose molecules uniformly, preventing many-conformer molecules dominating training."""
    return records[int(torch.randint(len(records), (), generator=generator))]


def save(path: Path, model: ConformerGenerator, ema: ConformerGenerator, config: dict, step: int) -> None:
    """Save both training and sampling weights."""
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "config": config, "step": step}, path)


def main() -> None:
    """Run clean one-shot Gaussian drift regression over graph-conditioned ensembles."""
    args = arguments()
    config = json.loads(args.config.read_text())
    if args.steps is not None:
        config["training"]["steps"] = args.steps
    if args.output is not None:
        config["output"] = str(args.output)
    training, model_config = config["training"], config["model"]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = select_device(args.device)
    records = load_split(Path(config["data"]), "train", training.get("max_conformers"))
    model = ConformerGenerator(**model_config).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"device": str(device), "molecules": len(records), "parameters": sum(value.numel() for value in model.parameters())}))
    for step in range(1, int(training["steps"]) + 1):
        record = sample_graph(records, generator)
        atoms = record.atomic_numbers.to(device)
        edge_index, edge_order = record.bond_index.to(device), record.bond_order.to(device)
        references = record.conformers
        bandwidth = median_bandwidth(references) * float(training.get("bandwidth_scale", 1.0))
        selected = torch.randint(len(references), (int(training["positive_references"]),), generator=generator)
        positive = references[selected].to(device)
        coordinate_noise = float(training["coordinate_noise_scale"]) * torch.randn(int(training["batch_size"]), len(atoms), 3, device=device)
        generated = model(coordinate_noise, atoms, edge_index, edge_order)
        field, diagnostics = unnormalised_drift(generated.detach(), positive, bandwidth, float(training.get("repulsion", 1.0)))
        target = (generated.detach() + float(training["eta"]) * field).detach()
        loss = (generated - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 10.0)))
        optimizer.step()
        with torch.no_grad():
            for target_value, source_value in zip(ema.parameters(), model.parameters(), strict=True):
                target_value.lerp_(source_value, 1.0 - float(training.get("ema_decay", 0.999)))
        if step == 1 or step % int(training.get("log_every", 100)) == 0:
            print(json.dumps({"step": step, "molecule": record.identifier, "atoms": len(atoms), "bandwidth": bandwidth, "loss": float(loss), "gradient_norm": float(gradient_norm), **diagnostics}))
        if step % int(training.get("checkpoint_every", 1000)) == 0 or step == int(training["steps"]):
            save(output / f"checkpoint_{step:07d}.pt", model, ema, config, step)
            save(output / "latest.pt", model, ema, config, step)


if __name__ == "__main__":
    main()
