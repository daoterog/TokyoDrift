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
from .drift import angle_indices, descriptor_statistics, median_bandwidth, three_hop_pair_indices, torsion_indices, unnormalised_drift
from .model import ConformerGenerator


def arguments() -> argparse.Namespace:
    """Parse the minimal experiment interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
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


def save(path: Path, model: ConformerGenerator, ema: ConformerGenerator, config: dict, step: int,
         optimizer: torch.optim.Optimizer, generator: torch.Generator) -> None:
    """Save weights and continuation state, replacing the destination atomically."""
    state = {"model": model.state_dict(), "ema": ema.state_dict(), "config": config, "step": step,
             "optimizer": optimizer.state_dict(), "generator_rng": generator.get_state(),
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "numpy_rng": np.random.get_state()}
    device = next(model.parameters()).device
    if device.type == "mps":
        state["mps_rng"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    """Run clean one-shot Gaussian drift regression over graph-conditioned ensembles."""
    args = arguments()
    config = json.loads(args.config.read_text())
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = copy.deepcopy(checkpoint["config"])
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
    if config.get("molecule") is not None:
        records = [record for record in records if record.identifier == config["molecule"]]
        if len(records) != 1:
            raise ValueError("Single-molecule selection must match exactly one training record")
    model = ConformerGenerator(**model_config).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))
    start_step = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        start_step = int(checkpoint["step"])
        if int(training["steps"]) <= start_step:
            raise ValueError("--steps must exceed the checkpoint step (it is the total target).")
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "generator_rng" in checkpoint:
            generator.set_state(checkpoint["generator_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            random.setstate(checkpoint["python_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            if device.type == "mps" and "mps_rng" in checkpoint:
                torch.mps.set_rng_state(checkpoint["mps_rng"])
            elif device.type == "cuda" and "cuda_rng" in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        else:
            # Legacy checkpoints did not save RNG state; use a fresh stream.
            torch.manual_seed(seed + start_step)
            generator.manual_seed(seed + start_step)
        print(json.dumps({"resumed_from": str(args.resume), "step": start_step,
                          "optimizer_restored": "optimizer" in checkpoint,
                          "rng_restored": "generator_rng" in checkpoint}))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"device": str(device), "molecules": len(records), "parameters": sum(value.numel() for value in model.parameters())}))
    for step in range(start_step + 1, int(training["steps"]) + 1):
        record = sample_graph(records, generator)
        atoms = record.atomic_numbers.to(device)
        edge_index, edge_order = record.bond_index.to(device), record.bond_order.to(device)
        angles = angle_indices(edge_index, len(atoms))
        torsions = torsion_indices(edge_index, len(atoms))
        three_hop_pairs = three_hop_pair_indices(edge_index, len(atoms))
        references = record.conformers
        descriptor_mean, descriptor_scale = descriptor_statistics(
            references, edge_index.cpu(), angles.cpu(), torsions.cpu(), three_hop_pairs.cpu(),
            minimum_scale=float(training.get("internal_coordinate_scale_floor", 0.1)),
        )
        bandwidth = median_bandwidth(
            references,
            bond_index=edge_index.cpu(),
            angles=angles.cpu(),
            torsions=torsions.cpu(),
            three_hop_pairs=three_hop_pairs.cpu(),
            mean=descriptor_mean,
            scale=descriptor_scale,
        ) * float(training.get("bandwidth_scale", 1.0))
        requested_references = training.get("positive_references")
        if requested_references is None or int(requested_references) >= len(references):
            positive = references.to(device)
        else:
            selected = torch.randperm(len(references), generator=generator)[: int(requested_references)]
            positive = references[selected].to(device)
        coordinate_noise = float(training["coordinate_noise_scale"]) * torch.randn(int(training["batch_size"]), len(atoms), 3, device=device)
        generated = model(coordinate_noise, atoms, edge_index, edge_order)
        field, diagnostics = unnormalised_drift(
            generated.detach(),
            positive,
            bandwidth,
            float(training.get("repulsion", 1.0)),
            bond_index=edge_index,
            angles=angles,
            torsions=torsions,
            three_hop_pairs=three_hop_pairs,
            mean=descriptor_mean.to(device),
            scale=descriptor_scale.to(device),
        )
        target = (generated.detach() + float(training["eta"]) * field).detach()
        loss = (generated - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 10.0)))
        optimizer.step()
        with torch.no_grad():
            for target_value, source_value in zip(ema.parameters(), model.parameters(), strict=True):
                target_value.lerp_(source_value, 1.0 - float(training.get("ema_decay", 0.999)))
        if step == start_step + 1 or step % int(training.get("log_every", 100)) == 0:
            coordinate_scales = model.coordinate_update_scales().detach()
            print(json.dumps({
                "step": step, "molecule": record.identifier, "atoms": len(atoms), "bandwidth": bandwidth,
                "loss": float(loss.detach()), "gradient_norm": float(gradient_norm.detach()),
                "coordinate_scale_mean": float(coordinate_scales.mean()),
                "coordinate_scale_max": float(coordinate_scales.abs().max()),
                **diagnostics,
            }))
        if step % int(training.get("checkpoint_every", 1000)) == 0 or step == int(training["steps"]):
            save(output / f"checkpoint_{step:07d}.pt", model, ema, config, step, optimizer, generator)
            save(output / "latest.pt", model, ema, config, step, optimizer, generator)


if __name__ == "__main__":
    main()
