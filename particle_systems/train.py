"""Train an equivariant direct generator with raw unnormalized drift."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from .drift import UnnormalizedDrift, median_bandwidth
from .io import build_model, device_summary, load_config, load_dataset, select_device
from .systems import get_system


def arguments() -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a raw unnormalized-drift generator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None, help="Override config steps.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--output", type=Path, default=None, help="Override output directory.")
    return parser.parse_args()


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Update an exponential-moving-average model in place."""
    with torch.no_grad():
        for ema_value, value in zip(
            ema.state_dict().values(), model.state_dict().values(), strict=True
        ):
            if ema_value.is_floating_point():
                ema_value.lerp_(value.detach(), 1.0 - decay)
            else:
                ema_value.copy_(value)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    bandwidth: float,
    step: int,
) -> None:
    """Write a restartable model and optimizer checkpoint."""
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "bandwidth": bandwidth,
            "step": step,
        },
        path,
    )


def main() -> None:
    """Run unnormalized-drift training from a JSON configuration."""
    args = arguments()
    config = load_config(args.config)
    if args.steps is not None:
        config["training"]["steps"] = args.steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output is not None:
        config["output"] = str(args.output)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = select_device(args.device)
    system = get_system(config["system"])
    train_data, metadata = load_dataset(Path(config["data"]), "train")
    if metadata["system"] != system.name:
        raise ValueError("config and dataset systems differ")
    model = build_model(config).to(device)
    ema = copy.deepcopy(model).eval()
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    bandwidth_config = training["bandwidth"]
    bandwidth = (
        median_bandwidth(train_data) if bandwidth_config == "auto" else float(bandwidth_config)
    )
    drift = UnnormalizedDrift(bandwidth).to(device)
    outdir = Path(config["output"])
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "resolved_config.json").open("w") as stream:
        json.dump({**config, "resolved_bandwidth": bandwidth}, stream, indent=2)

    batch_size = int(training["batch_size"])
    references = int(training["positive_references"])
    steps = int(training["steps"])
    coordinate_scale = float(training["coordinate_noise_scale"])
    feature_dim = int(config["model"]["feature_dim"])
    eta = float(training["eta"])
    repulsion = float(training["repulsion"])
    log_every = int(training["log_every"])
    checkpoint_every = int(training["checkpoint_every"])
    generator = torch.Generator().manual_seed(seed)
    print(
        json.dumps(
            {
                "system": system.name,
                **device_summary(device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "bandwidth": bandwidth,
            }
        )
    )

    for step in range(1, steps + 1):
        reference_indices = torch.randint(len(train_data), (references,), generator=generator)
        positive = train_data[reference_indices].to(device)
        coordinate_noise = coordinate_scale * torch.randn(
            batch_size, system.particles, system.dimensions, device=device
        )
        feature_noise = torch.randn(batch_size, system.particles, feature_dim, device=device)
        generated = model(coordinate_noise, feature_noise)
        field, drift_metrics = drift(
            generated.detach(),
            positive,
            generated.detach(),
            torch.arange(batch_size, device=device),
            repulsion,
        )
        target = (generated.detach() + eta * field).detach()
        loss = (generated - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip"])
        )
        optimizer.step()
        update_ema(ema, model, float(training["ema_decay"]))

        if step == 1 or step % log_every == 0:
            with torch.no_grad():
                energy = system.energy(generated)
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "energy_mean": float(energy.mean().detach()),
                **{name: float(value) for name, value in drift_metrics.items()},
            }
            print(json.dumps(record))
        if step % checkpoint_every == 0 or step == steps:
            save_checkpoint(
                outdir / f"checkpoint_{step:07d}.pt",
                model,
                ema,
                optimizer,
                config,
                bandwidth,
                step,
            )
            save_checkpoint(outdir / "latest.pt", model, ema, optimizer, config, bandwidth, step)


if __name__ == "__main__":
    main()
