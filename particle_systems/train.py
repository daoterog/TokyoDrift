"""Train an equivariant generator with alignment-free unnormalized drift."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .io import build_model, device_summary, load_config, load_dataset, select_device
from .systems import get_system
from .unnormalized_drifting import DirectCoordinateDrift, median_bandwidth
from .validation import ValidationEvaluator


@dataclass
class EnergyStratifiedReferenceSampler:
    """Unbiased, low-variance target-reference sampling across energy strata."""

    samples: torch.Tensor
    bin_indices: list[torch.Tensor]
    bin_masses: torch.Tensor
    boundaries: torch.Tensor

    @classmethod
    def build(
        cls, samples: torch.Tensor, system, quantiles: list[float]
    ) -> "EnergyStratifiedReferenceSampler":
        if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
            raise ValueError("stratification quantiles must lie strictly between zero and one")
        if quantiles != sorted(set(quantiles)):
            raise ValueError("stratification quantiles must be strictly increasing")
        energies = system.energy(samples)
        boundaries = torch.quantile(energies, torch.tensor(quantiles, dtype=energies.dtype))
        labels = torch.bucketize(energies, boundaries, right=True)
        indices = [torch.where(labels == bin_index)[0] for bin_index in range(len(quantiles) + 1)]
        if any(len(value) == 0 for value in indices):
            raise ValueError("energy stratification produced an empty bin")
        masses = torch.tensor([len(value) / len(samples) for value in indices], dtype=samples.dtype)
        return cls(samples=samples, bin_indices=indices, bin_masses=masses, boundaries=boundaries)

    def sample(self, count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        if count < len(self.bin_indices):
            raise ValueError("positive reference count must be at least the number of strata")
        counts = torch.full((len(self.bin_indices),), count // len(self.bin_indices), dtype=torch.long)
        counts[: count % len(self.bin_indices)] += 1
        selected: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for indices, mass, bin_count in zip(self.bin_indices, self.bin_masses, counts, strict=True):
            choice = indices[torch.randint(len(indices), (int(bin_count),), generator=generator)]
            selected.append(choice)
            weights.append(
                torch.full((int(bin_count),), float(mass / bin_count), dtype=self.samples.dtype)
            )
        return self.samples[torch.cat(selected)], torch.cat(weights)


def bandwidth_scale(training: dict, step: int) -> float:
    """Return the cosine-annealed coordinate bandwidth multiplier."""
    start_scale = float(training.get("bandwidth_scale_start", training.get("bandwidth_scale", 1.0)))
    end_scale = float(training.get("bandwidth_scale_end", start_scale))
    anneal_start = int(training.get("bandwidth_anneal_start", 0))
    anneal_end = int(training.get("bandwidth_anneal_end", anneal_start))
    if start_scale <= 0 or end_scale <= 0:
        raise ValueError("bandwidth scales must be positive")
    if anneal_end < anneal_start:
        raise ValueError("bandwidth_anneal_end must not precede bandwidth_anneal_start")
    if step <= anneal_start:
        return start_scale
    if step >= anneal_end or anneal_end == anneal_start:
        return end_scale
    progress = (step - anneal_start) / (anneal_end - anneal_start)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end_scale + (start_scale - end_scale) * cosine_weight


def learning_rate(training: dict, step: int) -> float:
    """Return a cosine-decayed learning rate with constant-rate compatibility."""
    start = float(training["learning_rate"])
    end = float(training.get("learning_rate_end", start))
    decay_start = int(training.get("learning_rate_decay_start", 0))
    decay_end = int(training.get("learning_rate_decay_end", decay_start))
    if start <= 0 or end <= 0:
        raise ValueError("learning rates must be positive")
    if decay_end < decay_start:
        raise ValueError("learning_rate_decay_end must not precede its start")
    if step <= decay_start:
        return start
    if step >= decay_end or decay_end == decay_start:
        return end
    progress = (step - decay_start) / (decay_end - decay_start)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end + (start - end) * cosine_weight


def arguments() -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an unnormalized-drift generator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None, help="Override config steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument(
        "--positive-references",
        type=int,
        default=None,
        help="Override the number of target samples used in the attractive field estimate.",
    )
    parser.add_argument(
        "--positive-reference-sampling",
        choices=("iid", "energy-stratified"),
        default=None,
        help="Sample target references i.i.d. or from energy strata with unbiased weights.",
    )
    parser.add_argument(
        "--positive-reference-energy-quantiles",
        type=float,
        nargs="+",
        default=None,
        help="Strictly increasing energy quantiles defining stratification boundaries.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--output", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume a training checkpoint.")
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
    drift_definition = {
        "space": "particle_coordinates",
        "kernel": "gaussian",
        "normalized": False,
    }
    configured_drift = config.setdefault("drift", drift_definition)
    if configured_drift != drift_definition:
        raise ValueError(f"this training pipeline requires drift={drift_definition}")
    if args.steps is not None:
        config["training"]["steps"] = args.steps
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.positive_references is not None:
        config["training"]["positive_references"] = args.positive_references
    if args.positive_reference_sampling is not None:
        config["training"]["positive_reference_sampling"] = args.positive_reference_sampling
    if args.positive_reference_energy_quantiles is not None:
        config["training"]["positive_reference_energy_quantiles"] = args.positive_reference_energy_quantiles
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
    validation_holdout = int(training.get("validation_holdout", 0))
    validator = None
    if validation_holdout:
        if validation_holdout >= len(train_data):
            raise ValueError("validation_holdout must be smaller than the training split")
        validation_data = train_data[-validation_holdout:]
        train_data = train_data[:-validation_holdout]
        validator = ValidationEvaluator.build(
            validation_data,
            system,
            int(config["model"]["feature_dim"]),
            float(training["coordinate_noise_scale"]),
            training,
            seed,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    start_step = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint["config"]
        if checkpoint_config["system"] != config["system"]:
            raise ValueError("resume checkpoint and config systems differ")
        if checkpoint_config["model"] != config["model"]:
            raise ValueError("resume checkpoint and config model definitions differ")
        if checkpoint_config.get("drift") != drift_definition:
            raise ValueError("resume checkpoint was not trained with direct-coordinate drift")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = float(training["learning_rate"])
            group["weight_decay"] = float(training["weight_decay"])
        start_step = int(checkpoint["step"]) + 1
    bandwidth_config = training["bandwidth"]
    base_bandwidth = (
        median_bandwidth(train_data)
        if bandwidth_config == "auto"
        else float(bandwidth_config)
    )
    initial_scale = bandwidth_scale(training, 1)
    final_scale = bandwidth_scale(training, int(training["steps"]))
    initial_bandwidth = base_bandwidth * initial_scale
    final_bandwidth = base_bandwidth * final_scale
    drift = DirectCoordinateDrift(initial_bandwidth).to(device)
    reference_radius = float(train_data.square().sum(dim=-1).mean().sqrt())
    minimum_radius = reference_radius * float(training.get("min_radius_fraction", 0.0))
    reference_sampling = str(training.get("positive_reference_sampling", "iid"))
    quantiles = list(training.get("positive_reference_energy_quantiles", (0.5, 0.9, 0.99)))
    stratified_references = (
        EnergyStratifiedReferenceSampler.build(train_data, system, quantiles)
        if reference_sampling == "energy-stratified"
        else None
    )
    if reference_sampling not in {"iid", "energy-stratified"}:
        raise ValueError("positive_reference_sampling must be 'iid' or 'energy-stratified'")
    outdir = Path(config["output"])
    outdir.mkdir(parents=True, exist_ok=True)
    resolved_parameters = {
        **config,
        "base_bandwidth": base_bandwidth,
        "initial_bandwidth": initial_bandwidth,
        "final_bandwidth": final_bandwidth,
        "reference_radius": reference_radius,
        "minimum_radius": minimum_radius,
        "positive_reference_sampler": (
            {
                "type": "energy-stratified",
                "energy_quantiles": quantiles,
                "energy_boundaries": stratified_references.boundaries.tolist(),
                "bin_masses": stratified_references.bin_masses.tolist(),
            }
            if stratified_references is not None
            else {"type": "iid"}
        ),
    }
    serialized_parameters = json.dumps(resolved_parameters, indent=2) + "\n"
    (outdir / "parameters.json").write_text(serialized_parameters)
    # Keep the established filename for compatibility with existing tooling.
    (outdir / "resolved_config.json").write_text(serialized_parameters)

    batch_size = int(training["batch_size"])
    references = int(training["positive_references"])
    steps = int(training["steps"])
    if start_step > steps:
        raise ValueError(
            f"training ends at step {steps}, before resumed step {start_step}; "
            "increase training.steps or pass --steps"
        )
    coordinate_scale = float(training["coordinate_noise_scale"])
    feature_dim = int(config["model"]["feature_dim"])
    eta = float(training["eta"])
    repulsion = float(training["repulsion"])
    log_every = int(training["log_every"])
    checkpoint_every = int(training["checkpoint_every"])
    validation_every = int(training.get("validation_every", checkpoint_every))
    best_validation_score = float("inf")
    generator = torch.Generator().manual_seed(seed)
    print(
        json.dumps(
            {
                "system": system.name,
                **device_summary(device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "base_bandwidth": base_bandwidth,
                "initial_bandwidth": initial_bandwidth,
                "final_bandwidth": final_bandwidth,
                "drift_space": "particle_coordinates",
                "reference_radius": reference_radius,
                "minimum_radius": minimum_radius,
                "resume": str(args.resume) if args.resume is not None else None,
                "start_step": start_step,
            }
        )
    )

    for step in range(start_step, steps + 1):
        current_learning_rate = learning_rate(training, step)
        for group in optimizer.param_groups:
            group["lr"] = current_learning_rate
        current_scale = bandwidth_scale(training, step)
        current_bandwidth = base_bandwidth * current_scale
        drift.bandwidth = current_bandwidth
        if stratified_references is None:
            reference_indices = torch.randint(len(train_data), (references,), generator=generator)
            positive = train_data[reference_indices].to(device)
            positive_weights = None
        else:
            positive, positive_weights = stratified_references.sample(references, generator)
            positive = positive.to(device)
            positive_weights = positive_weights.to(device)
        coordinate_noise = coordinate_scale * torch.randn(
            batch_size, system.particles, system.dimensions, device=device
        )
        feature_noise = torch.randn(batch_size, system.particles, feature_dim, device=device)
        generated = model(coordinate_noise, feature_noise)
        generated_radius = generated.detach().square().sum(dim=-1).mean().sqrt()
        if minimum_radius and float(generated_radius) < minimum_radius:
            raise RuntimeError(
                f"generator collapse detected at step {step}: radius "
                f"{float(generated_radius):.6g} is below {minimum_radius:.6g}; "
                "do not continue this run"
            )
        field, drift_metrics = drift(
            generated.detach(),
            positive,
            generated.detach(),
            torch.arange(batch_size, device=device),
            repulsion,
            positive_weights,
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
                "bandwidth_scale": current_scale,
                "bandwidth": current_bandwidth,
                "learning_rate": current_learning_rate,
                "generated_radius": float(generated_radius),
                "reference_radius": reference_radius,
                **{name: float(value) for name, value in drift_metrics.items()},
            }
            print(json.dumps(record))
        if validator is not None and (step % validation_every == 0 or step == steps):
            validation_metrics = validator.evaluate(ema, drift, device)
            validation_record = {"step": step, "validation": validation_metrics}
            print(json.dumps(validation_record))
            with (outdir / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(validation_record) + "\n")
            score = validation_metrics["energy_wasserstein_1"]
            if score < best_validation_score:
                best_validation_score = score
                save_checkpoint(
                    outdir / "best_validation.pt",
                    model,
                    ema,
                    optimizer,
                    config,
                    current_bandwidth,
                    step,
                )
                with (outdir / "best_validation.json").open("w") as stream:
                    json.dump(validation_record, stream, indent=2)
        if step % checkpoint_every == 0 or step == steps:
            save_checkpoint(
                outdir / f"checkpoint_{step:07d}.pt",
                model,
                ema,
                optimizer,
                config,
                current_bandwidth,
                step,
            )
            save_checkpoint(
                outdir / "latest.pt", model, ema, optimizer, config, current_bandwidth, step
            )


if __name__ == "__main__":
    main()
