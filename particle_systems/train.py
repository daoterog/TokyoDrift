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
from .systems import ParticleSystem, get_system
from .unnormalized_drifting import DirectCoordinateDrift, median_bandwidth
from .validation import ValidationEvaluator


@dataclass
class EnergyStratifiedReferenceSampler:
    """Energy partitions used to build balanced, without-replacement epoch batches."""

    samples: torch.Tensor
    bin_indices: list[torch.Tensor]
    bin_masses: torch.Tensor
    boundaries: torch.Tensor

    @classmethod
    def build(
        cls, samples: torch.Tensor, system: ParticleSystem, quantiles: list[float]
    ) -> EnergyStratifiedReferenceSampler:
        """Partition reference indices at the requested energy quantiles."""
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

    def epoch_batches(
        self, batch_size: int, generator: torch.Generator
    ) -> list[torch.Tensor]:
        """Return stratified batches containing every reference exactly once."""
        batch_count = math.ceil(len(self.samples) / batch_size)
        shuffled_bins = [
            indices[torch.randperm(len(indices), generator=generator)]
            for indices in self.bin_indices
        ]
        split_bins = [torch.tensor_split(indices, batch_count) for indices in shuffled_bins]
        batches = []
        for batch_index in range(batch_count):
            indices = torch.cat([parts[batch_index] for parts in split_bins])
            batches.append(indices[torch.randperm(len(indices), generator=generator)])
        return batches


def epoch_reference_batches(
    sample_count: int, batch_size: int, generator: torch.Generator
) -> list[torch.Tensor]:
    """Shuffle references and split them into one complete, without-replacement epoch."""
    if sample_count <= 0:
        raise ValueError("at least one training reference is required")
    if batch_size <= 0:
        raise ValueError("positive_references must be positive")
    return list(torch.randperm(sample_count, generator=generator).split(batch_size))


def bandwidth_scale(training: dict, epoch: int) -> float:
    """Return the cosine-annealed coordinate bandwidth multiplier for an epoch."""
    start_scale = float(training.get("bandwidth_scale_start", training.get("bandwidth_scale", 1.0)))
    end_scale = float(training.get("bandwidth_scale_end", start_scale))
    anneal_start = int(training.get("bandwidth_anneal_start", 0))
    anneal_end = int(training.get("bandwidth_anneal_end", anneal_start))
    if start_scale <= 0 or end_scale <= 0:
        raise ValueError("bandwidth scales must be positive")
    if anneal_end < anneal_start:
        raise ValueError("bandwidth_anneal_end must not precede bandwidth_anneal_start")
    if epoch <= anneal_start:
        return start_scale
    if epoch >= anneal_end or anneal_end == anneal_start:
        return end_scale
    progress = (epoch - anneal_start) / (anneal_end - anneal_start)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end_scale + (start_scale - end_scale) * cosine_weight


def learning_rate(training: dict, epoch: int) -> float:
    """Return the cosine-decayed learning rate for an epoch."""
    start = float(training["learning_rate"])
    end = float(training.get("learning_rate_end", start))
    decay_start = int(training.get("learning_rate_decay_start", 0))
    decay_end = int(training.get("learning_rate_decay_end", decay_start))
    if start <= 0 or end <= 0:
        raise ValueError("learning rates must be positive")
    if decay_end < decay_start:
        raise ValueError("learning_rate_decay_end must not precede its start")
    if epoch <= decay_start:
        return start
    if epoch >= decay_end or decay_end == decay_start:
        return end
    progress = (epoch - decay_start) / (decay_end - decay_start)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end + (start - end) * cosine_weight


def arguments() -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an unnormalized-drift generator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument(
        "--positive-references",
        type=int,
        default=None,
        help="Override the reference-batch size used to partition each full epoch.",
    )
    parser.add_argument(
        "--positive-reference-sampling",
        choices=("shuffled", "energy-stratified"),
        default=None,
        help="Shuffle each full epoch globally or construct energy-stratified epoch batches.",
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
    epoch: int,
    global_step: int,
    generator: torch.Generator,
) -> None:
    """Write a restartable model and optimizer checkpoint."""
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "bandwidth": bandwidth,
            "epoch": epoch,
            "global_step": global_step,
            # Retained for evaluation tools and old checkpoint consumers.
            "step": global_step,
            "reference_generator_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            **({"cuda_rng": torch.cuda.get_rng_state_all()} if torch.cuda.is_available() else {}),
            **(
                {"mps_rng": torch.mps.get_rng_state()}
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else {}
            ),
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
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
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
    if "steps" in training:
        raise ValueError("training.steps has been replaced by training.epochs")
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
    reference_batch_size = int(training["positive_references"])
    if reference_batch_size <= 0:
        raise ValueError("positive_references must be positive")
    epochs = int(training["epochs"])
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    batches_per_epoch = math.ceil(len(train_data) / reference_batch_size)
    generator = torch.Generator().manual_seed(seed)
    start_epoch = 1
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "epoch" not in checkpoint:
            raise ValueError(
                "cannot resume a step-based checkpoint in full-epoch training; start a new run"
            )
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
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        generator.set_state(checkpoint["reference_generator_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        if device.type == "cuda" and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        elif device.type == "mps" and "mps_rng" in checkpoint:
            torch.mps.set_rng_state(checkpoint["mps_rng"])
    bandwidth_config = training["bandwidth"]
    base_bandwidth = (
        median_bandwidth(train_data)
        if bandwidth_config == "auto"
        else float(bandwidth_config)
    )
    initial_scale = bandwidth_scale(training, 1)
    final_scale = bandwidth_scale(training, epochs)
    initial_bandwidth = base_bandwidth * initial_scale
    final_bandwidth = base_bandwidth * final_scale
    drift = DirectCoordinateDrift(initial_bandwidth).to(device)
    reference_radius = float(train_data.square().sum(dim=-1).mean().sqrt())
    minimum_radius = reference_radius * float(training.get("min_radius_fraction", 0.0))
    reference_sampling = str(training.get("positive_reference_sampling", "shuffled"))
    quantiles = list(training.get("positive_reference_energy_quantiles", (0.5, 0.9, 0.99)))
    stratified_references = (
        EnergyStratifiedReferenceSampler.build(train_data, system, quantiles)
        if reference_sampling == "energy-stratified"
        else None
    )
    if reference_sampling not in {"shuffled", "energy-stratified"}:
        raise ValueError(
            "positive_reference_sampling must be 'shuffled' or 'energy-stratified'"
        )
    outdir = Path(config["output"])
    outdir.mkdir(parents=True, exist_ok=True)
    resolved_parameters = {
        **config,
        "training_samples": len(train_data),
        "batches_per_epoch": batches_per_epoch,
        "optimizer_steps": epochs * batches_per_epoch,
        "base_bandwidth": base_bandwidth,
        "initial_bandwidth": initial_bandwidth,
        "final_bandwidth": final_bandwidth,
        "reference_radius": reference_radius,
        "minimum_radius": minimum_radius,
        "positive_reference_sampler": (
            {
                "type": "energy-stratified-full-epoch",
                "energy_quantiles": quantiles,
                "energy_boundaries": stratified_references.boundaries.tolist(),
                "bin_masses": stratified_references.bin_masses.tolist(),
            }
            if stratified_references is not None
            else {"type": "shuffled-full-epoch"}
        ),
    }
    serialized_parameters = json.dumps(resolved_parameters, indent=2) + "\n"
    (outdir / "parameters.json").write_text(serialized_parameters)
    # Keep the established filename for compatibility with existing tooling.
    (outdir / "resolved_config.json").write_text(serialized_parameters)

    batch_size = int(training["batch_size"])
    if start_epoch > epochs:
        raise ValueError(
            f"training ends at epoch {epochs}, before resumed epoch {start_epoch}; "
            "increase training.epochs or pass --epochs"
        )
    coordinate_scale = float(training["coordinate_noise_scale"])
    feature_dim = int(config["model"]["feature_dim"])
    eta = float(training["eta"])
    repulsion = float(training["repulsion"])
    log_every = int(training["log_every"])
    checkpoint_every = int(training["checkpoint_every"])
    validation_every = int(training.get("validation_every", checkpoint_every))
    best_validation_score = float("inf")
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
                "training_samples": len(train_data),
                "batches_per_epoch": batches_per_epoch,
                "epochs": epochs,
                "optimizer_steps": epochs * batches_per_epoch,
                "resume": str(args.resume) if args.resume is not None else None,
                "start_epoch": start_epoch,
                "global_step": global_step,
            }
        )
    )

    for epoch in range(start_epoch, epochs + 1):
        current_learning_rate = learning_rate(training, epoch)
        for group in optimizer.param_groups:
            group["lr"] = current_learning_rate
        current_scale = bandwidth_scale(training, epoch)
        current_bandwidth = base_bandwidth * current_scale
        drift.bandwidth = current_bandwidth
        if stratified_references is None:
            reference_batches = epoch_reference_batches(
                len(train_data), reference_batch_size, generator
            )
        else:
            reference_batches = stratified_references.epoch_batches(
                reference_batch_size, generator
            )

        epoch_totals: dict[str, float] = {}
        for batch_index, reference_indices in enumerate(reference_batches, start=1):
            positive = train_data[reference_indices].to(device)
            coordinate_noise = coordinate_scale * torch.randn(
                batch_size, system.particles, system.dimensions, device=device
            )
            feature_noise = torch.randn(batch_size, system.particles, feature_dim, device=device)
            generated = model(coordinate_noise, feature_noise)
            generated_radius = generated.detach().square().sum(dim=-1).mean().sqrt()
            if minimum_radius and float(generated_radius) < minimum_radius:
                raise RuntimeError(
                    f"generator collapse detected at epoch {epoch}, batch {batch_index}: radius "
                    f"{float(generated_radius):.6g} is below {minimum_radius:.6g}; "
                    "do not continue this run"
                )
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
            global_step += 1

            with torch.no_grad():
                energy = system.energy(generated)
            batch_metrics = {
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "energy_mean": float(energy.mean().detach()),
                "generated_radius": float(generated_radius),
                **{name: float(value) for name, value in drift_metrics.items()},
            }
            for name, value in batch_metrics.items():
                epoch_totals[name] = epoch_totals.get(name, 0.0) + value

        if epoch == start_epoch or epoch % log_every == 0:
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "batches": len(reference_batches),
                **{
                    name: value / len(reference_batches)
                    for name, value in epoch_totals.items()
                },
                "bandwidth_scale": current_scale,
                "bandwidth": current_bandwidth,
                "learning_rate": current_learning_rate,
                "reference_radius": reference_radius,
            }
            print(json.dumps(record))
        if validator is not None and (epoch % validation_every == 0 or epoch == epochs):
            validation_metrics = validator.evaluate(ema, drift, device)
            validation_record = {
                "epoch": epoch,
                "global_step": global_step,
                "validation": validation_metrics,
            }
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
                    epoch,
                    global_step,
                    generator,
                )
                with (outdir / "best_validation.json").open("w") as stream:
                    json.dump(validation_record, stream, indent=2)
        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(
                outdir / f"checkpoint_epoch_{epoch:07d}.pt",
                model,
                ema,
                optimizer,
                config,
                current_bandwidth,
                epoch,
                global_step,
                generator,
            )
            save_checkpoint(
                outdir / "latest.pt",
                model,
                ema,
                optimizer,
                config,
                current_bandwidth,
                epoch,
                global_step,
                generator,
            )


if __name__ == "__main__":
    main()
