"""Plot a PCA parameter-space loss landscape across training checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters

from .drift import UnnormalizedDrift
from .io import build_model, load_dataset, select_device
from .systems import get_system


def arguments() -> argparse.Namespace:
    """Parse loss-landscape command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot a PCA loss landscape.")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--center", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--references", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--state", choices=("ema", "model"), default="ema")
    return parser.parse_args()


def parameter_vector(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten checkpoint tensors corresponding to trainable model parameters."""
    return torch.cat([state[name].detach().cpu().flatten() for name, _ in model.named_parameters()])


def fixed_batch_loss(
    model: torch.nn.Module,
    drift: UnnormalizedDrift,
    coordinate_noise: torch.Tensor,
    feature_noise: torch.Tensor,
    references: torch.Tensor,
    eta: float,
    repulsion: float,
) -> float:
    """Evaluate the deterministic detached-target loss used during training."""
    with torch.no_grad():
        generated = model(coordinate_noise, feature_noise)
    field, _ = drift(
        generated,
        references,
        generated,
        torch.arange(generated.shape[0], device=generated.device),
        repulsion,
    )
    return float((eta * field).square().mean())


def main() -> None:
    """Evaluate and save a PCA loss surface and its underlying values."""
    args = arguments()
    if args.grid_size < 3:
        raise ValueError("grid-size must be at least 3")
    device = select_device(args.device)
    center_checkpoint = torch.load(args.center, map_location="cpu", weights_only=False)
    config = center_checkpoint["config"]
    system = get_system(config["system"])
    model = build_model(config).to(device).eval()
    checkpoint_values = [
        torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints
    ]
    vectors = torch.stack(
        [parameter_vector(model, checkpoint[args.state]) for checkpoint in checkpoint_values]
    ).double()
    center_vector = parameter_vector(model, center_checkpoint[args.state]).double()
    centered_trajectory = vectors - vectors.mean(dim=0, keepdim=True)
    _, singular_values, directions = torch.linalg.svd(centered_trajectory, full_matrices=False)
    pc1, pc2 = directions[:2]
    explained = singular_values.square()
    explained = explained / explained.sum()
    trajectory = torch.stack(
        ((vectors - center_vector) @ pc1, (vectors - center_vector) @ pc2), dim=1
    )

    low = torch.minimum(trajectory.min(dim=0).values, torch.zeros(2))
    high = torch.maximum(trajectory.max(dim=0).values, torch.zeros(2))
    padding = (high - low).clamp_min((high[0] - low[0]).clamp_min(1e-6) * 0.15) * 0.15
    axes = [
        torch.linspace(
            float(low[index] - padding[index]),
            float(high[index] + padding[index]),
            args.grid_size,
        )
        for index in range(2)
    ]

    training = config["training"]
    train_data, _ = load_dataset(Path(config["data"]), "train")
    generator = torch.Generator().manual_seed(args.seed)
    reference_indices = torch.randperm(len(train_data), generator=generator)[: args.references]
    references = train_data[reference_indices].to(device)
    coordinate_noise = float(training["coordinate_noise_scale"]) * torch.randn(
        args.batch_size,
        system.particles,
        system.dimensions,
        generator=generator,
    ).to(device)
    feature_noise = torch.randn(
        args.batch_size,
        system.particles,
        int(config["model"]["feature_dim"]),
        generator=generator,
    ).to(device)
    drift = UnnormalizedDrift(float(center_checkpoint["bandwidth"])).to(device)
    eta = float(training["eta"])
    repulsion = float(training["repulsion"])
    surface = torch.empty(args.grid_size, args.grid_size)
    pc1_device = pc1.to(device=device, dtype=next(model.parameters()).dtype)
    pc2_device = pc2.to(device=device, dtype=next(model.parameters()).dtype)
    center_device = center_vector.to(device=device, dtype=next(model.parameters()).dtype)
    for row, y_value in enumerate(axes[1]):
        for column, x_value in enumerate(axes[0]):
            candidate = center_device + float(x_value) * pc1_device + float(y_value) * pc2_device
            vector_to_parameters(candidate, model.parameters())
            surface[row, column] = fixed_batch_loss(
                model,
                drift,
                coordinate_noise,
                feature_noise,
                references,
                eta,
                repulsion,
            )

    trajectory_loss = []
    for checkpoint in checkpoint_values:
        model.load_state_dict(checkpoint[args.state])
        trajectory_loss.append(
            fixed_batch_loss(
                model,
                drift,
                coordinate_noise,
                feature_noise,
                references,
                eta,
                repulsion,
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "loss_landscape.npz",
        pc1=axes[0].numpy(),
        pc2=axes[1].numpy(),
        loss=surface.numpy(),
        trajectory=trajectory.numpy(),
        trajectory_loss=np.asarray(trajectory_loss),
        steps=np.asarray([checkpoint["step"] for checkpoint in checkpoint_values]),
        explained_variance=explained[:2].numpy(),
    )
    metadata = {
        "center": str(args.center),
        "state": args.state,
        "bandwidth": float(center_checkpoint["bandwidth"]),
        "batch_size": args.batch_size,
        "references": args.references,
        "seed": args.seed,
        "pc1_explained_variance": float(explained[0]),
        "pc2_explained_variance": float(explained[1]),
        "minimum_loss": float(surface.min()),
        "maximum_loss": float(surface.max()),
    }
    with (args.output / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    x_grid, y_grid = np.meshgrid(axes[0].numpy(), axes[1].numpy())
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(x_grid, y_grid, surface.numpy(), cmap="viridis", alpha=0.85)
    values = trajectory.numpy()
    axis.plot(
        values[:, 0],
        values[:, 1],
        trajectory_loss,
        color="crimson",
        marker="o",
        linewidth=2,
        label="EMA checkpoints",
    )
    axis.set(xlabel="PCA direction 1", ylabel="PCA direction 2", zlabel="fixed-batch loss")
    axis.set_title("DW4 unnormalized-drift loss landscape")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "loss_landscape.png", dpi=200)
    plt.close(figure)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
