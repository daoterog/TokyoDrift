"""Clean one-shot Gaussian-kernel drift benchmark on a 40-mode 2D mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2)
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class GMM40:
    """A 5-by-8 grid of isotropic Gaussian modes.

    ``distribution='imbalanced'`` keeps the support fixed while assigning
    8 frequent (8%), 12 medium (2%), and 20 rare (0.6%) components.  The
    tiers are spread across the grid, so rarity is not confounded with a
    particular spatial region.
    """

    def __init__(
        self,
        device: torch.device,
        standard_deviation: float = 0.35,
        distribution: str = "uniform",
    ) -> None:
        x, y = torch.meshgrid(torch.linspace(-4.5, 4.5, 8), torch.linspace(-3.0, 3.0, 5), indexing="xy")
        self.centers = torch.stack((x.flatten(), y.flatten()), dim=1).to(device)
        self.standard_deviation = standard_deviation
        if distribution == "uniform":
            self.weights = torch.full((40,), 1 / 40, device=device)
            self.tiers = torch.full((40,), 1, dtype=torch.long, device=device)
        elif distribution == "imbalanced":
            # A coprime stride disperses each tier over the lattice.
            ordering = (17 * torch.arange(40)) % 40
            self.tiers = torch.empty(40, dtype=torch.long, device=device)
            self.tiers[ordering[:8].to(device)] = 0  # frequent
            self.tiers[ordering[8:20].to(device)] = 1  # medium
            self.tiers[ordering[20:].to(device)] = 2  # rare
            tier_weights = torch.tensor((0.08, 0.02, 0.006), device=device)
            self.weights = tier_weights[self.tiers]
        else:
            raise ValueError(f"unknown distribution: {distribution}")

    def sample(self, count: int) -> torch.Tensor:
        component = torch.multinomial(self.weights, count, replacement=True)
        return self.centers[component] + self.standard_deviation * torch.randn(count, 2, device=self.centers.device)

    def mode_index(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.cdist(samples, self.centers).argmin(dim=1)


def drift_field(query: torch.Tensor, positive: torch.Tensor, bandwidth: float) -> torch.Tensor:
    query = query.detach().requires_grad_(True)
    positive_density = torch.exp(-torch.cdist(query, positive).square() / (2 * bandwidth**2)).mean(dim=1)
    negative_density = torch.exp(-torch.cdist(query, query.detach()).square() / (2 * bandwidth**2)).mean(dim=1)
    (field,) = torch.autograd.grad((positive_density - negative_density).sum(), query)
    return field.detach()


def sliced_w2(generated: torch.Tensor, reference: torch.Tensor, projections: int = 64) -> float:
    directions = torch.randn(projections, 2, device=generated.device)
    directions = directions / directions.square().sum(dim=1, keepdim=True).sqrt()
    generated_projection = (generated @ directions.T).sort(dim=0).values
    reference_projection = (reference @ directions.T).sort(dim=0).values
    return float((generated_projection - reference_projection).square().mean().sqrt())


def metrics(target: GMM40, generated: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Validity-first diagnostics; per-mode mass is retained only as an audit."""
    distances = torch.cdist(generated, target.centers)
    nearest_distance, nearest_mode = distances.min(dim=1)
    valid = nearest_distance <= 2 * target.standard_deviation
    generated_mass = torch.bincount(nearest_mode, minlength=40).float() / len(generated)
    valid_mode_counts = torch.bincount(nearest_mode[valid], minlength=40)
    tier_mass = torch.stack(
        [generated_mass[target.tiers == tier].sum() for tier in range(3)]
    )
    target_tier_mass = torch.stack(
        [target.weights[target.tiers == tier].sum() for tier in range(3)]
    )
    rare = target.tiers == 2
    return {
        "validity_rate": float(valid.float().mean()),
        "mean_distance_to_mode": float(nearest_distance.mean()),
        "valid_mode_coverage": float((valid_mode_counts >= 10).float().mean()),
        "rare_valid_mode_coverage": float((valid_mode_counts[rare] >= 10).float().mean()),
        "frequent_mass": float(tier_mass[0]),
        "medium_mass": float(tier_mass[1]),
        "rare_mass": float(tier_mass[2]),
        "target_frequent_mass": float(target_tier_mass[0]),
        "target_medium_mass": float(target_tier_mass[1]),
        "target_rare_mass": float(target_tier_mass[2]),
        "mode_mass_l1_audit": float((generated_mass - target.weights).abs().sum()),
        "mode_coverage": float((generated_mass > 0.005).float().mean()),
        "sliced_w2": sliced_w2(generated, reference),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--bandwidth", type=float, default=0.7)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--distribution", choices=("uniform", "imbalanced"), default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    target = GMM40(device, distribution=args.distribution)
    reference = target.sample(20_000)
    model = Generator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    with (args.output / "metrics.jsonl").open("w") as stream:
        for step in range(args.steps + 1):
            generated = model(5.0 * torch.randn(args.batch_size, 2, device=device))
            field = drift_field(generated, target.sample(args.batch_size), args.bandwidth)
            loss = (generated - (generated.detach() + args.eta * field)).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 1_000 == 0:
                with torch.no_grad():
                    samples = model(5.0 * torch.randn(20_000, 2, device=device))
                record = {
                    "step": step,
                    "distribution": args.distribution,
                    "loss": float(loss.detach()),
                    **metrics(target, samples, reference),
                }
                print(json.dumps(record), flush=True)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                torch.save({"step": step, "model": model.state_dict()}, args.output / f"checkpoint-{step:06d}.pt")


if __name__ == "__main__":
    main()
