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
    """Equal-weight 5-by-8 grid of isotropic Gaussian modes."""

    def __init__(self, device: torch.device, standard_deviation: float = 0.35) -> None:
        x, y = torch.meshgrid(torch.linspace(-4.5, 4.5, 8), torch.linspace(-3.0, 3.0, 5), indexing="xy")
        self.centers = torch.stack((x.flatten(), y.flatten()), dim=1).to(device)
        self.standard_deviation = standard_deviation

    def sample(self, count: int) -> torch.Tensor:
        component = torch.randint(len(self.centers), (count,), device=self.centers.device)
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
    generated_mass = torch.bincount(target.mode_index(generated), minlength=40).float() / len(generated)
    target_mass = torch.full_like(generated_mass, 1 / 40)
    return {
        "mode_mass_l1": float((generated_mass - target_mass).abs().sum()),
        "mode_coverage": float((generated_mass > 0.005).float().mean()),
        "sliced_w2": sliced_w2(generated, reference),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--bandwidth", type=float, default=0.7)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    target = GMM40(device)
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
                record = {"step": step, "loss": float(loss.detach()), **metrics(target, samples, reference)}
                print(json.dumps(record), flush=True)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                torch.save({"step": step, "model": model.state_dict()}, args.output / f"checkpoint-{step:06d}.pt")


if __name__ == "__main__":
    main()
