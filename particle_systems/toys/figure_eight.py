"""One-shot drift on a balanced two-ring figure-eight control distribution."""

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
            nn.Linear(2, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 2)
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    centers = torch.tensor([[-0.75, 0.0], [0.75, 0.0]], device=device)
    crossings = torch.tensor([[0.0, (1 - 0.75**2) ** 0.5], [0.0, -(1 - 0.75**2) ** 0.5]], device=device)

    def sample_target(count: int) -> torch.Tensor:
        component = torch.randint(2, (count,), device=device)
        angle = 2 * torch.pi * torch.rand(count, device=device)
        radius = 1 + 0.045 * torch.randn(count, device=device)
        return centers[component] + radius[:, None] * torch.stack((angle.cos(), angle.sin()), dim=1)

    def field(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        query = query.detach().requires_grad_(True)
        attraction = torch.exp(-torch.cdist(query, reference).square() / (2 * 0.20**2)).mean(1)
        repulsion = torch.exp(-torch.cdist(query, query.detach()).square() / (2 * 0.20**2)).mean(1)
        (result,) = torch.autograd.grad((attraction - repulsion).sum(), query)
        return result.detach()

    def metrics(samples: torch.Tensor) -> dict[str, float]:
        radial_error = (torch.cdist(samples, centers) - 1).abs().min(1).values
        return {
            "crossing_mass": float((torch.cdist(samples, crossings).min(1).values < 0.16).float().mean()),
            "ring_coverage": float((radial_error < 0.12).float().mean()),
            "interior_mass": float((torch.cdist(samples, centers).min(1).values < 0.70).float().mean()),
        }

    target, test = sample_target(40_000), sample_target(20_000)
    model = Generator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    with (args.output / "metrics.jsonl").open("w") as stream:
        stream.write(json.dumps({"step": 0, "target": metrics(test)}) + "\n")
        for step in range(args.steps + 1):
            generated = model(1.55 * torch.randn(512, 2, device=device))
            loss = (generated - (generated.detach() + 0.16 * field(generated, target[torch.randint(len(target), (512,), device=device)]))).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 1_000 == 0:
                with torch.no_grad():
                    samples = model(1.55 * torch.randn(20_000, 2, device=device))
                record = {"step": step, "loss": float(loss.detach()), **metrics(samples)}
                print(json.dumps(record), flush=True)
                stream.write(json.dumps(record) + "\n")
                stream.flush()


if __name__ == "__main__":
    main()
