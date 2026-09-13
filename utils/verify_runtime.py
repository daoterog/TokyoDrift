"""Verify that the selected PyTorch device can execute particle-system operations."""

from __future__ import annotations

import argparse
import json

import torch

from data.systems import get_system
from models.particle_generator import ParticleGenerator

from .io import device_summary, select_device


def arguments() -> argparse.Namespace:
    """Parse runtime-verification arguments."""
    parser = argparse.ArgumentParser(description="Verify the particle-system runtime.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    return parser.parse_args()


def main() -> None:
    """Run representative tensor and potential operations on the selected device."""
    args = arguments()
    device = select_device(args.device)
    system = get_system("dw4")
    model = ParticleGenerator(
        feature_dim=3, hidden_dim=16, layers=1, radial_basis=6, max_distance=8.0
    ).to(device)
    square = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]], device=device)
    positions = square.unsqueeze(0).repeat(4, 1, 1)
    features = torch.randn(4, system.particles, 3, device=device)
    generated = model(positions, features)
    loss = system.energy(generated).mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not torch.isfinite(loss) or not all(torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("The selected device produced non-finite verification values.")
    result = {
        **device_summary(device),
        "verification": "passed",
        "operation": "DW4 EGNN forward/backward and potential on a 4-sample batch",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
