"""Deterministic validation for particle-generator training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .evaluate import distribution_metrics
from .systems import ParticleSystem
from .unnormalized_drifting import DirectCoordinateDrift


@dataclass
class ValidationEvaluator:
    """Fixed noises and references for comparable checkpoint validation."""

    reference: torch.Tensor
    coordinate_noise: torch.Tensor
    feature_noise: torch.Tensor
    drift_references: torch.Tensor
    batch_size: int
    eta: float
    repulsion: float
    system: ParticleSystem

    @classmethod
    def build(
        cls,
        reference: torch.Tensor,
        system: ParticleSystem,
        feature_dim: int,
        coordinate_scale: float,
        training: dict,
        seed: int,
    ) -> ValidationEvaluator:
        """Create deterministic validation inputs without touching training RNG state."""
        samples = int(training.get("validation_generated_samples", 10_000))
        batch_size = int(training.get("validation_batch_size", 512))
        positive_references = int(training.get("validation_positive_references", 1024))
        if len(reference) < positive_references:
            raise ValueError("validation holdout is smaller than validation reference count")
        generator = torch.Generator().manual_seed(seed + 1_000_003)
        coordinate_noise = coordinate_scale * torch.randn(
            samples, system.particles, system.dimensions, generator=generator
        )
        feature_noise = torch.randn(
            samples, system.particles, feature_dim, generator=generator
        )
        indices = torch.randperm(len(reference), generator=generator)[:positive_references]
        return cls(
            reference=reference,
            coordinate_noise=coordinate_noise,
            feature_noise=feature_noise,
            drift_references=reference[indices],
            batch_size=batch_size,
            eta=float(training["eta"]),
            repulsion=float(training["repulsion"]),
            system=system,
        )

    @torch.no_grad()
    def generated_samples(self, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
        """Generate the same validation samples for every checkpoint."""
        values: list[torch.Tensor] = []
        for start in range(0, len(self.coordinate_noise), self.batch_size):
            stop = start + self.batch_size
            values.append(
                model(
                    self.coordinate_noise[start:stop].to(device),
                    self.feature_noise[start:stop].to(device),
                ).cpu()
            )
        return torch.cat(values)

    def evaluate(
        self, model: torch.nn.Module, drift: DirectCoordinateDrift, device: torch.device
    ) -> dict[str, float]:
        """Return fixed drift loss and held-out distribution metrics."""
        generated = self.generated_samples(model, device)
        drift_batch = generated[: self.batch_size].to(device)
        field, _ = drift(
            drift_batch,
            self.drift_references.to(device),
            drift_batch,
            torch.arange(len(drift_batch), device=device),
            self.repulsion,
        )
        metrics = distribution_metrics(generated, self.reference, self.system.name)
        return {
            "drift_loss": float((self.eta * field).square().mean()),
            "energy_mean": float(metrics["energy_mean"]),
            "energy_std": float(metrics["energy_std"]),
            "energy_wasserstein_1": float(metrics["energy_wasserstein_1"]),
            "energy_histogram_js": float(metrics["energy_histogram_js"]),
            "pair_distance_wasserstein_1": float(metrics["pair_distance_wasserstein_1"]),
            "radius_wasserstein_1": float(metrics["radius_wasserstein_1"]),
            "collision_fraction": float(metrics["collision_fraction_d_lt_0_5"]),
        }
