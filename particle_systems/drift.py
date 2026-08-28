"""Raw unnormalized kernel drift for particle configurations."""

from __future__ import annotations

import torch
from torch import nn

from .systems import center


class UnnormalizedDrift(nn.Module):
    """Raw, unaligned Gaussian-kernel density drift for point clouds.

    Both fields are empirical expectations of kernel gradients. Kernel weights
    are deliberately not divided by their mass, and configurations are not
    rotated, reflected, or permuted before comparison.
    """

    def __init__(self, bandwidth: float) -> None:
        """Initialize the Gaussian kernel with a positive bandwidth."""
        super().__init__()
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.bandwidth = float(bandwidth)

    def _field(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        self_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if references.shape[0] == 0:
            raise ValueError("at least one reference is required")
        residual = references.unsqueeze(0) - query.unsqueeze(1)
        particles = query.shape[-2]
        energy = residual.square().sum(dim=(-1, -2)) / (2.0 * particles * self.bandwidth**2)
        kernel = torch.exp(-energy)
        if self_indices is not None:
            valid = self_indices >= 0
            rows = torch.arange(query.shape[0], device=query.device)[valid]
            columns = self_indices[valid]
            if torch.any(columns >= references.shape[0]):
                raise ValueError("self index is outside the negative reference bank")
            kernel[rows, columns] = 0.0
        field = (kernel[..., None, None] * residual).mean(dim=1) / self.bandwidth**2
        return field, kernel.mean(dim=1)

    @torch.no_grad()
    def forward(
        self,
        generated: torch.Tensor,
        positive_references: torch.Tensor,
        negative_references: torch.Tensor | None = None,
        self_indices: torch.Tensor | None = None,
        repulsion: float = 0.3,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return attraction minus self-repulsion and diagnostic kernel masses."""
        positive, positive_mass = self._field(generated, positive_references)
        if negative_references is None:
            negative_references = generated
        if self_indices is None:
            self_indices = torch.arange(generated.shape[0], device=generated.device)
        negative, negative_mass = self._field(generated, negative_references, self_indices)
        value = center(positive - repulsion * negative)
        return value, {
            "drift_rms": value.square().mean().sqrt(),
            "positive_kernel_mass": positive_mass.mean(),
            "negative_kernel_mass": negative_mass.mean(),
        }


@torch.no_grad()
def median_bandwidth(samples: torch.Tensor, max_samples: int = 1024) -> float:
    """Median per-particle configuration distance, a stable KDE heuristic."""
    samples = samples[:max_samples]
    if samples.shape[0] < 2:
        raise ValueError("at least two samples are required to estimate bandwidth")
    flattened = samples.flatten(start_dim=1)
    distance2 = torch.pdist(flattened).square() / samples.shape[1]
    return float(distance2.median().sqrt().clamp_min(1e-3))
