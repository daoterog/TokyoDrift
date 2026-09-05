"""Clean Gaussian-kernel drift for conformers of one fixed molecular graph."""

from __future__ import annotations

import torch


def descriptor(positions: torch.Tensor) -> torch.Tensor:
    """Sorted pair distances: invariant without alignment or atom assignment."""
    atoms = positions.shape[1]
    pairs = torch.triu_indices(atoms, atoms, offset=1, device=positions.device)
    distances = (positions[:, pairs[0]] - positions[:, pairs[1]]).square().sum(-1).sqrt()
    return distances.sort(dim=-1).values


@torch.no_grad()
def median_bandwidth(samples: torch.Tensor, maximum: int = 512) -> float:
    """Return median RMS descriptor separation in a conformer ensemble."""
    values = descriptor(samples[:maximum])
    return float((torch.pdist(values).square() / values.shape[-1]).median().sqrt().clamp_min(1e-3))


def field(query: torch.Tensor, references: torch.Tensor, bandwidth: float, self_field: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate an unnormalised Gaussian kernel density in Cartesian space."""
    with torch.enable_grad():
        differentiable = query.detach().requires_grad_(True)
        residual = descriptor(references.detach())[None] - descriptor(differentiable)[:, None]
        squared_error = residual.square().mean(-1)
        kernel = torch.exp(-squared_error / (2.0 * bandwidth**2))
        if self_field:
            kernel = kernel * (1.0 - torch.eye(len(query), device=query.device, dtype=query.dtype))
        density = residual.shape[-1] * kernel.mean(1)
        (value,) = torch.autograd.grad(density.sum(), differentiable)
    return value.detach(), kernel.detach().mean()


def unnormalised_drift(generated: torch.Tensor, positive: torch.Tensor, bandwidth: float, repulsion: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Attraction toward reference conformers minus within-batch repulsion."""
    attraction, positive_mass = field(generated, positive, bandwidth, self_field=False)
    repulsive, negative_mass = field(generated, generated, bandwidth, self_field=True)
    value = attraction - repulsion * repulsive
    value = value - value.mean(1, keepdim=True)
    return value, {"positive_kernel_mass": float(positive_mass), "negative_kernel_mass": float(negative_mass), "drift_rms": float(value.square().mean().sqrt())}
