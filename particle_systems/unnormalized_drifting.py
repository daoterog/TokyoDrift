"""Unnormalized Gaussian drift evaluated directly in particle-coordinate space."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


@torch.no_grad()
def median_bandwidth(samples: torch.Tensor, max_samples: int = 1024) -> float:
    """Return the median Euclidean distance between flattened configurations."""
    if samples.ndim != 3:
        raise ValueError("samples must have shape [batch, particles, dimensions]")
    samples = samples[:max_samples]
    if len(samples) < 2:
        raise ValueError("at least two samples are required to estimate bandwidth")
    distances = torch.pdist(samples.flatten(start_dim=1))
    positive = distances[distances > 0]
    if len(positive) == 0:
        raise ValueError("cannot estimate bandwidth from identical configurations")
    return float(positive.median())


class DirectCoordinateDrift(nn.Module):
    """Multi-bandwidth Gaussian density-gradient drift without invariant descriptors.

    For a query configuration ``x`` and reference configurations ``y_j``, the
    attractive field at bandwidth ``h`` is the gradient of the empirical
    Gaussian kernel density:

    ``mean_j[k_h(x, y_j) * (y_j - x) / h**2]``.

    When multiple bandwidths are supplied, their fields and kernel-mass
    diagnostics are averaged with equal weight. A scalar remains supported as
    the single-bandwidth case.

    No division by the local kernel mass is performed. Consequently, this is an
    unnormalized density gradient rather than a KDE score. Coordinates are
    compared as flattened arrays, so particle ordering and global orientation
    must be meaningful and consistent across configurations.
    """

    def __init__(self, bandwidth: float | Sequence[float]) -> None:
        """Initialize a direct-coordinate Gaussian drift."""
        super().__init__()
        self.bandwidth = bandwidth

    @property
    def bandwidth(self) -> float | tuple[float, ...]:
        """Return one bandwidth as a scalar and multiple bandwidths as a tuple."""
        if len(self._bandwidths) == 1:
            return self._bandwidths[0]
        return self._bandwidths

    @bandwidth.setter
    def bandwidth(self, bandwidth: float | Sequence[float]) -> None:
        if isinstance(bandwidth, (int, float)):
            values = (float(bandwidth),)
        else:
            if isinstance(bandwidth, (str, bytes)):
                raise TypeError("bandwidth must be a number or a sequence of numbers")
            values = tuple(float(value) for value in bandwidth)
        if not values:
            raise ValueError("at least one bandwidth is required")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("bandwidths must be finite and positive")
        self._bandwidths = values

    @staticmethod
    def _validate_inputs(query: torch.Tensor, references: torch.Tensor) -> None:
        if query.ndim != 3 or references.ndim != 3:
            raise ValueError("positions must have shape [batch, particles, dimensions]")
        if query.shape[1:] != references.shape[1:]:
            raise ValueError("query and reference particle shapes must match")
        if len(references) == 0:
            raise ValueError("at least one reference configuration is required")
        if query.device != references.device:
            raise ValueError("query and references must be on the same device")

    @torch.no_grad()
    def _field(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        self_indices: torch.Tensor | None = None,
        reference_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a Gaussian density gradient and its mean kernel mass."""
        self._validate_inputs(query, references)
        residual = references.unsqueeze(0) - query.unsqueeze(1)
        squared_distance = residual.flatten(start_dim=2).square().sum(dim=-1)
        keep = None
        if self_indices is not None:
            if self_indices.shape != (len(query),):
                raise ValueError("self_indices must have one entry per query")
            if self_indices.device != query.device:
                raise ValueError("self_indices must be on the query device")
            valid = self_indices >= 0
            if torch.any(self_indices[valid] >= len(references)):
                raise ValueError("self index is outside the reference bank")
            keep = torch.ones_like(squared_distance)
            rows = torch.arange(len(query), device=query.device)[valid]
            keep[rows, self_indices[valid]] = 0.0
        if reference_weights is not None:
            if reference_weights.shape != (len(references),):
                raise ValueError("reference_weights must have one entry per reference")
            if reference_weights.device != references.device:
                raise ValueError("reference_weights must be on the reference device")
            if torch.any(reference_weights < 0):
                raise ValueError("reference_weights must be non-negative")
            expected = torch.ones((), dtype=reference_weights.dtype, device=references.device)
            if not torch.isclose(reference_weights.sum(), expected):
                raise ValueError("reference_weights must sum to one")

        fields = []
        masses = []
        for bandwidth in self._bandwidths:
            kernel = torch.exp(-squared_distance / (2.0 * bandwidth**2))
            if keep is not None:
                kernel = kernel * keep
            if reference_weights is None:
                weighted_kernel = kernel / len(references)
            else:
                weighted_kernel = kernel * reference_weights.unsqueeze(0)
            fields.append(
                (weighted_kernel[..., None, None] * residual).sum(dim=1) / bandwidth**2
            )
            masses.append(weighted_kernel.sum(dim=1))

        return torch.stack(fields).mean(dim=0), torch.stack(masses).mean(dim=0)

    def forward(
        self,
        generated: torch.Tensor,
        positive_references: torch.Tensor,
        negative_references: torch.Tensor | None = None,
        self_indices: torch.Tensor | None = None,
        repulsion: float = 1.0,
        positive_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return data attraction minus generated-sample repulsion."""
        if repulsion < 0:
            raise ValueError("repulsion must be non-negative")
        positive, positive_mass = self._field(
            generated, positive_references, reference_weights=positive_weights
        )
        if negative_references is None:
            negative_references = generated
        if self_indices is None and negative_references is generated:
            self_indices = torch.arange(len(generated), device=generated.device)
        negative, negative_mass = self._field(
            generated, negative_references, self_indices=self_indices
        )
        drift = positive - repulsion * negative
        return drift, {
            "drift_rms": drift.square().mean().sqrt(),
            "positive_field_rms": positive.square().mean().sqrt(),
            "negative_field_rms": negative.square().mean().sqrt(),
            "positive_kernel_mass": positive_mass.mean(),
            "negative_kernel_mass": negative_mass.mean(),
        }

    def step(
        self,
        generated: torch.Tensor,
        positive_references: torch.Tensor,
        step_size: float,
        negative_references: torch.Tensor | None = None,
        self_indices: torch.Tensor | None = None,
        repulsion: float = 1.0,
        positive_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply one explicit unnormalized-drift step to generated positions."""
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        drift, metrics = self(
            generated,
            positive_references,
            negative_references,
            self_indices,
            repulsion,
            positive_weights,
        )
        return generated.detach() + step_size * drift, metrics
