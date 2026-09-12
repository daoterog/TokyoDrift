"""Radial-kernel drift with optional local kernel-mass normalization."""

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
    """Multi-bandwidth radial-kernel drift without invariant descriptors.

    The Gaussian and Laplacian kernels omit bandwidth-dependent probability-
    density prefactors. For bandwidth ``h``:

    ``gaussian: k_h(x, y) = exp(-||x-y||**2 / (2*h**2))``

    ``laplacian: k_h(x, y) = exp(-||x-y|| / h)``

    The field is the gradient with respect to the query coordinates of the
    corresponding empirical kernel density.

    When multiple bandwidths are supplied, their complete attraction-minus-
    repulsion fields are averaged with equal weight. Kernel-mass diagnostics
    are averaged in the same way. A scalar remains supported as the
    single-bandwidth case.

    By default no division by the local kernel mass is performed. With
    ``normalized=True``, each attraction and repulsion field is divided by its
    own expected kernel value per query and bandwidth before subtraction and
    bandwidth averaging. This gives a difference of KDE scores. Coordinates are
    compared as flattened arrays, so particle ordering and global orientation
    must be meaningful and consistent across configurations.
    """

    def __init__(
        self,
        bandwidth: float | Sequence[float],
        kernel: str = "gaussian",
        normalized: bool = False,
    ) -> None:
        """Initialize a direct-coordinate Gaussian or Laplacian drift."""
        super().__init__()
        kernel = kernel.lower()
        if kernel not in {"gaussian", "laplacian"}:
            raise ValueError("kernel must be 'gaussian' or 'laplacian'")
        self.kernel = kernel
        if type(normalized) is not bool:
            raise ValueError("normalized must be true or false")
        self.normalized = normalized
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
    def _fields(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        self_indices: torch.Tensor | None = None,
        reference_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one density gradient and kernel mass per bandwidth."""
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
            if self.kernel == "gaussian":
                kernel = torch.exp(-squared_distance / (2.0 * bandwidth**2))
                kernel_gradient = residual / bandwidth**2
            else:
                distance = squared_distance.sqrt()
                kernel = torch.exp(-distance / bandwidth)
                safe_distance = distance.clamp_min(torch.finfo(query.dtype).eps)
                kernel_gradient = residual / (bandwidth * safe_distance[..., None, None])
            if keep is not None:
                kernel = kernel * keep
            if reference_weights is None:
                weighted_kernel = kernel / len(references)
            else:
                weighted_kernel = kernel * reference_weights.unsqueeze(0)
            fields.append((weighted_kernel[..., None, None] * kernel_gradient).sum(dim=1))
            masses.append(weighted_kernel.sum(dim=1))

        return torch.stack(fields), torch.stack(masses)

    def _normalize_fields(self, fields: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Optionally divide each field by its matching empirical kernel mean."""
        if not self.normalized:
            return fields
        # A fully excluded bank or complete kernel underflow contributes zero.
        # Use tiny, not eps: small representable kernel means still need normalization.
        denominator = masses.clamp_min(torch.finfo(masses.dtype).tiny)
        return fields / denominator[..., None, None]

    @torch.no_grad()
    def _field(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        self_indices: torch.Tensor | None = None,
        reference_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the selected bandwidth-averaged field and raw kernel masses."""
        fields, masses = self._fields(query, references, self_indices, reference_weights)
        return self._normalize_fields(fields, masses).mean(dim=0), masses.mean(dim=0)

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
        positive_fields, positive_masses = self._fields(
            generated, positive_references, reference_weights=positive_weights
        )
        if negative_references is None:
            negative_references = generated
        if self_indices is None and negative_references is generated:
            self_indices = torch.arange(len(generated), device=generated.device)
        negative_fields, negative_masses = self._fields(
            generated, negative_references, self_indices=self_indices
        )
        positive_fields = self._normalize_fields(positive_fields, positive_masses)
        negative_fields = self._normalize_fields(negative_fields, negative_masses)
        fields = positive_fields - repulsion * negative_fields
        temperature_rms = fields.square().mean(dim=(1, 2, 3)).sqrt()
        drift = fields.mean(dim=0)
        positive = positive_fields.mean(dim=0)
        negative = negative_fields.mean(dim=0)
        return drift, {
            "drift_rms": drift.square().mean().sqrt(),
            "positive_field_rms": positive.square().mean().sqrt(),
            "negative_field_rms": negative.square().mean().sqrt(),
            "temperature_field_rms_mean": temperature_rms.mean(),
            "temperature_field_rms_min": temperature_rms.min(),
            "temperature_field_rms_max": temperature_rms.max(),
            "positive_kernel_mass": positive_masses.mean(),
            "negative_kernel_mass": negative_masses.mean(),
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
        """Apply one explicit step using the selected drift normalization."""
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
