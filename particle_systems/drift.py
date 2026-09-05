"""Alignment-free unnormalized drift for particle configurations."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .systems import center, pair_distances


def invariant_descriptor(
    positions: torch.Tensor,
    energy_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    energy_mean: torch.Tensor | None = None,
    energy_standard_deviation: torch.Tensor | None = None,
    energy_feature_scale: float = 0.0,
) -> torch.Tensor:
    """Return invariant pair distances, optionally augmented by standardized energy."""
    descriptor = pair_distances(positions).sort(dim=-1).values
    if energy_feature_scale == 0.0:
        return descriptor
    if energy_function is None or energy_mean is None or energy_standard_deviation is None:
        raise ValueError("energy augmentation requires an energy function, mean, and standard deviation")
    energy = energy_function(positions).unsqueeze(-1)
    standardized_energy = (energy - energy_mean) / energy_standard_deviation
    return torch.cat((descriptor, energy_feature_scale * standardized_energy), dim=-1)


def descriptor_whitening(
    samples: torch.Tensor, shrinkage: float = 0.1, ridge: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed, regularized whitening transform for invariant descriptors."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("whitening shrinkage must lie in [0, 1]")
    if ridge <= 0:
        raise ValueError("whitening ridge must be positive")
    descriptor = invariant_descriptor(samples)
    if len(descriptor) < 2:
        raise ValueError("at least two samples are required to estimate descriptor whitening")
    mean = descriptor.mean(dim=0)
    centered = descriptor - mean
    covariance = centered.T @ centered / (len(descriptor) - 1)
    diagonal = torch.diag(torch.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    ridge_scale = covariance.diag().mean().clamp_min(torch.finfo(covariance.dtype).eps)
    covariance = covariance + ridge * ridge_scale * torch.eye(
        covariance.shape[0], dtype=covariance.dtype, device=covariance.device
    )
    cholesky = torch.linalg.cholesky(covariance)
    whitener = torch.linalg.inv(cholesky)
    return mean, whitener


class UnnormalizedDrift(nn.Module):
    """Unnormalized kernel density drift in invariant distance space.

    Kernel weights remain unnormalized: each field is the gradient of the
    empirical mean kernel density, not a kernel-mass-normalized score. The
    descriptor removes translation, rotation, reflection, and permutation
    without performing an alignment or assignment.
    """

    def __init__(
        self,
        bandwidth: float,
        kernel: str = "gaussian",
        imq_beta: float = 0.5,
        descriptor_mean: torch.Tensor | None = None,
        descriptor_whitener: torch.Tensor | None = None,
        broad_bandwidth: float | None = None,
        broad_weight: float = 0.0,
        minimum_bandwidth: float | None = None,
        minimum_weight: float = 0.0,
        energy_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        energy_mean: torch.Tensor | None = None,
        energy_standard_deviation: torch.Tensor | None = None,
        energy_feature_scale: float = 0.0,
    ) -> None:
        """Initialize the Gaussian kernel with a positive bandwidth."""
        super().__init__()
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if kernel not in {"gaussian", "imq"}:
            raise ValueError("kernel must be 'gaussian' or 'imq'")
        if imq_beta <= 0:
            raise ValueError("imq_beta must be positive")
        if (descriptor_mean is None) != (descriptor_whitener is None):
            raise ValueError("descriptor mean and whitener must be provided together")
        if broad_weight < 0 or minimum_weight < 0:
            raise ValueError("kernel component weights must be non-negative")
        if broad_weight and (broad_bandwidth is None or broad_bandwidth <= 0):
            raise ValueError("broad bandwidth must be positive when its weight is nonzero")
        if minimum_weight and (minimum_bandwidth is None or minimum_bandwidth <= 0):
            raise ValueError("minimum bandwidth must be positive when its weight is nonzero")
        if energy_feature_scale < 0:
            raise ValueError("energy feature scale must be non-negative")
        if energy_feature_scale and (
            energy_function is None or energy_mean is None or energy_standard_deviation is None
        ):
            raise ValueError("energy augmentation requires an energy function, mean, and standard deviation")
        self.bandwidth = float(bandwidth)
        self.kernel = kernel
        self.imq_beta = float(imq_beta)
        self.register_buffer("descriptor_mean", descriptor_mean)
        self.register_buffer("descriptor_whitener", descriptor_whitener)
        self.broad_bandwidth = broad_bandwidth
        self.broad_weight = float(broad_weight)
        self.minimum_bandwidth = minimum_bandwidth
        self.minimum_weight = float(minimum_weight)
        self.energy_function = energy_function
        self.energy_feature_scale = float(energy_feature_scale)
        self.register_buffer("energy_mean", energy_mean)
        self.register_buffer("energy_standard_deviation", energy_standard_deviation)

    def _kernel(self, squared_error: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """Evaluate the selected positive-definite radial kernel."""
        if self.kernel == "gaussian":
            return torch.exp(-squared_error / (2.0 * bandwidth**2))
        return (1.0 + squared_error / bandwidth**2).pow(-self.imq_beta)

    def _descriptor(self, positions: torch.Tensor) -> torch.Tensor:
        """Return raw or fixed-whitened invariant descriptors."""
        descriptor = invariant_descriptor(
            positions,
            energy_function=self.energy_function,
            energy_mean=self.energy_mean,
            energy_standard_deviation=self.energy_standard_deviation,
            energy_feature_scale=self.energy_feature_scale,
        )
        if self.descriptor_mean is None:
            return descriptor
        return (descriptor - self.descriptor_mean) @ self.descriptor_whitener.T

    def _field(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        self_indices: torch.Tensor | None = None,
        reference_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if references.shape[0] == 0:
            raise ValueError("at least one reference is required")
        if reference_weights is not None:
            if reference_weights.ndim != 1 or len(reference_weights) != len(references):
                raise ValueError("reference weights must have one entry per reference")
            if torch.any(reference_weights < 0) or not torch.isclose(
                reference_weights.sum(), torch.tensor(1.0, device=reference_weights.device)
            ):
                raise ValueError("reference weights must be non-negative and sum to one")

        # Build a small local graph to pull the invariant density gradient
        # back to Cartesian coordinates. It is detached before returning.
        with torch.enable_grad():
            differentiable_query = query.detach().requires_grad_(True)
            query_descriptor = self._descriptor(differentiable_query)
            reference_descriptor = self._descriptor(references.detach())
            residual = reference_descriptor.unsqueeze(0) - query_descriptor.unsqueeze(1)
            descriptor_size = query_descriptor.shape[-1]
            squared_error = residual.square().mean(dim=-1)
            kernel = self._kernel(squared_error, self.bandwidth)
            broad_kernel = (
                self._kernel(squared_error, float(self.broad_bandwidth))
                if self.broad_weight
                else None
            )
            minimum_residual = residual[..., 0]
            minimum_kernel = (
                self._kernel(minimum_residual.square(), float(self.minimum_bandwidth))
                if self.minimum_weight
                else None
            )
            if self_indices is not None:
                valid = self_indices >= 0
                rows = torch.arange(query.shape[0], device=query.device)[valid]
                columns = self_indices[valid]
                if torch.any(columns >= references.shape[0]):
                    raise ValueError("self index is outside the negative reference bank")
                keep = torch.ones_like(kernel)
                keep[rows, columns] = 0.0
                kernel = kernel * keep
                if broad_kernel is not None:
                    broad_kernel = broad_kernel * keep
                if minimum_kernel is not None:
                    minimum_kernel = minimum_kernel * keep

            def average(values: torch.Tensor) -> torch.Tensor:
                if reference_weights is None:
                    return values.mean(dim=1)
                return (values * reference_weights.detach().unsqueeze(0)).sum(dim=1)

            density = descriptor_size * average(kernel)
            if broad_kernel is not None:
                density = density + self.broad_weight * descriptor_size * average(broad_kernel)
            if minimum_kernel is not None:
                density = density + self.minimum_weight * average(minimum_kernel)
            (field,) = torch.autograd.grad(density.sum(), differentiable_query)
        masses = {"kernel_mass": average(kernel).detach()}
        if broad_kernel is not None:
            masses["broad_kernel_mass"] = average(broad_kernel).detach()
        if minimum_kernel is not None:
            masses["minimum_kernel_mass"] = average(minimum_kernel).detach()
        return field.detach(), masses

    def forward(
        self,
        generated: torch.Tensor,
        positive_references: torch.Tensor,
        negative_references: torch.Tensor | None = None,
        self_indices: torch.Tensor | None = None,
        repulsion: float = 1.0,
        positive_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return attraction minus self-repulsion and useful diagnostics."""
        positive, positive_masses = self._field(
            generated, positive_references, reference_weights=positive_weights
        )
        if negative_references is None:
            negative_references = generated
        if self_indices is None:
            self_indices = torch.arange(generated.shape[0], device=generated.device)
        negative, negative_masses = self._field(generated, negative_references, self_indices)
        value = center(positive - repulsion * negative)
        cosine = torch.nn.functional.cosine_similarity(
            positive.flatten(start_dim=1), negative.flatten(start_dim=1), dim=1
        ).mean()
        return value, {
            "drift_rms": value.square().mean().sqrt(),
            "positive_field_rms": positive.square().mean().sqrt(),
            "negative_field_rms": negative.square().mean().sqrt(),
            "positive_negative_cosine": cosine,
            **{f"positive_{name}": mass.mean() for name, mass in positive_masses.items()},
            **{f"negative_{name}": mass.mean() for name, mass in negative_masses.items()},
        }


@torch.no_grad()
def median_bandwidth(
    samples: torch.Tensor,
    max_samples: int = 1024,
    descriptor_mean: torch.Tensor | None = None,
    descriptor_whitener: torch.Tensor | None = None,
    energy_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    energy_mean: torch.Tensor | None = None,
    energy_standard_deviation: torch.Tensor | None = None,
    energy_feature_scale: float = 0.0,
) -> float:
    """Median RMS distance between invariant descriptors."""
    samples = samples[:max_samples]
    if samples.shape[0] < 2:
        raise ValueError("at least two samples are required to estimate bandwidth")
    if (descriptor_mean is None) != (descriptor_whitener is None):
        raise ValueError("descriptor mean and whitener must be provided together")
    descriptors = invariant_descriptor(
        samples,
        energy_function=energy_function,
        energy_mean=energy_mean,
        energy_standard_deviation=energy_standard_deviation,
        energy_feature_scale=energy_feature_scale,
    )
    if descriptor_mean is not None:
        descriptors = (descriptors - descriptor_mean) @ descriptor_whitener.T
    distance2 = torch.pdist(descriptors).square() / descriptors.shape[-1]
    return float(distance2.median().sqrt().clamp_min(1e-3))
