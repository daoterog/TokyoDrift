"""Invariant pair-distance descriptors and their coordinate-space kernel gradient."""

from __future__ import annotations

import math

import torch

from .unnormalized_drifting import DirectCoordinateDrift


def particle_descriptors(positions: torch.Tensor) -> torch.Tensor:
    """Return sorted unique pair distances divided by sqrt(number of pairs).

    Euclidean descriptor distance is the RMS difference of the sorted distance
    lists (the 1D Wasserstein-2 distance between empirical pair distributions).
    This is invariant to translations, orthogonal transforms and permutations.
    Sorting is differentiable almost everywhere; ties use PyTorch's subgradient.
    The representation is intended for identical particles of a fixed system size.
    """
    if positions.ndim != 3 or positions.shape[1] < 2:
        raise ValueError("positions must have shape [batch, particles >= 2, dimensions]")
    rows, columns = torch.triu_indices(
        positions.shape[1], positions.shape[1], offset=1, device=positions.device
    )
    distances = (positions[:, rows] - positions[:, columns]).norm(dim=-1)
    return distances.sort(dim=-1).values / math.sqrt(len(rows))


@torch.no_grad()
def descriptor_bandwidth(samples: torch.Tensor, max_samples: int = 1024) -> float:
    """Estimate the median positive distance in the exact descriptor kernel metric."""
    distances = torch.pdist(particle_descriptors(samples[:max_samples]))
    positive = distances[distances > 0]
    if len(positive) == 0:
        raise ValueError("at least two distinct descriptors are required to estimate bandwidth")
    return float(positive.median())


class DescriptorDrift(DirectCoordinateDrift):
    """Pull back descriptor KDE gradients to particle coordinates.

    For z = phi(x), the field is J_phi(x).T @ grad_z KDE(z). The inherited
    attraction, repulsion, self exclusion, bandwidth averaging and step interface
    therefore still return coordinate updates. References and output are detached;
    no second derivatives through the generator are needed. Optional kernel-mass
    normalization is inherited and preserves the descriptor field's equivariance.
    """

    @staticmethod
    def descriptors(positions: torch.Tensor) -> torch.Tensor:
        """Return comparison features; molecular subclasses preserve atom labels."""
        return particle_descriptors(positions)

    @torch.no_grad()
    def _fields(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        self_indices: torch.Tensor | None = None,
        reference_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute kernel fields without a [query, reference, descriptor] tensor."""
        self._validate_inputs(query, references)
        reference_features = self.descriptors(references.detach())
        # Enable only the descriptor Jacobian, including when called under no_grad.
        with torch.enable_grad():
            positions = query.detach().requires_grad_(True)
            features = self.descriptors(positions)
        query_features = features.detach()
        distance = torch.cdist(
            query_features, reference_features, compute_mode="donot_use_mm_for_euclid_dist"
        )
        keep = torch.ones_like(distance)
        if self_indices is not None:
            if self_indices.shape != (len(query),) or self_indices.device != query.device:
                raise ValueError("self_indices must have one entry per query on its device")
            valid = self_indices >= 0
            if torch.any(self_indices[valid] >= len(references)):
                raise ValueError("self index is outside the reference bank")
            keep[torch.arange(len(query), device=query.device)[valid], self_indices[valid]] = 0
        if reference_weights is not None:
            if reference_weights.shape != (len(references),):
                raise ValueError("reference_weights must have one entry per reference")
            if reference_weights.device != references.device:
                raise ValueError("reference_weights must be on the reference device")
            if torch.any(reference_weights < 0) or not torch.isclose(
                reference_weights.sum(), reference_weights.new_tensor(1.0)
            ):
                raise ValueError("reference_weights must be non-negative and sum to one")
            keep = keep * reference_weights[None]
        else:
            keep = keep / len(references)

        # A shared origin reduces cancellation in the weighted residual matmul.
        origin = reference_features.mean(dim=0, keepdim=True)
        query_features = query_features - origin
        reference_features = reference_features - origin
        fields, masses = [], []
        for index, bandwidth in enumerate(self._bandwidths):
            if self.kernel == "gaussian":
                weights = torch.exp(-distance.square() / (2 * bandwidth**2)) * keep
                gradient_weights = weights / bandwidth**2
            else:
                weights = torch.exp(-distance / bandwidth) * keep
                gradient_weights = weights / (
                    bandwidth * distance.clamp_min(torch.finfo(query.dtype).eps)
                )
                gradient_weights = gradient_weights.masked_fill(distance == 0, 0)
            feature_field = gradient_weights @ reference_features - (
                gradient_weights.sum(dim=1, keepdim=True) * query_features
            )
            coordinate_field = torch.autograd.grad(
                features,
                positions,
                grad_outputs=feature_field,
                retain_graph=index + 1 < len(self._bandwidths),
            )[0]
            fields.append(coordinate_field.detach())
            masses.append(weights.sum(dim=1))
        return torch.stack(fields), torch.stack(masses)
