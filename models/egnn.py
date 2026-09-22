"""E(n)-equivariant direct generator for fixed-size particle systems."""

from __future__ import annotations

import math

import torch
from torch import nn

from data.systems import center


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class EGNNLayer(nn.Module):
    """Complete-graph E(n)-equivariant message-passing layer."""

    def __init__(
        self,
        hidden_dim: int,
        radial_basis: int = 16,
        max_distance: float = 8.0,
        variant: str = "legacy",
        coordinate_range: float = 1.0,
    ) -> None:
        """Initialize one legacy or bounded message-passing layer.

        The bounded variant augments the local radial basis with transformed
        initial and current squared distances. Its coordinate messages use
        ``difference / (distance + 1)`` so their magnitude cannot grow with an
        already-large configuration. The legacy variant is retained so older
        checkpoints keep their original parameter shapes and behavior.
        """
        super().__init__()
        if radial_basis <= 0:
            raise ValueError("radial_basis must be positive")
        if not math.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("max_distance must be finite and positive")
        if variant not in {"legacy", "bounded"}:
            raise ValueError("EGNN variant must be 'legacy' or 'bounded'")
        if not math.isfinite(coordinate_range) or coordinate_range <= 0:
            raise ValueError("coordinate_range must be finite and positive")

        self.variant = variant
        self.coordinate_range = float(coordinate_range)
        self.distance_scale_squared = float(max_distance) ** 2
        distance_features = 2 if variant == "bounded" else 0
        self.edge_mlp = _mlp(
            2 * hidden_dim + radial_basis + distance_features,
            hidden_dim,
            hidden_dim,
        )
        self.edge_gate = nn.Linear(hidden_dim, 1)
        self.coordinate_mlp = _mlp(hidden_dim, hidden_dim, 1)
        self.node_mlp = _mlp(2 * hidden_dim, 2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        centers = torch.linspace(0.0, max_distance, radial_basis)
        spacing = max_distance / max(radial_basis - 1, 1)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max(spacing, 1e-3) ** 2
        nn.init.normal_(self.coordinate_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.coordinate_mlp[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        initial_squared_distance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update scalar node features and equivariant coordinates."""
        particles = positions.shape[1]
        difference = positions[:, :, None] - positions[:, None, :]
        squared_distance = difference.square().sum(dim=-1, keepdim=True)
        distance = squared_distance.clamp_min(1e-12).sqrt()
        radial = torch.exp(
            -self.gamma * (distance - self.centers.view(1, 1, 1, -1)).square()
        )
        if self.variant == "bounded":
            if initial_squared_distance is None:
                raise ValueError("bounded EGNN layers require initial squared distances")
            if initial_squared_distance.shape != squared_distance.shape:
                raise ValueError("initial and current squared distances must have matching shapes")
            # RBFs retain local resolution. These monotone features remain
            # informative beyond the final RBF center without exposing the MLP
            # directly to potentially enormous raw squared distances.
            distance_features = torch.cat(
                (
                    torch.log1p(initial_squared_distance / self.distance_scale_squared),
                    torch.log1p(squared_distance / self.distance_scale_squared),
                ),
                dim=-1,
            )
            edge_attributes = torch.cat((radial, distance_features), dim=-1)
            coordinate_vectors = difference / (distance + 1.0)
        else:
            edge_attributes = radial
            coordinate_vectors = difference

        left = features[:, :, None].expand(-1, -1, particles, -1)
        right = features[:, None, :].expand(-1, particles, -1, -1)
        messages = self.edge_mlp(torch.cat((left, right, edge_attributes), dim=-1))
        mask = (1.0 - torch.eye(particles, device=positions.device, dtype=positions.dtype))[
            None, :, :, None
        ]
        messages = messages * torch.sigmoid(self.edge_gate(messages)) * mask
        weights = torch.tanh(self.coordinate_mlp(messages)) * mask
        if self.variant == "bounded":
            weights = weights * self.coordinate_range
        update = (coordinate_vectors * weights).sum(dim=2) / max(particles - 1, 1)
        positions = center(positions + update)
        aggregate = messages.sum(dim=2) / max(particles - 1, 1)
        features = self.norm(features + self.node_mlp(torch.cat((features, aggregate), dim=-1)))
        return features, positions


class EGNN(nn.Module):
    """Direct equivariant map from mean-free Gaussian noise to positions."""

    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dim: int = 64,
        layers: int = 4,
        radial_basis: int = 16,
        max_distance: float = 8.0,
        fixed_atom_identity: int | None = None,
        variant: str = "legacy",
        coordinate_range: float = 1.0,
    ) -> None:
        """Initialize a stack of complete-graph EGNN layers."""
        super().__init__()
        self.feature_dim = feature_dim
        self.variant = variant
        self.coordinate_range = float(coordinate_range)
        if fixed_atom_identity is not None and feature_dim != fixed_atom_identity:
            raise ValueError("fixed atom identity requires feature_dim equal to atom count")
        self.register_buffer(
            "atom_identity",
            torch.eye(fixed_atom_identity) if fixed_atom_identity is not None else None,
        )
        self.embedding = _mlp(feature_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            EGNNLayer(
                hidden_dim,
                radial_basis,
                max_distance,
                variant=variant,
                coordinate_range=coordinate_range,
            )
            for _ in range(layers)
        )

    def forward(self, coordinate_noise: torch.Tensor, feature_noise: torch.Tensor) -> torch.Tensor:
        """Map coordinate and scalar Gaussian noise to centered positions."""
        positions = center(coordinate_noise)
        if self.atom_identity is not None:
            if positions.shape[1] != len(self.atom_identity):
                raise ValueError("coordinates do not match the configured atom identities")
            # Fixed topology labels distinguish chemically inequivalent atom slots.
            feature_noise = self.atom_identity.unsqueeze(0).expand(len(positions), -1, -1)
        features = self.embedding(feature_noise)
        initial_difference = positions[:, :, None] - positions[:, None, :]
        initial_squared_distance = initial_difference.square().sum(dim=-1, keepdim=True)
        for layer in self.layers:
            features, positions = layer(features, positions, initial_squared_distance)
        return center(positions)
