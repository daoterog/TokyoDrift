"""Non-geometric complete-graph generator for fixed-size particle systems."""

from __future__ import annotations

import torch
from torch import nn

from data.systems import center


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class GNNLayer(nn.Module):
    """Complete-graph message-passing layer without geometric equivariance."""

    def __init__(
        self,
        hidden_dim: int,
        dimensions: int,
        radial_basis: int = 16,
        max_distance: float = 8.0,
    ) -> None:
        """Initialize one feature and unconstrained coordinate-update layer."""
        super().__init__()
        self.dimensions = dimensions
        self.edge_mlp = _mlp(
            2 * hidden_dim + 2 * dimensions + radial_basis,
            hidden_dim,
            hidden_dim,
        )
        self.edge_gate = nn.Linear(hidden_dim, 1)
        self.coordinate_mlp = _mlp(hidden_dim, hidden_dim, dimensions)
        self.node_mlp = _mlp(2 * hidden_dim, 2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        centers = torch.linspace(0.0, max_distance, radial_basis)
        spacing = max_distance / max(radial_basis - 1, 1)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max(spacing, 1e-3) ** 2
        nn.init.normal_(self.coordinate_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.coordinate_mlp[-1].bias)

    def forward(
        self, features: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update node features and coordinates without an E(n) constraint."""
        if positions.shape[-1] != self.dimensions:
            raise ValueError(
                f"expected {self.dimensions} coordinate dimensions, got {positions.shape[-1]}"
            )
        particles = positions.shape[1]
        difference = positions[:, :, None] - positions[:, None, :]
        distance = difference.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        radial = torch.exp(-self.gamma * (distance - self.centers.view(1, 1, 1, -1)).square())
        left_features = features[:, :, None].expand(-1, -1, particles, -1)
        right_features = features[:, None, :].expand(-1, particles, -1, -1)
        left_positions = positions[:, :, None].expand(-1, -1, particles, -1)
        right_positions = positions[:, None, :].expand(-1, particles, -1, -1)
        messages = self.edge_mlp(
            torch.cat(
                (left_features, right_features, left_positions, right_positions, radial),
                dim=-1,
            )
        )
        mask = (1.0 - torch.eye(particles, device=positions.device, dtype=positions.dtype))[
            None, :, :, None
        ]
        messages = messages * torch.sigmoid(self.edge_gate(messages)) * mask
        coordinate_messages = torch.tanh(self.coordinate_mlp(messages)) * mask
        update = coordinate_messages.sum(dim=2) / max(particles - 1, 1)
        positions = center(positions + update)
        aggregate = messages.sum(dim=2) / max(particles - 1, 1)
        features = self.norm(features + self.node_mlp(torch.cat((features, aggregate), dim=-1)))
        return features, positions


class GNN(nn.Module):
    """Direct non-equivariant GNN map from Gaussian noise to positions."""

    def __init__(
        self,
        dimensions: int,
        feature_dim: int = 8,
        hidden_dim: int = 64,
        layers: int = 4,
        radial_basis: int = 16,
        max_distance: float = 8.0,
        fixed_atom_identity: int | None = None,
    ) -> None:
        """Initialize a stack of unconstrained complete-graph GNN layers."""
        super().__init__()
        self.feature_dim = feature_dim
        if fixed_atom_identity is not None and feature_dim != fixed_atom_identity:
            raise ValueError("fixed atom identity requires feature_dim equal to atom count")
        self.register_buffer(
            "atom_identity",
            torch.eye(fixed_atom_identity) if fixed_atom_identity is not None else None,
        )
        self.embedding = _mlp(feature_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GNNLayer(hidden_dim, dimensions, radial_basis, max_distance) for _ in range(layers)
        )

    def forward(self, coordinate_noise: torch.Tensor, feature_noise: torch.Tensor) -> torch.Tensor:
        """Map coordinate and node-feature noise to centered positions."""
        positions = center(coordinate_noise)
        if self.atom_identity is not None:
            if positions.shape[1] != len(self.atom_identity):
                raise ValueError("coordinates do not match the configured atom identities")
            feature_noise = self.atom_identity.unsqueeze(0).expand(len(positions), -1, -1)
        features = self.embedding(feature_noise)
        for layer in self.layers:
            features, positions = layer(features, positions)
        return center(positions)
