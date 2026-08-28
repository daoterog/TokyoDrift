"""E(n)-equivariant direct generator for fixed-size particle systems."""

from __future__ import annotations

import torch
from torch import nn

from .systems import center


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class EGNNLayer(nn.Module):
    """Small complete-graph E(n)-equivariant message-passing layer."""

    def __init__(self, hidden_dim: int, radial_basis: int = 16, max_distance: float = 8.0) -> None:
        """Initialize one message-passing and coordinate-update layer."""
        super().__init__()
        self.edge_mlp = _mlp(2 * hidden_dim + radial_basis, hidden_dim, hidden_dim)
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
        self, features: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update scalar node features and equivariant coordinates."""
        particles = positions.shape[1]
        difference = positions[:, :, None] - positions[:, None, :]
        distance = difference.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        radial = torch.exp(-self.gamma * (distance - self.centers.view(1, 1, 1, -1)).square())
        left = features[:, :, None].expand(-1, -1, particles, -1)
        right = features[:, None, :].expand(-1, particles, -1, -1)
        messages = self.edge_mlp(torch.cat((left, right, radial), dim=-1))
        mask = (1.0 - torch.eye(particles, device=positions.device, dtype=positions.dtype))[
            None, :, :, None
        ]
        messages = messages * torch.sigmoid(self.edge_gate(messages)) * mask
        weights = torch.tanh(self.coordinate_mlp(messages)) * mask
        update = (difference * weights).sum(dim=2) / max(particles - 1, 1)
        positions = center(positions + update)
        aggregate = messages.sum(dim=2) / max(particles - 1, 1)
        features = self.norm(features + self.node_mlp(torch.cat((features, aggregate), dim=-1)))
        return features, positions


class ParticleGenerator(nn.Module):
    """Direct equivariant map from mean-free Gaussian noise to positions."""

    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dim: int = 64,
        layers: int = 4,
        radial_basis: int = 16,
        max_distance: float = 8.0,
    ) -> None:
        """Initialize a stack of complete-graph EGNN layers."""
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding = _mlp(feature_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            EGNNLayer(hidden_dim, radial_basis, max_distance) for _ in range(layers)
        )

    def forward(self, coordinate_noise: torch.Tensor, feature_noise: torch.Tensor) -> torch.Tensor:
        """Map coordinate and scalar Gaussian noise to centered positions."""
        positions = center(coordinate_noise)
        features = self.embedding(feature_noise)
        for layer in self.layers:
            features, positions = layer(features, positions)
        return center(positions)
