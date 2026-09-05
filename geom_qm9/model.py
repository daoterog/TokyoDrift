"""A compact graph-conditioned E(3)-equivariant direct conformer generator."""

from __future__ import annotations

import torch
from torch import nn


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class GraphEGNNLayer(nn.Module):
    """Complete-graph EGNN layer, conditioned on atom and bond features."""

    def __init__(self, hidden_dim: int, radial_basis: int, max_distance: float) -> None:
        super().__init__()
        self.edge = _mlp(2 * hidden_dim + radial_basis + 2, hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.coordinate = _mlp(hidden_dim, hidden_dim, 1)
        self.node = _mlp(2 * hidden_dim, 2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        centers = torch.linspace(0.0, max_distance, radial_basis)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max(max_distance / max(radial_basis - 1, 1), 1e-3) ** 2
        nn.init.normal_(self.coordinate[-1].weight, std=1e-3)
        nn.init.zeros_(self.coordinate[-1].bias)

    def forward(
        self, features: torch.Tensor, positions: torch.Tensor, bonds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update nodes and coordinates while preserving E(3) equivariance."""
        atoms = positions.shape[1]
        difference = positions[:, :, None] - positions[:, None, :]
        distance = difference.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        radial = torch.exp(-self.gamma * (distance - self.centers.view(1, 1, 1, -1)).square())
        left = features[:, :, None].expand(-1, -1, atoms, -1)
        right = features[:, None, :].expand(-1, atoms, -1, -1)
        edge_features = torch.cat((left, right, radial, bonds.expand(len(positions), -1, -1, -1)), -1)
        messages = self.edge(edge_features)
        mask = 1.0 - torch.eye(atoms, device=positions.device, dtype=positions.dtype)[None, :, :, None]
        messages = messages * torch.sigmoid(self.gate(messages)) * mask
        coordinate_weight = torch.tanh(self.coordinate(messages)) * mask
        positions = positions + (difference * coordinate_weight).sum(2) / max(atoms - 1, 1)
        positions = positions - positions.mean(1, keepdim=True)
        aggregate = messages.sum(2) / max(atoms - 1, 1)
        features = self.norm(features + self.node(torch.cat((features, aggregate), -1)))
        return features, positions


class ConformerGenerator(nn.Module):
    """One-shot map from noise and a fixed molecular graph to coordinates."""

    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 6,
        radial_basis: int = 24,
        max_distance: float = 8.0,
        atom_embedding_size: int = 32,
    ) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(128, atom_embedding_size)
        self.input = _mlp(atom_embedding_size + 1, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GraphEGNNLayer(hidden_dim, radial_basis, max_distance) for _ in range(layers)
        )

    def forward(
        self, coordinate_noise: torch.Tensor, atomic_numbers: torch.Tensor, bond_index: torch.Tensor,
        bond_order: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a batch of centered conformers for one molecular graph."""
        atoms = len(atomic_numbers)
        positions = coordinate_noise - coordinate_noise.mean(1, keepdim=True)
        atom = self.atom_embedding(atomic_numbers).unsqueeze(0).expand(len(positions), -1, -1)
        features = self.input(torch.cat((atom, positions.square().sum(-1, keepdim=True)), -1))
        bonds = torch.zeros(atoms, atoms, 2, device=positions.device, dtype=positions.dtype)
        if bond_index.numel():
            src, dst = bond_index
            bonds[src, dst, 0] = 1.0
            bonds[dst, src, 0] = 1.0
            bonds[src, dst, 1] = bond_order
            bonds[dst, src, 1] = bond_order
        for layer in self.layers:
            features, positions = layer(features, positions, bonds)
        return positions - positions.mean(1, keepdim=True)
