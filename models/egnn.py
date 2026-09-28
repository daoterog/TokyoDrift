"""E(n)-equivariant block EGNN generator for fixed-size particle systems.

Each block applies scalar-feature message-passing sublayers (``TypeGCN``) followed by
one equivariant coordinate update (``PosGCN``). Edge attributes are the initial and
current squared distances, coordinate messages are ``(x_j - x_i) / (|x_j - x_i| + 1)``
scaled by a ``tanh``-bounded learned weight, and messages are summed (or averaged)
over neighbours. Graphs are complete without self-edges and are batched densely, so
entry ``[b, i, j]`` of every pair tensor is the message node ``i`` receives from ``j``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from data.systems import center

AGGREGATIONS = ("sum", "mean")
EDGE_ATTRIBUTES = 2


def _pair_input(features: torch.Tensor, edge_attributes: torch.Tensor) -> torch.Tensor:
    """Concatenate destination features, source features, and edge attributes."""
    particles = features.shape[1]
    destination = features[:, :, None].expand(-1, -1, particles, -1)
    source = features[:, None, :].expand(-1, particles, -1, -1)
    return torch.cat((destination, source, edge_attributes), dim=-1)


def _aggregate(messages: torch.Tensor, mask: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Reduce pair messages over sources, excluding self-edges."""
    total = (messages * mask).sum(dim=2)
    if aggregation == "mean":
        return total / max(messages.shape[2] - 1, 1)
    return total


def _validate_aggregation(aggregation: str) -> None:
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")


def compute_edge_properties(
    positions: torch.Tensor, norm_constant: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return squared distances ``[B, N, N, 1]`` and scaled directions ``[B, N, N, D]``."""
    direction = positions[:, None, :] - positions[:, :, None]
    squared_norm = direction.square().sum(dim=-1, keepdim=True)
    # The clamp only affects coincident points (including the masked self-edges),
    # whose direction is zero, and keeps the square-root gradient finite there.
    norm = squared_norm.clamp_min(1e-12).sqrt()
    return squared_norm, direction / (norm + norm_constant)


class TypeGCN(nn.Module):
    """Message-passing update of invariant node features with optional attention."""

    def __init__(self, hidden_dim: int, attention: bool = True, aggregation: str = "sum") -> None:
        """Initialize message, attention, and residual update networks."""
        super().__init__()
        _validate_aggregation(aggregation)
        message_dim = 2 * hidden_dim + EDGE_ATTRIBUTES
        self.aggregation = aggregation
        self.message_mlp = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.att_mlp = (
            nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid()) if attention else None
        )

    def forward(
        self, features: torch.Tensor, edge_attributes: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Return residually updated node features."""
        messages = self.message_mlp(_pair_input(features, edge_attributes))
        if self.att_mlp is not None:
            messages = messages * self.att_mlp(messages)
        aggregate = _aggregate(messages, mask, self.aggregation)
        return features + self.update_mlp(torch.cat((aggregate, features), dim=-1))


class PosGCN(nn.Module):
    """Equivariant coordinate update from direction-weighted pair messages."""

    def __init__(
        self,
        hidden_dim: int,
        tanh_coordinate_updates: bool = True,
        coordinate_range: float = 15.0,
        aggregation: str = "sum",
    ) -> None:
        """Initialize the invariant pair-weight network."""
        super().__init__()
        _validate_aggregation(aggregation)
        output = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(output.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + EDGE_ATTRIBUTES, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            output,
        )
        self.tanh_coordinate_updates = tanh_coordinate_updates
        self.coordinate_range = float(coordinate_range)
        self.aggregation = aggregation

    def forward(
        self,
        positions: torch.Tensor,
        features: torch.Tensor,
        edge_attributes: torch.Tensor,
        scaled_direction: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return positions shifted by the aggregated equivariant messages."""
        weights = self.coord_mlp(_pair_input(features, edge_attributes))
        if self.tanh_coordinate_updates:
            weights = torch.tanh(weights) * self.coordinate_range
        return positions + _aggregate(scaled_direction * weights, mask, self.aggregation)


class EquivariantBlock(nn.Module):
    """Feature sublayers followed by one coordinate update."""

    def __init__(
        self,
        hidden_dim: int,
        layers: int = 1,
        attention: bool = True,
        tanh_coordinate_updates: bool = True,
        coordinate_range: float = 15.0,
        aggregation: str = "sum",
    ) -> None:
        """Initialize ``layers`` feature sublayers and one coordinate update."""
        super().__init__()
        self.type_update = nn.ModuleList(
            TypeGCN(hidden_dim, attention=attention, aggregation=aggregation)
            for _ in range(layers)
        )
        self.coord_update = PosGCN(
            hidden_dim,
            tanh_coordinate_updates=tanh_coordinate_updates,
            coordinate_range=coordinate_range,
            aggregation=aggregation,
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        edge_attributes: torch.Tensor,
        scaled_direction: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update features, then positions from the updated features."""
        for type_gcn in self.type_update:
            features = type_gcn(features, edge_attributes, mask)
        positions = self.coord_update(positions, features, edge_attributes, scaled_direction, mask)
        return features, positions


class EGNN(nn.Module):
    """Direct equivariant map from mean-free Gaussian noise to positions."""

    variant = "block"

    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dim: int = 256,
        layers: int = 9,
        layers_per_block: int = 1,
        attention: bool = True,
        tanh_coordinate_updates: bool = True,
        coordinate_range: float = 15.0,
        aggregation: str = "sum",
        fixed_atom_identity: int | None = None,
    ) -> None:
        """Initialize a linear feature embedding and ``layers`` equivariant blocks."""
        super().__init__()
        if layers <= 0 or layers_per_block <= 0:
            raise ValueError("layers and layers_per_block must be positive")
        if not math.isfinite(coordinate_range) or coordinate_range <= 0:
            raise ValueError("coordinate_range must be finite and positive")
        _validate_aggregation(aggregation)
        if fixed_atom_identity is not None and feature_dim != fixed_atom_identity:
            raise ValueError("fixed atom identity requires feature_dim equal to atom count")
        self.feature_dim = feature_dim
        self.coordinate_range = float(coordinate_range)
        self.aggregation = aggregation
        self.register_buffer(
            "atom_identity",
            torch.eye(fixed_atom_identity) if fixed_atom_identity is not None else None,
        )
        self.type_embedding = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            EquivariantBlock(
                hidden_dim,
                layers=layers_per_block,
                attention=attention,
                tanh_coordinate_updates=tanh_coordinate_updates,
                coordinate_range=coordinate_range,
                aggregation=aggregation,
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
        features = self.type_embedding(feature_noise)
        particles = positions.shape[1]
        mask = (1.0 - torch.eye(particles, device=positions.device, dtype=positions.dtype))[
            None, :, :, None
        ]
        initial_squared_norm, _ = compute_edge_properties(positions)
        for block in self.blocks:
            squared_norm, scaled_direction = compute_edge_properties(positions)
            edge_attributes = torch.cat((initial_squared_norm, squared_norm), dim=-1)
            features, positions = block(features, positions, edge_attributes, scaled_direction, mask)
        # Updates need not preserve the centroid; distances are translation invariant,
        # so removing it once at the end equals removing it after every block.
        return center(positions)
