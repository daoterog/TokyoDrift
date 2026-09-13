"""Generator architectures."""

from .egnn import EGNN, EGNNLayer
from .gnn import GNN, GNNLayer

__all__ = ["EGNN", "GNN", "EGNNLayer", "GNNLayer"]
