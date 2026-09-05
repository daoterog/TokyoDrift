"""Dataset format and loading for fixed-graph conformer ensembles.

The trainer deliberately operates on one molecular graph at a time: its
attraction field is estimated only from conformers of that graph.  This avoids
mixing incomparable variable-length distance descriptors across molecules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class MoleculeEnsemble:
    """A molecular graph and its centered heavy-atom conformer ensemble."""

    atomic_numbers: torch.Tensor  # [atoms]
    bond_index: torch.Tensor  # [2, bonds], undirected edges stored once
    bond_order: torch.Tensor  # [bonds]
    conformers: torch.Tensor  # [conformers, atoms, 3]
    identifier: str

    def validate(self) -> None:
        """Raise a useful error for an invalid processed record."""
        atoms = len(self.atomic_numbers)
        if self.atomic_numbers.ndim != 1 or atoms < 2:
            raise ValueError(f"{self.identifier}: atomic_numbers must have at least two atoms")
        if self.conformers.ndim != 3 or self.conformers.shape[1:] != (atoms, 3):
            raise ValueError(f"{self.identifier}: invalid conformer tensor shape {tuple(self.conformers.shape)}")
        if len(self.conformers) < 2:
            raise ValueError(f"{self.identifier}: at least two conformers are required")
        if self.bond_index.ndim != 2 or self.bond_index.shape[0] != 2:
            raise ValueError(f"{self.identifier}: bond_index must have shape [2, bonds]")
        if self.bond_order.shape != (self.bond_index.shape[1],):
            raise ValueError(f"{self.identifier}: bond_order must have one entry per bond")
        if self.bond_index.numel() and (self.bond_index.min() < 0 or self.bond_index.max() >= atoms):
            raise ValueError(f"{self.identifier}: bond index outside atom range")


def _record(value: dict[str, Any]) -> MoleculeEnsemble:
    record = MoleculeEnsemble(
        atomic_numbers=torch.as_tensor(value["atomic_numbers"], dtype=torch.long),
        bond_index=torch.as_tensor(value["bond_index"], dtype=torch.long),
        bond_order=torch.as_tensor(value["bond_order"], dtype=torch.float32),
        conformers=torch.as_tensor(value["conformers"], dtype=torch.float32),
        identifier=str(value.get("identifier", "unknown")),
    )
    record.validate()
    # The global origin carries no molecular information.
    centered = record.conformers - record.conformers.mean(dim=1, keepdim=True)
    return MoleculeEnsemble(
        record.atomic_numbers, record.bond_index, record.bond_order, centered, record.identifier
    )


def load_split(path: Path, split: str, max_conformers: int | None = None) -> list[MoleculeEnsemble]:
    """Load ``{split}.pt`` created by :mod:`geom_qm9.prepare`.

    Files are intentionally simple Torch dictionaries: ``{"molecules": [...]}``,
    with each record containing the five fields of :class:`MoleculeEnsemble`.
    """
    payload = torch.load(path / f"{split}.pt", map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("molecules"), list):
        raise TypeError(f"{path / f'{split}.pt'} is not a GEOM-QM9 processed split")
    result = [_record(item) for item in payload["molecules"]]
    if max_conformers is not None:
        result = [
            MoleculeEnsemble(
                item.atomic_numbers,
                item.bond_index,
                item.bond_order,
                item.conformers[:max_conformers],
                item.identifier,
            )
            for item in result
        ]
    if not result:
        raise ValueError(f"{split} split is empty")
    return result
