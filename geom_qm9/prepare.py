"""Create the portable GEOM-QM9 processed format from exported tensors.

This intentionally does not entangle model code with RDKit or a particular
version of GEOM's raw MessagePack files.  The source export is a list of
dictionary records written with ``torch.save``.  The format is documented in
``data/README.md`` and can be produced from any GEOM reader in a few lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import _record


def main() -> None:
    """Validate and copy a portable list of molecule records into split files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Torch-saved list of molecule records.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--max-molecules", type=int, default=None)
    args = parser.parse_args()
    values = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(values, list):
        raise TypeError("input must be a torch-saved list of molecule dictionaries")
    validated = []
    for value in values[: args.max_molecules]:
        item = _record(value)
        validated.append({
            "atomic_numbers": item.atomic_numbers,
            "bond_index": item.bond_index,
            "bond_order": item.bond_order,
            "conformers": item.conformers,
            "identifier": item.identifier,
        })
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"molecules": validated}, args.output / f"{args.split}.pt")
    print(f"wrote {len(validated)} molecules to {args.output / f'{args.split}.pt'}")


if __name__ == "__main__":
    main()
