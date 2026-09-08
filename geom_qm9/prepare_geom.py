"""Convert official GEOM-QM9 MessagePack archives into training splits.

The crude archive contains atom types and coordinates, while its MessagePack
keys are canonical SMILES. Bond labels come exactly from that SMILES graph;
RDKit only maps those graph atoms to the archive's coordinate atom order.
"""

from __future__ import annotations

import argparse
import random
import tarfile
from pathlib import Path
from typing import Any

import msgpack
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def map_records(path: Path) -> msgpack.Unpacker:
    """Return a streaming unpacker for the one MessagePack member in an archive."""
    archive = tarfile.open(path, mode="r:gz")
    member = archive.next()
    if member is None:
        archive.close()
        raise ValueError(f"{path} contains no member")
    stream = archive.extractfile(member)
    if stream is None:
        archive.close()
        raise ValueError(f"cannot read {member.name} from {path}")
    # Retain the archive through the stream's lifetime.
    stream._geom_archive = archive  # type: ignore[attr-defined]
    return msgpack.Unpacker(stream, raw=False)


def labelled_graph(xyz: torch.Tensor, smiles: str) -> dict[str, torch.Tensor]:
    """Map the archive coordinates onto the exact graph encoded by SMILES."""
    lines = [str(len(xyz)), "GEOM-QM9"]
    periodic_table = Chem.GetPeriodicTable()
    lines.extend(
        f"{periodic_table.GetElementSymbol(int(atom))} {x:.8f} {y:.8f} {z:.8f}"
        for atom, x, y, z in xyz.tolist()
    )
    molecule = Chem.MolFromXYZBlock("\n".join(lines))
    if molecule is None:
        raise ValueError("RDKit could not construct a molecule from GEOM coordinates")
    # The archive does not store its coordinate atom order in the SMILES key.
    # Perceive a temporary graph only to recover that atom correspondence; bond
    # types below always come from the source SMILES, never from this inference.
    rdDetermineBonds.DetermineBonds(molecule)
    source = Chem.MolFromSmiles(smiles)
    if source is None:
        raise ValueError(f"invalid GEOM SMILES: {smiles}")
    source = Chem.AddHs(source)
    # Match entries map QUERY (source SMILES) atoms to TARGET (XYZ) atoms.
    coordinate_index = molecule.GetSubstructMatch(source)
    if len(coordinate_index) != source.GetNumAtoms() or len(coordinate_index) != len(xyz):
        raise ValueError("could not map source-SMILES atoms to GEOM coordinate order")
    atomic_numbers = xyz[:, 0].long()
    heavy = atomic_numbers > 1
    remap = torch.full((len(atomic_numbers),), -1, dtype=torch.long)
    remap[heavy] = torch.arange(int(heavy.sum()))
    edges: list[list[int]] = []
    orders: list[float] = []
    order = {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.AROMATIC: 1.5,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
    }
    for bond in source.GetBonds():
        left = coordinate_index[bond.GetBeginAtomIdx()]
        right = coordinate_index[bond.GetEndAtomIdx()]
        if heavy[left] and heavy[right]:
            edges.append([int(remap[left]), int(remap[right])])
            orders.append(order[bond.GetBondType()])
    return {
        "atomic_numbers": atomic_numbers[heavy],
        "bond_index": torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T,
        "bond_order": torch.tensor(orders, dtype=torch.float32),
        "heavy_mask": heavy,
    }


def molecule_record(identifier: str, source: dict[str, Any], maximum: int) -> dict[str, Any] | None:
    """Build one centered heavy-atom ensemble, or reject an unusable molecule."""
    conformers = source["conformers"][:maximum]
    if len(conformers) < 2:
        return None
    first_xyz = torch.tensor(conformers[0]["xyz"], dtype=torch.float32)
    try:
        graph = labelled_graph(first_xyz, identifier)
    except Exception:
        return None
    values = []
    atomic_numbers = graph["atomic_numbers"]
    for conformer in conformers:
        xyz = torch.tensor(conformer["xyz"], dtype=torch.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 4:
            return None
        if not torch.equal(xyz[:, 0].long()[graph["heavy_mask"]], atomic_numbers):
            return None
        positions = xyz[graph["heavy_mask"], 1:]
        values.append(positions - positions.mean(dim=0, keepdim=True))
    return {
        "identifier": identifier,
        "atomic_numbers": atomic_numbers,
        "bond_index": graph["bond_index"],
        "bond_order": graph["bond_order"],
        "conformers": torch.stack(values),
    }


def conformer_bandwidth(record: dict[str, Any]) -> float:
    """Return the median RMS separation of invariant conformer descriptors."""
    positions = record["conformers"]
    atoms = positions.shape[1]
    pairs = torch.triu_indices(atoms, atoms, offset=1)
    descriptor = (positions[:, pairs[0]] - positions[:, pairs[1]]).square().sum(-1).sqrt()
    descriptor = descriptor.sort(dim=-1).values
    return float((torch.pdist(descriptor).square() / descriptor.shape[-1]).median().sqrt())


def main() -> None:
    """Write reproducible molecule-level GEOM-QM9 train/validation/test splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crude", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int, default=500)
    parser.add_argument("--max-conformers", type=int, default=50)
    parser.add_argument(
        "--minimum-bandwidth",
        type=float,
        default=0.1,
        help="Discard near-rigid ensembles whose median descriptor separation is below this value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_molecules < 20 or args.max_conformers < 2:
        raise ValueError("need at least 20 molecules and two conformers per molecule")
    records = []
    unpacker = map_records(args.crude)
    for _ in range(unpacker.read_map_header()):
        identifier, source = unpacker.unpack(), unpacker.unpack()
        record = molecule_record(identifier, source, args.max_conformers)
        if record is not None and conformer_bandwidth(record) >= args.minimum_bandwidth:
            records.append(record)
        if len(records) >= args.max_molecules:
            break
    if len(records) < 20:
        raise RuntimeError(f"only found {len(records)} molecules with at least two conformers")
    random.Random(args.seed).shuffle(records)
    validation_end = max(1, round(0.1 * len(records)))
    test_end = max(validation_end + 1, round(0.2 * len(records)))
    splits = {
        "validation": records[:validation_end],
        "test": records[validation_end:test_end],
        "train": records[test_end:],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        torch.save({"molecules": values}, args.output / f"{name}.pt")
    print({name: len(values) for name, values in splits.items()})


if __name__ == "__main__":
    main()
