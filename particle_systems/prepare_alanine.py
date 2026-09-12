"""Download and prepare the published FAB 300 K alanine train/val/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .alanine import ATOM_NAMES, BONDS, ELEMENTS, TOPOLOGY_SHA256, TOPOLOGY_URL, geometry_rules
from .prepare_data import download, sha256

RECORD = "https://zenodo.org/records/6993124"
SOURCES = {
    "train": ("train.h5", "8d34fdda8694ee8d6745fc45b9eb3380", 1_000_000),
    "validation": ("val.h5", "7f93e4931ae4b2ef424d15b42218c9db", 1_000_000),
    "test": ("test.h5", "060e0165dcaba7c596293fc04f274081", 10_000_000),
}


def verify_md5(path: Path, expected: str) -> None:
    """Verify the publisher's pinned checksum before opening the HDF5 file."""
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "md5").hexdigest()
    if actual != expected:
        raise ValueError(f"MD5 mismatch for {path}: {actual}; expected {expected}")


def read_frames(path: Path, count: int, seed: int, expected_frames: int | None = None) -> tuple:
    """Read a deterministic subset in bounded chunks and convert nm to angstroms."""
    import h5py

    with h5py.File(path, "r") as source:
        coordinates = source["coordinates"]
        if coordinates.ndim != 3 or coordinates.shape[1:] != (22, 3):
            raise ValueError(f"expected all 22 atoms, got {coordinates.shape}")
        if expected_frames is not None and len(coordinates) != expected_frames:
            raise ValueError(f"unexpected frame count in {path}: {len(coordinates)}")
        units = coordinates.attrs.get("units", "")
        if isinstance(units, bytes):
            units = units.decode()
        if units not in {"nanometers", "nanometer", "nm"}:
            raise ValueError(f"expected published nanometer coordinates, found units={units!r}")
        topology = json.loads(source["topology"][0])
        atoms = [
            atom
            for chain in topology["chains"]
            for residue in chain["residues"]
            for atom in residue["atoms"]
        ]
        atoms.sort(key=lambda atom: atom["index"])
        if tuple(atom["name"] for atom in atoms) != ATOM_NAMES:
            raise ValueError("HDF5 atom order does not match the pinned alanine topology")
        if tuple(atom["element"] for atom in atoms) != ELEMENTS:
            raise ValueError("HDF5 elements do not match the pinned alanine topology")
        bonds = {tuple(sorted(bond[:2])) for bond in topology["bonds"]}
        if bonds != set(BONDS):
            raise ValueError("HDF5 bond topology differs from the benchmark")
        if count <= 0 or count > len(coordinates):
            raise ValueError(f"requested {count} frames from {len(coordinates)}")
        indices = np.sort(
            np.random.default_rng(seed).choice(len(coordinates), count, replace=False)
        )
        values = np.empty((count, 22, 3), dtype=np.float32)
        for start in range(0, len(coordinates), 65_536):
            stop = min(start + 65_536, len(coordinates))
            low, high = np.searchsorted(indices, [start, stop])
            if high > low:
                values[low:high] = coordinates[start:stop][indices[low:high] - start]
    if not np.isfinite(values).all():
        raise ValueError("source coordinates contain NaN or infinity")
    values *= 10.0
    values -= values.mean(axis=1, keepdims=True)
    return values, indices


def main() -> None:
    """Download checksummed coordinates/topology and keep the official split boundaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("particle_systems/data/aldp.npz"))
    parser.add_argument("--source-directory", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prepared data: {args.output}")
    sizes = {"train": args.train_size, "validation": args.validation_size, "test": args.test_size}
    for split, count in sizes.items():
        if not 0 < count <= SOURCES[split][2]:
            raise ValueError(f"invalid size for {split}: {count}")
    source_dir = args.source_directory or args.output.parent / "downloads/aldp"
    arrays, provenance = {}, {}
    for offset, (split, (filename, checksum, available)) in enumerate(SOURCES.items()):
        path = source_dir / filename
        url = f"{RECORD}/files/{filename}?download=1"
        if not path.exists():
            print(f"downloading {url} to {path}", flush=True)
            download(url, path)
        verify_md5(path, checksum)
        arrays[split], arrays[f"{split}_source_indices"] = read_frames(
            path, sizes[split], args.seed + offset, available
        )
        provenance[split] = {"url": url, "md5": checksum, "available_frames": available}
    # Keep topology beside the prepared data so the archive is relocatable as a directory.
    topology_path = args.output.parent / "alanine-dipeptide.prmtop"
    if not topology_path.exists():
        download(TOPOLOGY_URL, topology_path)
    if sha256(topology_path) != TOPOLOGY_SHA256:
        raise ValueError("alanine topology SHA-256 mismatch")
    metadata = {
        "system": "aldp",
        "particles": 22,
        "dimensions": 3,
        "dataset": "FAB alanine dipeptide in implicit solvent at 300 K",
        "record": RECORD,
        "license": "CC-BY-4.0",
        "sources": provenance,
        "coordinate_units": "angstrom",
        "source_coordinate_units": "nanometer",
        "temperature_kelvin": 300,
        "forcefield": "AMBER ff96 / OBC1 GBSA",
        "topology_file": topology_path.name,
        "topology_url": TOPOLOGY_URL,
        "topology_sha256": TOPOLOGY_SHA256,
        "atom_names": ATOM_NAMES,
        "elements": ELEMENTS,
        "bonds": BONDS,
        "seed": args.seed,
        "split_method": "independent deterministic subsets of the official separate files",
        "geometry_rules": geometry_rules(torch.from_numpy(arrays["train"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".npz.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.array(json.dumps(metadata)))
    partial.replace(args.output)
    print(f"wrote {args.output}: " + ", ".join(f"{s}={arrays[s].shape}" for s in SOURCES))


if __name__ == "__main__":
    main()
