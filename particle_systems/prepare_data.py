"""Download, verify, reshape, and split the official DW4/LJ13 arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from .systems import get_system


def arguments() -> argparse.Namespace:
    """Parse data-preparation command-line arguments."""
    parser = argparse.ArgumentParser(description="Download and split an official particle dataset.")
    parser.add_argument("system", choices=("dw4", "lj13"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--source", type=Path, default=None, help="Use an existing official NPY file."
    )
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--test-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Download atomically to a destination via a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "particle-drift-benchmark/1.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(destination)


def load_official_source(path: Path, system_name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load one of the two known official source formats."""
    if system_name == "dw4":
        # This legacy file is an object array containing a torch.Tensor and an
        # index permutation. It is loaded only after its official hash passes.
        payload = np.load(path, allow_pickle=True)
        coordinates = payload[0].numpy()
        order = np.asarray(payload[1], dtype=np.int64)
        return coordinates, order
    return np.load(path, mmap_mode="r"), None


def reshape_and_center(values: np.ndarray, particles: int, dimensions: int) -> np.ndarray:
    """Reshape flattened coordinates and remove each geometric center."""
    values = np.asarray(values, dtype=np.float32).reshape(-1, particles, dimensions)
    return values - values.mean(axis=1, keepdims=True)


def prepare(
    system_name: str,
    source: Path,
    output: Path,
    train_size: int,
    test_size: int,
    seed: int,
) -> None:
    """Verify and convert an official file into deterministic train/test splits."""
    system = get_system(system_name)
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train-size and test-size must be positive")
    actual_hash = sha256(source)
    if actual_hash != system.sha256:
        raise ValueError(f"SHA-256 mismatch for {source}: {actual_hash}")

    coordinates, official_order = load_official_source(source, system_name)
    if coordinates.shape != system.source_shape:
        raise ValueError(f"expected source shape {system.source_shape}, got {coordinates.shape}")
    requested = train_size + test_size
    if requested > coordinates.shape[0]:
        raise ValueError(f"requested {requested} samples from a dataset of {coordinates.shape[0]}")

    if official_order is not None:
        train_indices = official_order[:train_size]
        test_indices = official_order[train_size:requested]
    else:
        selection = np.random.default_rng(seed).choice(
            coordinates.shape[0], size=requested, replace=False
        )
        train_indices = np.sort(selection[:train_size])
        test_indices = np.sort(selection[train_size:])
    # Sorted LJ13 indices turn random 1.56 GB memmap seeks into a sequential
    # scan. Sample order is immaterial because training resamples uniformly.
    train = reshape_and_center(coordinates[train_indices], system.particles, system.dimensions)
    test = reshape_and_center(coordinates[test_indices], system.particles, system.dimensions)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "system": system.name,
            "particles": system.particles,
            "dimensions": system.dimensions,
            "source_url": system.osf_url,
            "source_sha256": system.sha256,
            "seed": seed,
        }
    )
    np.savez_compressed(output, train=train, test=test, metadata=np.array(metadata))
    print(f"wrote {output}: train={train.shape}, test={test.shape}")


def main() -> None:
    """Prepare the selected official benchmark dataset."""
    args = arguments()
    system = get_system(args.system)
    output = args.output or Path("particle_systems/data") / f"{args.system}.npz"
    source = args.source
    if source is None:
        source = output.parent / "downloads" / f"{args.system}-official.npy"
        if not source.exists():
            print(f"downloading {system.osf_url} to {source}")
            download(system.osf_url, source)
    prepare(args.system, source, output, args.train_size, args.test_size, args.seed)


if __name__ == "__main__":
    main()
