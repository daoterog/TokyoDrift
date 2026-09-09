"""Download, verify, reshape, and split the official particle-system arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .systems import get_system


def arguments() -> argparse.Namespace:
    """Parse data-preparation command-line arguments."""
    parser = argparse.ArgumentParser(description="Download and split an official particle dataset.")
    parser.add_argument("system", choices=("dw4", "lj13", "lj55"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        default=None,
        help="Use existing official NPY file(s), in the published part order.",
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


def load_official_sources(
    paths: Sequence[Path], system_name: str
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Memory-map ordinary arrays or load DW4's legacy object payload."""
    if system_name == "dw4":
        # This legacy file is an object array containing a torch.Tensor and an
        # index permutation. It is loaded only after its official hash passes.
        payload = np.load(paths[0], allow_pickle=True)
        coordinates = payload[0].numpy()
        order = np.asarray(payload[1], dtype=np.int64)
        return [coordinates], order
    return [np.load(path, mmap_mode="r") for path in paths], None


def select_rows(parts: Sequence[np.ndarray], indices: np.ndarray) -> np.ndarray:
    """Select globally indexed rows from one or more source arrays."""
    if not parts:
        raise ValueError("at least one source part is required")
    if indices.ndim != 1:
        raise ValueError("source indices must be one-dimensional")
    available = sum(len(part) for part in parts)
    if np.any(indices < 0) or np.any(indices >= available):
        raise IndexError("source index is outside the multipart dataset")
    columns = parts[0].shape[1]
    selected = np.empty((len(indices), columns), dtype=np.float32)
    offset = 0
    for part in parts:
        within_part = (indices >= offset) & (indices < offset + len(part))
        local_indices = indices[within_part] - offset
        selected[within_part] = np.asarray(part[local_indices], dtype=np.float32)
        offset += len(part)
    return selected


def reshape_and_center(values: np.ndarray, particles: int, dimensions: int) -> np.ndarray:
    """Reshape flattened coordinates and remove each geometric center."""
    values = np.asarray(values, dtype=np.float32).reshape(-1, particles, dimensions)
    return values - values.mean(axis=1, keepdims=True)


def prepare(
    system_name: str,
    sources: Path | Sequence[Path],
    output: Path,
    train_size: int,
    test_size: int,
    seed: int,
) -> None:
    """Verify and convert official source file(s) into deterministic splits."""
    system = get_system(system_name)
    source_paths = [sources] if isinstance(sources, Path) else list(sources)
    if len(source_paths) != len(system.sources):
        raise ValueError(
            f"{system.name} requires {len(system.sources)} source file(s), got {len(source_paths)}"
        )
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train-size and test-size must be positive")
    for path, source_definition in zip(source_paths, system.sources, strict=True):
        actual_hash = sha256(path)
        if actual_hash != source_definition.sha256:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual_hash}")

    coordinate_parts, official_order = load_official_sources(source_paths, system_name)
    for coordinates, source_definition in zip(
        coordinate_parts, system.sources, strict=True
    ):
        if coordinates.shape != source_definition.shape:
            raise ValueError(
                f"expected {source_definition.filename} shape {source_definition.shape}, "
                f"got {coordinates.shape}"
            )
    available = sum(len(coordinates) for coordinates in coordinate_parts)
    requested = train_size + test_size
    if requested > available:
        raise ValueError(f"requested {requested} samples from a dataset of {available}")

    if official_order is not None:
        train_indices = official_order[:train_size]
        test_indices = official_order[train_size:requested]
    else:
        selection = np.random.default_rng(seed).choice(available, size=requested, replace=False)
        train_indices = np.sort(selection[:train_size])
        test_indices = np.sort(selection[train_size:])
    # Sorted Lennard-Jones indices make selection from large memmaps mostly sequential.
    train = reshape_and_center(
        select_rows(coordinate_parts, train_indices), system.particles, system.dimensions
    )
    test = reshape_and_center(
        select_rows(coordinate_parts, test_indices), system.particles, system.dimensions
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "system": system.name,
            "particles": system.particles,
            "dimensions": system.dimensions,
            "sources": [
                {
                    "filename": source.filename,
                    "url": source.url,
                    "sha256": source.sha256,
                }
                for source in system.sources
            ],
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
    sources = args.source
    if sources is None:
        sources = []
        for source_definition in system.sources:
            source = output.parent / "downloads" / source_definition.filename
            if not source.exists():
                print(f"downloading {source_definition.url} to {source}")
                download(source_definition.url, source)
            sources.append(source)
    prepare(args.system, sources, output, args.train_size, args.test_size, args.seed)


if __name__ == "__main__":
    main()
