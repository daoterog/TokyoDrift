"""Aggregate repeated evaluation JSON files into means and deviations."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    """Parse summary command-line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate repeated particle-drift evaluations.")
    parser.add_argument("metrics", nargs="+", help="Metric files or wildcard patterns.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def flatten_numeric(prefix: str, value: object, output: dict[str, float]) -> None:
    """Flatten nested numeric values into dotted metric names."""
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else key, nested, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def expand_metric_paths(patterns: list[str]) -> list[Path]:
    """Expand wildcards consistently on POSIX shells and Windows PowerShell."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
            else:
                raise FileNotFoundError(f"no metric files matched {pattern!r}")
    return list(dict.fromkeys(paths))


def main() -> None:
    """Aggregate compatible evaluation records."""
    args = arguments()
    records = []
    for path in expand_metric_paths(args.metrics):
        with path.open() as stream:
            records.append(json.load(stream))
    systems = {record["system"] for record in records}
    if len(systems) != 1:
        raise ValueError(f"all evaluations must use one system, got {sorted(systems)}")
    flattened: list[dict[str, float]] = []
    for record in records:
        values: dict[str, float] = {}
        flatten_numeric("sample_metrics", record["sample_metrics"], values)
        values["endpoint_transport_distance_per_particle"] = record[
            "endpoint_transport_distance_per_particle"
        ]
        flattened.append(values)
    common = sorted(set.intersection(*(set(record) for record in flattened)))
    summary = {
        "system": records[0]["system"],
        "runs": len(records),
        "metrics": {
            key: {
                "mean": float(np.mean([record[key] for record in flattened])),
                "std": float(np.std([record[key] for record in flattened], ddof=1))
                if len(records) > 1
                else 0.0,
            }
            for key in common
        },
        "paper_reference_results": records[0]["paper_metrics"]["reference_results"],
        "likelihood_metric_note": records[0]["paper_metrics"]["unnormalized_drift"]["reason"],
    }
    rendered = json.dumps(summary, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
