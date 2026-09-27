from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from data.prepare import arguments, prepare, reshape_and_center, sha256
from data.systems import DatasetSource, ParticleSystem


class DataPreparationTests(unittest.TestCase):
    def test_particle_defaults_create_the_shared_three_way_split(self) -> None:
        with patch("sys.argv", ["data.prepare", "lj13"]):
            args = arguments()
        self.assertEqual(
            (args.train_size, args.validation_size, args.test_size),
            (100_000, 400_000, 500_000),
        )

    def test_particle_configs_use_validation_tracking(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for system in ("dw4", "lj13", "lj55"):
            with self.subTest(system=system):
                config = json.loads((root / f"configs/{system}_config.json").read_text())
                training = config["training"]
                self.assertEqual(training["validation_split"], "validation")
                self.assertTrue(training["track_train_validation_metrics"])
                self.assertNotIn("track_train_test_metrics", training)

    def test_ordered_preparation_writes_train_validation_and_test_splits(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.npy"
            output_path = root / "dataset.npz"
            coordinates = np.arange(40, dtype=np.float32).reshape(10, 4)
            coordinates[:, 0] += np.arange(10, dtype=np.float32) ** 2
            np.save(source_path, coordinates)
            system = ParticleSystem(
                name="toy",
                particles=2,
                dimensions=2,
                energy=None,
                sources=(
                    DatasetSource(
                        filename=source_path.name,
                        url="https://example.invalid/source.npy",
                        sha256=sha256(source_path),
                        shape=coordinates.shape,
                    ),
                ),
                paper_reference={},
            )

            with patch("data.prepare.get_system", return_value=system):
                prepare(
                    "toy",
                    source_path,
                    output_path,
                    train_size=2,
                    validation_size=3,
                    test_size=5,
                    seed=2023,
                    ordered=True,
                )

            with np.load(output_path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"train", "validation", "test", "metadata"})
                np.testing.assert_array_equal(
                    archive["train"], reshape_and_center(coordinates[:2], 2, 2)
                )
                np.testing.assert_array_equal(
                    archive["validation"], reshape_and_center(coordinates[2:5], 2, 2)
                )
                np.testing.assert_array_equal(
                    archive["test"], reshape_and_center(coordinates[5:], 2, 2)
                )


if __name__ == "__main__":
    unittest.main()
