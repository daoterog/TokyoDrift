from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from data.summarize import evenly_spaced_indices, summarize_dataset


class DataSummaryTests(unittest.TestCase):
    def test_evenly_spaced_indices_are_unique_and_cover_the_split(self) -> None:
        indices = evenly_spaced_indices(100, 4)
        np.testing.assert_array_equal(indices, np.array([0, 25, 50, 75]))

    def test_summary_writes_statistics_and_split_plots(self) -> None:
        rng = np.random.default_rng(7)
        splits = {}
        for name, size in (("train", 8), ("validation", 10), ("test", 12)):
            values = rng.normal(size=(size, 4, 2)).astype(np.float32)
            splits[name] = values - values.mean(axis=1, keepdims=True)
        metadata = np.array(json.dumps({"system": "dw4"}))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.npz"
            np.savez_compressed(dataset, **splits, metadata=metadata)
            output = summarize_dataset(
                dataset,
                root / "reports",
                max_configurations=6,
                max_observations=100,
                batch_size=3,
                bins=12,
            )

            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["splits"]["train"]["configurations"], 8)
            self.assertEqual(report["splits"]["validation"]["sampled_configurations"], 6)
            self.assertIn("energy", report["splits"]["test"]["metrics"])
            for filename in (
                "all_splits_distributions.png",
                "train_distributions.png",
                "validation_distributions.png",
                "test_distributions.png",
            ):
                self.assertTrue((output / filename).is_file())


if __name__ == "__main__":
    unittest.main()
