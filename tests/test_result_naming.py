import json
import unittest
from pathlib import Path

from utils.result_naming import result_folder_name


class ResultFolderNamingTests(unittest.TestCase):
    def test_all_variants(self):
        for normalized, descriptors, expected in (
            (False, False, "17_gnn_unnorm_nodescr"),
            (False, True, "17_gnn_unnorm_descr"),
            (True, False, "17_gnn_norm_nodescr"),
            (True, True, "17_gnn_norm_descr"),
        ):
            with self.subTest(normalized=normalized, descriptors=descriptors):
                config = {
                    "model": {"architecture": "gnn"},
                    "drift": {"normalized": normalized, "descriptors": descriptors},
                }
                self.assertEqual(result_folder_name(config, "17"), expected)

    def test_checked_in_particle_configs_have_expected_names(self):
        root = Path(__file__).parents[1]
        expected = {
            "dw4": "example_egnn_unnorm_nodescr",
            "lj13": "example_egnn_unnorm_descr",
            "lj55": "example_egnn_unnorm_descr",
        }
        for system, folder_name in expected.items():
            with self.subTest(system=system):
                config = json.loads((root / f"configs/{system}_config.json").read_text())
                self.assertEqual(result_folder_name(config, "example"), folder_name)

    def test_requires_explicit_descriptor_setting(self):
        config = {
            "model": {"architecture": "egnn"},
            "drift": {"normalized": False},
        }
        with self.assertRaisesRegex(ValueError, "drift.descriptors"):
            result_folder_name(config, "example")

    def test_run_id_may_contain_underscores(self):
        config = {
            "model": {"architecture": "egnn"},
            "drift": {"normalized": False, "descriptors": False},
        }
        self.assertEqual(
            result_folder_name(config, "experiment_17"),
            "experiment_17_egnn_unnorm_nodescr",
        )


if __name__ == "__main__":
    unittest.main()
