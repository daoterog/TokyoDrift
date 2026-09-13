from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from utils.io import device_summary, select_device


class DeviceSelectionTests(unittest.TestCase):
    def test_explicit_cpu_is_always_valid(self) -> None:
        self.assertEqual(select_device("cpu"), torch.device("cpu"))

    @patch("utils.io.torch.cuda.is_available", return_value=False)
    def test_explicit_unavailable_cuda_has_actionable_error(self, _mock: object) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDA dependency group"):
            select_device("cuda")

    @patch("utils.io.torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda(self, _mock: object) -> None:
        self.assertEqual(select_device("auto"), torch.device("cuda"))

    @patch("utils.io._mps_available", return_value=True)
    @patch("utils.io.torch.cuda.is_available", return_value=False)
    def test_auto_uses_mps_after_cuda(self, _cuda: object, _mps: object) -> None:
        self.assertEqual(select_device("auto"), torch.device("mps"))

    def test_cpu_summary_is_serializable(self) -> None:
        summary = device_summary(torch.device("cpu"))
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["device_name"], "CPU")


if __name__ == "__main__":
    unittest.main()
