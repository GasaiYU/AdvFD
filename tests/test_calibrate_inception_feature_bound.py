from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.calibrate_inception_feature_bound import (
    extract_pool3_norms,
    resolve_imagefolder_root,
    select_dataset_indices,
    summarize_norms,
)


class CalibrateInceptionFeatureBoundTest(unittest.TestCase):
    def test_feature_norm_extraction_uses_pool3_and_float64_reduction(self) -> None:
        class FakeInception(torch.nn.Module):
            def forward(self, images: torch.Tensor):
                pool3 = torch.zeros(images.shape[0], 2048, dtype=torch.float32)
                pool3[:, 0] = images[:, 0, 0, 0]
                pool3[:, 1] = images[:, 0, 0, 1]
                return pool3, torch.zeros(images.shape[0], 1008)

        images = torch.zeros(2, 3, 2, 2)
        images[0, 0, 0, :2] = torch.tensor([3.0, 4.0])
        images[1, 0, 0, :2] = torch.tensor([5.0, 12.0])
        loader = DataLoader(TensorDataset(images, torch.zeros(2)), batch_size=2)

        norms = extract_pool3_norms(FakeInception(), loader, torch.device("cpu"), rank=1)

        self.assertEqual(norms.dtype, torch.float64)
        torch.testing.assert_close(norms, torch.tensor([5.0, 13.0], dtype=torch.float64))

    def test_random_sampling_is_unique_exact_and_reproducible(self) -> None:
        targets = np.repeat(np.arange(5), 20)
        first = select_dataset_indices(targets, 37, "random", seed=11)
        second = select_dataset_indices(targets, 37, "random", seed=11)

        self.assertEqual(first.size, 37)
        self.assertEqual(np.unique(first).size, 37)
        np.testing.assert_array_equal(first, second)

    def test_stratified_sampling_preserves_balanced_classes(self) -> None:
        targets = np.repeat(np.arange(10), 100)
        selected = select_dataset_indices(targets, 100, "stratified", seed=7)
        counts = np.bincount(targets[selected], minlength=10)

        self.assertEqual(selected.size, 100)
        np.testing.assert_array_equal(counts, np.full(10, 10))

    def test_summary_uses_requested_quantile_and_margin(self) -> None:
        norms = np.arange(1.0, 101.0)
        summary = summarize_norms(norms, calibration_quantile=0.9, margin=1.05)
        expected_quantile = float(np.quantile(norms, 0.9))

        self.assertAlmostEqual(summary["calibration_quantile_value"], expected_quantile)
        self.assertAlmostEqual(summary["recommended_B"], 1.05 * expected_quantile)
        self.assertEqual(summary["count"], 100)

    def test_resolve_imagefolder_root_accepts_parent_or_train_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train"
            train.mkdir()

            self.assertEqual(resolve_imagefolder_root(root), train)
            self.assertEqual(resolve_imagefolder_root(train), train)


if __name__ == "__main__":
    unittest.main()
