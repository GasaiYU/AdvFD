from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hacking.apply_fourier_pattern import (
    get_args_parser as get_apply_args_parser,
    load_spatial_pattern,
    main as apply_main,
)
from hacking.extract_spectral_pattern import (
    decode_cache_images,
    discover_cache,
    discover_image_directory,
    extract_spectral_pattern,
    gaussian_highpass,
    synthesize_pattern,
    write_outputs,
)
from hacking.pmf_fourier_universal import SpatialBandpassPattern


def _write_synthetic_cache(
    root: Path,
    *,
    world_size: int = 2,
    images_per_rank: int = 6,
    resolution: int = 16,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    all_images: list[torch.Tensor] = []
    fingerprint = "synthetic-cache-fingerprint"
    for rank in range(world_size):
        rank_dir = root / "optimization" / f"rank_{rank:05d}"
        rank_dir.mkdir(parents=True)
        base = torch.randint(
            0,
            256,
            (images_per_rank, 1, resolution, resolution),
            generator=generator,
            dtype=torch.uint8,
        )
        images = torch.cat(
            [
                base,
                ((base.to(torch.int16) * 3 + 17) % 256).to(torch.uint8),
                (255 - base),
            ],
            dim=1,
        )
        all_images.append(images)
        entries: list[dict[str, object]] = []
        for chunk_index, start in enumerate(range(0, images_per_rank, 3)):
            chunk = images[start : start + 3]
            chunk_path = rank_dir / f"chunk_{chunk_index:05d}.pt"
            torch.save({"images": chunk}, chunk_path)
            entries.append(
                {
                    "path": str(chunk_path.relative_to(root)),
                    "start": start,
                    "count": chunk.shape[0],
                }
            )
        index = {
            "version": 1,
            "fingerprint": fingerprint,
            "split": "optimization",
            "rank": rank,
            "world_size": world_size,
            "count": images_per_rank,
            "dtype": "uint8",
            "entries": entries,
        }
        with (rank_dir / "index.json").open("w") as handle:
            json.dump(index, handle)
    with (root / "manifest.json").open("w") as handle:
        json.dump({"fingerprint": fingerprint, "spec": {}}, handle)
    return torch.cat(all_images, dim=0)


class SpectralPatternTest(unittest.TestCase):
    def test_spatial_bandpass_pattern_is_nonzero_unit_rms(self) -> None:
        pattern = SpatialBandpassPattern(
            size=16,
            min_radius=0.1,
            max_radius=0.5,
        )
        generator = torch.Generator().manual_seed(87)
        with torch.no_grad():
            pattern.coeff.copy_(
                torch.randn(
                    pattern.coeff.shape,
                    generator=generator,
                )
            )
            pattern.normalize_coefficients_()
        applied = pattern.patch(normalize=True)
        self.assertEqual(tuple(applied.shape), (1, 3, 16, 16))
        torch.testing.assert_close(
            applied.mean(dim=(-2, -1)),
            torch.zeros(1, 3),
            atol=1e-6,
            rtol=0,
        )
        self.assertAlmostEqual(
            float(applied.detach().square().mean().sqrt()),
            1.0,
            places=5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "spatial.pth"
            torch.save(
                {"spatial_pattern": applied[0]},
                checkpoint_path,
            )
            loaded = load_spatial_pattern(checkpoint_path)
            torch.testing.assert_close(loaded, applied[0])

    def test_parallel_pattern_application_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            generator = np.random.default_rng(31)
            for index in range(12):
                image = generator.integers(
                    0, 256, size=(16, 16, 3), dtype=np.uint8
                )
                Image.fromarray(image).save(
                    input_dir / f"{index:06d}.png"
                )
            pattern = generator.normal(
                0.0, 0.02, size=(3, 16, 16)
            ).astype(np.float32)
            pattern_path = root / "pattern.npy"
            np.save(pattern_path, pattern)

            base_arguments = [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--pattern",
                str(pattern_path),
                "--alpha",
                "0.2",
                "--alpha_space",
                "pixel",
                "--preserve_pattern_scale",
                "--output_format",
                "png",
                "--num_workers",
                "4",
                "--log_every",
                "5",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                apply_main(
                    get_apply_args_parser().parse_args(base_arguments)
                )
            outputs = sorted(output_dir.glob("*.png"))
            self.assertEqual(len(outputs), 12)
            outputs[3].write_bytes(b"truncated png")

            with contextlib.redirect_stdout(io.StringIO()):
                apply_main(
                    get_apply_args_parser().parse_args(
                        [*base_arguments, "--resume"]
                    )
                )
            with (
                output_dir / "apply_pattern_manifest.json"
            ).open() as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["processed_this_run"], 1)
            self.assertEqual(manifest["resumed_existing"], 11)
            self.assertEqual(manifest["repaired_invalid_images"], 1)
            self.assertEqual(len(list(output_dir.glob("*.png"))), 12)
            with Image.open(outputs[3]) as repaired:
                repaired.load()

    def test_flat_image_directory_matches_uint8_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_root = root / "cache"
            expected_images = _write_synthetic_cache(cache_root)
            image_root = root / "images"
            image_root.mkdir()
            for index, tensor in enumerate(expected_images):
                array = tensor.permute(1, 2, 0).numpy()
                Image.fromarray(array).save(
                    image_root / f"{index:06d}.png"
                )

            cache_inventory = discover_cache(
                cache_root, "optimization"
            )
            image_inventory = discover_image_directory(image_root)
            self.assertEqual(
                image_inventory.total_images, expected_images.shape[0]
            )
            kwargs = {
                "num_images": expected_images.shape[0],
                "batch_size": 5,
                "device": torch.device("cpu"),
                "blur_sigma": 1.0,
                "seed": 2026,
                "log_every": 100,
            }
            from_cache = extract_spectral_pattern(
                cache_inventory, **kwargs
            )
            from_images = extract_spectral_pattern(
                image_inventory, **kwargs
            )
            np.testing.assert_allclose(
                from_cache.pattern,
                from_images.pattern,
                rtol=3e-4,
                atol=3e-4,
            )
            np.testing.assert_allclose(
                from_cache.cross_spectrum,
                from_images.cross_spectrum,
                rtol=1e-5,
                atol=1e-8,
            )

    def test_cache_discovery_and_deterministic_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            expected_images = _write_synthetic_cache(root)
            inventory = discover_cache(root, "optimization")
            self.assertEqual(inventory.total_images, expected_images.shape[0])
            self.assertEqual(inventory.dtype_name, "uint8")
            self.assertEqual(
                [(entry.rank, entry.start) for entry in inventory.entries],
                [
                    (0, 0),
                    (0, 3),
                    (1, 0),
                    (1, 3),
                ],
            )

            kwargs = {
                "num_images": expected_images.shape[0],
                "batch_size": 4,
                "device": torch.device("cpu"),
                "blur_sigma": 1.0,
                "seed": 2026,
                "log_every": 100,
            }
            first = extract_spectral_pattern(inventory, **kwargs)
            second = extract_spectral_pattern(inventory, **kwargs)
            np.testing.assert_array_equal(first.pattern, second.pattern)
            np.testing.assert_allclose(
                first.cross_spectrum,
                second.cross_spectrum,
                rtol=0,
                atol=0,
            )
            self.assertEqual(first.pattern.shape, (3, 16, 16))
            self.assertEqual(
                first.cross_spectrum.shape, (16, 9, 3, 3)
            )
            np.testing.assert_allclose(
                first.pattern.mean(axis=(1, 2)),
                np.zeros(3),
                atol=1e-6,
            )
            self.assertAlmostEqual(
                float(np.sqrt(np.mean(first.pattern**2))),
                1.0,
                places=5,
            )
            self.assertTrue(np.isfinite(first.radial_psd_cosine))

            hermitian_error = np.max(
                np.abs(
                    first.cross_spectrum
                    - first.cross_spectrum.conj().swapaxes(-1, -2)
                )
            )
            self.assertLess(float(hermitian_error), 1e-10)
            eigenvalues = np.linalg.eigvalsh(first.cross_spectrum)
            self.assertGreater(float(eigenvalues.min()), -1e-9)

    def test_outputs_are_compatible_and_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            _write_synthetic_cache(root)
            inventory = discover_cache(root, "optimization")
            result = extract_spectral_pattern(
                inventory,
                num_images=12,
                batch_size=5,
                device=torch.device("cpu"),
                blur_sigma=1.0,
                seed=2026,
                log_every=100,
            )
            output = Path(temporary) / "output"
            paths = write_outputs(
                result,
                inventory,
                output,
                overwrite=False,
                elapsed_seconds=0.1,
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))
            loaded = load_spatial_pattern(paths["spectral_pattern.npy"])
            np.testing.assert_allclose(
                loaded.numpy(),
                result.pattern,
                atol=1e-6,
            )
            with paths["manifest.json"].open() as handle:
                manifest = json.load(handle)
            self.assertIsNone(manifest["training"]["optimizer"])
            self.assertFalse(manifest["training"]["backward"])
            self.assertFalse(
                manifest["training"]["metric_guided_selection"]
            )
            with self.assertRaises(FileExistsError):
                write_outputs(
                    result,
                    inventory,
                    output,
                    overwrite=False,
                    elapsed_seconds=0.1,
                )

    def test_decode_highpass_and_seeded_synthesis(self) -> None:
        images = torch.tensor(
            [[[[0, 255], [128, 64]]] * 3],
            dtype=torch.uint8,
        )
        decoded = decode_cache_images(
            images,
            dtype_name="uint8",
            device=torch.device("cpu"),
        )
        self.assertGreaterEqual(float(decoded.min()), 0.0)
        self.assertLessEqual(float(decoded.max()), 1.0)

        constant = torch.ones(2, 3, 16, 16)
        residual = gaussian_highpass(constant, sigma=1.0)
        self.assertLess(float(residual.abs().max()), 1e-6)

        identity_spectrum = torch.eye(
            3, dtype=torch.complex128
        ).expand(16, 9, 3, 3).clone()
        first, _ = synthesize_pattern(
            identity_spectrum,
            spatial_width=16,
            seed=1,
        )
        repeated, _ = synthesize_pattern(
            identity_spectrum,
            spatial_width=16,
            seed=1,
        )
        different, _ = synthesize_pattern(
            identity_spectrum,
            spatial_width=16,
            seed=2,
        )
        preserved, _ = synthesize_pattern(
            identity_spectrum,
            spatial_width=16,
            seed=1,
            normalize_rms=False,
        )
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assertFalse(torch.equal(first, different))
        preserved_rms = preserved.square().mean().sqrt()
        torch.testing.assert_close(
            first,
            preserved / preserved_rms,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_pattern_loader_can_preserve_stored_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pattern_path = Path(temporary) / "pattern.npy"
            generator = np.random.default_rng(9)
            stored = generator.normal(
                0.0, 0.025, size=(3, 16, 16)
            ).astype(np.float32)
            stored -= stored.mean(axis=(1, 2), keepdims=True)
            np.save(pattern_path, stored)

            preserved = load_spatial_pattern(
                pattern_path, normalize_rms=False
            )
            normalized = load_spatial_pattern(pattern_path)
            np.testing.assert_allclose(
                preserved.numpy(), stored, rtol=1e-6, atol=1e-7
            )
            self.assertAlmostEqual(
                float(normalized.square().mean().sqrt()),
                1.0,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
