from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from my_tools.make_selected_qualitative_grid import (
    get_args_parser,
    run,
)


class MakeSelectedQualitativeGridTest(unittest.TestCase):
    def test_left_and_right_halves_keep_matching_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "pairs"
            input_dir.mkdir()
            filenames = []
            for index in range(8):
                filename = f"{index:05d}_pair.png"
                filenames.append(filename)
                pair = Image.new("RGB", (32, 16))
                pair.paste((index + 1, 0, 0), (0, 0, 16, 16))
                pair.paste((0, index + 1, 0), (16, 0, 32, 16))
                pair.save(input_dir / filename)

            output_path = root / "grid.png"
            args = get_args_parser().parse_args(
                [
                    "--input_dir",
                    str(input_dir),
                    "--filenames",
                    *filenames,
                    "--output",
                    str(output_path),
                    "--seed",
                    "17",
                    "--unit",
                    "4",
                    "--group_gap",
                    "8",
                    "--outer_padding",
                    "2",
                    "--no_titles",
                ]
            )
            run(args)

            manifest_path = output_path.with_suffix(".json")
            with manifest_path.open() as handle:
                manifest = json.load(handle)
            self.assertEqual(len(manifest["layout"]), 8)
            self.assertTrue(manifest["shuffled"])

            with Image.open(output_path) as grid:
                self.assertEqual(grid.size, (132, 36))
                for item in manifest["layout"]:
                    source_index = int(item["filename"].split("_")[0])
                    left_box = item["left_box"]
                    right_box = item["right_box"]
                    left_pixel = grid.getpixel(
                        (
                            left_box["x"] + left_box["width"] // 2,
                            left_box["y"] + left_box["height"] // 2,
                        )
                    )
                    right_pixel = grid.getpixel(
                        (
                            right_box["x"] + right_box["width"] // 2,
                            right_box["y"] + right_box["height"] // 2,
                        )
                    )
                    self.assertEqual(
                        left_pixel,
                        (source_index + 1, 0, 0),
                    )
                    self.assertEqual(
                        right_pixel,
                        (0, source_index + 1, 0),
                    )


if __name__ == "__main__":
    unittest.main()
