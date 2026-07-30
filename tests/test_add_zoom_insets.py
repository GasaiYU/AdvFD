import unittest
from pathlib import Path

from PIL import Image

from my_tools.add_zoom_insets import (
    _default_output_path,
    get_args_parser,
    make_zoom_strip,
)


class MakeZoomStripTest(unittest.TestCase):
    def test_default_output_is_next_to_input(self) -> None:
        input_path = Path("/images/example.png")
        self.assertEqual(
            _default_output_path(input_path),
            Path("/images/example_1x4.png"),
        )

    def test_output_argument_is_optional(self) -> None:
        args = get_args_parser().parse_args(
            ["--input", "example.png", "--roi", "0", "0", "1", "1"]
        )
        self.assertIsNone(args.output)

    def test_builds_four_columns_in_expected_order(self) -> None:
        left = Image.new("RGB", (4, 4), "red")
        right = Image.new("RGB", (4, 4), "blue")
        left.putpixel((1, 1), (0, 255, 0))
        right.putpixel((2, 2), (255, 255, 0))

        pair = Image.new("RGB", (9, 4), "black")
        pair.paste(left, (0, 0))
        pair.paste(right, (5, 0))
        result = make_zoom_strip(
            pair,
            left_roi=(1, 1, 2, 2),
            right_roi=(2, 2, 3, 3),
            panel_gap=1,
            draw_roi=False,
            roi_width=1,
            roi_color=(255, 215, 0),
        )

        self.assertEqual(result.size, (16, 4))
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result.getpixel((4, 0)), (0, 255, 0))
        self.assertEqual(result.getpixel((8, 0)), (0, 0, 255))
        self.assertEqual(result.getpixel((12, 0)), (255, 255, 0))

    def test_roi_boxes_are_only_drawn_on_full_image_columns(self) -> None:
        pair = Image.new("RGB", (8, 4), "black")
        result = make_zoom_strip(
            pair,
            left_roi=(1, 1, 3, 3),
            right_roi=(1, 1, 3, 3),
            panel_gap=0,
            draw_roi=True,
            roi_width=1,
            roi_color=(255, 215, 0),
        )

        self.assertEqual(result.getpixel((1, 1)), (255, 215, 0))
        self.assertEqual(result.getpixel((9, 1)), (255, 215, 0))
        self.assertEqual(result.getpixel((5, 1)), (0, 0, 0))
        self.assertEqual(result.getpixel((13, 1)), (0, 0, 0))

    def test_rejects_roi_outside_a_panel(self) -> None:
        pair = Image.new("RGB", (8, 4), "black")
        with self.assertRaisesRegex(ValueError, "exceeds panel size"):
            make_zoom_strip(
                pair,
                left_roi=(3, 3, 5, 5),
                right_roi=(0, 0, 1, 1),
                panel_gap=0,
                draw_roi=False,
                roi_width=1,
                roi_color=(255, 215, 0),
            )

    def test_parser_no_longer_accepts_inset_options(self) -> None:
        parser = get_args_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--input",
                    "input.png",
                    "--output",
                    "output.png",
                    "--roi",
                    "0",
                    "0",
                    "1",
                    "1",
                    "--inset_size",
                    "10",
                    "10",
                ]
            )


if __name__ == "__main__":
    unittest.main()
