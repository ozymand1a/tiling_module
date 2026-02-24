import argparse
from pathlib import Path

import cv2
from src.slice import slice_image
from src.utils.constants import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Slice an image and save tiles.")
    parser.add_argument(
        "--image",
        default=str(PROJECT_ROOT / "data/1.jpg"),
        help="Input image path",
    )
    parser.add_argument(
        "--result-dir",
        default=PROJECT_ROOT / "result",
        help="Base folder for outputs; artifacts go under result/slice/<image_name>/",
    )
    parser.add_argument("--slice-height", type=int, default=640)
    parser.add_argument("--slice-width", type=int, default=640)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.2)
    args = parser.parse_args()

    image_path = args.image
    image_stem = Path(image_path).stem
    out_dir = Path(args.result_dir) / "slice" / image_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_image_result = slice_image(
        image=image_path,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        auto_slice_resolution=False,
    )

    for sliced_image in slice_image_result.sliced_image_list:
        x, y = sliced_image.starting_pixel[0], sliced_image.starting_pixel[1]
        path = out_dir / f"sliced_image_{x}_{y}.jpg"
        cv2.imwrite(
            str(path),
            cv2.cvtColor(sliced_image.image, cv2.COLOR_RGB2BGR),
        )

    print(f"Saved {len(slice_image_result.sliced_image_list)} slices to {out_dir}")


if __name__ == "__main__":
    main()

"""
PYTHONPATH="." python3 scripts/check_slicing.py
"""
