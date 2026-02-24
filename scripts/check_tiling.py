import argparse
from pathlib import Path

import cv2
import numpy as np
from sahi.models.ultralytics import UltralyticsDetectionModel
from src.slice.slicing import slice_image
from src.tile.tiling import get_prediction, get_sliced_prediction
from src.visualizations.visualize import draw_bboxes
from src.utils.constants import PROJECT_ROOT

# Same slice params as tiled prediction (keep in sync)
SLICE_HEIGHT = 480
SLICE_WIDTH = 480
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2


def main():
    parser = argparse.ArgumentParser(
        description="Run tiled and full-image detection, visualize both for comparison."
    )
    parser.add_argument(
        "--image",
        default=str(PROJECT_ROOT / "data/1.jpg"),
        help="Input image path",
    )
    parser.add_argument(
        "--result-dir",
        default=PROJECT_ROOT / "result",
        help="Base folder for outputs; artifacts go under result/tile/<image_name>/",
    )
    parser.add_argument(
        "--no-label-names",
        action="store_true",
        help="Do not draw class names on bboxes",
    )
    parser.add_argument(
        "--no-confs",
        action="store_true",
        help="Do not draw confidence scores on bboxes",
    )
    args = parser.parse_args()

    image_path = args.image
    image_stem = Path(image_path).stem
    out_dir = Path(args.result_dir) / "tile" / image_stem
    tiles_dir = out_dir / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    detection_model = UltralyticsDetectionModel(
        model_path="yolo26m.pt",
        confidence_threshold=0.3,
        device="cpu",
    )
    draw_opts = {
        "with_label_names": not args.no_label_names,
        "with_confs": not args.no_confs,
    }

    # 1) Tiled (sliced) prediction
    print("Running tiled prediction...")
    result_tiled = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        auto_slice_resolution=False,
        postprocess_type="NMM",
    )
    n_tiled = len(result_tiled.object_prediction_list)
    print(f"  Tiled: {n_tiled} objects")

    image_rgb = np.asarray(result_tiled.image)
    vis_tiled = draw_bboxes(image_rgb, result_tiled.object_prediction_list, **draw_opts)
    path_tiled = out_dir / "tiled.jpg"
    cv2.imwrite(str(path_tiled), cv2.cvtColor(vis_tiled, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {path_tiled}")

    # 2) Run detector on each tile and save (detector runs per-tile via get_prediction)
    print(f"Saving tiles with detections to {tiles_dir}...")
    slice_result = slice_image(
        image=image_path,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        auto_slice_resolution=False,
    )
    for i, (slice_img, start_xy) in enumerate(
        zip(slice_result.images, slice_result.starting_pixels)
    ):
        h, w = slice_img.shape[:2]
        # Run detector on this tile only; bboxes are in tile-local coordinates
        result_slice = get_prediction(
            image=slice_img,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=[h, w],
        )
        vis_slice = draw_bboxes(
            slice_img, result_slice.object_prediction_list, **draw_opts
        )
        tile_path = tiles_dir / f"tile_{i:04d}_x{start_xy[0]}_y{start_xy[1]}.jpg"
        cv2.imwrite(str(tile_path), cv2.cvtColor(vis_slice, cv2.COLOR_RGB2BGR))
    print(f"  Saved {len(slice_result.images)} tiles to {tiles_dir}")

    # 3) Full-image prediction (no tiling)
    print("Running full-image prediction (no tiling)...")
    result_full = get_prediction(image=image_path, detection_model=detection_model)
    n_full = len(result_full.object_prediction_list)
    print(f"  Full:  {n_full} objects")

    image_rgb_full = np.asarray(result_full.image)
    vis_full = draw_bboxes(
        image_rgb_full, result_full.object_prediction_list, **draw_opts
    )
    path_full = out_dir / "full.jpg"
    cv2.imwrite(str(path_full), cv2.cvtColor(vis_full, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {path_full}")

    print(f"Compare: {path_tiled} vs {path_full}")


if __name__ == "__main__":
    main()

"""
PYTHONPATH="." python3 scripts/check_tiling.py
PYTHONPATH="." python3 scripts/check_tiling.py --no-label-names --no-confs
"""
