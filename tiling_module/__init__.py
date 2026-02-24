"""
Tiling module: image slicing and sliced object-detection prediction.

Primary API for use as a submodule:

- slice_image: slice a large image into overlapping windows.
- get_sliced_prediction: slice image, run detection on each slice, merge results.

Usage:

    from tiling_module import slice_image, get_sliced_prediction
    # or
    from tiling_module import slice_image
    from tiling_module import get_sliced_prediction
"""

from src.slice import get_slice_bboxes, slice_image
from src.slice.objects import SliceImageResult, SlicedImage
from src.tile import ObjectPrediction, get_prediction, get_sliced_prediction
from src.tile.objects import PredictionResult

__all__ = [
    "get_slice_bboxes",
    "get_prediction",
    "get_sliced_prediction",
    "ObjectPrediction",
    "PredictionResult",
    "SliceImageResult",
    "slice_image",
    "SlicedImage",
]
