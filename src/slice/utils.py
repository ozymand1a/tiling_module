from typing import Literal

import numpy as np


def get_resolution_selector(
    res: str, height: int, width: int
) -> tuple[int, int, int, int]:
    """Get resolution selector."""
    orientation = calc_aspect_ratio_orientation(width=width, height=height)
    x_overlap, y_overlap, slice_width, slice_height = calc_slice_and_overlap_params(
        resolution=res, height=height, width=width, orientation=orientation
    )

    return x_overlap, y_overlap, slice_width, slice_height


def calc_resolution_factor(resolution: int) -> int:
    """
    According to image resolution calculate power(2,n) and return the closest smaller `n`.
    Args:
        resolution: the width and height of the image multiplied. such as 1024x720 = 737280

    Returns:

    """
    expo = 0
    while np.power(2, expo) < resolution:
        expo += 1

    return expo - 1


def calc_slice_and_overlap_params(
    resolution: str, height: int, width: int, orientation: str
) -> tuple[int, int, int, int]:
    """
    This function calculate according to image resolution slice and overlap params.
    Args:
        resolution: str
        height: int
        width: int
        orientation: str

    Returns:
        x_overlap, y_overlap, slice_width, slice_height
    """

    if resolution == "medium":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = (
            calc_ratio_and_slice(orientation, slide=1, ratio=0.8)
        )

    elif resolution == "high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = (
            calc_ratio_and_slice(orientation, slide=2, ratio=0.4)
        )

    elif resolution == "ultra-high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = (
            calc_ratio_and_slice(orientation, slide=4, ratio=0.4)
        )
    else:  # low condition
        split_col = 1
        split_row = 1
        overlap_width_ratio = 1
        overlap_height_ratio = 1

    slice_height = height // split_col
    slice_width = width // split_row

    x_overlap = int(slice_width * overlap_width_ratio)
    y_overlap = int(slice_height * overlap_height_ratio)

    return x_overlap, y_overlap, slice_width, slice_height


def calc_ratio_and_slice(
    orientation: Literal["vertical", "horizontal", "square"],
    slide: int = 1,
    ratio: float = 0.1,
):
    """Return (slice_row, slice_col, overlap_height_ratio, overlap_width_ratio) for given orientation."""
    mapping = {
        "vertical": (slide, slide * 2, ratio, ratio),
        "horizontal": (slide * 2, slide, ratio, ratio),
        "square": (slide, slide, ratio, ratio),
    }
    if orientation not in mapping:
        raise ValueError(
            f"Invalid orientation: {orientation}. Must be one of 'vertical', 'horizontal', or 'square'."
        )
    return mapping[orientation]


def calc_aspect_ratio_orientation(width: int, height: int) -> str:
    """Return image orientation: 'vertical', 'horizontal', or 'square'."""
    return (
        "vertical" if width < height else "horizontal" if width > height else "square"
    )


def get_auto_slice_params(height: int, width: int) -> tuple[int, int, int, int]:
    """
    According to Image HxW calculate overlap sliding window and buffer params
    factor is the power value of 2 closest to the image resolution.
        factor <= 18: low resolution image such as 300x300, 640x640
        18 < factor <= 21: medium resolution image such as 1024x1024, 1336x960
        21 < factor <= 24: high resolution image such as 2048x2048, 2048x4096, 4096x4096
        factor > 24: ultra-high resolution image such as 6380x6380, 4096x8192
    Args:
        height:
        width:

    Returns:
        slicing overlap params x_overlap, y_overlap, slice_width, slice_height
    """
    factor = calc_resolution_factor(height * width)
    if factor <= 18:
        res = "low"
    elif factor < 21:
        res = "medium"
    elif factor < 24:
        res = "high"
    else:
        res = "ultra-high"
    return get_resolution_selector(res, height=height, width=width)
