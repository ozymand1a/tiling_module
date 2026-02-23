"""
Draw detection boxes on image. Uses only cv2 and numpy.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.onnx_engine import Detection

_COLORS = [
    (255, 56, 56),
    (168, 153, 44),
    (31, 112, 255),
    (255, 127, 65),
    (48, 245, 10),
    (23, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (255, 157, 151),
    (0, 194, 255),
    (52, 69, 147),
    (255, 29, 29),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]
def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    thickness: int | None = None,
    font_scale: float | None = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image. Modifies a copy; image can be RGB or BGR.

    Args:
        image: HWC numpy array (RGB or BGR).
        detections: List of Detection.
        thickness: Line thickness; default auto from image size.
        font_scale: Text scale; default auto.
        hide_labels: If True, do not draw class names.
        hide_conf: If True, do not draw confidence in label.

    Returns:
        Image with drawings (copy).
    """
    img = image.copy()
    if thickness is None:
        thickness = max(round(min(img.shape[:2]) * 0.003), 2)
    if font_scale is None:
        font_scale = thickness / 3
    text_thickness = max(thickness - 1, 1)

    for det in detections:
        x1, y1, x2, y2 = [int(round(x)) for x in det.bbox]
        cid = det.class_id % len(_COLORS)
        color = _COLORS[cid]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if not hide_labels:
            label = det.class_name
            if not hide_conf:
                label += f" {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            outside = y1 - th - 3 >= 0
            y_label = y1 - 2 if outside else y1 + th + 2
            cv2.rectangle(
                img,
                (x1, y1 - th - 3 if outside else y1),
                (x1 + tw, y1 + 3 if outside else y1 + th + 3),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1, y_label),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness,
            )
    return img
