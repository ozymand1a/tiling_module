"""
Draw bounding boxes on images for tiling/slicing prediction results.
Works with SAHI-style ObjectPrediction (bbox, category.name, score).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

# Distinct, readable palette (BGR for cv2) – avoid pure white/yellow on light bg
_PALETTE_BGR = [
    (41, 128, 255),  # orange
    (255, 128, 41),  # blue
    (41, 255, 128),  # green
    (255, 41, 128),  # magenta
    (128, 41, 255),  # purple
    (128, 255, 41),  # lime
    (255, 200, 100),  # light blue
    (100, 200, 255),  # peach
    (200, 255, 100),  # mint
    (180, 130, 70),  # teal
    (70, 130, 180),  # brown
    (130, 70, 180),  # violet
]


def _bbox_xyxy(obj: Any) -> tuple[int, int, int, int]:
    """Get (x1, y1, x2, y2) from SAHI ObjectPrediction or any .bbox."""
    bbox = obj.bbox
    coords = bbox.to_xyxy() if hasattr(bbox, "to_xyxy") else bbox
    return tuple(int(round(x)) for x in coords[:4])


def _get_color(
    category_id: int | None, category_name: str | None
) -> tuple[int, int, int]:
    """Stable color per class (BGR)."""
    if category_id is not None:
        idx = category_id % len(_PALETTE_BGR)
    else:
        idx = hash(category_name or "") % len(_PALETTE_BGR)
    return _PALETTE_BGR[idx % len(_PALETTE_BGR)]


def _get_score(obj: Any) -> float:
    """Get confidence from ObjectPrediction."""
    s = getattr(obj, "score", None)
    if s is None:
        return 0.0
    return getattr(s, "value", s) if not isinstance(s, (int, float)) else float(s)


def draw_bboxes(
    image: np.ndarray,
    object_predictions: Sequence[Any],
    *,
    with_label_names: bool = True,
    with_confs: bool = True,
    box_thickness: int | None = None,
    font_scale: float | None = None,
    label_padding: int = 4,
    alpha_bg: float = 0.85,
) -> np.ndarray:
    """
    Draw bounding boxes on a copy of the image.

    Works with SAHI-style objects that have:
      .bbox (list [x1,y1,x2,y2] or object with .to_xyxy()),
      .category.name (or .category_name),
      .category.id (or .category_id),
      .score (.value if not a number).

    Args:
        image: HWC numpy array (RGB or BGR).
        object_predictions: List of ObjectPrediction-like objects.
        with_label_names: If True, draw class name above the box.
        with_confs: If True, append confidence to the label (e.g. "car 0.92").
        box_thickness: Line thickness; default scales with image size.
        font_scale: Text scale; default scales with image size.
        label_padding: Padding around label text (px).
        alpha_bg: Opacity of label background (0–1).

    Returns:
        New image with drawings (RGB if input was RGB, BGR if BGR).
    """
    img = image.copy()
    h, w = img.shape[:2]
    scale = min(h, w)

    if box_thickness is None:
        box_thickness = max(round(scale * 0.004), 2)
    if font_scale is None:
        font_scale = max(scale / 800, 0.35)
    text_thickness = max(box_thickness, 1)

    for obj in object_predictions:
        x1, y1, x2, y2 = _bbox_xyxy(obj)
        # Clip to image
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue

        name = (
            getattr(getattr(obj, "category", None), "name", None)
            or getattr(obj, "category_name", "")
            or "?"
        )
        category_id = getattr(getattr(obj, "category", None), "id", None) or getattr(
            obj, "category_id", None
        )
        color = _get_color(category_id, name)
        score = _get_score(obj)

        # Label text
        if with_label_names:
            label = name
            if with_confs:
                label += f" {score:.2f}"
        elif with_confs:
            label = f"{score:.2f}"
        else:
            label = None

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        if label:
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )
            th += label_padding * 2
            tw += label_padding * 2
            # Label above box if space, else below
            if y1 - th - 2 >= 0:
                ly1 = y1 - th - 2
                ly2 = y1 - 2
            else:
                ly1 = y1 + 2
                ly2 = y1 + th + 2
            lx1 = x1
            lx2 = min(x1 + tw, w)
            ly1 = max(0, ly1)
            ly2 = min(h, ly2)
            if ly2 > ly1 and lx2 > lx1:
                overlay = img.copy()
                cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
                cv2.addWeighted(overlay, alpha_bg, img, 1 - alpha_bg, 0, img)
                # Border around label
                cv2.rectangle(
                    img, (lx1, ly1), (lx2, ly2), color, max(1, box_thickness - 1)
                )
            tx = x1 + label_padding
            if y1 - th - 2 >= 0:
                ty = y1 - 2 - label_padding  # baseline just above box
            else:
                ty = ly1 + th - label_padding  # baseline inside label below box
            ty = max(th, min(ty, h - 2))
            cv2.putText(
                img,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

    return img
