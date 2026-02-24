import numpy as np

from src.tile.objects import ObjectPrediction


def box_intersection_area_ij(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, i: int, j: int
) -> float:
    """Intersection area of two axis-aligned boxes at indices i and j (arrays are x1,y1,x2,y2 per box)."""
    a = np.array([float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])])
    b = np.array([float(x1[j]), float(y1[j]), float(x2[j]), float(y2[j])])
    return calculate_intersection_area(a, b)


def query_overlapping_indices(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    current_idx: int,
    n: int,
) -> list[int]:
    """Indices j != current_idx where box j overlaps (AABB) the box at current_idx."""
    x1_c, y1_c, x2_c, y2_c = (
        x1[current_idx],
        y1[current_idx],
        x2[current_idx],
        y2[current_idx],
    )
    return [
        j
        for j in range(n)
        if j != current_idx
        and x1[j] < x2_c
        and x1_c < x2[j]
        and y1[j] < y2_c
        and y1_c < y2[j]
    ]


def calculate_box_union(
    box1: list[int] | np.ndarray, box2: list[int] | np.ndarray
) -> list[int]:
    """Union of two boxes [x1, y1, x2, y2]."""
    a, b = np.array(box1), np.array(box2)
    return list(np.concatenate((np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:]))))


def calculate_area(box: list[int] | np.ndarray) -> float:
    """Area of box [x1, y1, x2, y2]."""
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_intersection_area(box1: np.ndarray, box2: np.ndarray) -> float:
    """Intersection area of two boxes [x1, y1, x2, y2]."""
    lt = np.maximum(box1[:2], box2[:2])
    rb = np.minimum(box1[2:], box2[2:])
    wh = (rb - lt).clip(min=0)
    return wh[0] * wh[1]


def calculate_bbox_iou(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
    """Returns the ratio of intersection area to the union."""
    box1 = np.array(pred1.bbox.to_xyxy())
    box2 = np.array(pred2.bbox.to_xyxy())
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    intersect = calculate_intersection_area(box1, box2)
    return intersect / (area1 + area2 - intersect)


def calculate_bbox_ios(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
    """Returns the ratio of intersection area to the smaller box's area."""
    box1 = np.array(pred1.bbox.to_xyxy())
    box2 = np.array(pred2.bbox.to_xyxy())
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    intersect = calculate_intersection_area(box1, box2)
    smaller_area = np.minimum(area1, area2)
    return intersect / smaller_area


def has_match(
    pred1: ObjectPrediction,
    pred2: ObjectPrediction,
    match_type: str = "IOU",
    match_threshold: float = 0.5,
) -> bool:
    calc = {"IOU": calculate_bbox_iou, "IOS": calculate_bbox_ios}
    if match_type not in calc:
        raise ValueError(f"match_type must be IOU or IOS, got {match_type}")
    return calc[match_type](pred1, pred2) > match_threshold
