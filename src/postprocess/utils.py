import numpy as np

from src.tile.objects import ObjectPrediction


def calculate_box_union(box1: list[int] | np.ndarray, box2: list[int] | np.ndarray) -> list[int]:
    """
    Args:
        box1 (List[int]): [x1, y1, x2, y2]
        box2 (List[int]): [x1, y1, x2, y2]
    """
    box1 = np.array(box1)
    box2 = np.array(box2)
    left_top = np.minimum(box1[:2], box2[:2])
    right_bottom = np.maximum(box1[2:], box2[2:])
    return list(np.concatenate((left_top, right_bottom)))


def calculate_area(box: list[int] | np.ndarray) -> float:
    """
    Args:
        box (List[int]): [x1, y1, x2, y2]
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_intersection_area(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Args:
        box1 (np.ndarray): np.array([x1, y1, x2, y2])
        box2 (np.ndarray): np.array([x1, y1, x2, y2])
    """
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])
    width_height = (right_bottom - left_top).clip(min=0)
    return width_height[0] * width_height[1]

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
    pred1: ObjectPrediction, pred2: ObjectPrediction, match_type: str = "IOU", match_threshold: float = 0.5
) -> bool:
    if match_type == "IOU":
        threshold_condition = calculate_bbox_iou(pred1, pred2) > match_threshold
    elif match_type == "IOS":
        threshold_condition = calculate_bbox_ios(pred1, pred2) > match_threshold
    else:
        raise ValueError()
    return threshold_condition

