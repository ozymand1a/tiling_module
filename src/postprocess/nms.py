import torch
from shapely import STRtree, box
import numpy as np

from src.tile.objects import ObjectPrediction
from src.postprocess import PostprocessPredictions, ObjectPredictionList


def batched_nms(predictions: torch.tensor, match_metric: str = "IOU", match_threshold: float = 0.5):
    """Apply non-maximum suppression to avoid detecting too many overlapping bounding boxes for a given object.

    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    scores = predictions[:, 4].squeeze()
    category_ids = predictions[:, 5].squeeze()
    keep_mask = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_indices = nms(predictions[curr_indices], match_metric, match_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    # sort selected indices by their scores
    keep_indices = keep_indices[scores[keep_indices].sort(descending=True)[1]].tolist()
    return keep_indices


def nms(
    predictions: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Optimized non-maximum suppression for axis-aligned bounding boxes using STRTree.

    Args:
        predictions: (tensor) The location preds for the image along with the class
            predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.

    Returns:
        A list of filtered indexes, Shape: [ ,]
    """
    if len(predictions) == 0:
        return []

    # Ensure predictions are on CPU and convert to numpy
    if predictions.device.type != "cpu":
        predictions = predictions.cpu()

    predictions_np = predictions.numpy()

    # Extract coordinates and scores
    x1 = predictions_np[:, 0]
    y1 = predictions_np[:, 1]
    x2 = predictions_np[:, 2]
    y2 = predictions_np[:, 3]
    scores = predictions_np[:, 4]

    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes (vectorized)
    boxes = box(x1, y1, x2, y2)

    # Sort indices by score (descending)
    sorted_idxs = np.argsort(scores)[::-1]

    # Build STRtree
    tree = STRtree(boxes)

    keep = []
    suppressed = set()

    for current_idx in sorted_idxs:
        if current_idx in suppressed:
            continue

        keep.append(current_idx)
        current_box = boxes[current_idx]
        current_area = areas[current_idx]

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx or candidate_idx in suppressed:
                continue

            # Skip candidates with higher scores (already processed)
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                # Use box coordinates for stable ordering
                current_coords = (
                    x1[current_idx],
                    y1[current_idx],
                    x2[current_idx],
                    y2[current_idx],
                )
                candidate_coords = (
                    x1[candidate_idx],
                    y1[candidate_idx],
                    x2[candidate_idx],
                    y2[candidate_idx],
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx] - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx])
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError("Invalid match_metric")

            # Suppress if overlap exceeds threshold
            if metric >= match_threshold:
                suppressed.add(candidate_idx)

    return keep


class NMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep = nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, match_metric=self.match_metric
            )
        else:
            keep = batched_nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, match_metric=self.match_metric
            )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions
