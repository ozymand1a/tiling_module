import numpy as np
import torch

from src.postprocess import ObjectPredictionList, PostprocessPredictions
from src.postprocess.utils import box_intersection_area_ij, query_overlapping_indices
from src.tile.objects import ObjectPrediction


def batched_nms(
    predictions: torch.tensor, match_metric: str = "IOU", match_threshold: float = 0.5
):
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
        curr_keep_indices = nms(
            predictions[curr_indices], match_metric, match_threshold
        )
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
    Non-maximum suppression for axis-aligned bounding boxes.

    Args:
        predictions: (tensor) [num_boxes, 6] - x1, y1, x2, y2, score, category_id.
        match_metric: IOU or IOS
        match_threshold: overlap threshold.

    Returns:
        List of kept indices.
    """
    if len(predictions) == 0:
        return []

    predictions_np = (
        predictions.cpu().numpy()
        if predictions.device.type != "cpu"
        else predictions.numpy()
    )
    x1 = predictions_np[:, 0]
    y1 = predictions_np[:, 1]
    x2 = predictions_np[:, 2]
    y2 = predictions_np[:, 3]
    scores = predictions_np[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    n = len(predictions_np)
    sorted_idxs = np.argsort(scores)[::-1]

    keep = []
    suppressed = set()

    for current_idx in sorted_idxs:
        if current_idx in suppressed:
            continue
        keep.append(current_idx)
        current_area = float(areas[current_idx])
        candidate_idxs = query_overlapping_indices(x1, y1, x2, y2, current_idx, n)

        for candidate_idx in candidate_idxs:
            if candidate_idx in suppressed:
                continue
            if scores[candidate_idx] > scores[current_idx]:
                continue
            if scores[candidate_idx] == scores[current_idx]:
                current_coords = (
                    float(x1[current_idx]),
                    float(y1[current_idx]),
                    float(x2[current_idx]),
                    float(y2[current_idx]),
                )
                candidate_coords = (
                    float(x1[candidate_idx]),
                    float(y1[candidate_idx]),
                    float(x2[candidate_idx]),
                    float(y2[candidate_idx]),
                )
                if candidate_coords > current_coords:
                    continue

            intersection = box_intersection_area_ij(
                x1, y1, x2, y2, current_idx, candidate_idx
            )
            area_j = float(areas[candidate_idx])

            if match_metric == "IOU":
                union = current_area + area_j - intersection
                metric = intersection / union if union > 0 else 0.0
            elif match_metric == "IOS":
                smaller = min(current_area, area_j)
                metric = intersection / smaller if smaller > 0 else 0.0
            else:
                raise ValueError("Invalid match_metric")

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
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep = batched_nms(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        sel = object_prediction_list[keep].tolist()
        return sel if isinstance(sel, list) else [sel]
