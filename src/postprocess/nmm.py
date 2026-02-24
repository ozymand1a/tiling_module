import numpy as np
import torch

from src.postprocess import ObjectPredictionList, PostprocessPredictions
from src.postprocess.annotation import BoundingBox
from src.postprocess.utils import (
    box_intersection_area_ij,
    calculate_box_union,
    has_match,
    query_overlapping_indices,
)
from src.tile.objects import ObjectPrediction


def batched_nmm(
    object_predictions_as_tensor: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """Apply non-maximum merging per category to avoid detecting too many overlapping bounding boxes for a given object.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = {}
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = nmm(
            object_predictions_as_tensor[curr_indices], match_metric, match_threshold
        )
        curr_indices_list = curr_indices.tolist()
        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_indices_list[curr_keep]
            merge_list = [
                curr_indices_list[curr_merge_ind] for curr_merge_ind in curr_merge_list
            ]
            keep_to_merge_list[keep] = merge_list
    return keep_to_merge_list


def nmm(
    object_predictions_as_tensor: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """Apply non-maximum merging to avoid detecting too many overlapping bounding boxes for a given object.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    # Extract coordinates and scores as tensors
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]
    scores = object_predictions_as_tensor[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    # Work with numpy for indexing in helpers
    x1_np = x1.numpy() if torch.is_tensor(x1) else np.asarray(x1)
    y1_np = y1.numpy() if torch.is_tensor(y1) else np.asarray(y1)
    x2_np = x2.numpy() if torch.is_tensor(x2) else np.asarray(x2)
    y2_np = y2.numpy() if torch.is_tensor(y2) else np.asarray(y2)
    areas_np = areas.numpy() if torch.is_tensor(areas) else np.asarray(areas)
    scores_np = scores.numpy() if torch.is_tensor(scores) else np.asarray(scores)

    n = len(object_predictions_as_tensor)
    sorted_idxs = torch.argsort(scores, descending=True).tolist()

    keep_to_merge_list = {}
    merge_to_keep = {}

    for current_idx in sorted_idxs:
        current_area = float(areas_np[current_idx])
        candidate_idxs = query_overlapping_indices(
            x1_np, y1_np, x2_np, y2_np, current_idx, n
        )

        matched_box_indices = []
        for candidate_idx in candidate_idxs:
            if scores_np[candidate_idx] > scores_np[current_idx]:
                continue

            if scores_np[candidate_idx] == scores_np[current_idx]:
                current_coords = (
                    float(x1_np[current_idx]),
                    float(y1_np[current_idx]),
                    float(x2_np[current_idx]),
                    float(y2_np[current_idx]),
                )
                candidate_coords = (
                    float(x1_np[candidate_idx]),
                    float(y1_np[candidate_idx]),
                    float(x2_np[candidate_idx]),
                    float(y2_np[candidate_idx]),
                )
                if candidate_coords > current_coords:
                    continue

            intersection = box_intersection_area_ij(
                x1_np, y1_np, x2_np, y2_np, current_idx, candidate_idx
            )
            area_j = float(areas_np[candidate_idx])

            if match_metric == "IOU":
                union = current_area + area_j - intersection
                metric = intersection / union if union > 0 else 0.0
            elif match_metric == "IOS":
                smaller = min(current_area, area_j)
                metric = intersection / smaller if smaller > 0 else 0.0
            else:
                raise ValueError("Invalid match_metric")

            if metric >= match_threshold:
                matched_box_indices.append(candidate_idx)

        # Convert current_idx to native Python int
        current_idx_native = int(current_idx)

        # Create keep_ind to merge_ind_list mapping
        if current_idx_native not in merge_to_keep:
            keep_to_merge_list[current_idx_native] = []

            for matched_box_idx in matched_box_indices:
                matched_box_idx_native = int(matched_box_idx)
                if matched_box_idx_native not in merge_to_keep:
                    keep_to_merge_list[current_idx_native].append(
                        matched_box_idx_native
                    )
                    merge_to_keep[matched_box_idx_native] = current_idx_native
        else:
            keep_idx = merge_to_keep[current_idx_native]
            for matched_box_idx in matched_box_indices:
                matched_box_idx_native = int(matched_box_idx)
                if (
                    matched_box_idx_native not in keep_to_merge_list.get(keep_idx, [])
                    and matched_box_idx_native not in merge_to_keep
                ):
                    if keep_idx not in keep_to_merge_list:
                        keep_to_merge_list[keep_idx] = []
                    keep_to_merge_list[keep_idx].append(matched_box_idx_native)
                    merge_to_keep[matched_box_idx_native] = keep_idx

    return keep_to_merge_list


def get_merged_score(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
    return max(pred1.score.value, pred2.score.value)


def get_merged_bbox(pred1: ObjectPrediction, pred2: ObjectPrediction):
    return BoundingBox(
        box=calculate_box_union(pred1.bbox.to_xyxy(), pred2.bbox.to_xyxy())
    )


def get_merged_category(pred1: ObjectPrediction, pred2: ObjectPrediction):
    return pred1.category if pred1.score.value > pred2.score.value else pred2.category


def merge_object_prediction_pair(
    pred1: ObjectPrediction, pred2: ObjectPrediction
) -> ObjectPrediction:
    cat = get_merged_category(pred1, pred2)
    return ObjectPrediction(
        bbox=get_merged_bbox(pred1, pred2).to_xyxy(),
        score=get_merged_score(pred1, pred2),
        category_id=cat.id,
        category_name=cat.name,
        segmentation=None,
        shift_amount=pred1.bbox.shift_amount,
        full_shape=None,
    )


class NMMPostprocess(PostprocessPredictions):
    def __call__(self, object_predictions: list[ObjectPrediction]):
        object_prediction_list = ObjectPredictionList(object_predictions)
        tensor = object_prediction_list.totensor()
        nmm_fn = nmm if self.class_agnostic else batched_nmm
        keep_to_merge_list = nmm_fn(
            tensor,
            match_threshold=self.match_threshold,
            match_metric=self.match_metric,
        )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    self.match_metric,
                    self.match_threshold,
                ):
                    object_prediction_list[keep_ind] = merge_object_prediction_pair(
                        object_prediction_list[keep_ind].tolist(),
                        object_prediction_list[merge_ind].tolist(),
                    )
            selected_object_predictions.append(
                object_prediction_list[keep_ind].tolist()
            )

        return selected_object_predictions
