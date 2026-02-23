import torch
import numpy as np
from shapely import STRtree, box

from src.tile.objects import ObjectPrediction
from src.postprocess import PostprocessPredictions, ObjectPredictionList
from src.postprocess.utils import has_match, calculate_box_union
from src.postprocess.annotation import BoundingBox


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
        curr_keep_to_merge_list = nmm(object_predictions_as_tensor[curr_indices], match_metric, match_threshold)
        curr_indices_list = curr_indices.tolist()
        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_indices_list[curr_keep]
            merge_list = [curr_indices_list[curr_merge_ind] for curr_merge_ind in curr_merge_list]
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

    # Calculate areas as tensor (vectorized operation)
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes only once
    boxes = []
    for i in range(len(object_predictions_as_tensor)):
        boxes.append(
            box(
                x1[i].item(),  # Convert only individual values
                y1[i].item(),
                x2[i].item(),
                y2[i].item(),
            )
        )

    # Sort indices by score (descending) using torch
    sorted_idxs = torch.argsort(scores, descending=True).tolist()

    # Build STRtree
    tree = STRtree(boxes)

    keep_to_merge_list = {}
    merge_to_keep = {}

    for current_idx in sorted_idxs:
        current_box = boxes[current_idx]
        current_area = areas[current_idx].item()  # Convert only when needed

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        matched_box_indices = []
        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx:
                continue

            # Only consider candidates with lower or equal score
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                # Use box coordinates for stable ordering
                current_coords = (
                    x1[current_idx].item(),
                    y1[current_idx].item(),
                    x2[current_idx].item(),
                    y2[current_idx].item(),
                )
                candidate_coords = (
                    x1[candidate_idx].item(),
                    y1[candidate_idx].item(),
                    x2[candidate_idx].item(),
                    y2[candidate_idx].item(),
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx].item() - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx].item())
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError("Invalid match_metric")

            # Add to matched list if overlap exceeds threshold
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
                    keep_to_merge_list[current_idx_native].append(matched_box_idx_native)
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


def get_merged_score(
    pred1: ObjectPrediction,
    pred2: ObjectPrediction,
) -> float:
    scores: list[float] = [pred.score.value for pred in (pred1, pred2)]
    return max(scores)


def get_merged_bbox(pred1: ObjectPrediction, pred2: ObjectPrediction):
    box1: list[int] = pred1.bbox.to_xyxy()
    box2: list[int] = pred2.bbox.to_xyxy()
    bbox = BoundingBox(box=calculate_box_union(box1, box2))
    return bbox


def get_merged_category(pred1: ObjectPrediction, pred2: ObjectPrediction):
    if pred1.score.value > pred2.score.value:
        return pred1.category
    else:
        return pred2.category


def merge_object_prediction_pair(
    pred1: ObjectPrediction,
    pred2: ObjectPrediction,
) -> ObjectPrediction:
    shift_amount = pred1.bbox.shift_amount
    merged_bbox = get_merged_bbox(pred1, pred2)
    merged_score: float = get_merged_score(pred1, pred2)
    merged_category = get_merged_category(pred1, pred2)
    return ObjectPrediction(
        bbox=merged_bbox.to_xyxy(),
        score=merged_score,
        category_id=merged_category.id,
        category_name=merged_category.name,
        segmentation=None,
        shift_amount=shift_amount,
        full_shape=None,
    )


class NMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep_to_merge_list = nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep_to_merge_list = batched_nmm(
                object_predictions_as_torch,
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
                        object_prediction_list[keep_ind].tolist(), object_prediction_list[merge_ind].tolist()
                    )
            selected_object_predictions.append(object_prediction_list[keep_ind].tolist())

        return selected_object_predictions