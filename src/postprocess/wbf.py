import warnings

import numpy as np

from src.tile.objects import ObjectPrediction
from src.postprocess import PostprocessPredictions


# -----------------------------------------------------------------------------
# Core WBF logic (unchanged)
# -----------------------------------------------------------------------------


def prefilter_boxes(boxes, scores, labels, weights, thr):
    """Create dict with boxes stored by its label."""
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print(
                "Error. Length of boxes arrays not equal to length of scores array: {} != {}".format(
                    len(boxes[t]), len(scores[t])
                )
            )
            exit()

        if len(boxes[t]) != len(labels[t]):
            print(
                "Error. Length of boxes arrays not equal to length of labels array: {} != {}".format(
                    len(boxes[t]), len(labels[t])
                )
            )
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn("X2 < X1 value in box. Swap them.")
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn("Y2 < Y1 value in box. Swap them.")
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn("X1 < 0 in box. Set it to 0.")
                x1 = 0
            if x1 > 1:
                warnings.warn("X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.")
                x1 = 1
            if x2 < 0:
                warnings.warn("X2 < 0 in box. Set it to 0.")
                x2 = 0
            if x2 > 1:
                warnings.warn("X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.")
                x2 = 1
            if y1 < 0:
                warnings.warn("Y1 < 0 in box. Set it to 0.")
                y1 = 0
            if y1 > 1:
                warnings.warn("Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.")
                y1 = 1
            if y2 < 0:
                warnings.warn("Y2 < 0 in box. Set it to 0.")
                y2 = 0
            if y2 > 1:
                warnings.warn("Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.")
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type="avg"):
    """
    Create weighted box for set of boxes.

    Args:
        boxes: set of boxes to fuse
        conf_type: type of confidence one of 'avg' or 'max'
    Returns:
        weighted box (label, score, weight, model index, x1, y1, x2, y2)
    """
    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += b[1] * b[4:]
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ("avg", "box_and_model_avg", "absent_model_aware_avg"):
        box[1] = conf / len(boxes)
    elif conf_type == "max":
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = -1  # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """
    Reimplementation of find_matching_box with numpy instead of loops.
    Gives significant speed up for larger arrays (~100x).
    """

    def bb_iou_array(boxes, new_box):
        # bb intersection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    boxes = boxes_list
    ious = bb_iou_array(boxes[:, 4:], new_box[4:])
    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion(
    boxes_list,
    scores_list,
    labels_list,
    weights=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    allows_overflow=False,
):
    """
    Weighted boxes fusion.

    Args:
        boxes_list: list of boxes predictions from each model, each box is 4 numbers.
            Shape (models_number, model_preds, 4). Order: x1, y1, x2, y2. Normalized [0; 1].
        scores_list: list of scores for each model
        labels_list: list of labels for each model
        weights: list of weights per model. Default: None (weight 1 per model)
        iou_thr: IoU threshold for boxes to be a match
        skip_box_thr: exclude boxes with score lower than this
        conf_type: 'avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'
        allows_overflow: if False, confidence score is capped at 1.0

    Returns:
        boxes: (N, 4) x1, y1, x2, y2
        scores: (N,)
        labels: (N,)
    """
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print(
            "Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.".format(
                len(weights), len(boxes_list)
            )
        )
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ["avg", "max", "box_and_model_avg", "absent_model_aware_avg"]:
        print(
            'Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(
                conf_type
            )
        )
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 8))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            if conf_type == "box_and_model_avg":
                clustered_boxes = np.array(clustered_boxes)
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                weighted_boxes[i, 1] = (
                    weighted_boxes[i, 1] * clustered_boxes[idx, 2].sum() / weights.sum()
                )
            elif conf_type == "absent_model_aware_avg":
                clustered_boxes = np.array(clustered_boxes)
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                weighted_boxes[i, 1] = (
                    weighted_boxes[i, 1]
                    * len(clustered_boxes)
                    / (weighted_boxes[i, 2] + weights[mask].sum())
                )
            elif conf_type == "max":
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = (
                    weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
                )
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels


# -----------------------------------------------------------------------------
# ObjectPrediction conversion helpers (same style as nmm.py)
# -----------------------------------------------------------------------------


def object_predictions_to_wbf_input(
    object_predictions: list[ObjectPrediction],
    normalize: bool = True,
):
    """
    Convert list of ObjectPrediction to WBF input format.

    Args:
        object_predictions: list of ObjectPrediction
        normalize: if True, normalize box coords to [0, 1] by max extent (required by prefilter_boxes)

    Returns:
        boxes_list: list of one array (N, 4) for single-model
        scores_list: list of one array (N,)
        labels_list: list of one array (N,) int category_id
        scale: (width, height) used for normalization; use to denormalize after WBF
        label_id_to_name: dict category_id -> category_name for building ObjectPredictions
    """
    if not object_predictions:
        return [], [], [], (1.0, 1.0), {}

    boxes = np.array([obj.bbox.to_xyxy() for obj in object_predictions], dtype=np.float32)
    scores = np.array([obj.score.value for obj in object_predictions], dtype=np.float32)
    labels = np.array([obj.category.id for obj in object_predictions], dtype=np.int32)
    label_id_to_name = {obj.category.id: obj.category.name for obj in object_predictions}

    if normalize:
        width = float(max(boxes[:, 2].max(), 1.0))
        height = float(max(boxes[:, 3].max(), 1.0))
        scale = (width, height)
        boxes = boxes.copy()
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height
    else:
        scale = (1.0, 1.0)

    boxes_list = [boxes]
    scores_list = [scores]
    labels_list = [labels]
    return boxes_list, scores_list, labels_list, scale, label_id_to_name


def wbf_output_to_object_predictions(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    scale: tuple[float, float],
    label_id_to_name: dict[int, str],
    shift_amount: tuple[int, int] = (0, 0),
) -> list[ObjectPrediction]:
    """
    Convert WBF output (boxes, scores, labels) back to list[ObjectPrediction].

    Args:
        boxes: (N, 4) x1,y1,x2,y2 (normalized if scale != (1,1))
        scores: (N,)
        labels: (N,) category_id
        scale: (width, height) to denormalize coords
        label_id_to_name: mapping category_id -> name
        shift_amount: shift for bbox (e.g. slice offset)
    """
    width, height = scale
    result = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]
        x1 = x1 * width
        x2 = x2 * width
        y1 = y1 * height
        y2 = y2 * height
        category_id = int(labels[i])
        category_name = label_id_to_name.get(category_id, "?")
        result.append(
            ObjectPrediction(
                bbox=[x1, y1, x2, y2],
                score=float(scores[i]),
                category_id=category_id,
                category_name=category_name,
                segmentation=None,
                shift_amount=list(shift_amount),
                full_shape=None,
            )
        )
    return result


# -----------------------------------------------------------------------------
# Postprocess class (same interface as NMMPostprocess)
# -----------------------------------------------------------------------------


class WBFPostprocess(PostprocessPredictions):
    """Postprocess predictions using Weighted Boxes Fusion."""

    def __init__(
        self,
        match_threshold: float = 0.1,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
        *,
        iou_thr: float | None = None,
        skip_box_thr: float = 0.0,
        conf_type: str = "avg",
        allows_overflow: bool = False,
    ):
        super().__init__(
            match_threshold=match_threshold,
            match_metric=match_metric,
            class_agnostic=class_agnostic,
        )
        self.iou_thr = iou_thr if iou_thr is not None else match_threshold
        self.skip_box_thr = skip_box_thr
        self.conf_type = conf_type
        self.allows_overflow = allows_overflow

    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ) -> list[ObjectPrediction]:
        if not object_predictions:
            return []

        (
            boxes_list,
            scores_list,
            labels_list,
            scale,
            label_id_to_name,
        ) = object_predictions_to_wbf_input(object_predictions, normalize=True)

        shift_amount = (0, 0)
        if hasattr(object_predictions[0].bbox, "shift_amount"):
            shift_amount = tuple(object_predictions[0].bbox.shift_amount)

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=None,
            iou_thr=self.iou_thr,
            skip_box_thr=self.skip_box_thr,
            conf_type=self.conf_type,
            allows_overflow=self.allows_overflow,
        )

        return wbf_output_to_object_predictions(
            boxes,
            scores,
            labels,
            scale=scale,
            label_id_to_name=label_id_to_name,
            shift_amount=shift_amount,
        )
