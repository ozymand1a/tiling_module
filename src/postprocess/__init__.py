import importlib.util
from collections.abc import Sequence

import numpy as np
import torch

from src.tile.objects import ObjectPrediction


def check_requirements(package_names):
    """Raise ImportError if any package is not installed."""
    missing = [p for p in package_names if importlib.util.find_spec(p) is None]
    if missing:
        raise ImportError(
            f"The following packages are required to use this module: {missing}"
        )


class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions."""

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ):
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric

        check_requirements(["torch"])

    def __call__(self, predictions: list[ObjectPrediction]):
        raise NotImplementedError()


class ObjectPredictionList(Sequence):
    def __init__(self, list):
        self.list = list
        super().__init__()

    def __getitem__(self, i):
        i = i.tolist() if torch.is_tensor(i) or isinstance(i, np.ndarray) else i
        if isinstance(i, int):
            return ObjectPredictionList([self.list[i]])
        if isinstance(i, (tuple, list)):
            return ObjectPredictionList([self.list[j] for j in i])
        raise NotImplementedError(f"{type(i)}")

    def __setitem__(self, i, elem):
        i = i.tolist() if torch.is_tensor(i) or isinstance(i, np.ndarray) else i
        if isinstance(i, int):
            self.list[i] = elem
        elif isinstance(i, (tuple, list)):
            if len(i) != len(elem):
                raise ValueError()
            vals = elem.list if isinstance(elem, ObjectPredictionList) else elem
            for j, el in enumerate(vals):
                self.list[i[j]] = el
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self):
        return len(self.list)

    def __str__(self):
        return str(self.list)

    def extend(self, object_prediction_list):
        self.list.extend(object_prediction_list.list)

    def totensor(self):
        return object_prediction_list_to_torch(self)

    def tonumpy(self):
        return object_prediction_list_to_numpy(self)

    def tolist(self):
        return self.list[0] if len(self.list) == 1 else self.list


def object_prediction_list_to_torch(
    object_prediction_list: ObjectPredictionList,
) -> torch.tensor:
    """Return tensor of shape N x 6: [x1, y1, x2, y2, score, category_id]."""
    n = len(object_prediction_list)
    out = torch.zeros([n, 6], dtype=torch.float32)
    for i, pred in enumerate(object_prediction_list.list):
        out[i, :4] = torch.tensor(pred.bbox.to_xyxy(), dtype=torch.float32)
        out[i, 4] = pred.score.value
        out[i, 5] = pred.category.id
    return out


def object_prediction_list_to_numpy(
    object_prediction_list: ObjectPredictionList,
) -> np.ndarray:
    """Return array of shape N x 6: [x1, y1, x2, y2, score, category_id]."""
    n = len(object_prediction_list)
    out = np.zeros([n, 6], dtype=np.float32)
    for i, pred in enumerate(object_prediction_list.list):
        out[i, :4] = np.array(pred.bbox.to_xyxy(), dtype=np.float32)
        out[i, 4] = pred.score.value
        out[i, 5] = pred.category.id
    return out


from src.postprocess.nmm import NMMPostprocess
from src.postprocess.nms import NMSPostprocess
from src.postprocess.wbf import WBFPostprocess

__all__ = [
    "PostprocessPredictions",
    "NMMPostprocess",
    "NMSPostprocess",
    "WBFPostprocess",
]
