import torch
import numpy as np
import importlib.util
from collections.abc import Sequence

from src.tile.objects import ObjectPrediction

def check_requirements(package_names):
    """Raise error if module is not installed."""
    missing_packages = []
    for package_name in package_names:
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package_name)
    if missing_packages:
        raise ImportError(f"The following packages are required to use this module: {missing_packages}")
    yield


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
        if torch.is_tensor(i) or isinstance(i, np.ndarray):
            i = i.tolist()
        if isinstance(i, int):
            return ObjectPredictionList([self.list[i]])
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(self.list.__getitem__, i)
            return ObjectPredictionList(list(accessed_mapping))
        else:
            raise NotImplementedError(f"{type(i)}")

    def __setitem__(self, i, elem):
        if torch.is_tensor(i) or isinstance(i, np.ndarray):
            i = i.tolist()
        if isinstance(i, int):
            self.list[i] = elem
        elif isinstance(i, (tuple, list)):
            if len(i) != len(elem):
                raise ValueError()
            if isinstance(elem, ObjectPredictionList):
                for ind, el in enumerate(elem.list):
                    self.list[i[ind]] = el
            else:
                for ind, el in enumerate(elem):
                    self.list[i[ind]] = el
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
        if len(self.list) == 1:
            return self.list[0]
        else:
            return self.list


def object_prediction_list_to_torch(object_prediction_list: ObjectPredictionList) -> torch.tensor:
    """
    Returns:
        torch.tensor of size N x [x1, y1, x2, y2, score, category_id]
    """
    num_predictions = len(object_prediction_list)
    torch_predictions = torch.zeros([num_predictions, 6], dtype=torch.float32)
    for ind, object_prediction in enumerate(object_prediction_list):
        torch_predictions[ind, :4] = torch.tensor(object_prediction.tolist().bbox.to_xyxy(), dtype=torch.float32)
        torch_predictions[ind, 4] = object_prediction.tolist().score.value
        torch_predictions[ind, 5] = object_prediction.tolist().category.id
    return torch_predictions


def object_prediction_list_to_numpy(object_prediction_list: ObjectPredictionList) -> np.ndarray:
    """
    Returns:
        np.ndarray of size N x [x1, y1, x2, y2, score, category_id]
    """
    num_predictions = len(object_prediction_list)
    numpy_predictions = np.zeros([num_predictions, 6], dtype=np.float32)
    for ind, object_prediction in enumerate(object_prediction_list):
        numpy_predictions[ind, :4] = np.array(object_prediction.tolist().bbox.to_xyxy(), dtype=np.float32)
        numpy_predictions[ind, 4] = object_prediction.tolist().score.value
        numpy_predictions[ind, 5] = object_prediction.tolist().category.id
    return numpy_predictions


from src.postprocess.nmm import NMMPostprocess
from src.postprocess.nms import NMSPostprocess
from src.postprocess.wbf import WBFPostprocess

__all__ = ["PostprocessPredictions", "NMMPostprocess", "NMSPostprocess", "WBFPostprocess"]
