import copy
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any
from src.utils.cv import read_image_as_pil
from src.postprocess.annotation import BoundingBox


class Category:
    """Minimal category with id and name for ObjectPrediction."""

    def __init__(self, id: int | None = None, name: str | None = None):
        self.id = id
        self.name = name or "?"


class PredictionScore:
    def __init__(self, value: float | np.ndarray):
        """
        Args:
            score: prediction score between 0 and 1
        """
        # if score is a numpy object, convert it to python variable
        if type(value).__module__ == "numpy":
            value = copy.deepcopy(value).tolist()
        # set score
        self.value = value

    def is_greater_than_threshold(self, threshold):
        """Check if score is greater than threshold."""
        return self.value > threshold

    def __eq__(self, threshold):
        return self.value == threshold

    def __gt__(self, threshold):
        return self.value > threshold

    def __lt__(self, threshold):
        return self.value < threshold

    def __repr__(self):
        return f"PredictionScore: <value: {self.value}>"


class ObjectPrediction:
    """Class for handling detection model predictions."""

    def __init__(
        self,
        bbox: list[int] | None = None,
        category_id: int | None = None,
        category_name: str | None = None,
        segmentation: list[list[float]] | None = None,
        score: float = 0.0,
        shift_amount: list[int] | None = [0, 0],
        full_shape: list[int] | None = None,
    ):
        """Creates ObjectPrediction from bbox, score, category_id, category_name, segmentation.

        Args:
            bbox: list
                [minx, miny, maxx, maxy]
            score: float
                Prediction score between 0 and 1
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            segmentation: List[List]
                [
                    [x1, y1, x2, y2, x3, y3, ...],
                    [x1, y1, x2, y2, x3, y3, ...],
                    ...
                ]
            shift_amount: list
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        self.score = PredictionScore(score)
        shift = shift_amount if shift_amount is not None else [0, 0]
        self.bbox = BoundingBox(box=bbox or [0, 0, 0, 0], shift_amount=(shift[0], shift[1]))
        self.category = Category(id=category_id, name=category_name)
        self.mask = None
        self.full_shape = full_shape

    def get_shifted_object_prediction(self):
        """Returns shifted version ObjectPrediction.

        Shifts bbox and mask coords. Used for mapping sliced predictions over full image.
        """
        if self.mask:
            shifted_mask = self.mask.get_shifted_mask()
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                score=self.score.value,
                segmentation=shifted_mask.segmentation,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=shifted_mask.full_shape,
            )
        else:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                score=self.score.value,
                segmentation=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=None,
            )

    def __repr__(self):
        return f"""ObjectPrediction<
    bbox: {self.bbox},
    mask: {self.mask},
    score: {self.score},
    category: {self.category}>"""


class PredictionResult:
    def __init__(
        self,
        object_prediction_list: list[ObjectPrediction],
        image: Image.Image | str | np.ndarray,
        durations_in_seconds: dict[str, Any] = dict(),
    ):
        self.image: Image.Image = read_image_as_pil(image)
        self.image_width, self.image_height = self.image.size
        self.object_prediction_list: list[ObjectPrediction] = object_prediction_list
        self.durations_in_seconds = durations_in_seconds

    def export_visuals(
        self,
        export_dir: str,
        text_size: float | None = None,
        rect_th: int | None = None,
        hide_labels: bool = False,
        hide_conf: bool = False,
        file_name: str = "prediction_visual",
    ):
        """

        Args:
            export_dir: directory for resulting visualization to be exported
            text_size: size of the category name over box
            rect_th: rectangle thickness
            hide_labels: hide labels
            hide_conf: hide confidence
            file_name: saving name
        Returns:

        """
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        visualize_object_predictions(
            image=np.ascontiguousarray(self.image),
            object_prediction_list=self.object_prediction_list,
            rect_th=rect_th,
            text_size=text_size,
            text_th=None,
            color=None,
            hide_labels=hide_labels,
            hide_conf=hide_conf,
            output_dir=export_dir,
            file_name=file_name,
            export_format="png",
        )
