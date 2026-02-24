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
        if getattr(type(value), "__module__", None) == "numpy":
            value = copy.deepcopy(value).tolist()
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
        """Create ObjectPrediction from bbox [minx, miny, maxx, maxy], score, category, optional segmentation."""
        self.score = PredictionScore(score)
        shift = shift_amount if shift_amount is not None else [0, 0]
        self.bbox = BoundingBox(box=bbox or [0, 0, 0, 0], shift_amount=(shift[0], shift[1]))
        self.category = Category(id=category_id, name=category_name)
        self.mask = None
        self.full_shape = full_shape

    def get_shifted_object_prediction(self):
        """Return shifted ObjectPrediction (bbox/mask in full-image coords)."""
        seg, full = (None, None)
        if self.mask:
            shifted_mask = self.mask.get_shifted_mask()
            seg, full = shifted_mask.segmentation, shifted_mask.full_shape
        return ObjectPrediction(
            bbox=self.bbox.get_shifted_box().to_xyxy(),
            category_id=self.category.id,
            score=self.score.value,
            category_name=self.category.name,
            segmentation=seg,
            shift_amount=[0, 0],
            full_shape=full,
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
        durations_in_seconds: dict[str, Any] | None = None,
    ):
        self.image = read_image_as_pil(image)
        self.image_width, self.image_height = self.image.size
        self.object_prediction_list = object_prediction_list
        self.durations_in_seconds = durations_in_seconds or {}

    def export_visuals(
        self,
        export_dir: str,
        text_size: float | None = None,
        rect_th: int | None = None,
        hide_labels: bool = False,
        hide_conf: bool = False,
        file_name: str = "prediction_visual",
    ):
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
