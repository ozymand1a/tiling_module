import time

import numpy as np
from tqdm import tqdm

from src.postprocess import PostprocessPredictions
from src.postprocess.nmm import NMMPostprocess
from src.postprocess.nms import NMSPostprocess
from src.postprocess.wbf import WBFPostprocess
from src.slice import slice_image
from src.tile.objects import ObjectPrediction, PredictionResult
from src.utils.cv import read_image_as_pil

POSTPROCESS_NAME_TO_CLASS = {
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "WBF": WBFPostprocess,
}


def filter_predictions(
    object_prediction_list, exclude_classes_by_name, exclude_classes_by_id
):
    return [
        obj_pred
        for obj_pred in object_prediction_list
        if obj_pred.category.name not in (exclude_classes_by_name or [])
        and obj_pred.category.id not in (exclude_classes_by_id or [])
    ]


def get_prediction(
    image,
    detection_model,
    shift_amount: list | None = None,
    full_shape=None,
    postprocess: PostprocessPredictions | None = None,
    verbose: int = 0,
    exclude_classes_by_name: list[str] | None = None,
    exclude_classes_by_id: list[int] | None = None,
) -> PredictionResult:
    """Function for performing prediction for given image using given detection_model.

    Args:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionMode
        shift_amount: List
            To shift the box and mask predictions from sliced image to full
            sized image, should be in the form of [shift_x, shift_y]
        full_shape: List
            Size of the full image, should be in the form of [height, width]
        postprocess: sahi.postprocess.combine.PostprocessPredictions
        verbose: int
            0: no print (default)
            1: print prediction duration
        exclude_classes_by_name: Optional[List[str]]
            None: if no classes are excluded
            List[str]: set of classes to exclude using its/their class label name/s
        exclude_classes_by_id: Optional[List[int]]
            None: if no classes are excluded
            List[int]: set of classes to exclude using one or more IDs
    Returns:
        A dict with fields:
            object_prediction_list: a list of ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    durations_in_seconds = {}
    image_as_pil = read_image_as_pil(image)
    shift_amount = shift_amount or [0, 0]

    time_start = time.perf_counter()
    detection_model.perform_inference(np.ascontiguousarray(image_as_pil))
    durations_in_seconds["prediction"] = time.perf_counter() - time_start

    full_shape = full_shape or [image_as_pil.height, image_as_pil.width]

    # process prediction
    time_start = time.perf_counter()
    # works only with 1 batch
    detection_model.convert_original_predictions(
        shift_amount=shift_amount,
        full_shape=full_shape,
    )
    object_prediction_list: list[ObjectPrediction] = (
        detection_model.object_prediction_list
    )
    object_prediction_list = filter_predictions(
        object_prediction_list, exclude_classes_by_name, exclude_classes_by_id
    )

    # postprocess matching predictions
    if postprocess is not None:
        object_prediction_list = postprocess(object_prediction_list)

    time_end = time.perf_counter() - time_start
    durations_in_seconds["postprocess"] = time_end

    if verbose == 1:
        print("Prediction performed in", durations_in_seconds["prediction"], "seconds.")

    return PredictionResult(
        image=image,
        object_prediction_list=object_prediction_list,
        durations_in_seconds=durations_in_seconds,
    )


def get_sliced_prediction(
    image,
    detection_model=None,
    slice_height: int | None = None,
    slice_width: int | None = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    perform_standard_pred: bool = True,
    postprocess_type: str = "NMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
    merge_buffer_length: int | None = None,
    auto_slice_resolution: bool = True,
    slice_export_prefix: str | None = None,
    slice_dir: str | None = None,
    exclude_classes_by_name: list[str] | None = None,
    exclude_classes_by_id: list[int] | None = None,
    progress_bar: bool = False,
    progress_callback=None,
) -> PredictionResult:
    """Function for slice image + get predicion for each slice + combine predictions in full image.

    Args:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionModel
        slice_height: int
            Height of each slice.  Defaults to ``None``.
        slice_width: int
            Width of each slice.  Defaults to ``None``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        perform_standard_pred: bool
            Perform a standard prediction on top of sliced predictions to increase large object
            detection accuracy. Default: True.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GREEDYNMM' or 'NMS'. Default is 'GREEDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        verbose: int
            0: no print
            1: print number of slices (default)
            2: print number of slices and slice/prediction durations
        merge_buffer_length: int
            The length of buffer for slices to be used during sliced prediction, which is suitable for low memory.
            It may affect the AP if it is specified. The higher the amount, the closer results to the non-buffered.
            scenario. See [the discussion](https://github.com/obss/sahi/pull/445).
        auto_slice_resolution: bool
            if slice parameters (slice_height, slice_width) are not given,
            it enables automatically calculate these params from image resolution and orientation.
        slice_export_prefix: str
            Prefix for the exported slices. Defaults to None.
        slice_dir: str
            Directory to save the slices. Defaults to None.
        exclude_classes_by_name: Optional[List[str]]
            None: if no classes are excluded
            List[str]: set of classes to exclude using its/their class label name/s
        exclude_classes_by_id: Optional[List[int]]
            None: if no classes are excluded
            List[int]: set of classes to exclude using one or more IDs
        progress_bar: bool
            Whether to show progress bar for slice processing. Default: False.
        progress_callback: callable
            A callback function that will be called after each slice is processed.
            The function should accept two arguments: (current_slice, total_slices)
    Returns:
        A Dict with fields:
            object_prediction_list: a list of sahi.prediction.ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """

    durations_in_seconds = {}
    time_start = time.perf_counter()
    slice_image_result = slice_image(
        image=image,
        output_file_name=slice_export_prefix,
        output_dir=slice_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=auto_slice_resolution,
    )
    num_slices = len(slice_image_result)
    durations_in_seconds["slice"] = time.perf_counter() - time_start

    if postprocess_type not in POSTPROCESS_NAME_TO_CLASS:
        raise ValueError(
            f"postprocess_type should be one of {list(POSTPROCESS_NAME_TO_CLASS)} but given as {postprocess_type}"
        )
    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic,
    )

    postprocess_time = 0
    time_start = time.perf_counter()
    num_group = num_slices  # num_batch is 1
    if verbose in (1, 2):
        tqdm.write(f"Performing prediction on {num_slices} slices.")

    slice_iterator = (
        tqdm(range(num_group), desc="Processing slices", total=num_group)
        if progress_bar
        else range(num_group)
    )
    full_shape = [
        slice_image_result.original_image_height,
        slice_image_result.original_image_width,
    ]

    object_prediction_list = []
    for group_ind in slice_iterator:
        prediction_result = get_prediction(
            image=slice_image_result.images[group_ind],
            detection_model=detection_model,
            shift_amount=slice_image_result.starting_pixels[group_ind],
            full_shape=full_shape,
            exclude_classes_by_name=exclude_classes_by_name,
            exclude_classes_by_id=exclude_classes_by_id,
        )
        for object_prediction in prediction_result.object_prediction_list:
            if object_prediction:
                object_prediction_list.append(
                    object_prediction.get_shifted_object_prediction()
                )

        # merge matching predictions during sliced prediction
        if (
            merge_buffer_length is not None
            and len(object_prediction_list) > merge_buffer_length
        ):
            postprocess_time_start = time.time()
            object_prediction_list = postprocess(object_prediction_list)
            postprocess_time += time.time() - postprocess_time_start

        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(group_ind + 1, num_group)

    if num_slices > 1 and perform_standard_pred:
        prediction_result = get_prediction(
            image=image,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=full_shape,
            postprocess=None,
            exclude_classes_by_name=exclude_classes_by_name,
            exclude_classes_by_id=exclude_classes_by_id,
        )
        object_prediction_list.extend(prediction_result.object_prediction_list)

    # merge matching predictions
    if len(object_prediction_list) > 1:
        postprocess_time_start = time.time()
        object_prediction_list = postprocess(object_prediction_list)
        postprocess_time += time.time() - postprocess_time_start

    time_end = time.perf_counter() - time_start
    durations_in_seconds["prediction"] = time_end - postprocess_time
    durations_in_seconds["postprocess"] = postprocess_time

    if verbose == 2:
        for key, label in [
            ("slice", "Slicing"),
            ("prediction", "Prediction"),
            ("postprocess", "Postprocessing"),
        ]:
            print(f"{label} performed in {durations_in_seconds[key]} seconds.")

    return PredictionResult(
        image=image,
        object_prediction_list=object_prediction_list,
        durations_in_seconds=durations_in_seconds,
    )
