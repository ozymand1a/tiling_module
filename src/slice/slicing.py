from PIL import Image
from pathlib import Path
import numpy as np
import concurrent.futures
from loguru import logger
from typing import Literal

from src.utils.cv import read_image_as_pil
from src.slice.utils import get_auto_slice_params
from src.slice.objects import SliceImageResult, SlicedImage


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int | None = None,
    slice_width: int | None = None,
    auto_slice_resolution: bool | None = True,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
) -> list[list[int]]:
    """Generate bounding boxes for slicing an image into crops.

    The function calculates the coordinates for each slice based on the provided
    image dimensions, slice size, and overlap ratios. If slice size is not provided
    and auto_slice_resolution is True, the function will automatically determine
    appropriate slice parameters.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool, optional): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these parameters from image resolution and orientation.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0

    if slice_height and slice_width:
        if overlap_height_ratio is not None and overlap_height_ratio >= 1.0:
            raise ValueError("Overlap ratio must be less than 1.0")
        if overlap_width_ratio is not None and overlap_width_ratio >= 1.0:
            raise ValueError("Overlap ratio must be less than 1.0")
        y_overlap = int((overlap_height_ratio if overlap_height_ratio is not None else 0.2) * slice_height)
        x_overlap = int((overlap_width_ratio if overlap_width_ratio is not None else 0.2) * slice_width)
    elif auto_slice_resolution:
        x_overlap, y_overlap, slice_width, slice_height = get_auto_slice_params(height=image_height, width=image_width)
    else:
        raise ValueError("Compute type is not auto and slice width and height are not provided.")

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def slice_image(
    image: str | Image.Image,
    output_file_name: str | None = None,
    output_dir: str | None = None,
    slice_height: int | None = None,
    slice_width: int | None = None,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
    auto_slice_resolution: bool | None = True,
    min_area_ratio: float | None = 0.1,
    out_ext: str | None = None,
    verbose: bool | None = False,
    exif_fix: bool = True,
) -> SliceImageResult:
    """Slice a large image into smaller windows. If output_file_name and output_dir is given, export sliced images.

    Args:
        image (str or PIL.Image): File path of image or Pillow Image to be sliced.
        coco_annotation_list (List[CocoAnnotation], optional): List of CocoAnnotation objects.
        output_file_name (str, optional): Root name of output files (coordinates will
            be appended to this)
        output_dir (str, optional): Output directory
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool, optional): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these params from image resolution and orientation.
        min_area_ratio (float, optional): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix for lossless image formats and png for lossy formats ('.jpg','.jpeg').
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.
        exif_fix (bool): Whether to apply an EXIF fix to the image.

    Returns:
        sliced_image_result: SliceImageResult:
                                sliced_image_list: list of SlicedImage
                                image_dir: str
                                    Directory of the sliced image exports.
                                original_image_size: list of int
                                    Size of the unsliced original image in [height, width]
    """

    # define verboseprint
    verboselog = logger.info if verbose else lambda *a, **k: None

    def _export_single_slice(image: np.ndarray, output_dir: str, slice_file_name: str):
        image_pil = read_image_as_pil(image, exif_fix=exif_fix)
        slice_file_path = str(Path(output_dir) / slice_file_name)
        # export sliced image
        image_pil.save(slice_file_path)
        image_pil.close()  # to fix https://github.com/obss/sahi/issues/565
        verboselog("sliced image path: " + slice_file_path)

    # create outdir if not present
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read image
    image_pil = read_image_as_pil(image, exif_fix=exif_fix)
    verboselog("image.shape: " + str(image_pil.size))

    image_width, image_height = image_pil.size
    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"invalid image size: {image_pil.size} for 'slice_image'.")
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        auto_slice_resolution=auto_slice_resolution,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    n_ims = 0

    # init images and annotations lists
    sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width], image_dir=output_dir)

    image_pil_arr = np.asarray(image_pil)
    # iterate over slices
    for slice_bbox in slice_bboxes:
        n_ims += 1

        # extract image
        tlx = slice_bbox[0]
        tly = slice_bbox[1]
        brx = slice_bbox[2]
        bry = slice_bbox[3]
        image_pil_slice = image_pil_arr[tly:bry, tlx:brx]

        # set image file suffixes
        slice_suffixes = "_".join(map(str, slice_bbox))
        if out_ext:
            suffix = out_ext
        elif hasattr(image_pil, "filename"):
            suffix = Path(getattr(image_pil, "filename")).suffix
            if suffix in IMAGE_EXTENSIONS_LOSSY:
                suffix = ".png"
            elif suffix in IMAGE_EXTENSIONS_LOSSLESS:
                suffix = Path(image_pil.filename).suffix
        else:
            suffix = ".png"

        # set image file name and path
        slice_file_name = f"{output_file_name}_{slice_suffixes}{suffix}"

        # create coco image
        slice_width = slice_bbox[2] - slice_bbox[0]
        slice_height = slice_bbox[3] - slice_bbox[1]

        # create sliced image and append to sliced_image_result
        sliced_image = SlicedImage(
            image=image_pil_slice, starting_pixel=[slice_bbox[0], slice_bbox[1]]
        )
        sliced_image_result.add_sliced_image(sliced_image)

    # export slices if output directory is provided
    if output_file_name and output_dir:
        # Use a context-managed ThreadPoolExecutor for clean shutdown and
        # limit workers based on CPU count to avoid oversubscription.
        max_workers = min(MAX_WORKERS, len(sliced_image_result))
        max_workers = max(1, max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map will schedule tasks and wait for completion when the context exits
            list(
                executor.map(
                    _export_single_slice,
                    sliced_image_result.images,
                    [output_dir] * len(sliced_image_result),
                    sliced_image_result.filenames,
                )
            )

    verboselog(
        "Num slices: " + str(n_ims) + " slice_height: " + str(slice_height) + " slice_width: " + str(slice_width)
    )

    return sliced_image_result
