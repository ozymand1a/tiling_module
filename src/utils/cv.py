from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from loguru import logger


def read_image_as_pil(image: Image.Image | str | np.ndarray, exif_fix: bool = True) -> Image.Image:
    """Loads an image as PIL.Image.Image.

    Args:
        image (Union[Image.Image, str, np.ndarray]): The image to be loaded. It can be an image path or URL (str),
            a numpy image (np.ndarray), or a PIL.Image object.
        exif_fix (bool, optional): Whether to apply an EXIF fix to the image. Defaults to False.

    Returns:
        PIL.Image.Image: The loaded image as a PIL.Image object.
    """
    # https://stackoverflow.com/questions/56174099/how-to-load-images-larger-than-max-image-pixels-with-pil
    Image.MAX_IMAGE_PIXELS = None

    if isinstance(image, Image.Image):
        image_pil = image
    elif isinstance(image, str):
        # read image if str image path is provided
        try:
            image_pil = Image.open(
                BytesIO(requests.get(image, stream=True).content) if str(image).startswith("http") else image
            ).convert("RGB")
            if exif_fix:
                ImageOps.exif_transpose(image_pil, in_place=True)
        except Exception as e:  # handle large/tiff image reading
            logger.error(f"PIL failed reading image with error {e}, trying skimage instead")
            try:
                import skimage.io
            except ImportError:
                raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")
            image_sk = skimage.io.imread(image).astype(np.uint8)
            if len(image_sk.shape) == 2:  # b&w
                image_pil = Image.fromarray(image_sk, mode="1")
            elif image_sk.shape[2] == 4:  # rgba
                image_pil = Image.fromarray(image_sk, mode="RGBA")
            elif image_sk.shape[2] == 3:  # rgb
                image_pil = Image.fromarray(image_sk, mode="RGB")
            else:
                raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")
    elif isinstance(image, np.ndarray):
        # check if image is in CHW format (Channels, Height, Width)
        # heuristic: 3 dimensions, first dim (channels) < 5, last dim (width) > 4
        if image.ndim == 3 and image.shape[0] < 5:  # image in CHW
            if image.shape[2] > 4:
                # convert CHW to HWC (Height, Width, Channels)
                image = np.transpose(image, (1, 2, 0))
        image_pil = Image.fromarray(image)
    else:
        raise TypeError("read image with 'pillow' using 'Image.open()'")
    return image_pil