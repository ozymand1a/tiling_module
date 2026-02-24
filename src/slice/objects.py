import numpy as np


class SlicedImage:
    def __init__(self, image: np.ndarray, starting_pixel: list[int]):
        """
        image: np.array
            Sliced image.
        starting_pixel: list of list of int
            Starting pixel coordinates of the sliced image.
        """
        self.image = image
        self.starting_pixel = starting_pixel


class SliceImageResult:
    def __init__(self, original_image_size: list[int], image_dir: str | None = None):
        self.original_image_height = original_image_size[0]
        self.original_image_width = original_image_size[1]
        self.image_dir = image_dir
        self._sliced_image_list: list[SlicedImage] = []

    def add_sliced_image(self, sliced_image: SlicedImage):
        if not isinstance(sliced_image, SlicedImage):
            raise TypeError("sliced_image must be a SlicedImage instance")
        self._sliced_image_list.append(sliced_image)

    @property
    def sliced_image_list(self):
        return self._sliced_image_list

    @property
    def images(self):
        """Returns sliced images as a list of np.ndarray."""
        return [s.image for s in self._sliced_image_list]

    @property
    def starting_pixels(self) -> list[list[int]]:
        """Returns starting pixel coords [x, y] for each slice."""
        return [s.starting_pixel for s in self._sliced_image_list]

    @property
    def filenames(self) -> list[str]:
        """Returns filename for each slice."""
        return [s.coco_image.file_name for s in self._sliced_image_list]

    def _item_at(self, i: int):
        return {
            "image": self.images[i],
            "starting_pixel": self.starting_pixels[i],
            "filename": self.filenames[i],
        }

    def __getitem__(self, i):
        i = i.tolist() if isinstance(i, np.ndarray) else i
        if isinstance(i, int):
            return self._item_at(i)
        if isinstance(i, slice):
            return [self._item_at(j) for j in range(*i.indices(len(self)))]
        if isinstance(i, (tuple, list)):
            return [self._item_at(j) for j in i]
        raise NotImplementedError(f"{type(i)}")

    def __len__(self):
        return len(self._sliced_image_list)
