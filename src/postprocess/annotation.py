from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """BoundingBox represents a rectangular region in 2D space, typically used for object detection annotations.

    Attributes:
        box (Tuple[float, float, float, float]): The bounding box coordinates in the format (minx, miny, maxx, maxy).
            - minx (float): Minimum x-coordinate (left).
            - miny (float): Minimum y-coordinate (top).
            - maxx (float): Maximum x-coordinate (right).
            - maxy (float): Maximum y-coordinate (bottom).
        shift_amount (Tuple[int, int], optional): The amount to shift the bounding box in the x and y directions.
            Defaults to (0, 0).

    !!! example "BoundingBox Usage Example"
        ```python
        bbox = BoundingBox((10.0, 20.0, 50.0, 80.0))
        area = bbox.area
        expanded_bbox = bbox.get_expanded_box(ratio=0.2)
        shifted_bbox = bbox.get_shifted_box()
        coco_format = bbox.to_coco_bbox()
        ```
    """

    box: tuple[float, float, float, float] | list[float]
    shift_amount: tuple[int, int] = (0, 0)

    def __post_init__(self):
        if len(self.box) != 4 or any(coord < 0 for coord in self.box):
            raise ValueError("box must be 4 non-negative floats: [minx, miny, maxx, maxy]")
        if len(self.shift_amount) != 2:
            raise ValueError("shift_amount must be 2 integers: [shift_x, shift_y]")

    @property
    def minx(self):
        return self.box[0]

    @property
    def miny(self):
        return self.box[1]

    @property
    def maxx(self):
        return self.box[2]

    @property
    def maxy(self):
        return self.box[3]

    @property
    def shift_x(self):
        return self.shift_amount[0]

    @property
    def shift_y(self):
        return self.shift_amount[1]

    @property
    def area(self):
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    def get_expanded_box(self, ratio: float = 0.1, max_x: int | None = None, max_y: int | None = None):
        """Returns an expanded bounding box by increasing its size by a given ratio. The expansion is applied equally in
        all directions. Optionally, the expanded box can be clipped to maximum x and y boundaries.

        Args:
            ratio (float, optional): The proportion by which to expand the box size.
                Default is 0.1 (10%).
            max_x (int, optional): The maximum allowed x-coordinate for the expanded box.
                If None, no maximum is applied.
            max_y (int, optional): The maximum allowed y-coordinate for the expanded box.
                If None, no maximum is applied.

        Returns:
            BoundingBox: A new BoundingBox instance representing the expanded box.
        """

        w = self.maxx - self.minx
        h = self.maxy - self.miny
        y_mar = int(h * ratio)
        x_mar = int(w * ratio)
        maxx = min(max_x, self.maxx + x_mar) if max_x else self.maxx + x_mar
        minx = max(0, self.minx - x_mar)
        maxy = min(max_y, self.maxy + y_mar) if max_y else self.maxy + y_mar
        miny = max(0, self.miny - y_mar)
        box: list[float] = [minx, miny, maxx, maxy]
        return BoundingBox(box)

    def to_xywh(self):
        """Returns [xmin, ymin, width, height]

        Returns:
            List[float]: A list containing the bounding box in the format [xmin, ymin, width, height].
        """

        return [self.minx, self.miny, self.maxx - self.minx, self.maxy - self.miny]

    def to_coco_bbox(self):
        """
        Returns the bounding box in COCO format: [xmin, ymin, width, height]

        Returns:
            List[float]: A list containing the bounding box in COCO format.
        """
        return self.to_xywh()

    def to_xyxy(self):
        """
        Returns: [xmin, ymin, xmax, ymax]

        Returns:
            List[float]: A list containing the bounding box in the format [xmin, ymin, xmax, ymax].
        """
        return [self.minx, self.miny, self.maxx, self.maxy]

    def to_voc_bbox(self):
        """
        Returns the bounding box in VOC format: [xmin, ymin, xmax, ymax]

        Returns:
            List[float]: A list containing the bounding box in VOC format.
        """
        return self.to_xyxy()

    def get_shifted_box(self):
        """Returns shifted BoundingBox.

        Returns:
            BoundingBox: A new BoundingBox instance representing the shifted box.
        """
        box = [
            self.minx + self.shift_x,
            self.miny + self.shift_y,
            self.maxx + self.shift_x,
            self.maxy + self.shift_y,
        ]
        return BoundingBox(box)

    def __repr__(self):
        return (
            f"BoundingBox: <{(self.minx, self.miny, self.maxx, self.maxy)}, "
            f"w: {self.maxx - self.minx}, h: {self.maxy - self.miny}>"
        )
