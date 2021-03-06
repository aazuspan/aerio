import numpy as np

from aerio.BoundingBox import BoundingBox
from aerio.Fiducial import Fiducial


# This class manages all of the fiducials for a single photo
class Fiducials:
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

    def __init__(self, img):
        self.img = img

        self.fiducials = [None for i in range(4)]

    @property
    def top(self):
        return self.fiducials[0]

    @property
    def right(self):
        return self.fiducials[1]

    @property
    def bottom(self):
        return self.fiducials[2]

    @property
    def left(self):
        return self.fiducials[3]

    def locate(self, size, kernel_size=None, iterations=4, threshold=False, block_size=999):
        """
        Extract fiducials from image and store. Filter fiducials and use corner-finding to locate
        exact fiducial positions.
        """
        self.fiducials = self._crop_fiducials(size)

        return self._calculate_coordinates(kernel_size, iterations, threshold, block_size)

    def preview(self, ax=None, index=None):
        for fiducial in self.fiducials:
            if fiducial:
                fiducial.preview(ax=ax, index=index)

    def get_fiducial_bboxes(self, size):
        """
        Return a list of Bounding Boxes containing the fiducials
        """
        boxes = []

        for position in [self.TOP, self.RIGHT, self.BOTTOM, self.LEFT]:
            bbox = self._get_fiducial_bbox(self.img, position, size)
            boxes.append(bbox)

        return boxes

    def _get_fiducial_bbox(self, img, position, size):
        """
        Generate a bounding box for a fiducial
        """
        img_width = img.shape[0]
        img_height = img.shape[1]

        if position in [self.TOP, self.BOTTOM]:
            left = img_width // 2 - size[1] // 2
            right = img_width // 2 + size[1] // 2

            if position == self.TOP:
                top = 0
                bottom = size[0]
            else:
                top = img_height - size[0]
                bottom = img_height

        elif position in [self.RIGHT, self.LEFT]:
            top = img_height // 2 - size[1] // 2
            bottom = img_height // 2 + size[1] // 2

            if position == self.RIGHT:
                left = img_width - size[0]
                right = img_width
            else:
                left = 0
                right = size[0]

        coords = [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom]
        ]

        return BoundingBox(np.array(coords), self.img)

    def _crop_fiducials(self, size):
        fiducial_boxes = self.get_fiducial_bboxes(size)

        fiducials = []
        for box in fiducial_boxes:
            extent = box.extent
            crop = self.img[extent[0]:extent[1], extent[2]:extent[3]]
            # Top-left corner coordinates for the fiducial
            corner = (extent[0], extent[2])
            fiducials.append(Fiducial(crop, corner))

        return fiducials

    def _calculate_coordinates(self, kernel_size, iterations, threshold, block_size):
        """
        Run image processing and corner finding to locate coordinates of each fiducial corner
        """
        if not kernel_size:
            # This is a good default size
            kernel_size = self.img.shape[0] // 200

        for fiducial in [self.top, self.right, self.bottom, self.left]:
            fiducial._calculate_coordinates(kernel_size, iterations,
                                            threshold, block_size)

        return self.coordinates

    @property
    def coordinates(self):
        """
        Return the corner coordinates of each fiducial (top, right, bottom, left). Returns None
        for any fiducial that has not been located.
        """
        coordinates = []
        for fiducial in [self.top, self.right, self.bottom, self.left]:
            if not fiducial:
                coordinates.append(None)
            else:
                coordinates.append(fiducial.coordinates)

        return coordinates
