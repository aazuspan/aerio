import numpy as np

from modules.Fiducial import Fiducial


# This class manages all of the fiducials for a single photo
class Fiducials:
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

    def __init__(self, img, size):
        self.img = img
        self.size = size

        self.top, self.right, self.bottom, self.left = self._calculate_fiducials(
            self.img, self.size)

    def calculate_coordinates(self, kernel_size=None, iterations=4, threshold=False, block_size=999):
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
        Return the corner coordinates of each fiducial (top, right, bottom, left)
        """
        coordinates = []
        for fiducial in [self.top, self.right, self.bottom, self.left]:
            coordinates.append(fiducial.coordinates)

        return coordinates

    def _get_fiducial_bbox(self, img, position, size):
        """
        Calculate the coordinates of the bounding box for a fiducial.
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

        return (top, bottom, left, right)

    def _calculate_fiducials(self, img, size):
        """
        Extract, instantiate, and return all four fiducials
        """
        fiducials = []

        for position in [self.TOP, self.RIGHT, self.BOTTOM, self.LEFT]:
            bbox = self._get_fiducial_bbox(self.img, position, size)
            crop = self.img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            # Top-left corner coordinates for the fiducial
            position = (bbox[0], bbox[2])
            fiducials.append(Fiducial(crop, position))

        return fiducials
