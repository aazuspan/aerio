import numpy as np

from Fiducial import Fiducial


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

    def _extract_fiducial(self, img, position, size):
        """
        Crop and return a single fiducial from the image.
        """
        # Rotate the image to put the proper fiducial at the top
        img = np.rot90(img, position)

        img_width = img.shape[0]

        top = 0
        bottom = size[0]
        left = img_width // 2 - size[1] // 2
        right = img_width // 2 + size[1] // 2

        crop = img[top:bottom, left:right]

        # Undo the image rotation
        crop = np.rot90(crop, 4 - position)

        return crop

    def _calculate_fiducials(self, img, size):
        """
        Extract, instantiate, and return all four fiducials
        """
        fiducials = []

        for position in [self.TOP, self.RIGHT, self.BOTTOM, self.LEFT]:
            fiducial_crop = self._extract_fiducial(self.img, position, size)
            fiducials.append(Fiducial(fiducial_crop))

        return fiducials
