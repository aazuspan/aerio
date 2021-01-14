import copy
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import statistics
from skimage.exposure import match_histograms

from aerio.BoundingBoxCollection import BoundingBoxCollection
from aerio.Fiducials import Fiducials
from aerio import utils


# TODO: Allow directly loading cv2.imread images
class Photo:
    def __init__(self, path, dpi=None, photo_size=None, pixel_size=None, dtype=np.uint8):
        self.path = path
        self.img = dtype(cv2.imread(self.path, cv2.IMREAD_GRAYSCALE))
        self.dtype = dtype

        self._dpi = dpi
        self._photo_size = photo_size
        self._pixel_size = pixel_size

        self._fiducials = Fiducials(self.img)

    @property
    def fiducials(self):
        return self._fiducials

    def _is_underdefined(self):
        """
        Check if the photo has at least one of the necessary attributes defined.
        """
        if not any((self._dpi, self._photo_size, self._pixel_size)):
            return True
        return False

    @property
    def dpi(self):
        """
        Return scanning dots per inch.
        """
        if self._is_underdefined():
            return None

        if self._dpi:
            return self._dpi

        # Calculate height and width DPI
        h, w = [self.size[i] / self.photo_size[i] * 25.4 for i in range(2)]

        return statistics.mean((h, w))

    @property
    def photo_size(self):
        """
        Return photo size (height, width) in millimeters.
        """
        if self._is_underdefined():
            return None

        if self._photo_size:
            h, w = self._photo_size

        elif self._dpi:
            h, w = [x / self.dpmm for x in self.size]

        else:
            h, w = [self.pixel_size[i] * self.size[i] for i in range(2)]

        return (h, w)

    @property
    def pixel_size(self):
        """
        Return pixel size (height, width) in millimeters.
        """
        if self._is_underdefined():
            return None

        if self._pixel_size:
            h, w = self._pixel_size

        elif self._dpi:
            h, w = [1 / self.dpmm for i in range(2)]

        else:
            h, w = [self.photo_size[i] / self.size[i] for i in range(2)]

        return (h, w)

    @property
    def dpmm(self):
        """
        Convert dots per inch to dots per millimeter
        """
        if self._is_underdefined():
            return None

        return self.dpi / 25.4

    def __repr__(self):
        """
        Print out photo specs if they're available.
        """
        if not self._is_underdefined():
            height = self.height
            width = self.width
            dpi = round(self.dpi, 4)
            photo_height = round(self.photo_size[0], 4)
            photo_width = round(self.photo_size[1], 4)
            pixel_height = round(self.pixel_size[0], 4)
            pixel_width = round(self.pixel_size[1], 4)
        else:
            height = width = dpi = photo_height = photo_width = pixel_height = pixel_width = "Undefined"

        return f"{self.filename}\n"\
            f"Resolution (px): {height} (H) x {width} (W)\n"\
            f"DPI: {dpi}\n"\
            f"Size (mm): {photo_height} (H) x {photo_width} (W)\n"\
            f"Pixel size (mm): {pixel_height} (H) x {pixel_width} (W)\n"

    @property
    def height(self):
        """
        Return image height in pixels
        """
        return self.img.shape[0]

    @property
    def width(self):
        """
        Return image width in pixels
        """
        return self.img.shape[1]

    @property
    def size(self):
        return (self.height, self.width)

    @property
    def filename(self):
        """
        Return the base filename without extension
        """
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def extension(self):
        """
        Return the file extension
        """
        return os.path.splitext(os.path.basename(self.path))[1]

    def crop(self, height, width, fill=0):
        """
        Crop the image to a new size, anchored at the top left. If the crop size
        is larger than the current image, new pixels will be added with the fill value.
        """
        cropped = np.full((height, width), fill)

        copy_height = min(height, self.img.shape[0])
        copy_width = min(width, self.img.shape[1])

        cropped[0:copy_height, 0:copy_width] = self.img[0:copy_height, 0:copy_width]

        self.img = cropped

    def preview(self, size=(8, 8), cmap="gray", ax=None, index=None):
        if ax is None:
            _, ax = plt.subplots(figsize=size)
            ax.imshow(self.img, cmap=cmap)
        else:
            ax[index].imshow(self.img, cmap=cmap)

        self.fiducials.preview(ax=ax, index=index)

    def _match_histogram(self, reference):
        """
        Match the photo histogram to a reference photo.
        """
        self.img = self.dtype(match_histograms(self.img, reference.img))

    def border_box(self, width):
        """
        Return the bounding box of the image border at a given width
        """
        exterior = [
            [0, 0],
            [self.width, 0],
            [self.width, self.height],
            [0, self.height],
            [0, 0]
        ]

        interior = [
            [width, width],
            [self.width - width, width],
            [self.width - width, self.height - width],
            [width, self.height - width],
            [width, width]
        ]

        return BoundingBoxCollection([exterior + interior], self)

    def fiducial_boxes(self, size):
        """
        Return the bounding boxes around the side fiducials.
        """
        boxes = self.fiducials.get_fiducial_bboxes(size)
        return BoundingBoxCollection(boxes, self)

    def save(self, path, suffix="_processed", dtype=np.uint8):
        out_path = os.path.join(path, self.filename + self.extension)
        out_path = utils.add_suffix(out_path, suffix)

        self.img = dtype(self.img)

        cv2.imwrite(out_path, self.img)
