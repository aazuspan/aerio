import cv2
import matplotlib.pyplot as plt
import os
import statistics

from Fiducials import Fiducials


# TODO: Raise warning if fiducial coordinates can't be located. Format coordinates for Agisoft.
class Photo:
    def __init__(self, path, dpi=None, photo_size=None, pixel_size=None):
        self.path = path
        self.img = cv2.imread(self.path, 0)

        self._dpi = dpi
        self._photo_size = photo_size
        self._pixel_size = pixel_size

        self._fiducials = None

        if self._is_underdefined():
            raise AttributeError("This photo is under-defined. You must specify"
                                 " either dpi, photo_size, or pixel_size so that"
                                 " others can be calculated.")

    def calculate_fiducials(self, fiducial_size):
        self._fiducials = Fiducials(self.img, fiducial_size)
        return self._fiducials

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
        return self.dpi / 25.4

    # Print out all of the photo specs
    def __repr__(self):
        return f"""
        {self.filename}
        Resolution (px): {self.height} (H) x {self.width} (W)
        DPI: {round(self.dpi, 4)}
        Size (mm): {round(self.photo_size[0], 4)} (H) x {round(self.photo_size[1], 4)} (W)
        Pixel size (mm): {round(self.pixel_size[0], 4)} (H) x {round(self.pixel_size[1], 4)} (W)
        """

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

    def preview(self, size=(8, 8), cmap="gray"):
        _, ax = plt.subplots(figsize=size)
        ax.imshow(self.img, cmap=cmap)

        if self.fiducials and self.fiducials.coordinates:
            for coord in self.fiducials.coordinates:
                ax.plot(coord[0], coord[1], marker="+",
                        color="yellow", markersize=20)
