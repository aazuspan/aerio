import cv2
import numpy as np


class BoundingBox:
    def __init__(self, coords, img):
        self.coords = coords
        self.img = img

    @property
    def x(self):
        """
        Return a list of x values for all points
        """
        return [coord[0] for coord in self.coords]

    @property
    def y(self):
        """
        Return a list of y values for all points
        """
        return [coord[1] for coord in self.coords]

    @property
    def height(self):
        """
        Return the height of the bounding box
        """
        return max(self.y) - min(self.y)

    @property
    def width(self):
        """
        Return the width of the bounding box
        """
        return max(self.x) - min(self.x)

    @property
    def area(self):
        return self.height * self.width

    @property
    def centroid(self):
        """
        Return the centroid point in (x, y)
        """
        return (np.mean(self.x), np.mean(self.y))

    @property
    def hw_ratio(self):
        """
        Return the height to width ratio 
        """
        return self.height / self.width

    @property
    def edge_distance(self):
        """
        Calculate the minimum distance to an edge of the container
        """
        x, y = self.centroid
        h, w = self.img.shape[0], self.img.shape[1]

        return min(x, w - x, y, h - y)

    @property
    def extent(self):
        """
        Return the extent coordinates of the bounding box (top, bottom, left, right).
        Useful for cropping arrays to the bounding box.
        """
        top = min(self.y)
        bottom = max(self.y)
        left = min(self.x)
        right = max(self.x)

        return (top, bottom, left, right)

    def _draw(self, fills, lines, line_color, fill_color, line_width):
        """
        Draw bounding box fills and lines onto arrays
        """
        cv2.fillPoly(fills, pts=[self.coords],
                     color=fill_color, lineType=None)

        cv2.polylines(lines, [self.coords], True, line_color, line_width)
