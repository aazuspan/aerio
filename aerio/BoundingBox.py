import cv2
import numpy as np


class BoundingBox:
    def __init__(self, coords, collection):
        self.coords = coords
        self.collection = collection

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
    def distance_from_edge(self):
        """
        Calculate the minimum distance to an edge of the container
        """
        x, y = self.centroid
        h, w = self.collection.photo.height, self.collection.photo.width

        return min(x, w - x, y, h - y)

    def preview(self, img, color, line_width):
        cv2.polylines(img, [self.coords], True, color, line_width)
