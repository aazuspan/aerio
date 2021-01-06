import cv2
import matplotlib.pyplot as plt
import numpy as np

from aerio.BoundingBox import BoundingBox


class BoundingBoxCollection:
    def __init__(self, boxes, photo):
        # The photo that contains the bounding boxes
        self.photo = photo
        self.boxes = self.from_list(boxes)

    def from_list(self, box_list):
        """
        Take a list of lists and convert it to a list of BoundingBox objects
        """
        boxes = []

        for coord in box_list:
            # Remove redundant dimensions
            coord = np.squeeze(coord)
            box = BoundingBox(coord, self)
            boxes.append(box)

        return boxes

    def generate_mask(self, bg=255, fg=0, dtype=np.uint8):
        """
        Convert the bounding boxes to a raster mask
        """
        mask = np.full(self.photo.img.shape, fill_value=bg, dtype=dtype)

        for box in self.boxes:
            cv2.fillPoly(mask, pts=[box.coords], color=fg, lineType=None)

        return mask

    def filter(self, min_area=-np.Inf,
               max_area=np.Inf,
               min_edge_distance=-np.Inf,
               max_edge_distance=np.Inf,
               min_hw_ratio=-np.Inf,
               max_hw_ratio=np.Inf):
        """
        Remove all boxes that don't match specified criteria
        """
        self.filter_area((min_area, max_area))
        self.filter_edge_distance((min_edge_distance, max_edge_distance))
        self.filter_hw_ratio((min_hw_ratio, max_hw_ratio))

    def filter_edge_distance(self, range):
        self.boxes = [box for box in self.boxes if box.distance_from_edge >
                      range[0] and box.distance_from_edge < range[1]]

    def filter_area(self, range):
        self.boxes = [box for box in self.boxes if box.area >
                      range[0] and box.area < range[1]]

    def filter_hw_ratio(self, range):
        self.boxes = [box for box in self.boxes if box.hw_ratio >
                      range[0] and box.hw_ratio < range[1]]

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, i):
        return self.boxes[i]

    def preview(self, size=(8, 8), color=(0, 255, 0), line_width=2):
        img = self.photo.img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Add the boxes to the image
        for box in self.boxes:
            if box:
                box.preview(img, color, line_width)

        _, ax = plt.subplots(figsize=size)
        ax.imshow(img)

    def collapse(self, kernel=np.ones((5, 5), np.uint8), iterations=3):
        """
        Use morphological opening to collapse and combine bounding boxes
        """
        mask = self.generate_mask()
        mask_collapsed = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations)

        contours, _ = cv2.findContours(
            255 - mask_collapsed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        self.boxes = self.from_list(contours)
