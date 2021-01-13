import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from aerio.BoundingBox import BoundingBox
from aerio import utils


class BoundingBoxCollection:
    def __init__(self, boxes, photo):
        # The photo that contains the bounding boxes
        self.photo = photo

        self.boxes = self._load_boxes(boxes)

    def __add__(self, other):
        """
        Combine two BoundingBoxCollections by joining their boxes
        """
        if isinstance(other, BoundingBoxCollection):
            if self.photo != other.photo:
                raise ValueError(
                    "To combine two BoundingBoxCollections, they must use the same photo.")
            other_boxes = self._load_boxes(other.boxes)
        elif isinstance(other, (list, tuple, np.ndarray)):
            other_boxes = self._load_boxes(other)
        else:
            raise ValueError("A BoundingBoxCollection can only be combined with another BoundingBoxCollection"
                             " or a list of coordinate lists.")

        return BoundingBoxCollection(np.concatenate((self.boxes, other_boxes)), self.photo)

    def __getitem__(self, i):
        return self.boxes[i]

    def __len__(self):
        return len(self.boxes)

    def _load_boxes(self, boxes):
        """
        Take a list of BoundingBox objects or coordinates and convert to a list of BoundingBox objects.
        """
        loaded = []

        for box in boxes:
            if isinstance(box, BoundingBox):
                loaded.append(box)
            elif isinstance(box, (list, tuple, np.ndarray)):
                box = np.squeeze(box)
                loaded.append(BoundingBox(box, self.photo.img))
        return np.array(loaded)

    def generate_mask(self, bg=255, fg=0, dtype=np.uint8):
        """
        Convert the bounding boxes to a raster mask
        """
        mask = self._bbox_to_array(self.boxes, bg, fg, dtype)

        return mask

    def save_mask(self, path, suffix="_mask", bg=255, fg=0, dtype=np.uint8):
        out_path = os.path.join(
            path, self.photo.filename + self.photo.extension)
        out_path = utils.add_suffix(out_path, suffix)

        mask = self.generate_mask(bg, fg, dtype)

        cv2.imwrite(out_path, mask)

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
        filtered_boxes = self.boxes.copy()

        areas = [box.area for box in filtered_boxes]
        filtered_boxes = self._filter_boxes(
            filtered_boxes, min_area, max_area, areas)

        edge_distances = [box.edge_distance for box in filtered_boxes]
        filtered_boxes = self._filter_boxes(
            filtered_boxes, min_edge_distance, max_edge_distance, edge_distances)

        hw_ratios = [box.hw_ratio for box in filtered_boxes]
        filtered_boxes = self._filter_boxes(
            filtered_boxes, min_hw_ratio, max_hw_ratio, hw_ratios)

        self.boxes = filtered_boxes

    def _filter_boxes(self, boxes, min_val, max_val, vals):
        """
        Filter boxes based on a list of box attributes and an allowable range.
        """
        filter_vector = np.greater(vals, min_val) & np.less(vals, max_val)
        return boxes[filter_vector]

    def preview(self, size=(8, 8), line_color=(255, 0, 0), fill_color=(255, 0, 0), line_width=2, line_alpha=1, fill_alpha=0.25):
        """
        Draw fills and outlines of each bounding box. Then blend them onto the photo image and display it.
        """
        img = self.photo.img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        fills = np.zeros(img.shape, dtype=np.uint8)
        lines = np.zeros(img.shape, dtype=np.uint8)

        for box in self.boxes:
            if box:
                box._draw(fills, lines, line_color, fill_color, line_width)

        # Mask the pixels that contain fills or lines
        fill_mask = np.all(fills == fill_color, axis=-1)
        line_mask = np.all(lines == line_color, axis=-1)

        # Blend the fill with the original image
        img[fill_mask] = (fills[fill_mask] *
                          fill_alpha) + (img[fill_mask] * (1 - fill_alpha))

        # Blend the lines with the blended image
        img[line_mask] = (lines[line_mask] *
                          line_alpha) + (img[line_mask] * (1 - line_alpha))

        _, ax = plt.subplots(figsize=size)
        ax.imshow(img)

    def collapse(self, kernel=np.ones((5, 5), np.uint8), iterations=3):
        """
        Use morphological opening to collapse and combine bounding boxes
        """
        mask = self.generate_mask()
        mask_collapsed = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations)

        self.boxes = self._array_to_bbox(mask_collapsed)

    def _bbox_to_array(self, bbox, bg=255, fg=0, dtype=np.uint8):
        """
        Convert a list of Bounding Box objects into a 2D array. Boxes will be
        filled with the foreground value.
        """
        array = np.full(self.photo.img.shape, fill_value=bg, dtype=dtype)

        for box in bbox:
            cv2.fillPoly(array, pts=[box.coords], color=fg, lineType=None)

        return array

    def _array_to_bbox(self, array, bg=255, fg=0):
        """
        Convert an 2D array into a list of Bounding Box objects. The foreground
        values will become the boxes.
        """
        working_array = array.copy()
        # CV2 wants black background with white objects
        working_array[array == bg] = 0
        working_array[array == fg] = 255

        contours, _ = cv2.findContours(
            working_array, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = self._load_boxes(contours)

        return boxes
