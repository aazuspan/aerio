import matplotlib.pyplot as plt
from aerio.Photo import Photo


class PhotoCollection:
    def __init__(self, photo_paths, dpi=None, photo_size=None, pixel_size=None):
        self.photos = self._load_photos(
            photo_paths, dpi, photo_size, pixel_size)

    def _load_photos(self, photo_paths, dpi, photo_size, pixel_size):
        """
        Instantiate and return all photos
        """
        photos = [Photo(path, dpi, photo_size, pixel_size)
                  for path in photo_paths]

        return photos

    def crop(self, height=None, width=None):
        """
        Crop all photos in the collection to the same size. If no size is
        provided, the smallest dimensions from all photos will be used.
        """
        if not height:
            height = min([photo.height for photo in self.photos])
        if not width:
            width = min([photo.width for photo in self.photos])

        for photo in self.photos:
            photo.crop(height, width)

    def __getitem__(self, i):
        return self.photos[i]

    def match_histograms(self, reference_index=0):
        """
        Histogram match all photos, using one of the photos as a reference. 
        @param {int, default 0} reference_index The index of the photo to use as reference.
        """
        reference = self.photos[reference_index]

        for photo in self.photos:
            photo._match_histogram(reference)

    def __repr__(self):
        return repr(self.photos)

    def __len__(self):
        return len(self.photos)

    def preview(self, size=(8, 8)):
        _, ax = plt.subplots(nrows=len(self.photos), figsize=size)

        for i, photo in enumerate(self.photos):
            photo.preview(cmap="gray", ax=ax, index=i)

    def locate_fiducials(self, size, kernel_size=None, iterations=4, threshold=False, block_size=999):
        for photo in self.photos:
            photo.fiducials.locate(
                size, kernel_size, iterations, threshold, block_size)
