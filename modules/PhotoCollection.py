from modules.Photo import Photo


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
