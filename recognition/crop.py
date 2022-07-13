from autocrop.autocrop import Cropper, open_file, ImageReadError, check_underexposed, bgr_to_rbg
import cv2


class CustomCrop(Cropper):
    def __init__(self, width=500,
                 height=500,
                 face_percent=50,
                 padding=None,
                 fix_gamma=True):
        super().__init__(width=width, height=height, face_percent=face_percent, padding=padding, fix_gamma=fix_gamma)

    def crop(self, path_or_array, left, top, right, bottom):

        if isinstance(path_or_array, str):
            image = open_file(path_or_array)
        else:
            image = path_or_array

        # Some grayscale color profiles can throw errors, catch them
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = image

        try:
            img_height, img_width = image.shape[:2]
        except AttributeError:
            raise ImageReadError

        x, y, w, h = left, top, right-left, bottom-top
        pos = self._crop_positions(
            img_height,
            img_width,
            x,
            y,
            w,
            h,
        )

        # ====== Actual cropping ======
        image = image[pos[0]: pos[1], pos[2]: pos[3]]

        # Resize
        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_AREA
        )

        # Underexposition
        if self.gamma:
            image = check_underexposed(image, gray)
        return bgr_to_rbg(image)
