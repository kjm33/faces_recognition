from autocrop.autocrop import Cropper, open_file, ImageReadError, check_underexposed, bgr_to_rbg
import cv2
import sys
from pathlib import Path
cwd = Path(__file__).absolute()
sys.path.append(cwd.parent)

from common import EXPECTED_SIZE, FACE_PERCENT


def crop(path_or_array, left, top, right, bottom):
    return CustomCrop().crop(path_or_array, left, top, right, bottom)


class CustomCrop(Cropper):
    def __init__(self, width=EXPECTED_SIZE[0],
                 height=EXPECTED_SIZE[0],
                 face_percent=FACE_PERCENT,
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
