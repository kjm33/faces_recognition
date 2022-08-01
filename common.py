from pathlib import Path
from PIL import Image
import numpy as np

EXPECTED_SIZE = (224, 224)
FACE_PERCENT = 90
GIRLS = ('noam', 'shira', 'other')


def load_img(img_path:Path):
    img = Image.open(img_path)
    img = img.resize(EXPECTED_SIZE)
    img_array = np.asarray(img, dtype=np.float32)
    return img_array
